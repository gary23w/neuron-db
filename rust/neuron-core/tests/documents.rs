#![cfg(all(feature = "sqlite", feature = "semantic"))]
//! The "document storage" pattern (the chat-lab's design): each pasted document goes into its
//! OWN scope (e.g. `s__doc1`, `s__doc2`). These tests pin the storage behavior that pattern
//! relies on — per-document isolation, in-scope recall, independent forget — and a sanity bound
//! on the resident footprint (the semantic space is shared and vocab-bound, NOT per-document).
use neuron_core::db::NeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_docs_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}

#[test]
fn documents_isolated_per_scope() {
    let db = NeuronDB::open(&tmp(), 1_000_000);
    db.observe_many("s__doc1", &[
        "The aurora module compresses telemetry before upload.".to_string(),
        "The aurora retention window is ninety days.".to_string()]);
    db.observe_many("s__doc2", &[
        "The beacon scheduler batches jobs every minute.".to_string(),
        "The beacon retry limit is five attempts.".to_string()]);

    // storage is isolated: each scope holds only its own facts
    assert_eq!(db.stats("s__doc1").facts, 2);
    assert_eq!(db.stats("s__doc2").facts, 2);

    // recall works within each document's scope
    assert!(db.recall("s__doc1", "what does the aurora module do?").is_some());
    assert!(db.recall("s__doc2", "what is the beacon retry limit?").is_some());

    // isolation: aurora content cannot be recalled from doc2's scope
    let cross = db.recall("s__doc2", "aurora telemetry compression");
    assert!(cross.is_none() || !cross.unwrap().fact.to_lowercase().contains("aurora"));

    // forgetting one document leaves the other intact
    db.forget("s__doc1", None);
    assert_eq!(db.stats("s__doc1").facts, 0);
    assert_eq!(db.stats("s__doc2").facts, 2);
    assert!(db.recall("s__doc2", "beacon scheduler").is_some());
}

#[test]
fn many_documents_each_recall_independently() {
    let db = NeuronDB::open(&tmp(), 5_000_000);
    for d in 0..30 {
        let doc: Vec<String> = (0..10).map(|i|
            format!("Document {d} sentence {i}: the svc{d} component processes the payload and writes the ledger.")).collect();
        assert_eq!(db.observe_many(&format!("doc{d}"), &doc), 10);
    }
    // a distinctive token recalls within its own document only
    let r = db.recall("doc17", "what does svc17 do?").unwrap();
    assert!(r.fact.contains("svc17"));
    assert_eq!(db.stats("doc0").facts, 10);
    assert_eq!(db.stats("doc29").facts, 10);
}

#[test]
fn context_stitches_neighbors_in_document_order() {
    let db = NeuronDB::open(&tmp(), 1_000_000);
    let mut doc: Vec<String> = (0..12).map(|i|
        format!("Chapter sentence number {i}: the walkthrough continues with step {i} of the pipeline.")).collect();
    doc[6] = "The kraken surfaced beside the lighthouse at dusk.".to_string();
    db.observe_many("s__doc1", &doc);
    let p = db.recall_context("s__doc1", "kraken lighthouse", 3, 2, 3, false);
    assert_eq!(p.len(), 1, "one passage for one hit region: {:?}", p);
    let pass = &p[0];
    assert_eq!(pass.scope, "s__doc1");
    assert_eq!(pass.start, 4, "window starts before(2) episodes ahead of the hit");
    assert_eq!(pass.facts.len(), 6, "2 before + hit + 3 after");
    assert!(pass.facts[pass.hit_pos].contains("kraken"), "hit_pos marks the matched sentence");
    assert!(pass.facts[0].contains("number 4") && pass.facts[5].contains("number 9"),
        "neighbors come back in insertion (= document) order: {:?}", pass.facts);
}

#[test]
fn context_across_reaches_document_subscopes_without_requoting_hits() {
    let db = NeuronDB::open(&tmp(), 1_000_000);
    db.observe_many("base", &["The hive holds general shared notes about the project.".to_string()]);
    let doc: Vec<String> = (0..10).map(|i|
        format!("Passage line {i} about the migration ritual of the silver herons.")).collect();
    db.observe_many("base__doc-herons", &doc);
    // querying the BASE scope with across reaches the document child
    let p = db.recall_context("base", "silver herons migration ritual", 4, 1, 1, true);
    assert!(!p.is_empty(), "across must reach base__doc-herons");
    assert!(p.iter().all(|x| x.scope == "base__doc-herons"), "hits come from the doc child: {:?}", p);
    // the overlap dedupe: no passage's HIT may fall inside another passage's window
    for (i, a) in p.iter().enumerate() {
        let hit = a.start + a.hit_pos;
        for (j, b) in p.iter().enumerate() {
            if i == j { continue; }
            assert!(!(hit >= b.start && hit < b.start + b.facts.len()),
                "hit {} re-quoted inside window [{}, {})", hit, b.start, b.start + b.facts.len());
        }
    }
}

#[test]
fn assoc_across_reaches_doc_children_and_prefers_seeds() {
    let db = NeuronDB::open(&tmp(), 1_000_000);
    db.observe_many("hive", &["General note: the standup happens at nine.".to_string()]);
    db.observe_many("hive__doc-ops", &[
        "The failover runbook says promote the replica first.".to_string(),
        "After promoting the replica, rotate the pager schedule.".to_string()]);
    let hits = db.recall_assoc_across("hive", "failover replica runbook", 6, 2);
    assert!(!hits.is_empty(), "across assoc must reach the doc child");
    assert_eq!(hits[0].0, "hive__doc-ops");
    assert!(hits[0].1.seed && hits[0].1.fact.contains("runbook"),
        "the direct match ranks first as a seed: {:?}", hits);
}

#[test]
fn scope_page_windows_in_insertion_order() {
    let db = NeuronDB::open(&tmp(), 1_000_000);
    let doc: Vec<String> = (0..25).map(|i|
        format!("Ordered fact number {i} in the long stored document body.")).collect();
    db.observe_many("d", &doc);
    let (total, page) = db.scope_facts_page("d", 0, 10);
    assert_eq!(total, 25);
    assert_eq!(page.len(), 10);
    assert!(page[0].contains("number 0 ") && page[9].contains("number 9 "), "first page in order");
    let (_, p2) = db.scope_facts_page("d", 20, 10);
    assert_eq!(p2.len(), 5, "last partial page clamps to the end");
    assert!(p2[4].contains("number 24"));
    let (t3, p3) = db.scope_facts_page("d", 999, 10);
    assert_eq!(t3, 25);
    assert!(p3.is_empty(), "past-the-end from returns an empty page, total intact");
}

/// The resident footprint must stay BOUNDED: the SQLite store is text-sized and the semantic
/// space is shared + vocabulary-bound, so storing many documents does not multiply the space.
#[cfg(feature = "semantic")]
#[test]
fn doc_storage_footprint_is_bounded() {
    let path = tmp();
    let db = NeuronDB::open(&path, 5_000_000);
    let mut facts = 0;
    for d in 0..25 {
        let doc: Vec<String> = (0..24).map(|i|
            format!("The svc{d} module handles request {d}_{i} by validating input and appending a ledger record.")).collect();
        facts += db.observe_many(&format!("doc{d}"), &doc);
    }
    let (vocab, _tokens, bytes) = db.semantic_stats();
    assert!(facts >= 500, "stored {facts}");
    assert!(vocab > 0);
    // ~1KB/vocab word; for this corpus the shared space must be well under an absurd bound
    assert!(bytes < 64_000_000, "semantic footprint {bytes} bytes is unexpectedly large");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}-wal", path));
    let _ = std::fs::remove_file(format!("{}-shm", path));
}
