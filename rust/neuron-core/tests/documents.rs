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
