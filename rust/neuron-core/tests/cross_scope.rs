#![cfg(feature = "sqlite")]
//! Cross-scope recall: `recall_many_across` reaches a base scope AND its document sub-scopes
//! (`base__doc1`, …) so "tell me about X" can find a document the user filed under its own scope,
//! while typed sub-scopes (`base::var`, `base::stance`, …) and other users stay isolated.
#![cfg(feature = "sqlite")]
use neuron_core::db::NeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_xscope_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}
fn rm(p: &str) {
    let _ = std::fs::remove_file(p);
    let _ = std::fs::remove_file(format!("{}-wal", p));
    let _ = std::fs::remove_file(format!("{}-shm", p));
}

#[test]
fn recall_across_reaches_documents_but_not_typed_or_other_users() {
    let path = tmp();
    let db = NeuronDB::open(&path, 1000);
    db.observe("user:lab", "i like black cats");
    db.observe("user:lab__doc1", "the maritime cluster groups whale ship boat and sea");
    db.note_stance("user:lab::stance", "whales", "a private feeling about whales");
    db.observe("user:other", "the maritime archive holds whale records");

    // across reaches the document sub-scope
    let hits = db.recall_many_across("user:lab", "what is the maritime cluster of whales", 5);
    assert!(hits.iter().any(|h| h.fact.contains("maritime cluster")),
            "cross-scope recall must reach the document sub-scope");

    // a plain base-scope recall does NOT see the document
    let base = db.recall_many("user:lab", "maritime cluster whale", 5);
    assert!(!base.iter().any(|h| h.fact.contains("maritime cluster")),
            "base-scope recall stays isolated from document sub-scopes");

    // typed ::stance sub-scopes and other users never leak into cross-scope recall
    for h in &hits {
        assert!(!h.fact.contains("private feeling"), "typed sub-scopes must not leak: {}", h.fact);
        assert!(!h.fact.contains("maritime archive"), "other users must not leak: {}", h.fact);
    }
    rm(&path);
}
