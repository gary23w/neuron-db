#![cfg(feature = "sqlite")]
//! Opt-in write-behind (open_with_flush): single observes defer the SQLite blob rewrite for
//! throughput, but recall is unaffected (reads the in-memory cache) and no write is lost — deferred
//! writes are flushed on flush_all(), on Drop, and on LRU eviction. Default (open) stays immediate.
#![cfg(feature = "sqlite")]
use neuron_core::db::NeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_wb_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}
fn rm(p: &str) { for e in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{}{}", p, e)); } }

#[test]
fn write_behind_recall_immediate_and_survives_flush() {
    let path = tmp();
    {
        let db = NeuronDB::open_with_flush(&path, 100_000, 64); // defer: persist every 64 writes
        for i in 0..200 { db.observe("u", &format!("server number {i} holds value alpha{i}")); }
        // recall works NOW (reads the cache), even though most writes are still deferred on disk
        assert!(db.recall("u", "what value does server number 142 hold").is_some(), "recall must work mid-flight");
        db.flush_all();
    }
    // reopen: every flushed write is durable
    let db2 = NeuronDB::open(&path, 100_000);
    assert_eq!(db2.stats("u").facts, 200, "all writes must persist after flush_all");
    assert!(db2.recall("u", "server number 142").is_some());
    rm(&path);
}

#[test]
fn write_behind_flushes_on_drop() {
    let path = tmp();
    {
        let db = NeuronDB::open_with_flush(&path, 100_000, 100_000); // threshold so high nothing auto-persists
        db.observe("u", "the wifi password is hunter2 for the office");
        // no explicit flush — rely on Drop
    } // Drop flushes here
    let db2 = NeuronDB::open(&path, 100_000);
    assert!(db2.recall("u", "what is the wifi password?").is_some(), "Drop must flush deferred writes");
    rm(&path);
}
