#![cfg(feature = "sqlite")]
//! Append-log durability: a single observe is one durable INSERT (no whole-blob rewrite), recall reads
//! the in-memory cache, and on reopen the snapshot + replayed log reconstruct every fact — through
//! compaction, capacity eviction, deletes, and batch ingest. Every test reopens a fresh NeuronDB to
//! prove the on-disk state is correct, not just the in-memory cache.
use neuron_core::db::NeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_al_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}
fn rm(p: &str) { for e in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{}{}", p, e)); } }

#[test]
fn single_observe_is_durable_without_flush() {
    let path = tmp();
    {
        let db = NeuronDB::open(&path, 1_000_000);
        db.observe("u", "the wifi password is hunter2 in the office");
        // NO flush_all and NO explicit close — only Drop. The fact is logged immediately, so it survives.
    }
    let db2 = NeuronDB::open(&path, 1_000_000);
    assert!(db2.recall("u", "what is the wifi password?").is_some(), "a single observe must be durable");
    assert_eq!(db2.stats("u").facts, 1);
    rm(&path);
}

#[test]
fn many_observes_replay_from_log() {
    let path = tmp();
    {
        let db = NeuronDB::open(&path, 1_000_000);
        for i in 0..500 { db.observe("u", &format!("server node{i} maps to value tok{i}foo")); }
    }
    let db2 = NeuronDB::open(&path, 1_000_000);
    assert_eq!(db2.stats("u").facts, 500, "every appended fact replays on reopen");
    assert!(db2.recall("u", "what does server node250 map to").is_some());
    rm(&path);
}

#[test]
fn compaction_then_reopen_keeps_all() {
    let path = tmp();
    {
        let db = NeuronDB::open(&path, 1_000_000);
        // well past COMPACT_FLOOR (256) single observes -> at least one log -> snapshot compaction fires
        for i in 0..1200 { db.observe("u", &format!("entity ent{i} holds secret val{i}xz")); }
    }
    let db2 = NeuronDB::open(&path, 1_000_000);
    assert_eq!(db2.stats("u").facts, 1200, "facts survive compaction + reopen");
    assert!(db2.recall("u", "what secret does entity ent1100 hold").is_some());
    rm(&path);
}

#[test]
fn capacity_eviction_resnapshots_correctly() {
    let path = tmp();
    {
        let db = NeuronDB::open(&path, 100); // cap 100 facts/scope -> the front-drain shifts indices
        for i in 0..300 { db.observe("u", &format!("record rec{i} maps to data dat{i}qq")); }
    }
    let db2 = NeuronDB::open(&path, 100);
    assert_eq!(db2.stats("u").facts, 100, "capacity holds after eviction + reopen");
    assert!(db2.recall("u", "rec299").is_some(), "the newest fact survives");
    assert!(db2.recall("u", "rec0").is_none(), "the oldest was evicted");
    rm(&path);
}

#[test]
fn forget_then_observe_persists() {
    let path = tmp();
    {
        let db = NeuronDB::open(&path, 1_000_000);
        db.observe("u", "alpha apple is red and tasty");
        db.observe("u", "beta banana is yellow and sweet");
        db.forget("u", Some("banana")); // a delete must re-snapshot (indices shift)
        db.observe("u", "gamma grape is green and small");
    }
    let db2 = NeuronDB::open(&path, 1_000_000);
    assert_eq!(db2.stats("u").facts, 2, "delete + later observe persist correctly");
    assert!(db2.recall("u", "what color is the apple?").is_some());
    assert!(db2.recall("u", "grape").is_some());
    assert!(db2.recall("u", "banana").is_none(), "the deleted fact stays gone after reopen");
    rm(&path);
}

#[test]
fn var_set_survives_with_appended_facts() {
    let path = tmp();
    {
        let db = NeuronDB::open(&path, 1_000_000);
        db.observe("u", "the project deadline is friday afternoon");
        db.var_set("u", "region", "us-west-2");
        db.var_set("u", "region", "eu-central-1"); // upsert: delete old + write new (re-snapshot)
        db.observe("u", "the api token is zeta nine nine three one");
    }
    let db2 = NeuronDB::open(&path, 1_000_000);
    assert_eq!(db2.var_get("u", "region").as_deref(), Some("eu-central-1"), "the latest var value persists");
    assert!(db2.recall("u", "when is the project deadline?").is_some());
    assert!(db2.recall("u", "what is the api token?").is_some());
    rm(&path);
}

#[test]
fn observe_many_batch_durable() {
    let path = tmp();
    {
        let db = NeuronDB::open(&path, 1_000_000);
        let batch: Vec<String> = (0..2000).map(|i| format!("widget wid{i} costs price{i}usd")).collect();
        db.observe_many("u", &batch);
        // then a few single appends on top of the snapshot, no flush
        for i in 0..10 { db.observe("u", &format!("extra item x{i} tagged late{i}zz")); }
    }
    let db2 = NeuronDB::open(&path, 1_000_000);
    assert_eq!(db2.stats("u").facts, 2010, "batch + later single appends all persist");
    assert!(db2.recall("u", "how much does widget wid1500 cost").is_some());
    assert!(db2.recall("u", "x7").is_some());
    rm(&path);
}
