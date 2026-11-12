#![cfg(feature = "sqlite")]
//! Keyed facts on the durable store: replace and remove by exact key, across a reopen.
use neuron_core::db::NeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};
fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_keyed_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}

#[test]
fn upsert_and_forget_key_survive_reopen() {
    let path = tmp();
    {
        let db = NeuronDB::open(&path, 500);
        assert_eq!(db.upsert("crm", "customer:1", "customer Dana Reyes of Oakville note see rec12 file"), 1);
        assert_eq!(db.upsert("crm", "customer:12", "customer Omar Haddad of Milton"), 1);
        db.observe("crm", "the office is in Toronto on King Street");
        assert_eq!(db.upsert("crm", "customer:12", "customer Omar Haddad of Burlington"), 1);
        assert_eq!(db.stats("crm").facts, 3);
    }
    let db = NeuronDB::open(&path, 500);
    assert_eq!(db.stats("crm").facts, 3, "the replace is durable, with no duplicate replayed from the log");
    // a fresh upsert after reopen goes through the append log; reopen again to replay it
    assert_eq!(db.upsert("crm", "customer:30", "customer Lena Park of Ajax"), 1);
    drop(db);
    let db = NeuronDB::open(&path, 500);
    assert_eq!(db.upsert("crm", "customer:30", "customer Lena Park of Whitby"), 1, "a key replayed from the log still replaces");
    assert_eq!(db.stats("crm").facts, 4);
    assert_eq!(db.forget_key("crm", "customer:1"), (1, 3), "customer:1 must not match customer:12");
    assert_eq!(db.forget_key("crm", "customer:99"), (0, 3));
    drop(db);
    let db = NeuronDB::open(&path, 500);
    assert_eq!(db.stats("crm").facts, 3);
    assert_eq!(db.forget_key("crm", "customer:12"), (1, 2));
    assert_eq!(db.forget_key("crm", "customer:30"), (1, 1));
    let _ = std::fs::remove_file(&path);
}
