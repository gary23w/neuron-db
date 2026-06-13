#![cfg(feature = "sqlite")]
use neuron_core::db::NeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};
fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_db_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}

#[test]
fn observe_recall_get() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("user:1", "the deploy region is us-west-2");
    assert_eq!(db.get("user:1", "what is the deploy region?").as_deref(), Some("us-west-2"));
}

#[test]
fn turn_statement_then_question() {
    let db = NeuronDB::open(&tmp(), 500);
    let t = db.turn("u", "my plan is pro");
    assert_eq!(t.kind, "ack"); assert!(t.wrote >= 1); assert_eq!(t.facts, 1);
    let q = db.turn("u", "what plan am i on?");
    assert_eq!(q.kind, "recall"); assert!(q.reply.contains("pro"));
}

#[test]
fn scopes_isolated_and_persisted() {
    let path = tmp();
    { let db = NeuronDB::open(&path, 500); db.observe("a", "my color is teal"); db.observe("b", "my color is amber"); }
    let db = NeuronDB::open(&path, 500); // reopen -> durable
    assert_eq!(db.get("a", "what is my color?").as_deref(), Some("teal"));
    assert_eq!(db.get("b", "what is my color?").as_deref(), Some("amber"));
    let mut ids = db.neurons(); ids.sort();
    assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn forget_and_stats() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "the gate code is 4471");
    db.observe("u", "the vault code is 9920");
    assert_eq!(db.stats("u").facts, 2);
    let (forgot, remaining) = db.forget("u", Some("gate"));
    assert_eq!(forgot, 1); assert_eq!(remaining, 1);
    assert_eq!(db.get("u", "what is the vault code?").as_deref(), Some("9920"));
}

#[test]
fn batch_observe_and_recall_many() {
    let db = NeuronDB::open(&tmp(), 10_000);
    let facts: Vec<String> = (0..500).map(|i| format!("the metric{} reading is v{}", i, i)).collect();
    assert_eq!(db.observe_many("s", &facts), 500);
    assert_eq!(db.get("s", "what is the metric314 reading?").as_deref(), Some("v314"));
    db.observe("u", "my plan is pro"); db.observe("u", "my city is Halifax");
    let hits = db.recall_many("u", "my plan and city", 3);
    assert!(hits.len() >= 2, "top-k should return multiple facts");
}
