#![cfg(all(feature = "sqlite", feature = "semantic"))]
//! NeuronDB semantic fallback: paraphrase recall that shares no words with the stored fact.
#![cfg(all(feature = "sqlite", feature = "semantic"))]
use neuron_core::db::NeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};
fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_sem_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}
const CORPUS: &str = "
    I use wifi to get online. The wifi connects my laptop to the internet.
    Being online means connected to the internet through wifi or a router.
    The router broadcasts wifi so devices reach the web and browse the internet.
    We browse the web online using the wireless wifi network from the router.
";

#[test]
fn paraphrase_recall_via_semantic_space() {
    let db = NeuronDB::open(&tmp(), 500);
    for _ in 0..30 { db.train_semantic(CORPUS); }   // ground meaning in a corpus
    db.observe("u", "the wifi password is vekam73");
    // query shares NO content words with the stored fact -- only meaning
    let r = db.recall("u", "what is the thing I use to get online?");
    assert!(r.is_some(), "semantic fallback should resolve the paraphrase");
    assert_eq!(r.unwrap().value, "vekam73");
}

#[test]
fn lexical_still_preferred_and_exact() {
    let db = NeuronDB::open(&tmp(), 500);
    for _ in 0..20 { db.train_semantic(CORPUS); }
    db.observe("u", "the wifi password is vekam73");
    db.observe("u", "the deploy region is us-west-2");
    // direct lexical question is unaffected by the semantic tier
    assert_eq!(db.get("u", "what is the deploy region?").as_deref(), Some("us-west-2"));
}

#[test]
fn truly_unrelated_query_abstains() {
    let db = NeuronDB::open(&tmp(), 500);
    for _ in 0..20 { db.train_semantic(CORPUS); }
    db.observe("u", "the wifi password is vekam73");
    // nothing about volcanoes in memory or corpus -> no fabrication
    assert!(db.get("u", "what is the boiling point of magma?").is_none());
}
