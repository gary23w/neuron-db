#![cfg(feature = "sqlite")]
//! `why` grounds a feeling in its cause — it connects a stance (in `<scope>::stance`) to the facts the
//! base scope holds about that topic, so "why do I feel this?" has an evidence-backed answer.

use neuron_core::db::NeuronDB;

fn tmp(tag: &str) -> String {
    let p = std::env::temp_dir().join(format!("ndb_grounding_{tag}.sqlite"));
    for s in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{}{}", p.display(), s)); }
    p.to_string_lossy().to_string()
}

#[test]
fn why_returns_the_feeling_and_its_grounding_facts() {
    let db = NeuronDB::open(&tmp("why"), 1000);
    // the mind learns some facts about a topic into the base scope
    db.observe("mind", "the climate summit ended without a binding agreement");
    db.observe("mind", "global emissions rose again this year");
    db.observe("mind", "the deploy command is make ship"); // an unrelated fact, should not be evidence
    // and forms a feeling about it (stances live in the ::stance sub-scope)
    db.note_stance("mind::stance", "climate", "quietly furious that nothing changes");

    let w = db.why("mind", "climate").expect("a stance on climate should exist");
    assert!(w.feeling.contains("furious"), "feeling carried through: {}", w.feeling);
    assert!(w.intensity >= 1.0, "intensity accumulated: {}", w.intensity);
    assert!(!w.evidence.is_empty(), "should surface grounding facts");
    assert!(w.evidence.iter().any(|f| f.contains("climate summit")), "evidence is about the topic: {:?}", w.evidence);
    assert!(!w.evidence.iter().any(|f| f.contains("deploy command")), "unrelated facts are not evidence: {:?}", w.evidence);
}

#[test]
fn whitespace_variants_of_a_topic_reinforce_one_stance() {
    let db = NeuronDB::open(&tmp("canon"), 1000);
    db.note_stance("u::stance", "climate change", "worried");
    db.note_stance("u::stance", "climate  change", "more worried"); // double space — must canonicalize to the same topic
    let w = db.why("u", "climate change").expect("stance exists");
    assert!(w.intensity >= 2.0, "spacing variants must accumulate, not fragment: {}", w.intensity);
}

#[test]
fn why_is_none_without_a_stance() {
    let db = NeuronDB::open(&tmp("none"), 1000);
    db.observe("mind", "some fact about widgets");
    assert!(db.why("mind", "widgets").is_none(), "no stance -> no why");
}
