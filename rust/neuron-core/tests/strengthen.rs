//! Strengthen-only plasticity: the positive mirror of forget. These tests pin the property the
//! desk chat's playbook depends on — outcome feedback can only re-rank facts a scope already
//! learned, it can NEVER mint a memory out of its own key (the recalled-lesson pollution bug:
//! `reinforce` keyed on raw user prompts minted "<prompt>: worked" facts straight into the
//! lesson scope). Also pins strength-aware spreading: reinforced facts genuinely out-rank.
use neuron_core::{Neuron, STRENGTH_CAP};

#[test]
fn strengthen_never_mints() {
    let mut n = Neuron::new(1000);
    // empty scope: nothing to strengthen, nothing appears
    assert_eq!(n.strengthen_matching("deep dive into the repo and write documentation", 1.0), 0);
    assert_eq!(n.fact_count(), 0);
    // non-matching needle over a real fact: zero touched, nothing minted, text intact
    n.observe("the deploy pipeline uploads artifacts to the staging bucket");
    assert_eq!(n.strengthen_matching("scopes_for_coinbase_at_2026.csv", 1.0), 0);
    assert_eq!(n.fact_count(), 1);
    assert!(n.episodes[0].t.contains("staging bucket"));
}

#[test]
fn strengthen_bumps_matching_without_rewriting() {
    let mut n = Neuron::new(1000);
    n.observe("fix: `curl -s https://x` failed (exit code 6) — works as: `curl -sL https://api.x`");
    let before_text = n.episodes[0].t.clone();
    assert_eq!(n.strengthen_matching("works as: `curl -sL", 1.0), 1);
    assert_eq!(n.episodes[0].t, before_text, "strengthen must never rewrite fact text");
    assert!(n.episodes[0].strength > 1.5, "strength must accumulate, got {}", n.episodes[0].strength);
    // repeated feedback saturates at the cap instead of growing unboundedly
    for _ in 0..40 { n.strengthen_matching("works as:", 1.0); }
    assert!(n.episodes[0].strength <= STRENGTH_CAP);
}

#[test]
fn strength_weights_spreading_ranking() {
    let mut n = Neuron::new(1000);
    n.observe("deploy fails with kernel panic on the blue cluster");
    n.observe("deploy fails with kernel panic on the green cluster");
    // untouched scope: default strength 1.0 keeps the pre-existing order (insertion tiebreak)
    let base = n.recall_spreading("deploy kernel panic", 5, 2);
    assert!(base.len() >= 2, "both facts should light up, got {:?}", base);
    assert!(base[0].fact.contains("blue"), "default ranking must be unchanged, got {:?}", base[0].fact);
    // outcome feedback on the green fact flips the ranking — reinforced facts out-rank
    assert_eq!(n.strengthen_matching("green cluster", 1.0), 1);
    let after = n.recall_spreading("deploy kernel panic", 5, 2);
    assert!(after[0].fact.contains("green"), "strengthened fact must lead, got {:?}", after[0].fact);
}

/// The desk pollution shape end-to-end: a lesson scope holds a verified fix; Hebbian feedback
/// arrives keyed on a RAW USER PROMPT (the old bug's key). With strengthen it must be a no-op,
/// and a later failing-command recall must surface only real lessons — never prompt text.
#[test]
fn outcome_feedback_on_raw_prompt_cannot_pollute_lessons() {
    let mut n = Neuron::new(1000);
    n.observe("fix: `findstr features index.html full path` failed (exit code 1) — works as: `findstr features index.html`");
    let touched = n.strengthen_matching(
        "deep dive into the desk folder of this repo and write markdown documentation for every zig source file", 1.0);
    assert_eq!(touched, 0, "a raw prompt key must strengthen nothing");
    assert_eq!(n.fact_count(), 1, "and must never mint");
    let hits = n.recall_spreading("findstr features index.html failed", 8, 3);
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.fact.starts_with("fix:")),
        "only real lessons may surface, got {:?}", hits.iter().map(|h| &h.fact).collect::<Vec<_>>());
}

#[cfg(feature = "sqlite")]
#[test]
fn strengthen_persists_across_reopen() {
    use neuron_core::db::NeuronDB;
    let p = std::env::temp_dir().join(format!("strengthen_reopen_{}.db", std::process::id()));
    let p = p.to_string_lossy().to_string();
    let _ = std::fs::remove_file(&p);
    {
        let d = NeuronDB::open(&p, 1000);
        d.observe("s", "deploy fails with kernel panic on the blue cluster");
        d.observe("s", "deploy fails with kernel panic on the green cluster");
        assert_eq!(d.strengthen("s", "green cluster", 1.0), 1);
    }
    let d = NeuronDB::open(&p, 1000);
    let hits = d.recall_associative("s", "deploy kernel panic", 5, 2);
    assert!(!hits.is_empty());
    assert!(hits[0].fact.contains("green"),
        "strength must survive reopen and keep leading, got {:?}", hits.iter().map(|h| &h.fact).collect::<Vec<_>>());
    let _ = std::fs::remove_file(&p);
}
