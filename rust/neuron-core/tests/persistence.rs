//! dump/load is the persistence boundary; it now carries per-fact strength. These tests pin the
//! allocation-free parser: strength survives a roundtrip, legacy "flag\ttext" lines load at 1.0,
//! and a numeric trailing WORD in the text is not mistaken for a strength field.
use neuron_core::Neuron;

#[test]
fn dump_load_preserves_strength() {
    let mut n = Neuron::new(1000);
    n.observe("the aurora module compresses telemetry before upload");
    n.reinforce_prefix("riskpattern", "this insecure pattern keeps shipping", 1.0); // create at 1
    n.reinforce_prefix("riskpattern", "and it failed again, worse", 1.0);           // accumulate -> 2
    let max_before = n.episodes.iter().map(|e| e.strength).fold(0.0f32, f32::max);
    assert_eq!(max_before, 2.0);

    let blob = n.dump();
    let n2 = Neuron::load(&blob, 1000);
    assert_eq!(n2.fact_count(), n.fact_count());
    let max_after = n2.episodes.iter().map(|e| e.strength).fold(0.0f32, f32::max);
    assert_eq!(max_after, 2.0, "strength must survive dump/load");
}

#[test]
fn load_handles_legacy_and_numeric_text() {
    // legacy format had no strength field; both facts must load at strength 1.0
    let n = Neuron::load("0\tthe meeting is at 5\n0\tmy plan is pro", 1000);
    assert_eq!(n.fact_count(), 2, "legacy lines must still load");
    assert!(n.episodes.iter().all(|e| e.strength == 1.0), "legacy lines default to strength 1.0");
    // "5" is the value of the first fact, not a strength field — it must remain recallable
    let mut n = n;
    assert!(n.recall("when is the meeting?").is_some());
}
