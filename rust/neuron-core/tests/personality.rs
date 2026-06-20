#![cfg(all(feature = "sqlite", feature = "personality"))]
//! End-to-end proof the opt-in personality layer MODULATES behavior — and that, with no persona, the
//! store stays exactly as neutral as before.

use neuron_core::db::NeuronDB;
use neuron_core::persona::{BigFive, Persona};

fn tmp(tag: &str) -> String {
    let p = std::env::temp_dir().join(format!("ndb_persona_test_{tag}.sqlite"));
    for s in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{}{}", p.display(), s)); }
    p.to_string_lossy().to_string()
}

#[test]
fn a_volatile_mind_spikes_harder_than_a_calm_one_on_identical_input() {
    let db = NeuronDB::open(&tmp("react"), 1000);

    let mut volatile = Persona::from_traits(BigFive { neuroticism: 1.0, ..Default::default() });
    volatile.temperament.reactivity = 1.5;
    let calm = Persona::from_traits(BigFive { neuroticism: 0.0, ..Default::default() });

    // the SAME event, felt the same number of times, by two different temperaments
    for _ in 0..3 { db.note_stance_with("volatile", "layoffs", "this is wrong", &volatile); }
    for _ in 0..3 { db.note_stance_with("calm", "layoffs", "this is wrong", &calm); }

    let v = db.note_stance_with("volatile", "layoffs", "this is wrong", &volatile).0;
    let c = db.note_stance_with("calm", "layoffs", "this is wrong", &calm).0;
    assert!(v > c * 2.0, "a neurotic/reactive mind must accumulate a far stronger disposition: {v} vs {c}");
}

#[test]
fn the_persona_colors_the_directive_voice() {
    let db = NeuronDB::open(&tmp("voice"), 1000);
    let outgoing_warm = Persona::from_traits(BigFive { extraversion: 1.0, agreeableness: 1.0, neuroticism: 1.0, ..Default::default() });
    let reserved_calm = Persona::from_traits(BigFive { extraversion: 0.0, agreeableness: 0.0, neuroticism: 0.0, ..Default::default() });

    let d1 = db.affect_with("u1", None, &outgoing_warm);
    assert!(d1.contains("animated") && d1.contains("warm") && d1.contains("run hot"), "{d1}");
    let d2 = db.affect_with("u2", None, &reserved_calm);
    assert!(d2.contains("spare") && d2.contains("blunt") && d2.contains("even-keeled"), "{d2}");
}

#[test]
fn persona_round_trips_through_the_store() {
    let db = NeuronDB::open(&tmp("persist"), 1000);
    let mut p = Persona::from_traits(BigFive { openness: 0.8, conscientiousness: 0.2, extraversion: 0.7, agreeableness: 0.9, neuroticism: 0.6 });
    p.temperament.reactivity = 1.3;
    p.values = vec![("fairness".into(), 0.9)];
    db.set_persona("agent", &p);

    let back = db.get_persona("agent").expect("persona should load");
    assert_eq!(back, p, "a persona must survive set -> durable -> get");
    assert!(db.get_persona("nobody").is_none(), "a scope with no persona returns None");
}

#[test]
fn without_a_persona_the_store_is_unchanged() {
    let db = NeuronDB::open(&tmp("neutral"), 1000);
    // base stance bump is 1.0; a single neutral note must give exactly that, no persona involved
    let (intensity, created) = db.note_stance("plain", "topic", "a feeling");
    assert!(created && (intensity - 1.0).abs() < 1e-6, "neutral note_stance must use the base bump: {intensity}");
    // and the neutral directive carries no persona "voice" line
    let d = db.affect("plain", None);
    assert!(!d.contains("Your voice is") && !d.contains("run hot") && !d.contains("even-keeled"), "{d}");
}
