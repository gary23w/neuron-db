#![cfg(feature = "sqlite")]
//! Durable, accumulating stance strength — the affective layer. A stance about a topic
//! intensifies each time it is re-stated (a disposition deepening with repeated exposure),
//! and that accumulated strength PERSISTS across a database reopen (cross-restart / over weeks).
#![cfg(feature = "sqlite")]
use neuron_core::db::NeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_stance_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}
fn rm(p: &str) {
    let _ = std::fs::remove_file(p);
    let _ = std::fs::remove_file(format!("{}-wal", p));
    let _ = std::fs::remove_file(format!("{}-shm", p));
}

#[test]
fn reinforce_prefix_unencodable_stores_nothing() {
    // regression: a too-short stance must report not-stored (0.0, false) and create no phantom episode
    use neuron_core::Neuron;
    let mut n = Neuron::new(500);
    let (s, created) = n.reinforce_prefix("x", "y", 1.0);
    assert_eq!(s, 0.0);
    assert!(!created);
    assert_eq!(n.fact_count(), 0, "no phantom stance episode");
}

#[test]
fn stance_strength_accumulates_and_survives_reopen() {
    let path = tmp();
    {
        let db = NeuronDB::open(&path, 1000);
        let (s1, new1) = db.note_stance("u::stance", "unserialize auth", "I distrust cookie-fed unserialize calls");
        assert!(new1 && (s1 - 1.0).abs() < 1e-6, "first note forms at intensity 1, got {s1}");
        let (s2, new2) = db.note_stance("u::stance", "unserialize auth", "another live CVE proves the pattern is dangerous");
        assert!(!new2 && (s2 - 2.0).abs() < 1e-6, "re-stating intensifies to 2, got {s2}");
        let (s3, _) = db.note_stance("u::stance", "unserialize auth", "and again, six thousand stores exposed");
        assert!((s3 - 3.0).abs() < 1e-6, "intensity 3, got {s3}");
        // accumulation, not duplication: one neuron holds the deepening stance
        assert_eq!(db.stats("u::stance").facts, 1, "stance must accumulate into a single neuron");
        // an unrelated topic is independent
        let (o1, onew) = db.note_stance("u::stance", "rust ergonomics", "I find the borrow checker fair");
        assert!(onew && (o1 - 1.0).abs() < 1e-6);
        assert_eq!(db.stats("u::stance").facts, 2);
    }
    // reopen the SAME file: the accumulated strength must have persisted to disk. The intervening
    // 'rust ergonomics' note decayed 'unserialize auth' once (3.0 * 0.9 = 2.7), so one more note
    // here brings it to ~3.7 — accumulation persists, and so does the decay.
    {
        let db = NeuronDB::open(&path, 1000);
        let (s4, new4) = db.note_stance("u::stance", "unserialize auth", "still true weeks later");
        assert!(!new4, "the stance must already exist after reopen");
        assert!((s4 - 3.7).abs() < 1e-4, "accumulated-then-decayed intensity must survive reopen (expected ~3.7), got {s4}");
    }
    rm(&path);
}

#[test]
fn disposition_colors_mood_then_fades_when_neglected() {
    // a built-up disposition colors affect() once it clears the threshold; revisiting OTHER topics
    // decays the neglected one back below the threshold, so the active "culture" can shift over time.
    let path = tmp();
    let db = NeuronDB::open(&path, 1000);
    for f in ["this matters more than shipping", "still the hill i die on", "non-negotiable"] {
        db.note_stance("agent::stance", "ai safety", f);                  // -> strength 3.0
    }
    assert!(db.affect("agent").contains("ai safety"),
            "a disposition built up over time must color the mood: {}", db.affect("agent"));
    // pour energy into other topics; each new feeling decays the neglected ai-safety stance (0.9x)
    for t in ["coffee", "weather", "sports", "music"] {
        db.note_stance("agent::stance", t, "a passing thought");
        db.note_stance("agent::stance", t, "thought about twice");
    }
    let a = db.affect("agent");
    assert!(!a.contains("ai safety"),
            "a neglected disposition decays below the threshold and stops dominating: {a}");
    rm(&path);
}
