//! Scaling + incremental-index correctness. The recall hot-path optimization (binary_search +
//! precomputed positions) and the incremental inverted index must NOT change any result;
//! these tests pin the invariants that make "effectively unbounded context" safe.
//!
//! Keys use `code()` (distinct 5-char alpha words) so they don't collide under the stemmer,
//! which truncates to 5-6 chars (near-duplicate keys are a documented, separate limitation).
use neuron_core::Neuron;

/// distinct 5-char alpha key, one stem each (26^5 = 11.8M ids, no stem collision)
fn code(mut x: usize) -> String { let mut s = String::new(); for _ in 0..5 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; } s }

/// The realistic growing-memory pattern: append one fact, then recall it, many times. This
/// exercises the incremental index (each turn adds one fact instead of rebuilding). Every
/// freshly-added fact, and old ones, must still recall correctly.
#[test]
fn interleaved_grow_recall_is_correct() {
    let mut n = Neuron::new(1_000_000);
    for i in 0..2_000 {
        n.observe(&format!("the {} gate is val{}", code(i), i));
        let r = n.recall(&format!("what is the {} gate?", code(i))).expect("fresh fact recalls");
        assert_eq!(r.value, format!("val{}", i), "just-added fact {i}");
    }
    assert_eq!(n.recall(&format!("what is the {} gate?", code(0))).unwrap().value, "val0");
    assert_eq!(n.recall(&format!("what is the {} gate?", code(1500))).unwrap().value, "val1500");
}

/// Incrementally-built index (recalls interleaved with growth) must agree with a single full
/// rebuild over the same facts — proving the incremental path produces identical results.
#[test]
fn incremental_matches_full_rebuild() {
    let facts: Vec<String> = (0..3_000).map(|i| format!("the {} signal is s{}", code(i), i)).collect();
    let mut inc = Neuron::new(1_000_000);
    for (j, f) in facts.iter().enumerate() {
        inc.observe(f);
        if j % 250 == 0 { let _ = inc.recall(&format!("what is the {} signal?", code(0))); } // force incremental adds
    }
    let mut full = Neuron::new(1_000_000);
    for f in &facts { full.observe(f); }
    let _ = full.recall("warm up the index once"); // single full build
    for i in (0..3_000).step_by(137) {
        let q = format!("what is the {} signal?", code(i));
        assert_eq!(inc.recall(&q).map(|r| r.value), full.recall(&q).map(|r| r.value), "mismatch at {q}");
    }
}

/// The per-turn injected block stays bounded (<= k) no matter how large the store grows —
/// this is the flat-context property behind unbounded total memory.
#[test]
fn recall_many_block_is_bounded() {
    let mut n = Neuron::new(1_000_000);
    for i in 0..20_000 { n.observe(&format!("the shared signal is v{}", i)); } // all share the cue
    let block = n.recall_many("the shared signal", 15);
    assert!(block.len() <= 15, "per-turn block {} exceeded k", block.len());
    assert!(!block.is_empty());
}

/// A distinctive-key recall pulls the right value out of a large store (selective recall is
/// the scalable path); both the first and last fact resolve correctly at 50k facts.
#[test]
fn selective_recall_correct_at_scale() {
    let mut n = Neuron::new(1_000_000);
    for i in 0..50_000 { n.observe(&format!("the {} token is tok{}", code(i), i)); }
    assert_eq!(n.recall(&format!("what is the {} token?", code(49_999))).unwrap().value, "tok49999");
    assert_eq!(n.recall(&format!("what is the {} token?", code(0))).unwrap().value, "tok0");
}

/// Removal must invalidate the index (indices shift) so later recalls aren't corrupted by a
/// stale incremental index. Guards the NeuronDB forget() invalidation.
#[cfg(feature = "sqlite")]
#[test]
fn recall_correct_after_forget() {
    use neuron_core::db::NeuronDB;
    let uniq = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("ndb_scale_forget_{}_{}.db", std::process::id(), uniq));
    let p = path.to_string_lossy().into_owned();
    for ext in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{}{}", p, ext)); } // hermetic
    let db = NeuronDB::open(&p, 1_000_000);
    for i in 0..500 { db.observe("s", &format!("the {} key is k{}", code(i), i)); }
    assert_eq!(db.get("s", &format!("what is the {} key?", code(100))), Some("k100".to_string()));
    let (removed, _) = db.forget("s", Some(&code(100))); // remove -> index must invalidate
    assert_eq!(removed, 1);
    db.forget("s", Some(&code(200)));
    // surviving facts still recall correctly through the rebuilt index
    assert_eq!(db.get("s", &format!("what is the {} key?", code(101))), Some("k101".to_string()));
    assert_eq!(db.get("s", &format!("what is the {} key?", code(300))), Some("k300".to_string()));
    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_file(format!("{}-wal", p));
    let _ = std::fs::remove_file(format!("{}-shm", p));
}
