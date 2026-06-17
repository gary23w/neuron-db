#![cfg(feature = "sqlite")]
//! The "1000 books" invariant, as a hermetic test. The Krapivin/Yao load-factor wall is a
//! property of OPEN-ADDRESSING hash tables filled toward 100%. neuron-db stores a scope as an
//! append-only `Vec<Episode>` behind a stem->positions inverted index, so retrieval cost tracks
//! the matched POSTINGS, not how "full" anything is. These tests pin that:
//!
//!   - retrieving ONE specific fact out of a large library resolves correctly and stays fast,
//!   - retrieving ONE whole book returns exactly that book's facts and nothing from other books,
//!
//! at a library size (40k facts ~= 160 books) within the range tests/scale.rs already proves
//! collision-free. The heavy 1k..1M sweep + p50/p99 timing lives in examples/density_bench.rs.
use neuron_core::db::NeuronDB;
use std::time::Instant;

const FACTS_PER_BOOK: usize = 250;
const BOOKS: usize = 160; // 40_000 facts — proven collision-free range (see tests/scale.rs)

fn code(mut x: usize) -> String {
    let mut s = String::with_capacity(5);
    for _ in 0..5 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; }
    s
}
fn book_tag(b: usize) -> String { code(b) }
fn rec_tag(global: usize) -> String { code(2_000_000 + global) }
fn fact(b: usize, global: usize) -> String {
    format!("the {} entry of {} is val{}", rec_tag(global), book_tag(b), global)
}

fn tmp(tag: &str) -> String {
    let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_density_test_{}_{}_{}.db", tag, std::process::id(), n))
        .to_string_lossy().into_owned()
}
fn rm(p: &str) { for e in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{}{}", p, e)); } }

fn build(tag: &str) -> (String, NeuronDB) {
    let path = tmp(tag);
    rm(&path);
    let db = NeuronDB::open(&path, 5_000_000);
    let facts: Vec<String> = (0..BOOKS * FACTS_PER_BOOK).map(|g| fact(g / FACTS_PER_BOOK, g)).collect();
    db.observe_many("library", &facts);
    let _ = db.get("library", &rec_tag(0)); // warm the inverted index
    (path, db)
}

/// One specific fact resolves correctly regardless of where it sits in a 40k-fact library — the
/// distinctive cue lights a postings list of ~1, so the answer does not depend on library size.
#[test]
fn retrieve_one_fact_from_the_whole_library() {
    let (path, db) = build("one");
    let n = BOOKS * FACTS_PER_BOOK;
    for &g in &[0usize, 1, n / 2, n - 2, n - 1] {
        let b = g / FACTS_PER_BOOK;
        let hit = db.recall("library", &rec_tag(g)).expect("the unique fact must recall");
        assert!(hit.fact.contains(&rec_tag(g)), "got the wrong fact for record {g}: {}", hit.fact);
        assert!(hit.fact.contains(&book_tag(b)), "fact {g} should carry its book tag {}", book_tag(b));
    }
    rm(&path);
}

/// "Retrieve information based on a single book": a book cue returns exactly that book's facts
/// (postings == FACTS_PER_BOOK) and never leaks a fact from a different book — independent of how
/// many other books share the library.
#[test]
fn retrieve_one_whole_book_is_isolated() {
    let (path, db) = build("book");
    for &b in &[0usize, BOOKS / 2, BOOKS - 1] {
        let hits = db.recall_many("library", &book_tag(b), FACTS_PER_BOOK + 50);
        assert_eq!(hits.len(), FACTS_PER_BOOK, "book {b} should return exactly its {FACTS_PER_BOOK} facts");
        for h in &hits {
            assert!(h.fact.contains(&book_tag(b)), "leaked a fact lacking book {b}'s tag: {}", h.fact);
        }
    }
    rm(&path);
}

/// No load-factor wall: after warming, a selective recall in a 40k-fact library completes far
/// under a deliberately loose ceiling. This is a regression guard against an accidental O(N) or
/// O(N^2) path — NOT a tight micro-benchmark (that lives in examples/density_bench.rs), so the
/// 50 ms bound tolerates debug builds and noisy CI while still catching catastrophic blowups.
#[test]
fn selective_recall_does_not_blow_up_at_scale() {
    let (path, db) = build("flat");
    let n = BOOKS * FACTS_PER_BOOK;
    let mut worst = 0u128;
    for k in 0..200 {
        let g = (k * 199 + 7) % n; // spread across the library
        let t = Instant::now();
        let _ = db.recall("library", &rec_tag(g));
        worst = worst.max(t.elapsed().as_micros());
    }
    assert!(worst < 50_000, "selective recall p-worst {worst} us > 50 ms ceiling — possible O(N) regression");
    rm(&path);
}
