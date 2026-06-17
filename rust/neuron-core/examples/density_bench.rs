//! The "1000 books" density + retrieval benchmark.
//!
//! Motivation: the Krapivin/Yao "tiny pointers" result says an OPEN-ADDRESSING hash table
//! degrades as its load factor approaches 1 (the 99%-full parking lot). neuron-db has no such
//! table on the hot path — a scope is an append-only `Vec<Episode>` with a stem->positions
//! inverted index, sitting in a SQLite B-tree. There is no "free slot" to probe for, so there
//! is no load factor to push to 1. This bench measures what actually scales here:
//!
//!   1. density  — bytes on disk per stored fact, as the library grows.
//!   2. retrieval — p50/p99 latency to pull (a) one specific fact, (b) one whole book's worth
//!                  of facts, (c) a broad cue shared by every fact — as total facts go 1k..1M.
//!                  (a) and (b) should stay ~flat (no Yao wall); (c) is the genuine O(N) path.
//!   3. write cost — single observe() into a growing scope is O(scope): an O(S) exact-text
//!                  dedup scan + an O(S) blob re-serialize per write. This is the one real
//!                  "fills up -> slows down" effect, and it is LINEAR, not a probe blowup.
//!   4. disk full — characterizes the SQLite layer's out-of-space mode (SQLITE_FULL), which is
//!                  a hard error, NOT a speed curve. neuron-db's persist() currently .expect()s
//!                  it, so on a full disk a write surfaces as a panic, not graceful degradation.
//!
//! Run: cargo run --release --features sqlite --example density_bench [max_facts]
//!   max_facts caps the top tier (default 1_000_000). A book is modeled as 250 facts.
use neuron_core::db::NeuronDB;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const FACTS_PER_BOOK: usize = 250;

/// distinct 5-char alpha token, one stem each (26^5 = 11.88M ids) — collision-free under the
/// stemmer's 5-6 char truncation, same trick as tests/scale.rs.
fn code(mut x: usize) -> String {
    let mut s = String::with_capacity(5);
    for _ in 0..5 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; }
    s
}
/// A book's shared tag (book index b) — every fact in the book carries it, so a book-cue's
/// postings list is exactly FACTS_PER_BOOK regardless of how many other books exist.
fn book_tag(b: usize) -> String { code(b) }
/// A fact's unique tag (drawn from a disjoint range so it never collides with a book tag).
fn rec_tag(global: usize) -> String { code(2_000_000 + global) }

/// One library fact. Shared content stem across ALL facts: "entry" (the broad O(N) cue).
/// Distinctive stems: the book tag (postings = FACTS_PER_BOOK) and the record tag (postings = 1).
fn fact(b: usize, global: usize) -> String {
    format!("the {} entry of {} is val{}", rec_tag(global), book_tag(b), global)
}

fn tmp(tag: &str) -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_density_{}_{}_{}.db", tag, std::process::id(), n))
        .to_string_lossy().into_owned()
}
fn rm(p: &str) {
    for ext in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{}{}", p, ext)); }
}
fn pct(sorted: &[u128], q: f64) -> u128 {
    if sorted.is_empty() { return 0; }
    sorted[(((sorted.len() - 1) as f64) * q) as usize]
}
/// time `iters` recalls hitting pseudo-random items (no Math.random in std-only land; an LCG
/// over the call index gives a fixed, well-spread sequence), return (p50_us, p99_us).
fn time_recall(db: &NeuronDB, scope: &str, iters: usize, n: usize, mk: impl Fn(usize) -> String) -> (u128, u128) {
    let mut lat = Vec::with_capacity(iters);
    let mut seed = 0x9e3779b9u64;
    for _ in 0..iters {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let pick = (seed >> 33) as usize % n.max(1);
        let q = mk(pick);
        let t = Instant::now();
        std::hint::black_box(db.get(scope, &q));
        lat.push(t.elapsed().as_micros());
    }
    lat.sort_unstable();
    (pct(&lat, 0.50), pct(&lat, 0.99))
}

fn main() {
    let max_facts: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(1_000_000);
    // tiers chosen so the sweep contains clean x10 steps (1k->10k->100k->1M, 50k->500k) and a
    // clean x20 step (50k->1M) — to answer "how does it scale under a factor of 10 or 20?".
    let tiers: Vec<usize> = [1_000usize, 10_000, 50_000, 100_000, 500_000, 1_000_000]
        .into_iter().filter(|&n| n <= max_facts).collect();
    let mut rows: Vec<(usize, u128, u128, u128)> = Vec::new(); // (facts, sel_p50, book_p50, broad_p50)
    println!("== neuron-db density + \"1000 books\" retrieval benchmark ==");
    println!("   (a book = {} facts; SELECTIVE/BOOK cues should stay flat as the library grows;", FACTS_PER_BOOK);
    println!("    BROAD is the genuine O(N) path. No open addressing => no load-factor wall.)\n");

    // ---- 1 + 2: density and retrieval latency across the sweep ----
    println!("{:>9} {:>6} | {:>9} {:>8} | {:>9} {:>9} | {:>9} {:>9} | {:>11}",
             "facts", "books", "disk MB", "B/fact", "sel p50", "sel p99", "book p50", "book p99", "broad p50");
    println!("{}", "-".repeat(104));
    for &n in &tiers {
        let books = n / FACTS_PER_BOOK;
        let path = tmp(&format!("sweep{}", n));
        let db = NeuronDB::open(&path, 5_000_000);
        // one observe_many = one blob write (avoids O(N^2) re-serialize across chunks)
        let facts: Vec<String> = (0..n).map(|g| fact(g / FACTS_PER_BOOK, g)).collect();
        db.observe_many("library", &facts);
        let _ = db.get("library", &rec_tag(0)); // warm the inverted index (one O(N) build)

        let disk = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
            + std::fs::metadata(format!("{}-wal", path)).map(|m| m.len()).unwrap_or(0);

        // (a) one specific fact: cue = one unique record tag -> postings = 1
        let (sp50, sp99) = time_recall(&db, "library", 2_000, n, |i| rec_tag(i));
        // (b) one whole book: cue = one book tag -> postings = FACTS_PER_BOOK
        let (bp50, bp99) = time_recall(&db, "library", 2_000, books, |i| book_tag(i));
        // (c) broad cue shared by EVERY fact -> candidate set = N (the real O(N) path)
        let broad_iters = (50_000_000 / n).clamp(20, 500);
        let (xp50, _) = time_recall(&db, "library", broad_iters, 1, |_| "entry".to_string());

        println!("{:>9} {:>6} | {:>9.2} {:>8.1} | {:>7} us {:>7} us | {:>7} us {:>7} us | {:>9} us",
                 n, books, disk as f64 / 1e6, disk as f64 / n as f64,
                 sp50, sp99, bp50, bp99, xp50);
        rows.push((n, sp50, bp50, xp50));
        drop(db); rm(&path);
    }

    // ---- scaling factor: when the LIBRARY load grows xF, how much does retrieval time grow? ----
    // This is the breakthrough's question in its own vocabulary. Yao (linear): xF load -> xF time.
    // Krapivin (Elastic Hashing): xF load -> ~(log F)^2 more. neuron-db has no load factor at all,
    // so the honest test is the load MULTIPLIER -> time MULTIPLIER for a distinctive vs broad cue.
    println!("\n-- scaling factor (load xF  ->  retrieval-time xT), measured from the p50 columns --");
    println!("   {:>22} | {:>10} | {:>10} | {:>10}", "load step", "selective", "book", "broad");
    let find = |n: usize| rows.iter().find(|r| r.0 == n).copied();
    for &(a, b) in &[(1_000usize, 10_000usize), (10_000, 100_000), (100_000, 1_000_000),
                     (50_000, 500_000), (50_000, 1_000_000)] {
        if let (Some(lo), Some(hi)) = (find(a), find(b)) {
            let f = b / a;
            let t = |x: u128, y: u128| if x == 0 { 0.0 } else { y as f64 / x as f64 };
            println!("   {:>7} -> {:>7} (x{:>2}) | {:>9.2}x | {:>9.2}x | {:>9.2}x",
                     a, b, f, t(lo.1, hi.1), t(lo.2, hi.2), t(lo.3, hi.3));
        }
    }
    println!("   (selective/book ~1x means FLAT under load; broad >Fx means O(N log N), the worst path)");

    // ---- 3: single-observe write cost vs scope size (the linear "fills up -> slower writes") ----
    println!("\n-- single observe() write cost vs current scope size (O(scope): dedup scan + blob rewrite) --");
    {
        let path = tmp("write");
        let db = NeuronDB::open(&path, 5_000_000);
        let mut filled = 0usize;
        for &target in &[1_000usize, 10_000, 100_000, 500_000].iter().copied().filter(|&t| t <= max_facts).collect::<Vec<_>>() {
            // bulk-fill up to `target` cheaply, then measure ONE single observe() at that size
            if target > filled {
                let chunk: Vec<String> = (filled..target).map(|g| fact(g / FACTS_PER_BOOK, g)).collect();
                db.observe_many("hot", &chunk);
                filled = target;
            }
            let mut samples = Vec::new();
            for j in 0..20 {
                let g = filled + j;
                let t = Instant::now();
                db.observe("hot", &fact(g / FACTS_PER_BOOK, g));
                samples.push(t.elapsed().as_micros());
                filled += 1;
            }
            samples.sort_unstable();
            println!("   scope ~{:>7} facts: single observe() median {:>8} us", target, samples[samples.len() / 2]);
        }
        drop(db); rm(&path);
    }

    // ---- 4: disk-full (SQLITE_FULL) failure-mode characterization at the storage layer ----
    println!("\n-- disk-full mode: SQLite is a B-tree, not open addressing -- it does NOT slow down as\n   space runs out, it returns SQLITE_FULL (a hard error). Mirroring db.rs's persist() INSERT: --");
    disk_full_probe();

    println!("\n== done ==");
}

/// Reproduce db.rs's exact persist statement against a SQLite DB capped with PRAGMA
/// max_page_count, so we hit SQLITE_FULL deterministically and portably (no real disk filled).
fn disk_full_probe() {
    use rusqlite::{params, Connection};
    let path = tmp("full");
    let conn = Connection::open(&path).expect("open");
    let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA page_size=4096; PRAGMA max_page_count=32;");
    conn.execute("CREATE TABLE IF NOT EXISTS neurons (id TEXT PRIMARY KEY, facts TEXT NOT NULL DEFAULT '[]', created INTEGER NOT NULL, updated INTEGER NOT NULL, turns INTEGER NOT NULL DEFAULT 0);", []).expect("schema");
    // grow one scope's blob the way persist() does: INSERT ... ON CONFLICT ... SET facts=excluded.facts
    let mut blob = String::from("[");
    let mut rows = 0usize;
    loop {
        blob.push_str("\"the aaaaa entry of bbbbb is val\",");
        let r = conn.execute(
            "INSERT INTO neurons(id,facts,created,updated,turns) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(id) DO UPDATE SET facts=excluded.facts,updated=excluded.updated,turns=excluded.turns",
            params!["library", &blob, 0i64, 0i64, 0i64]);
        match r {
            Ok(_) => rows += 1,
            Err(e) => {
                println!("   wrote {} growing blobs (~{:.0} KB), then: {}", rows, blob.len() as f64 / 1024.0, e);
                println!("   -> db.rs persist() does `.expect(\"save\")` on this exact call, so on a full disk");
                println!("      neuron-db PANICS the write rather than degrading. (Graceful-degradation TODO.)");
                break;
            }
        }
        if rows > 1_000_000 { println!("   cap not reached (unexpected)"); break; }
    }
    drop(conn); rm(&path);
}
