//! stats_bench — the statistics tier's cost sheet and its headline capability, measured.
//!
//! (a) observe-path overhead: durable append rate with the tier LIVE (topic absorb + sampled
//!     scope moments on every write), plus the tier's per-op costs in isolation (absorb,
//!     query fold, axis solve).
//! (b) the capability: a theme buried ~20k facts deep — far beyond the 4000-fact blended
//!     window — recalled by a thematic query through the topic gate, timed; next to a
//!     window-bound query (un-foldable -> fail-open path), timed, for the cost comparison.
//!
//! Run: cargo run --release --features "sqlite semantic fisher topics" --example stats_bench
use neuron_core::db::NeuronDB;
use std::time::Instant;

fn main() {
    let path = std::env::temp_dir().join(format!("ndb_stats_bench_{}.db", std::process::id()));
    let path = path.to_string_lossy().into_owned();
    let _ = std::fs::remove_file(&path);
    let db = NeuronDB::open_with_flush(&path, 100_000, 512);

    // ---- (a) ingest with the tier live: theme A first, then a mountain of filler on top ----
    const THEME: usize = 200;
    const FILLER: usize = 20_000;
    let t0 = Instant::now();
    let theme: Vec<String> = (0..THEME).map(|i| format!("aurora launch note {i}: the rocket engine telemetry looked nominal today")).collect();
    db.observe_many("bench", &theme);
    let fill: Vec<String> = (0..FILLER).map(|i| format!("kitchen log {i}: the chef simmered garlic and basil in the copper pot")).collect();
    db.observe_many("bench", &fill);
    let ingest = t0.elapsed();
    let total = THEME + FILLER;
    println!("ingest      {total} facts with topics+fisher live: {:?}  ({:.0} facts/s)", ingest, total as f64 / ingest.as_secs_f64());

    // ---- the tier's per-op costs in isolation ----
    let (_k, docs, tokens, vocab) = db.topics_stats();
    println!("topic model {docs} docs, {tokens} tokens assigned, vocab {vocab}");
    let t = Instant::now();
    let mut folded = 0usize;
    for i in 0..1000 { if !db.scope_topics("bench", 1, 1).is_empty() { folded += 1; } let _ = i; }
    println!("scope_topics (postings warm): {:.1} µs/call ({folded}/1000 non-empty)", t.elapsed().as_secs_f64() * 1e3);

    // outcome axis: feed both sides from real ops, then time the solve via the public surface
    db.strengthen("bench", "rocket engine telemetry", 0.5);
    let junk: Vec<String> = (0..16).map(|i| format!("scratch note {i}: the zzdead experiment path was abandoned zzdead")).collect();
    db.observe_many("bench", &junk);
    let _ = db.forget("bench", Some("zzdead"));
    let t = Instant::now();
    let ax = db.outcome_axis();
    println!("outcome axis solve+read: {:?}  (armed: {})", t.elapsed(), ax.is_some());
    if let Some((posw, negw, np, nn)) = db.axis_words(5) {
        let p: Vec<&str> = posw.iter().map(|(w, _)| w.as_str()).collect();
        let n: Vec<&str> = negw.iter().map(|(w, _)| w.as_str()).collect();
        println!("axis n+ {np:.1} / n- {nn:.1}   helpful -> {}   harmful -> {}", p.join(" "), n.join(" "));
    }

    // ---- (b) the capability: reach the buried theme, timed ----
    let q_theme = "how did the rocket telemetry look?";
    let t = Instant::now();
    let hits = db.recall_blended("bench", q_theme, 3);
    let gated = t.elapsed();
    let reached = hits.first().map(|h| (h.idx, h.fact.contains("aurora"))).unwrap_or((usize::MAX, false));
    println!("gated blended recall: {:?}  top idx {} aurora-hit {}  (window floor was idx {})", gated, reached.0, reached.1, total - 4000);
    // an un-foldable query exercises the fail-open windowed path — the cost baseline
    let t = Instant::now();
    let _ = db.recall_blended("bench", "what is the boiling point of magma?", 3);
    println!("windowed blended recall (fail-open path): {:?}", t.elapsed());

    let _ = std::fs::remove_file(&path);
}
