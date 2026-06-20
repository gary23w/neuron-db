//! Concurrency benchmark: many threads recalling/observing across INDEPENDENT scope families.
//!
//! NeuronDB shards its in-memory cache + connection by top-level scope family, each shard behind its
//! own lock and its own WAL connection. Threads working on different families therefore never contend,
//! so recall throughput scales with cores instead of serializing on one global mutex. This bench loads
//! a set of tenant scopes, then measures single-thread vs multi-thread aggregate throughput for both
//! cache-hot recall (the read-mostly server profile) and durable observe.
//!
//! Run: cargo run --release --features sqlite --example concurrency_bench

use neuron_core::db::NeuronDB;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

// each call recalls a selective cue inside ONE family; threads split the families disjointly so they
// land on different shards and never contend. Free fn (not a closure) so it is 'static for thread::spawn.
fn recall_run(db: &NeuronDB, fam_lo: usize, fam_hi: usize, n: usize, facts_per: usize) -> u64 {
    let span = (fam_hi - fam_lo).max(1);
    let mut hits = 0u64;
    for q in 0..n {
        let f = fam_lo + (q % span);
        let i = (q * 7) % facts_per;
        if db.recall(&format!("tenant{f}"), &format!("deploy region for service {i}")).is_some() {
            hits += 1;
        }
    }
    hits
}

// independent families write in parallel: the in-memory encode/index work runs concurrently; only the
// brief WAL commit serializes (busy_timeout waits it out), so write throughput still rises with threads.
fn write_run(db: &NeuronDB, fam_lo: usize, fam_hi: usize, writes_per_family: usize) {
    for f in fam_lo..fam_hi {
        let scope = format!("wtenant{f}");
        for i in 0..writes_per_family {
            db.observe(&scope, &format!("event {i} for tenant {f} touched record {}", i % 131));
        }
    }
}

fn main() {
    let base = std::env::temp_dir().join("ndb_concurrency_bench.sqlite");
    let path = base.to_string_lossy().to_string();
    // start clean (also clear any WAL/shm sidecars from a prior run)
    for suffix in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{path}{suffix}")); }

    let families = 64usize; // independent tenants — each maps to its own shard family
    let facts_per = 1500usize;
    let cores = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let total_facts = families * facts_per;

    println!("== neuron-db concurrency benchmark ==");
    println!("cores {cores} ; {families} tenant scopes x {facts_per} facts = {total_facts} facts\n");

    let db = Arc::new(NeuronDB::open(&path, 200_000));

    // ---- preload ----
    let t = Instant::now();
    for f in 0..families {
        let scope = format!("tenant{f}");
        let texts: Vec<String> = (0..facts_per)
            .map(|i| format!("the deploy region for service {i} is zone {}", i % 9))
            .collect();
        db.observe_many(&scope, &texts);
    }
    let pre = t.elapsed().as_secs_f64();
    println!("[preload ] {total_facts} facts in {pre:.2}s  ({:.0} facts/s)", total_facts as f64 / pre);

    // warm the cache so recall is pure in-memory (the server steady state)
    for f in 0..families {
        let _ = db.recall(&format!("tenant{f}"), "deploy region for service 1");
    }

    // ---- recall scaling ----
    let queries = 24_000usize;
    let t = Instant::now();
    let _ = recall_run(&db, 0, families, queries, facts_per);
    let single = queries as f64 / t.elapsed().as_secs_f64();
    println!("\n[recall  ] 1 thread  : {single:>10.0} recalls/s   (baseline)");

    let mut counts: Vec<usize> = vec![2, 4, 8, cores].into_iter().filter(|&c| c > 1 && c <= families).collect();
    counts.sort_unstable();
    counts.dedup();
    for tc in counts {
        let per = queries / tc;
        let t = Instant::now();
        let handles: Vec<_> = (0..tc)
            .map(|ti| {
                let db = Arc::clone(&db);
                let lo = ti * families / tc;
                let hi = (ti + 1) * families / tc;
                thread::spawn(move || recall_run(&db, lo, hi, per, facts_per))
            })
            .collect();
        for h in handles { let _ = h.join(); }
        let agg = (per * tc) as f64 / t.elapsed().as_secs_f64();
        println!("[recall  ] {tc:>2} threads : {agg:>10.0} recalls/s   ({:.1}x)", agg / single);
    }

    // ---- durable write scaling ----
    let writes_per_family = 1_500usize;
    let t = Instant::now();
    write_run(&db, 0, 8, writes_per_family);
    let single_w = (8 * writes_per_family) as f64 / t.elapsed().as_secs_f64();
    println!("\n[observe ] 1 thread  : {single_w:>10.0} durable writes/s   (baseline)");

    let wtc = cores.min(8);
    if wtc > 1 {
        let t = Instant::now();
        let handles: Vec<_> = (0..wtc)
            .map(|ti| {
                let db = Arc::clone(&db);
                // disjoint families, offset past the single-thread set
                let lo = 8 + ti * 4;
                let hi = lo + 4;
                thread::spawn(move || write_run(&db, lo, hi, writes_per_family))
            })
            .collect();
        for h in handles { let _ = h.join(); }
        let agg_w = (wtc * 4 * writes_per_family) as f64 / t.elapsed().as_secs_f64();
        println!("[observe ] {wtc:>2} threads : {agg_w:>10.0} durable writes/s   ({:.1}x)", agg_w / single_w);
    }

    db.flush_all();
    for suffix in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{path}{suffix}")); }
    println!("\ndone.");
}
