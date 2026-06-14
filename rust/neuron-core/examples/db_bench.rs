//! NeuronDB (sqlite tier) benchmark. Measures the things that differ from the in-memory
//! core: per-commit write cost, single vs batch ingest, recall latency vs scope size,
//! cache hit vs cold reload, reopen/load cost, and concurrent write throughput.
//!
//! Run: cargo run --release --features sqlite --example db_bench
use neuron_core::db::NeuronDB;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn tmp(tag: &str) -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir()
        .join(format!("ndb_bench_{}_{}_{}.db", tag, std::process::id(), n))
        .to_string_lossy()
        .into_owned()
}
fn rm(path: &str) {
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(format!("{}-wal", path));
    let _ = std::fs::remove_file(format!("{}-shm", path));
}

fn main() {
    println!("== NeuronDB benchmark (sqlite, release) ==\n");

    // 1) single observe() into FRESH scopes (1 fact each) — pure per-commit overhead
    {
        let path = tmp("fresh");
        let db = NeuronDB::open(&path, 10_000);
        let nscopes = 5_000;
        let t = Instant::now();
        for i in 0..nscopes {
            db.observe(&format!("user{}", i), "the deploy region is us-west-2");
        }
        let secs = t.elapsed().as_secs_f64();
        println!("1) single observe(), fresh scope each: {:.0} writes/sec ({} writes in {:.2}s)",
                 nscopes as f64 / secs, nscopes, secs);
        rm(&path);
    }

    // 2) single observe() APPENDING to one scope — exposes the O(blob) re-serialize cost
    {
        let path = tmp("append");
        let db = NeuronDB::open(&path, 1_000_000);
        let block = 2_000;
        let t1 = Instant::now();
        for i in 0..block { db.observe("hot", &format!("the metric{} reading is v{}", i, i)); }
        let s1 = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        for i in block..2 * block { db.observe("hot", &format!("the metric{} reading is v{}", i, i)); }
        let s2 = t2.elapsed().as_secs_f64();
        println!("2) single observe() into ONE growing scope:");
        println!("     first  {} facts: {:.0} writes/sec", block, block as f64 / s1);
        println!("     second {} facts: {:.0} writes/sec  (slowdown shows blob-rewrite cost)",
                 block, block as f64 / s2);
        rm(&path);
    }

    // 3) batch observe_many() into one scope — one save amortizes all appends
    {
        let path = tmp("batch");
        let db = NeuronDB::open(&path, 1_000_000);
        let n = 50_000;
        let facts: Vec<String> = (0..n).map(|i| format!("the metric{} reading is v{}", i, i)).collect();
        let t = Instant::now();
        let wrote = db.observe_many("bulk", &facts);
        let secs = t.elapsed().as_secs_f64();
        println!("3) batch observe_many() {} facts: {:.0} writes/sec ({} stored in {:.3}s)",
                 n, wrote as f64 / secs, wrote, secs);
        rm(&path);
    }

    // 4) recall latency vs scope size: SELECTIVE cue (unique key, index hits ~1 episode)
    //    vs BROAD cue (a word shared by every fact, candidate set = whole scope = O(N)).
    {
        // unique 4-char alpha key per fact (no stem collision); covers up to 26^4 ids.
        fn code(mut x: usize) -> String {
            let mut s = String::new();
            for _ in 0..4 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; }
            s
        }
        for &n in &[1_000usize, 10_000, 50_000] {
            let path = tmp(&format!("recall{}", n));
            let db = NeuronDB::open(&path, 1_000_000);
            let facts: Vec<String> = (0..n).map(|i| format!("the {} signal is v{}", code(i), i)).collect();
            db.observe_many("s", &facts);
            let _ = db.get("s", &format!("what is {}?", code(0))); // warm index
            let iters = 5_000;

            let t = Instant::now();
            for k in 0..iters { std::hint::black_box(db.get("s", &format!("what is {}?", code(k % n)))); }
            let sel = t.elapsed().as_micros() as f64 / iters as f64;

            let t = Instant::now();
            for k in 0..iters { std::hint::black_box(db.get("s", &format!("what is the signal {}?", code(k % n)))); }
            let broad = t.elapsed().as_micros() as f64 / iters as f64;

            println!("4) scope {:>6} facts: selective cue {:>7.2} us/call ({:>6.0}/s) | broad/shared cue {:>8.2} us/call ({:>5.0}/s)",
                     n, sel, 1e6 / sel, broad, 1e6 / broad);
            rm(&path);
        }
    }

    // 5) cache hit vs cold reload (scope evicted from the 256-slot cache)
    {
        let path = tmp("evict");
        let db = NeuronDB::open(&path, 100_000);
        // build a target scope with some facts
        let facts: Vec<String> = (0..200).map(|i| format!("the metric{} reading is v{}", i, i)).collect();
        db.observe_many("target", &facts);
        // warm hit
        let iters = 5_000;
        let t = Instant::now();
        for _ in 0..iters { std::hint::black_box(db.get("target", "what is the metric7 reading?")); }
        let warm = t.elapsed().as_micros() as f64 / iters as f64;
        // cold: evict by touching > cap scopes, then time a single reload-recall, repeated
        let cycles = 200;
        let mut total = 0f64;
        for c in 0..cycles {
            for i in 0..300 { db.observe(&format!("filler{}_{}", c, i), "the filler value is noise"); }
            let t = Instant::now();
            std::hint::black_box(db.get("target", "what is the metric7 reading?"));
            total += t.elapsed().as_micros() as f64;
        }
        println!("5) cache hit: {:.2} us/call ;  cold reload (200 facts from sqlite): {:.1} us/call",
                 warm, total / cycles as f64);
        rm(&path);
    }

    // 6) reopen/load cost: time to open a db and first-recall a scope of N facts
    {
        let n = 50_000;
        let path = tmp("reopen");
        {
            let db = NeuronDB::open(&path, 1_000_000);
            let facts: Vec<String> = (0..n).map(|i| format!("the metric{} reading is v{}", i, i)).collect();
            db.observe_many("s", &facts);
        }
        let t = Instant::now();
        let db = NeuronDB::open(&path, 1_000_000);
        let _ = db.get("s", "what is the metric12345 reading?"); // forces load + index
        let secs = t.elapsed().as_secs_f64();
        println!("6) reopen + first-recall a {}-fact scope (load + index): {:.3}s", n, secs);
        rm(&path);
    }

    // 7) concurrent write throughput (all serialized by the global Mutex)
    {
        let path = tmp("concur");
        let db = Arc::new(NeuronDB::open(&path, 1_000_000));
        let threads = 8;
        let per = 2_000;
        let t = Instant::now();
        let mut handles = Vec::new();
        for tid in 0..threads {
            let db = db.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..per {
                    db.observe(&format!("u{}", tid), &format!("the metric{} reading is v{}", i, i));
                }
            }));
        }
        for h in handles { h.join().unwrap(); }
        let secs = t.elapsed().as_secs_f64();
        let total = threads * per;
        println!("7) concurrent observe, {} threads x {} writes: {:.0} writes/sec (lock-serialized)",
                 threads, per, total as f64 / secs);
        rm(&path);
    }

    // 8) recall_chain: server-side multi-hop. N hops = N microsecond recalls, ZERO model
    //    round-trips -- this is "infinite hops at no LLM cost".
    {
        let path = tmp("chain");
        let db = NeuronDB::open(&path, 1_000_000);
        let depth = 50usize;
        let facts: Vec<String> = (0..depth).map(|i| format!("node{} next is node{}", i, i + 1)).collect();
        db.observe_many("c", &facts);
        let _ = db.recall_chain("c", "node0", &["next".to_string()]); // warm
        for &hops in &[1usize, 5, 10, 20, 50] {
            let p: Vec<String> = std::iter::repeat("next".to_string()).take(hops).collect();
            let iters = 2000;
            let t = Instant::now();
            for _ in 0..iters { std::hint::black_box(db.recall_chain("c", "node0", &p)); }
            let us = t.elapsed().as_micros() as f64 / iters as f64;
            println!("8) recall_chain {:>2} hops: {:>8.2} us total ({:.2} us/hop)", hops, us, us / hops as f64);
        }
        rm(&path);
    }

    println!("\n== done ==");
}
