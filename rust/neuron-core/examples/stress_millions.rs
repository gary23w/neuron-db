//! Stress + scale benchmark: push neuron-db to millions of facts and large text blocks, measuring the
//! costs a real workload hits — selective recall latency at scale, block recall vs a whole-scope
//! ("markdown dump") read, dump/load serialization throughput, and durable write throughput
//! (single immediate vs batch vs write-behind). Writes a report to %TEMP%/ndb_stress.txt and stdout.
//!
//! Run: cargo run --release --example stress_millions --features sqlite [max_facts]   (default 1_000_000)

use neuron_core::db::NeuronDB;
use neuron_core::Neuron;
use std::fmt::Write as _;
use std::time::Instant;

fn pct(v: &mut Vec<u128>, p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable();
    let i = (((v.len() - 1) as f64) * p).round() as usize;
    v[i] as f64 / 1e3 // -> microseconds
}

// deterministic LCG so runs are comparable without a rand dependency
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n.max(1)
    }
}

// a 6-char base-36 tag — unique per i AND not collapsed by the 6-char stemmer, so selective recall
// hits exactly ONE candidate (the real entity-recall case: names/UUIDs, not collidable counters).
fn tag(i: usize) -> String {
    const D: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let (mut x, mut s) = (i, [b'a'; 6]);
    for k in (0..6).rev() {
        s[k] = D[x % 36];
        x /= 36;
    }
    String::from_utf8(s.to_vec()).unwrap()
}
// each fact carries: a unique cue tag(i), a value v{i}, a TOPIC tag shared by a block of 256 facts
// (the discriminative block query), and "registry" in every fact (the hub / broad-query word).
fn topic_tag(i: usize) -> String { tag(i / 256 + 0x4000_0000) }
fn fact(i: usize) -> String {
    format!("{} resolves to v{} in {} registry", tag(i), i, topic_tag(i))
}

fn main() {
    let max: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let mut rep = String::new();
    macro_rules! out {
        ($($a:tt)*) => {{ let line = format!($($a)*); println!("{}", line); let _ = writeln!(rep, "{}", line); }};
    }

    out!("=== neuron-db stress benchmark (up to {} facts) ===", max);

    // scale ladder up to `max`
    let steps: Vec<usize> = [100_000usize, 500_000, 1_000_000, 2_000_000, 4_000_000]
        .into_iter()
        .filter(|&n| n <= max)
        .collect();

    // ---------- A. selective recall flatness + fill/index throughput ----------
    out!("\n[A] in-memory: fill rate, index build, and selective recall (recall by a unique entity)");
    out!("{:>11}  {:>10}  {:>10}  {:>9}  {:>9}", "facts", "fill k/s", "index ms", "p50 us", "p99 us");
    for &n in &steps {
        let mut neu = Neuron::new(n + 16);
        let t = Instant::now();
        for i in 0..n {
            neu.observe(&fact(i));
        }
        let fill = n as f64 / t.elapsed().as_secs_f64() / 1e3;
        let ti = Instant::now();
        let _ = neu.recall(&tag(n / 2)); // first recall builds the index
        let idx_ms = ti.elapsed().as_secs_f64() * 1e3;
        let mut lat = Vec::with_capacity(3000);
        let mut r = Lcg(0x9E3779B9 ^ n as u64);
        for _ in 0..3000 {
            let q = tag(r.next(n));
            let t2 = Instant::now();
            let _ = neu.recall(&q);
            lat.push(t2.elapsed().as_nanos());
        }
        out!("{:>11}  {:>10.1}  {:>10.1}  {:>9.2}  {:>9.2}", n, fill, idx_ms, pct(&mut lat, 0.5), pct(&mut lat, 0.99));
    }

    // ---------- B. block recall: discriminative (df-gated, flat) vs hub vs whole-scope read ----------
    // A DISCRIMINATIVE query (a topic + the hub word) is df-gated to O(topic) and stays flat; a pure
    // HUB query matches every fact (O(scope)); the "markdown" baseline serializes the WHOLE scope.
    out!("\n[B] block recall: discriminative (flat) vs hub (O scope) vs whole-scope read (markdown)");
    out!("{:>10}  {:>11}  {:>11}  {:>14}  {:>10}  {:>11}", "facts", "disc us", "hub us", "wholescope us", "block KB", "whole KB");
    for &n in &steps {
        let mut neu = Neuron::new(n + 16);
        for i in 0..n {
            neu.observe(&fact(i));
        }
        let _ = neu.recall(&tag(0)); // warm index
        let mut r = Lcg(0x5151 ^ n as u64);
        // discriminative block: a topic (df = 256) + the hub word -> df-gating keeps it O(topic), flat
        let mut disc = Vec::with_capacity(60);
        for _ in 0..60 {
            let q = format!("{} registry", topic_tag(r.next(n)));
            let t = Instant::now();
            let hits = neu.recall_many(&q, 20);
            disc.push(t.elapsed().as_nanos());
            std::hint::black_box(&hits);
        }
        // hub-only block: every fact matches -> O(scope) scoring (the worst case)
        let mut hub = Vec::with_capacity(15);
        for _ in 0..15 {
            let t = Instant::now();
            let hits = neu.recall_many("registry", 20);
            hub.push(t.elapsed().as_nanos());
            std::hint::black_box(&hits);
        }
        let block_bytes: usize = neu.recall_many(&format!("{} registry", topic_tag(0)), 20).iter().map(|h| h.fact.len()).sum();
        // markdown baseline: serialize the whole scope (what a markdown-dump memory injects every turn)
        let mut wd = Vec::with_capacity(8);
        let mut whole_bytes = 0usize;
        for _ in 0..8 {
            let t = Instant::now();
            let blob = neu.dump();
            wd.push(t.elapsed().as_nanos());
            whole_bytes = blob.len();
            std::hint::black_box(&blob);
        }
        out!(
            "{:>10}  {:>11.1}  {:>11.1}  {:>14.0}  {:>10.1}  {:>11.0}",
            n, pct(&mut disc, 0.5), pct(&mut hub, 0.5), pct(&mut wd, 0.5),
            block_bytes as f64 / 1024.0, whole_bytes as f64 / 1024.0
        );
    }

    // ---------- C. dump / load serialization throughput ----------
    out!("\n[C] persistence: dump() + load() throughput and bytes/fact");
    out!("{:>11}  {:>12}  {:>12}  {:>12}", "facts", "dump M/s", "load M/s", "bytes/fact");
    for &n in &steps {
        let mut neu = Neuron::new(n + 16);
        for i in 0..n {
            neu.observe(&fact(i));
        }
        let t = Instant::now();
        let blob = neu.dump();
        let dump_rate = n as f64 / t.elapsed().as_secs_f64() / 1e6;
        let bpf = blob.len() as f64 / n as f64;
        let t = Instant::now();
        let n2 = Neuron::load(&blob, n + 16);
        let load_rate = n as f64 / t.elapsed().as_secs_f64() / 1e6;
        std::hint::black_box(&n2);
        out!("{:>11}  {:>12.2}  {:>12.2}  {:>12.1}", n, dump_rate, load_rate, bpf);
    }

    // ---------- D. durable write throughput: single vs batch vs write-behind ----------
    // The save cost. Each scope is one SQLite blob, so an immediate single observe rewrites the whole
    // blob (and dedups against the scope). Batch + write-behind amortize that.
    out!("\n[D] durable writes: 1000 observes into a scope already holding S facts (writes/sec)");
    out!("{:>11}  {:>14}  {:>14}  {:>16}", "scope S", "single immed", "write-behind", "batch (1 save)");
    let dir = std::env::temp_dir();
    for &pre in &[1_000usize, 10_000, 50_000] {
        if pre > max {
            continue;
        }
        // single immediate (flush_every=1): O(scope) dedup + whole-blob rewrite per observe
        let p1 = dir.join(format!("ndb_stress_s_{pre}.db"));
        let _ = std::fs::remove_file(&p1);
        let db1 = NeuronDB::open(p1.to_str().unwrap(), 5_000_000);
        let seed: Vec<String> = (0..pre).map(fact).collect();
        db1.observe_many("s", &seed);
        let t = Instant::now();
        for i in pre..pre + 1000 {
            db1.observe("s", &fact(i));
        }
        let single = 1000.0 / t.elapsed().as_secs_f64();
        drop(db1);
        let _ = std::fs::remove_file(&p1);

        // write-behind (flush_every large): defers the blob rewrite
        let p2 = dir.join(format!("ndb_stress_wb_{pre}.db"));
        let _ = std::fs::remove_file(&p2);
        let db2 = NeuronDB::open_with_flush(p2.to_str().unwrap(), 5_000_000, 100_000);
        db2.observe_many("s", &seed);
        let t = Instant::now();
        for i in pre..pre + 1000 {
            db2.observe("s", &fact(i));
        }
        let wb = 1000.0 / t.elapsed().as_secs_f64();
        db2.flush_all();
        drop(db2);
        let _ = std::fs::remove_file(&p2);

        // batch: one observe_many of 1000 (single save)
        let p3 = dir.join(format!("ndb_stress_b_{pre}.db"));
        let _ = std::fs::remove_file(&p3);
        let db3 = NeuronDB::open(p3.to_str().unwrap(), 5_000_000);
        db3.observe_many("s", &seed);
        let chunk: Vec<String> = (pre..pre + 1000).map(fact).collect();
        let t = Instant::now();
        db3.observe_many("s", &chunk);
        let batch = 1000.0 / t.elapsed().as_secs_f64();
        drop(db3);
        let _ = std::fs::remove_file(&p3);

        out!("{:>11}  {:>14.0}  {:>14.0}  {:>16.0}", pre, single, wb, batch);
    }

    // ---------- E. large text blocks (the "save a big document / code file" case) ----------
    out!("\n[E] large blocks: ingest a big multi-sentence document, then recall one line from it");
    out!("{:>10}  {:>13}  {:>9}  {:>11}  {:>11}", "doc KB", "ingest MB/s", "facts", "dump MB/s", "recall us");
    for &kb in &[64usize, 256, 1024] {
        let mut doc = String::with_capacity(kb * 1024 + 256);
        let mut i = 0usize;
        while doc.len() < kb * 1024 {
            doc.push_str(&format!("section {t} describes how component part{i} links module unit{i} in the pipeline. ", t = tag(i), i = i));
            i += 1;
        }
        let docbytes = doc.len();
        let mut neu = Neuron::new(2_000_000);
        let t = Instant::now();
        let facts = neu.observe(&doc);
        let ingest = docbytes as f64 / t.elapsed().as_secs_f64() / 1e6;
        let _ = neu.recall("tag0"); // warm index
        let t = Instant::now();
        let blob = neu.dump();
        let dmb = blob.len() as f64 / t.elapsed().as_secs_f64() / 1e6;
        std::hint::black_box(&blob);
        let mut lat = Vec::with_capacity(500);
        let mut r = Lcg(0xABCDEF ^ kb as u64);
        for _ in 0..500 {
            let q = tag(r.next(facts.max(1)));
            let t2 = Instant::now();
            let _ = neu.recall(&q);
            lat.push(t2.elapsed().as_nanos());
        }
        out!("{:>10}  {:>13.1}  {:>9}  {:>11.1}  {:>11.2}", kb, ingest, facts, dmb, pct(&mut lat, 0.5));
    }

    // ---------- report ----------
    let path = dir.join("ndb_stress.txt");
    let _ = std::fs::write(&path, &rep);
    out!("\nreport -> {}", path.display());
}
