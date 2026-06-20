//! Raw per-feature / per-interface metrics for neuron-db, in ONE reproducible harness:
//! creation throughput, recall latency vs scope size, serialized byte-size/density, and the
//! per-tier cost of Plastic / Router / Secure / Semantic. Numbers are machine-dependent;
//! run it where you care: cargo run --release --features "secure semantic" --example metrics
use neuron_core::Neuron;
use neuron_core::plastic::PlasticNeuron;
use neuron_core::router::NeuronRouter;
use neuron_core::secure::SecureNeuronDB;
use neuron_core::semantic::SemanticSpace;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn tmp(tag: &str) -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_metrics_{}_{}_{}.db", tag, std::process::id(), n)).to_string_lossy().into_owned()
}
fn rm(p: &str) { let _ = std::fs::remove_file(p); let _ = std::fs::remove_file(format!("{}-wal", p)); let _ = std::fs::remove_file(format!("{}-shm", p)); }
/// distinct 4-char alpha key, no stem collision (covers 26^4 ids)
fn code(mut x: usize) -> String { let mut s = String::new(); for _ in 0..4 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; } s }

fn main() {
    println!("== neuron-db raw metrics (release, single core) ==\n");

    // ---- core Neuron: creation throughput ----
    {
        let t = Instant::now(); let mut made = 0usize;
        while t.elapsed().as_millis() < 500 {
            let mut n = Neuron::new(500);
            n.observe("the api key is zeta-9931. the plan is pro. the region is us-west-2");
            std::hint::black_box(&n); made += 1;
        }
        println!("[core]    creation: {:.0} neurons/s (3 facts each, in-memory)", made as f64 / 0.5);
    }

    // ---- core recall latency vs scope size (distinct keys, selective cue) ----
    println!("\n[core]    recall latency (distinct keys, warm index):");
    for &n in &[100usize, 500, 1_000, 5_000, 50_000] {
        let mut neu = Neuron::new(2_000_000);
        for i in 0..n { neu.observe(&format!("the {} signal is v{}", code(i), i)); }
        let _ = neu.recall(&format!("what is the {} signal?", code(0))); // warm
        let iters = 20_000usize; let mut hits = 0usize;
        let t = Instant::now();
        for k in 0..iters {
            let q = format!("what is the {} signal?", code(k % n));
            if let Some(r) = neu.recall(&q) { if r.value == format!("v{}", k % n) { hits += 1; } }
        }
        let per = t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
        println!("            N={:>6}: {:>6.2} us/call ({:>8.0}/s)   recall@1 {}/{}", n, per, 1e6 / per, hits, iters);
    }

    // ---- serialized byte-size / density (the "no embedding bytes" claim) ----
    {
        let n = 20_000usize;
        let mut neu = Neuron::new(2_000_000);
        for i in 0..n { neu.observe(&format!("the {} access code is {}{}", code(i), 1000 + i, (b'A' + (i % 26) as u8) as char)); }
        let blob = neu.dump();
        let per = blob.len() as f64 / n as f64;
        println!("\n[density] {} facts -> dump() = {} bytes = {:.1} bytes/fact (flag + text only, NO embedding)", n, blob.len(), per);
        println!("            -> {:.1}M facts/GiB serialized", (1u64 << 30) as f64 / per / 1e6);
    }

    // ---- PlasticNeuron: recall latency (plastic overhead vs plain core) ----
    {
        let n = 500usize;
        let mut p = PlasticNeuron::new(2_000_000, Some(1e9), 8);
        for i in 0..n { p.observe(&format!("the {} signal is v{}", code(i), i)); }
        let _ = p.recall(&format!("what is the {} signal?", code(0)));
        let iters = 20_000usize;
        let t = Instant::now();
        for k in 0..iters { std::hint::black_box(p.recall(&format!("what is the {} signal?", code(k % n)))); }
        let per = t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
        println!("\n[plastic] recall N={}: {:.2} us/call ({:.0}/s)  (strength/decay/Hebbian side effects on)", n, per, 1e6 / per);
    }

    // ---- NeuronRouter: sharded fan-out recall ----
    {
        let n = 5_000usize;
        let mut r = NeuronRouter::new(128);
        for i in 0..n { r.observe(&format!("the {} gate code is v{}", code(i), i)); }
        let _ = r.get(&format!("what is the {} gate code?", code(0)));
        let iters = 20_000usize;
        let t = Instant::now();
        for k in 0..iters { std::hint::black_box(r.get(&format!("what is the {} gate code?", code(k % n)))); }
        let per = t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
        println!("\n[router]  recall over {} facts / {} shards: {:.2} us/call ({:.0}/s)", r.fact_count(), r.shard_count(), per, 1e6 / per);
    }

    // ---- SecureNeuronDB: AES-256-GCM put / get ----
    {
        let path = tmp("secure");
        let v = SecureNeuronDB::open(&path);
        let np = 2_000usize;
        let t = Instant::now();
        for i in 0..np { let _ = v.put("alice", "alice-secret", &format!("the {} password", code(i)), &format!("p{}", i)); }
        let puts = np as f64 / t.elapsed().as_secs_f64();
        let iters = 2_000usize;
        let t = Instant::now();
        for k in 0..iters { std::hint::black_box(v.get("alice", "alice-secret", &format!("what is the {} password?", code(k % np)))); }
        let getp = iters as f64 / t.elapsed().as_secs_f64();
        println!("\n[secure]  put (AES-256-GCM + keyed index): {:.0}/s ;  get (decrypt + recall): {:.0}/s", puts, getp);
        rm(&path);
    }

    // ---- SemanticSpace: train throughput + fuzzy embed/rank ----
    {
        let corpus = "the wifi router connects to the internet over the network. the password unlocks the wifi. \
                      the cat sat on the mat near the warm fire. the dog ran across the green field at dawn. \
                      the ship sailed the deep blue sea past the distant rocky shore.".repeat(400);
        let mut s = SemanticSpace::new();
        let words = corpus.split_whitespace().count();
        let t = Instant::now();
        s.train(&corpus);
        let train_s = t.elapsed().as_secs_f64();
        let t = Instant::now(); let iters = 5_000usize;
        for _ in 0..iters { std::hint::black_box(s.embed("the thing that connects me to the internet")); }
        let emb = t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
        println!("\n[semantic] train {:.0} words/s ;  embed a query: {:.2} us ;  vocab {} words", words as f64 / train_s, emb, s.vocab());

        // fuzzy recall over N facts: uncached rank() re-embeds every fact each query; rank_cached()
        // caches fact vectors (lazy) -> warm call pays the embed once, later calls are dot-only.
        let cands: Vec<String> = (0..5_000).map(|i| format!("memo {} on the wifi router and the internet network connection", i)).collect();
        let q = "the wifi router and the internet network";
        let cref: Vec<&str> = cands.iter().map(|s| s.as_str()).collect();   // rank_cached now borrows
        let t = Instant::now(); std::hint::black_box(s.rank(q, &cands)); let uncached = t.elapsed().as_micros();
        let t = Instant::now(); std::hint::black_box(s.rank_cached(q, &cref)); let warm = t.elapsed().as_micros();
        let t = Instant::now(); std::hint::black_box(s.rank_cached(q, &cref)); let cached = t.elapsed().as_micros();
        println!("[semantic] fuzzy rank over {} facts: rank() {} us | rank_cached warm {} us | rank_cached CACHED {} us  (cache {} facts, {:.1} MB)",
                 cands.len(), uncached, warm, cached, s.cached_embeddings(), s.bytes() as f64 / 1e6);
    }

    println!("\n== done ==");
}
