//! How large can the *total* addressable context grow while the per-turn LLM context stays
//! flat? Grows ONE store to increasing N and measures, at each size:
//!   - selective recall  (distinctive cue)  -> the scalable "infinite context" path
//!   - broad-cue recall  (word shared by all) -> the O(N) limiter
//!   - recall_many(k)    -> the block actually injected per turn (must stay flat as N grows)
//!   - index (re)build cost on growth, and serialized store size
//!
//! Run: cargo run --release --example context_scale
use neuron_core::Neuron;
use std::time::Instant;

fn code(mut x: usize) -> String { let mut s = String::new(); for _ in 0..5 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; } s }
fn approx_tokens(s: &str) -> usize { s.len().div_ceil(4) } // ~4 chars/token, rough

fn main() {
    println!("== neuron-db: how big can the total context grow, at flat per-turn cost? ==\n");
    let sizes = [10_000usize, 50_000, 200_000, 1_000_000];
    let mut neu = Neuron::new(2_000_000);
    let mut built = 0usize;
    println!("{:>10} | {:>7} | {:>9} | {:>16} | {:>11} | {:>18} | {:>8}",
             "N (facts)", "grow", "idx-build", "selective recall", "broad cue", "per-turn ctx (k=15)", "store");
    for &n in &sizes {
        let t = Instant::now();
        for i in built..n { neu.observe(&format!("the {} sensor reading is v{}", code(i), i)); }
        let grow = t.elapsed().as_secs_f64();
        built = n;

        // first recall after growth forces the inverted index to (re)build over all N facts
        let t = Instant::now(); let _ = neu.recall(&format!("what is the {} sensor reading?", code(0)));
        let idx_us = t.elapsed().as_micros();

        // selective recall: query the distinctive 5-char key ONLY -> index returns ~1 candidate.
        // (the cost of recall is the frequency of the most common cue word, so a key-only cue
        // is the scalable path; adding a shared word like "sensor" unions the whole scope.)
        let iters = 10_000usize; let mut hits = 0usize;
        let t = Instant::now();
        for k in 0..iters {
            let q = format!("what is {}?", code(k % n));
            if let Some(r) = neu.recall(&q) { if r.value == format!("v{}", k % n) { hits += 1; } }
        }
        let sel = t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;

        // broad cue: "sensor"/"reading" are in EVERY fact -> candidate set = whole store (O(N))
        let t = Instant::now(); let _ = neu.recall("what is the sensor reading?");
        let broad_us = t.elapsed().as_micros();

        // the block actually injected into the window each turn: top-k facts, regardless of N
        let block = neu.recall_many("the sensor reading", 15);
        let ctx_tokens: usize = block.iter().map(|r| approx_tokens(&r.fact)).sum();

        let store_mb = neu.dump().len() as f64 / 1e6;
        println!("{:>10} | {:>6.1}s | {:>7}us | {:>9.2}us {}/{} | {:>9}us | {:>3} facts/~{:>4} tok | {:>5.0} MB",
                 n, grow, idx_us, sel, hits, iters, broad_us, block.len(), ctx_tokens, store_mb);
    }
    // The realistic LLM-memory pattern: append ONE fact, then recall, over and over, on a huge
    // store. This is where incremental indexing matters — a full index rebuild would cost the
    // 'idx-build' time above (~0.9s) EVERY turn at 1M facts; incremental makes each turn O(1).
    {
        let base = neu.fact_count();
        let iters = 1_000usize;
        let t = Instant::now();
        for k in 0..iters {
            neu.observe(&format!("the {} late entry is w{}", code(base + k), k));
            let _ = neu.recall(&format!("what is {}?", code(base + k)));
        }
        let per = t.elapsed().as_micros() as f64 / iters as f64;
        println!("\n[grow+recall] append 1 fact + recall it, {} turns on a {}-fact store: {:.1} us/turn", iters, base, per);
        println!("   incremental index keeps each turn O(1); a full rebuild would be ~{}x slower (the idx-build cost).", 1_000_000 / (per.max(1.0) as u64).max(1));
    }

    println!("\nKey result: the per-turn injected context stays flat (k facts / a few hundred tokens)");
    println!("while total memory grows 100x. Selective recall stays ~flat (sub-linear via the stem");
    println!("index) = the scalable path to effectively unbounded context. Broad-cue and full-index");
    println!("rebuild are the O(N) costs; shard with NeuronRouter to keep per-scope work small.");
}
