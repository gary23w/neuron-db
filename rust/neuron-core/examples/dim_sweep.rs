//! (a) DIM sweep: how does the semantic space's dimensionality trade off recall quality vs size
//! vs speed? Builds the SAME corpus into spaces of several DIMs and measures:
//!
//!   - quality(neighbor): for each sampled word, fraction of its top-5 nearest words that are in
//!     the SAME topic (intrinsic separation of the space).
//!   - quality(paraphrase): a fact uses one half of a topic's vocabulary, the query uses the OTHER
//!     half (zero shared words, same topic). Does the right-topic fact rank #1 against ONE
//!     distractor fact per other topic (a 1-in-T choice)? Pure semantic recall, no lexical overlap.
//!   - size: resident bytes of the space.   - speed: average embed() time (per-recall cost).
//!
//! The task is deliberately HARD — many topics over a large vocabulary — because Random Indexing's
//! sparse index vectors live in DIM slots: with too few dimensions, distinct words collide onto the
//! same slots, topics blur, and separation collapses. The sweep shows the knee where adding
//! dimensions stops buying recall (and only costs bytes + time).
//!
//! Run: cargo run --release --features semantic --example dim_sweep
use neuron_core::semantic::SemanticSpace;
use std::collections::HashMap;
use std::time::Instant;

const T: usize = 100;   // topics (directions the space must separate)
const W: usize = 24;    // distinctive words per topic  -> vocab = T*W = 2400
const HALF: usize = 12; // fact words come from [0,HALF), query words from [HALF,W): disjoint

/// distinct 5-letter token for a (topic, word) — pure alpha so the semantic tokenizer keeps it whole.
fn code(mut g: usize) -> String { let mut s = String::with_capacity(5); for _ in 0..5 { s.push((b'a' + (g % 26) as u8) as char); g /= 26; } s }
fn word(t: usize, w: usize) -> String { code(t * 1000 + w) }

fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 16 }
fn pick(s: &mut u64, n: usize) -> usize { (lcg(s) as usize) % n.max(1) }
/// `take` distinct words drawn from topic `t`'s index range [lo,hi)
fn sentence(s: &mut u64, t: usize, lo: usize, hi: usize, take: usize) -> String {
    let mut idx: Vec<usize> = (lo..hi).collect();
    for i in (1..idx.len()).rev() { idx.swap(i, pick(s, i + 1)); }
    idx[..take.min(idx.len())].iter().map(|&w| word(t, w)).collect::<Vec<_>>().join(" ")
}

fn evaluate(dim: usize, w2t: &HashMap<String, usize>) -> (f64, f64, usize, f64) {
    // ---- train: many within-topic sentences over the FULL vocabulary (identical corpus per DIM) ----
    let mut space = SemanticSpace::with_dim(dim);
    let mut seed = 0xA5A5_1234u64;
    for _ in 0..150 { for t in 0..T { space.train(&sentence(&mut seed, t, 0, W, 6)); } }

    // ---- neighbor coherence: top-5 nearest of one sampled word per topic, fraction same-topic ----
    let (mut hit, mut tot) = (0usize, 0usize);
    for t in 0..T {
        let near = space.nearest(&word(t, 0), 5);
        for (nw, _) in &near { if w2t.get(nw) == Some(&t) { hit += 1; } tot += 1; }
    }
    let neighbor = hit as f64 / tot.max(1) as f64;

    // ---- paraphrase recall: fact from first half, query from second half (disjoint words) ----
    // single-word fact/query (no averaging to mask index-vector noise) — the sensitive case.
    let mut q = 0xBEEF_9001u64;
    let (mut correct, mut trials) = (0usize, 0usize);
    for _ in 0..500 {
        let t = pick(&mut q, T);
        let mut cands = vec![sentence(&mut q, t, 0, HALF, 1)];           // the right fact (one first-half word)
        let query = sentence(&mut q, t, HALF, W, 1);                     // query (one second-half word, no overlap)
        for ot in 0..T { if ot != t { cands.push(sentence(&mut q, ot, 0, HALF, 1)); } } // T-1 distractors
        if let Some(&(top, _)) = space.rank(&query, &cands).first() { if top == 0 { correct += 1; } }
        trials += 1;
    }
    let paraphrase = correct as f64 / trials.max(1) as f64;

    // ---- speed: average embed() of a query ----
    let probe = sentence(&mut q, 0, HALF, W, 5);
    let iters = 3_000;
    let t0 = Instant::now();
    for _ in 0..iters { std::hint::black_box(space.embed(&probe)); }
    let embed_us = t0.elapsed().as_micros() as f64 / iters as f64;

    (neighbor, paraphrase, space.bytes(), embed_us)
}

fn main() {
    println!("== (a) semantic DIM sweep: recall quality vs size vs speed ==");
    println!("   corpus: {} topics x {} words (vocab {}); identical data at every DIM.", T, W, T * W);
    println!("   paraphrase = top-1 recall of the right fact among {} (disjoint-word query).\n", T);
    let mut w2t = HashMap::new();
    for t in 0..T { for w in 0..W { w2t.insert(word(t, w), t); } }

    println!("   {:>5} | {:>14} | {:>16} | {:>10} | {:>11}", "DIM", "neighbor top-5", "paraphrase top-1", "size KB", "embed us");
    println!("   {}", "-".repeat(72));
    let mut prev = 0f64;
    for dim in [2usize, 4, 8, 16, 32, 64, 128, 256] {
        let (neighbor, paraphrase, bytes, embed_us) = evaluate(dim, &w2t);
        let delta = if prev == 0.0 { String::new() } else { format!("  ({:+.1} pts)", (paraphrase - prev) * 100.0) };
        println!("   {:>5} | {:>13.1}% | {:>15.1}% | {:>10.1} | {:>11.2}{}",
                 dim, neighbor * 100.0, paraphrase * 100.0, bytes as f64 / 1024.0, embed_us, delta);
        prev = paraphrase;
    }
    println!("\n   Reading it: quality climbs with DIM then saturates (the knee); size + embed cost grow");
    println!("   ~linearly. Below the knee, random-index collisions blur topics and recall drops. Pick");
    println!("   the smallest DIM at full quality to fit more on a small device — extending DIM past");
    println!("   the knee only spends bytes and time.");
}
