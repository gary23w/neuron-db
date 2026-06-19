//! Compression probe: how much room is actually squeezable out of each stored node?
//!
//! A node is ~42 B/fact, ~88% of which is the raw sentence text. LLM memory is highly redundant
//! (the same entities/relations/phrasings recur across facts), so the text cost SHOULD scale with
//! the scope's vocabulary, not its fact count. This sizes the prize before building any encoder, by
//! reporting a ladder of bytes/fact on a realistic, redundant memory corpus:
//!
//!   raw UTF-8                     - what we store today (the baseline text cost)
//!   interned + varint token-ids  - tokens interned once per scope; each fact = varint(freq-rank)s
//!                                  (the achievable simple scheme; tiny references into a shared table)
//!   entropy floor (Huffman/AC)   - Shannon entropy of the Zipfian token stream + the dictionary
//!                                  (the best any token-level coder can do)
//!
//! Plus two structural readings:
//!   - marginal bytes/fact: the cost of ONE more fact once the vocabulary is warm (the real capacity
//!     number — it should fall well below the per-fact average as the scope grows),
//!   - index postings: raw Vec<u32> vs delta+varint (the inverted index's pointers made "tiny" — the
//!     literal tiny-pointers analog; resident-only, so this is a RAM win for small devices).
//!
//! Synthetic but redundancy-calibrated to Heaps' law (vocab ~ n^0.5). Real ratios depend on the
//! actual memory's redundancy; this brackets the opportunity. Run: cargo run --release --example compression_probe
use std::collections::HashMap;

fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 16 }
fn pick(s: &mut u64, n: usize) -> usize { (lcg(s) as usize) % n.max(1) }
fn isqrt(n: usize) -> usize { (n as f64).sqrt() as usize }

/// alpha token for a pool id (distinct, lowercase) — stands in for an entity/value word
fn tok(mut x: usize) -> String { let mut s = String::with_capacity(5); for _ in 0..5 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; } s }

const RELS: [&str; 12] = ["is","prefers","owns","uses","reports to","depends on","lives in",
                          "status is","password is","region is","role is","email is"];

/// A realistic redundant memory fact. Entity/value pools grow ~sqrt(n) (Heaps' law), so vocabulary
/// saturates while facts keep re-referencing it — the property the compression exploits.
fn make_facts(n: usize) -> Vec<String> {
    let ent_pool = (30 * isqrt(n)).max(40);
    let val_pool = (60 * isqrt(n)).max(80);
    let mut s = 0xCAFE_1234u64;
    (0..n).map(|_| {
        let e = tok(pick(&mut s, ent_pool));
        let r = RELS[pick(&mut s, RELS.len())];
        let v = tok(1_000_000 + pick(&mut s, val_pool));
        format!("the {} {} {}", e, r, v)
    }).collect()
}

fn varint_len(mut x: usize) -> usize { let mut b = 1; while x >= 128 { x >>= 7; b += 1; } b }

/// (raw_bytes, interned_varint_bytes, entropy_floor_bytes, dict_bytes, n_tokens, vocab)
fn measure_text(facts: &[String]) -> (usize, usize, usize, usize, usize, usize) {
    let raw: usize = facts.iter().map(|f| f.len()).sum();
    // token frequencies
    let mut freq: HashMap<&str, u64> = HashMap::new();
    let mut ntok = 0usize;
    for f in facts { for t in f.split(' ') { *freq.entry(t).or_insert(0) += 1; ntok += 1; } }
    // rank tokens by frequency (so the commonest get the shortest varints)
    let mut toks: Vec<(&str, u64)> = freq.into_iter().collect();
    toks.sort_by(|a, b| b.1.cmp(&a.1));
    let rank: HashMap<&str, usize> = toks.iter().enumerate().map(|(i, (t, _))| (*t, i)).collect();
    let dict_bytes: usize = toks.iter().map(|(t, _)| t.len() + 1).sum(); // dictionary stored once
    // interned: each token -> varint(rank)
    let interned: usize = dict_bytes + facts.iter()
        .map(|f| f.split(' ').map(|t| varint_len(rank[t])).sum::<usize>()).sum::<usize>();
    // entropy floor: total tokens * H(token), + dictionary
    let total = ntok as f64;
    let h_bits: f64 = toks.iter().map(|(_, c)| { let p = *c as f64 / total; -(*c as f64) * p.log2() }).sum();
    let entropy = dict_bytes + (h_bits / 8.0).ceil() as usize;
    (raw, interned, entropy, dict_bytes, ntok, toks.len())
}

/// inverted-index postings: raw u32 array vs delta+varint. Returns (raw_bytes, compressed_bytes).
fn measure_index(facts: &[String]) -> (usize, usize) {
    let mut post: HashMap<&str, Vec<u32>> = HashMap::new();
    for (i, f) in facts.iter().enumerate() { for t in f.split(' ') { post.entry(t).or_default().push(i as u32); } }
    let mut raw = 0usize; let mut comp = 0usize;
    for (_, list) in post.iter() {
        raw += list.len() * 4;                 // Vec<u32>
        let mut prev = 0u32;
        for &p in list { comp += varint_len((p - prev) as usize); prev = p; } // sorted asc -> delta+varint
    }
    (raw, comp)
}

fn main() {
    println!("== compression probe: how much room can we squeeze out of each node? ==");
    println!("   corpus: redundant LLM-memory facts (entity/relation/value, Heaps-law vocab ~ n^0.5).");
    println!("   text is ~88% of a node; the rest (index, scalars) is small. Numbers are TEXT bytes.\n");

    println!("   {:>8} | {:>8} | {:>10} | {:>11} | {:>8} | {:>10}", "facts", "vocab", "raw B/fact", "intern B/fct", "floor", "vs raw");
    println!("   {}", "-".repeat(70));
    let sizes = [1_000usize, 10_000, 100_000];
    let mut prev: Option<(usize, usize)> = None; // (facts, interned_total) for marginal calc
    for &n in &sizes {
        let facts = make_facts(n);
        let (raw, interned, entropy, _dict, _ntok, vocab) = measure_text(&facts);
        println!("   {:>8} | {:>8} | {:>10.1} | {:>11.1} | {:>7.1}B | {:>9.2}x",
                 n, vocab, raw as f64 / n as f64, interned as f64 / n as f64,
                 entropy as f64 / n as f64, raw as f64 / interned as f64);
        if let Some((pn, pi)) = prev {
            let marginal = (interned - pi) as f64 / (n - pn) as f64;
            println!("   {:>8}   -> marginal interned cost of facts {}..{}: {:.1} B/fact (the warm-vocab capacity number)",
                     "", pn, n, marginal);
        }
        prev = Some((n, interned));
    }

    // index postings compression (the literal tiny-pointers analog) at the largest size
    let facts = make_facts(100_000);
    let (iraw, icomp) = measure_index(&facts);
    println!("\n   inverted-index postings (resident RAM, the 'tiny pointers' analog) @ 100k facts:");
    println!("     raw Vec<u32> .......... {:.2} MB", iraw as f64 / 1e6);
    println!("     delta + varint ........ {:.2} MB  ({:.2}x smaller)", icomp as f64 / 1e6, iraw as f64 / icomp as f64);

    println!("\n   Reading it: the gap between 'raw B/fact' and 'intern B/fct' is the squeezable room; the");
    println!("   'marginal' line is the real capacity win — once vocabulary is warm, each new fact costs");
    println!("   far less than the 42 B average, because it is mostly references into the shared table.");
    println!("   The entropy floor shows how much MORE an arithmetic/Huffman coder would buy on top.");
}
