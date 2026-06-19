//! (a)+(b)+(c) end-to-end. Take the REAL persisted blob (`Neuron::dump()`), compress it with the
//! lossless token-Huffman codec, PROVE exact recall survives a compress->decompress->load round-trip,
//! and measure the on-disk shrink, the full ladder (raw -> interned -> Huffman -> entropy floor), and
//! codec throughput. With a path arg it ingests real prose (Project Gutenberg .txt); otherwise a
//! synthetic redundant LLM-memory corpus (entities/relations/values, the density sweet spot).
//!
//! Nothing here changes dump()/load(); the codec is a drop-in wrapper, so this measures exactly what
//! neuron-db WOULD persist if it stored compressed blobs — at identical recall.
//!
//! Run: cargo run --release --features compress --example compression_bench [dir-or-file]
use neuron_core::{codec, Neuron};
use std::collections::HashMap;
use std::time::Instant;

fn strip(text: &str) -> &str {
    let s = text.find("*** START").and_then(|i| text[i..].find('\n').map(|j| i + j + 1)).unwrap_or(0);
    let e = text[s..].find("*** END").map(|j| s + j).unwrap_or(text.len());
    &text[s..e]
}
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        let c = if ch.is_whitespace() { ' ' } else { ch };
        if c == ' ' && cur.ends_with(' ') { continue; }
        cur.push(c);
        if matches!(ch, '.' | '!' | '?') {
            let s = cur.trim().to_string();
            if s.len() >= 30 && s.len() <= 300 && s.split_whitespace().count() >= 5 { out.push(s); }
            cur.clear();
        }
    }
    out
}

fn lcg(s: &mut u64) -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *s >> 16 }
fn pick(s: &mut u64, n: usize) -> usize { (lcg(s) as usize) % n.max(1) }
fn tok(mut x: usize) -> String { let mut s = String::with_capacity(5); for _ in 0..5 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; } s }
const RELS: [&str; 12] = ["is","prefers","owns","uses","reports to","depends on","lives in","status is","password is","region is","role is","email is"];
fn synth(n: usize) -> Vec<String> {
    let ent = (30 * (n as f64).sqrt() as usize).max(40);
    let val = (60 * (n as f64).sqrt() as usize).max(80);
    let mut s = 0xCAFE_1234u64;
    (0..n).map(|_| format!("the {} {} {}", tok(pick(&mut s, ent)), RELS[pick(&mut s, RELS.len())], tok(1_000_000 + pick(&mut s, val)))).collect()
}

/// entropy floor (token Huffman/arithmetic lower bound): H(token stream) + the dictionary.
fn entropy_floor(blob: &str) -> usize {
    // same tokenization class as the codec (alnum runs vs non-alnum runs), inlined
    let mut freq: HashMap<&str, u64> = HashMap::new();
    let mut total = 0u64;
    let mut start = 0usize; let mut cur: Option<bool> = None;
    for (i, ch) in blob.char_indices() {
        let a = ch.is_alphanumeric();
        if let Some(p) = cur {
            if p != a {
                let t = &blob[start..i];
                if !t.is_empty() { *freq.entry(t).or_insert(0) += 1; total += 1; }
                start = i; cur = Some(a);
            }
        } else { cur = Some(a); }
    }
    let t = &blob[start..];
    if !t.is_empty() { *freq.entry(t).or_insert(0) += 1; total += 1; }
    let dict: usize = freq.keys().map(|k| 1 + k.len()).sum();
    let tf = total as f64;
    let bits: f64 = freq.values().map(|&c| { let p = c as f64 / tf; -(c as f64) * p.log2() }).sum();
    dict + (bits / 8.0).ceil() as usize
}

fn main() {
    let arg = std::env::args().nth(1).or_else(|| std::env::var("NDB_BOOKS").ok());
    let (label, facts): (String, Vec<String>) = match arg {
        Some(p) => {
            let path = std::path::Path::new(&p);
            let mut files = Vec::new();
            if path.is_dir() {
                for e in std::fs::read_dir(path).unwrap_or_else(|_| panic!("cannot read dir {}", p)).flatten() {
                    if e.path().extension().is_some_and(|x| x == "txt") { files.push(e.path()); }
                }
            } else { files.push(path.to_path_buf()); }
            files.sort();
            let mut all = Vec::new();
            for f in &files { let raw = std::fs::read_to_string(f).unwrap_or_default(); all.extend(sentences(strip(&raw))); }
            (format!("real prose ({} file(s))", files.len()), all)
        }
        None => ("synthetic redundant memory".to_string(), synth(50_000)),
    };
    if facts.is_empty() { eprintln!("no facts to ingest"); return; }

    // ingest into the real store; the persisted blob is exactly Neuron::dump()
    let mut n = Neuron::new(5_000_000);
    for f in &facts { n.observe(f); }
    let stored = n.episodes.len();
    let blob = n.dump();
    let raw = blob.len();

    // the ladder
    let interned = codec::interned_bytes(&blob);
    let t = Instant::now(); let comp = codec::compress(&blob); let c_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now(); let back = codec::decompress(&comp); let d_ms = t.elapsed().as_secs_f64() * 1000.0;
    let floor = entropy_floor(&blob);

    // (b) PROOF the codec is lossless and recall survives
    assert_eq!(back, blob, "codec round-trip is NOT byte-exact — would corrupt persistence");
    let mut n2 = Neuron::load(&back, 5_000_000);
    let sample = stored.min(300);
    let (mut checked, mut agree) = (0usize, 0usize);
    let mut s = 0x1357u64;
    for _ in 0..sample {
        let i = pick(&mut s, stored);
        let q = n.episodes[i].t.clone();           // recall a stored fact by its own text
        let a = n.recall(&q).map(|r| r.value);
        let b = n2.recall(&q).map(|r| r.value);
        if a == b { agree += 1; } checked += 1;
    }
    let recall_ok = agree == checked;

    println!("== compression bench: lossless squeeze of the persisted dump() blob ==");
    println!("   corpus: {}  ({} facts stored)\n", label, stored);
    println!("   {:<34} {:>12} {:>12}", "stage", "bytes/fact", "total");
    println!("   {}", "-".repeat(60));
    let row = |name: &str, bytes: usize| println!("   {:<34} {:>12.2} {:>9.2} MB", name, bytes as f64 / stored as f64, bytes as f64 / 1e6);
    row("raw dump() (today)", raw);
    row("(b) interned + varint", interned);
    row("(c) interned + Huffman (the codec)", comp.len());
    row("    entropy floor (lower bound)", floor);
    println!("\n   on-disk shrink (raw -> codec): {:.2}x     ({:.2} MB -> {:.2} MB)", raw as f64 / comp.len() as f64, raw as f64 / 1e6, comp.len() as f64 / 1e6);
    println!("   codec headroom left to the floor: {:.1}%  (Huffman {} B vs floor {} B)", (comp.len() as f64 / floor as f64 - 1.0) * 100.0, comp.len(), floor);
    println!("   throughput: compress {:.0} MB/s, decompress {:.0} MB/s", raw as f64 / 1e6 / (c_ms / 1000.0), raw as f64 / 1e6 / (d_ms / 1000.0));
    println!("\n   lossless round-trip: {}   recall preserved: {} ({}/{} sampled facts agree)",
             if back == blob { "OK (byte-exact)" } else { "FAILED" },
             if recall_ok { "OK" } else { "MISMATCH" }, agree, checked);
    println!("\n   Reading it: the codec is a drop-in over dump()/load() — same recall, {:.2}x less disk.", raw as f64 / comp.len() as f64);
    println!("   Real prose compresses less than structured memory (less reuse); the synthetic run (no");
    println!("   arg) is the LLM-memory sweet spot where entity/relation reuse pays the most.");
}
