//! Lossless token-level compression for the persisted blob (the `dump()` text). Feature-gated
//! behind `compress`: pure std, wasm-safe, no new dependencies (the default/wasm build never
//! sees it; it's an opt-in transform).
//!
//! Two parts: (1) interning, a per-blob token dictionary so a recurring entity/relation/word is
//! stored once and referenced; (2) canonical Huffman to entropy-code the token stream so frequent
//! tokens cost few bits. `compress`/`decompress` is byte-exact, so it is a
//! drop-in wrapper over what `Neuron::dump()` already produces: `decompress(compress(dump())) ==
//! dump()`, which means recall after a compress/decompress round-trip is identical by construction.
//!
//! This is the LOSSLESS, exact-recall counterpart to the (rejected) HDC superposition idea: it
//! squeezes the text without giving up the precise sentence.
use std::collections::HashMap;

/// Reversible tokenization: a token is a maximal run of alphanumeric chars OR a maximal run of
/// non-alphanumeric chars. Concatenating the tokens reproduces the input byte-for-byte (unicode
/// safe — it splits on `char` class, never inside a multi-byte char).
fn tokenize(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut cur: Option<bool> = None; // Some(true)=alnum run, Some(false)=non-alnum run
    for (i, ch) in s.char_indices() {
        let a = ch.is_alphanumeric();
        match cur {
            Some(p) if p != a => { out.push(&s[start..i]); start = i; cur = Some(a); }
            None => cur = Some(a),
            _ => {}
        }
    }
    if start < s.len() { out.push(&s[start..]); }
    out
}

fn uvarint_len(mut x: u64) -> usize { let mut n = 1; while x >= 0x80 { x >>= 7; n += 1; } n }
fn put_uvarint(out: &mut Vec<u8>, mut x: u64) {
    while x >= 0x80 { out.push((x as u8 & 0x7f) | 0x80); x >>= 7; }
    out.push(x as u8);
}
fn get_uvarint(b: &[u8], pos: &mut usize) -> u64 {
    let (mut x, mut shift) = (0u64, 0u32);
    loop {
        let byte = b[*pos]; *pos += 1;
        x |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 { break; }
        shift += 7;
    }
    x
}

/// (b) the achievable INTERNING-only size, analytically: a frequency-ranked dictionary (stored once)
/// plus one varint(rank) per token occurrence. The common tokens get rank<128 -> a single byte.
pub fn interned_bytes(s: &str) -> usize {
    let toks = tokenize(s);
    if toks.is_empty() { return 0; }
    let mut freq: HashMap<&str, u64> = HashMap::new();
    for t in &toks { *freq.entry(t).or_insert(0) += 1; }
    let mut v: Vec<(&str, u64)> = freq.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0))); // by freq desc -> the commonest gets rank 0
    let rank: HashMap<&str, usize> = v.iter().enumerate().map(|(i, (t, _))| (*t, i)).collect();
    let dict: usize = v.iter().map(|(t, _)| uvarint_len(t.len() as u64) + t.len()).sum();
    let body: usize = toks.iter().map(|t| uvarint_len(rank[t] as u64)).sum();
    dict + body
}

/// Huffman code lengths (bits per symbol) for the given weights. Deterministic tie-break by id.
fn code_lengths(weights: &[u64]) -> Vec<u8> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let n = weights.len();
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![1]; }
    let mut kids: Vec<(i32, i32)> = vec![(-1, -1); n]; // leaves 0..n; internals appended
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = (0..n).map(|i| Reverse((weights[i], i))).collect();
    while heap.len() > 1 {
        let Reverse((w1, a)) = heap.pop().unwrap();
        let Reverse((w2, b)) = heap.pop().unwrap();
        let id = kids.len();
        kids.push((a as i32, b as i32));
        heap.push(Reverse((w1 + w2, id)));
    }
    let root = heap.pop().unwrap().0 .1;
    let mut len = vec![0u8; n];
    let mut stack = vec![(root, 0u8)];
    while let Some((node, d)) = stack.pop() {
        let (l, r) = kids[node];
        if l < 0 { len[node] = d.max(1); } // a leaf (index < n)
        else { stack.push((l as usize, d + 1)); stack.push((r as usize, d + 1)); }
    }
    len
}

const MAGIC: u8 = 0xC1;

/// Compress a blob losslessly. Layout: MAGIC, uvarint(nsym), uvarint(ntok), then per symbol in
/// canonical order [u8 codelen, uvarint(bytelen), bytes], then the MSB-first packed bitstream.
pub fn compress(s: &str) -> Vec<u8> {
    let toks = tokenize(s);
    let mut out = vec![MAGIC];
    if toks.is_empty() { put_uvarint(&mut out, 0); put_uvarint(&mut out, 0); return out; }

    // distinct tokens + counts
    let mut idx: HashMap<&str, usize> = HashMap::new();
    let mut syms: Vec<&str> = Vec::new();
    let mut cnt: Vec<u64> = Vec::new();
    for t in &toks {
        let i = *idx.entry(t).or_insert_with(|| { syms.push(t); cnt.push(0); syms.len() - 1 });
        cnt[i] += 1;
    }
    let nsym = syms.len();
    put_uvarint(&mut out, nsym as u64);
    put_uvarint(&mut out, toks.len() as u64);

    if nsym == 1 { // single distinct token: store it, repeat on decode (no bitstream)
        put_uvarint(&mut out, syms[0].len() as u64);
        out.extend_from_slice(syms[0].as_bytes());
        return out;
    }

    let lens = code_lengths(&cnt);
    // canonical order: by (codelen asc, token bytes asc)
    let mut order: Vec<usize> = (0..nsym).collect();
    order.sort_by(|&a, &b| lens[a].cmp(&lens[b]).then(syms[a].as_bytes().cmp(syms[b].as_bytes())));
    // assign canonical codes and write the symbol table in that order
    let mut codes = vec![0u64; nsym];
    let mut code = 0u64;
    let mut prev = lens[order[0]];
    for (k, &sym) in order.iter().enumerate() {
        if k > 0 { code = (code + 1) << (lens[sym] - prev); prev = lens[sym]; }
        codes[sym] = code;
        out.push(lens[sym]);
        put_uvarint(&mut out, syms[sym].len() as u64);
        out.extend_from_slice(syms[sym].as_bytes());
    }

    // bitstream, MSB-first
    let (mut acc, mut nbits) = (0u8, 0u8);
    for t in &toks {
        let s = idx[t];
        let (c, l) = (codes[s], lens[s]);
        let mut bit = l;
        while bit > 0 {
            bit -= 1;
            acc = (acc << 1) | (((c >> bit) & 1) as u8);
            nbits += 1;
            if nbits == 8 { out.push(acc); acc = 0; nbits = 0; }
        }
    }
    if nbits > 0 { out.push(acc << (8 - nbits)); }
    out
}

/// Inverse of `compress`. Byte-exact: `decompress(&compress(s)) == s`.
pub fn decompress(b: &[u8]) -> String {
    if b.is_empty() || b[0] != MAGIC { return String::new(); }
    let mut pos = 1usize;
    let nsym = get_uvarint(b, &mut pos) as usize;
    let ntok = get_uvarint(b, &mut pos) as usize;
    if nsym == 0 { return String::new(); }
    if nsym == 1 {
        let bl = get_uvarint(b, &mut pos) as usize;
        let tok = std::str::from_utf8(&b[pos..pos + bl]).unwrap_or("");
        return tok.repeat(ntok);
    }
    // read symbol table (already in canonical order); rebuild canonical codes identically
    let mut lens = Vec::with_capacity(nsym);
    let mut toks: Vec<String> = Vec::with_capacity(nsym);
    for _ in 0..nsym {
        let l = b[pos]; pos += 1;
        let bl = get_uvarint(b, &mut pos) as usize;
        toks.push(String::from_utf8_lossy(&b[pos..pos + bl]).into_owned());
        pos += bl;
        lens.push(l);
    }
    // build a decode tree from the canonical codes
    let mut c0: Vec<i32> = vec![-1]; let mut c1: Vec<i32> = vec![-1]; let mut leaf: Vec<i32> = vec![-1];
    let new_node = |c0: &mut Vec<i32>, c1: &mut Vec<i32>, leaf: &mut Vec<i32>| -> usize {
        c0.push(-1); c1.push(-1); leaf.push(-1); c0.len() - 1
    };
    let mut code = 0u64;
    let mut prev = lens[0];
    for k in 0..nsym {
        if k > 0 { code = (code + 1) << (lens[k] - prev); prev = lens[k]; }
        // insert token k at path `code` of length lens[k]
        let (mut node, l) = (0usize, lens[k]);
        let mut bit = l;
        while bit > 0 {
            bit -= 1;
            let go1 = (code >> bit) & 1 == 1;
            let child = if go1 { c1[node] } else { c0[node] };
            node = if child < 0 {
                let nn = new_node(&mut c0, &mut c1, &mut leaf) as i32;
                if go1 { c1[node] = nn } else { c0[node] = nn }
                nn as usize
            } else { child as usize };
        }
        leaf[node] = k as i32;
    }
    // walk the bitstream, emitting ntok tokens
    let mut out = String::new();
    let (mut node, mut produced) = (0usize, 0usize);
    'outer: for &byte in &b[pos..] {
        for s in (0..8).rev() {
            let go1 = (byte >> s) & 1 == 1;
            node = (if go1 { c1[node] } else { c0[node] }) as usize;
            if leaf[node] >= 0 {
                out.push_str(&toks[leaf[node] as usize]);
                node = 0; produced += 1;
                if produced == ntok { break 'outer; }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(s: &str) { assert_eq!(decompress(&compress(s)), s, "round-trip failed for {:?}", s); }

    #[test]
    fn roundtrip_lossless_varied() {
        rt("");
        rt("a");
        rt("aaaa");                                  // single alnum token
        rt("   ");                                   // single non-alnum token
        rt("the cat sat on the mat");
        rt("the api key is zeta-9931");
        rt("café ☕ 测试 — naïve façade");            // multibyte / unicode
        rt("0\tthe wifi password is vekam73\t1.0\n1\tthe plan is pro\t2.5\n"); // a real dump() blob shape
        rt("aaa\tbbb\nccc ddd 123 ddd ccc");
    }

    #[test]
    fn roundtrip_redundant_blob() {
        // many facts sharing vocabulary -> exactly the case the codec targets
        let mut blob = String::new();
        for i in 0..5000 { blob.push_str(&format!("0\tthe server node{} reports to cluster alpha\t1\n", i % 50)); }
        let c = compress(&blob);
        assert_eq!(decompress(&c), blob);
        assert!(c.len() < blob.len() / 3, "expected >3x shrink on redundant text, got {} -> {}", blob.len(), c.len());
    }

    #[test]
    fn interned_is_smaller_than_raw_on_redundant() {
        let mut blob = String::new();
        for i in 0..2000 { blob.push_str(&format!("the metric{} reading is value alpha beta\n", i % 30)); }
        assert!(interned_bytes(&blob) < blob.len());
    }

    // ---- end-to-end measurement (b)+(c) + real corpus (a). #[ignore]: heavy; run explicitly with
    //      `cargo test --features compress --lib -- --ignored --nocapture report_compression`.
    //      Writes the report to %TEMP%/ndb_compression_report.txt so capture can't hide the numbers.
    fn synth(n: usize) -> Vec<String> {
        let rels = ["is","prefers","owns","uses","reports to","depends on","status is","password is","region is","role is"];
        let tok = |mut x: usize| { let mut s = String::with_capacity(5); for _ in 0..5 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; } s };
        let ent = (30 * (n as f64).sqrt() as usize).max(40);
        let val = (60 * (n as f64).sqrt() as usize).max(80);
        let mut st = 0xCAFE_1234u64;
        let mut next = |m: usize| { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (st >> 16) as usize % m.max(1) };
        (0..n).map(|_| format!("the {} {} {}", tok(next(ent)), rels[next(rels.len())], tok(1_000_000 + next(val)))).collect()
    }
    fn entropy_floor(blob: &str) -> usize {
        let mut freq: std::collections::HashMap<&str, u64> = std::collections::HashMap::new();
        let mut total = 0u64; let mut start = 0usize; let mut cur: Option<bool> = None;
        for (i, ch) in blob.char_indices() {
            let a = ch.is_alphanumeric();
            if let Some(p) = cur { if p != a { let t = &blob[start..i]; if !t.is_empty() { *freq.entry(t).or_insert(0) += 1; total += 1; } start = i; cur = Some(a); } } else { cur = Some(a); }
        }
        let t = &blob[start..]; if !t.is_empty() { *freq.entry(t).or_insert(0) += 1; total += 1; }
        let dict: usize = freq.keys().map(|k| 1 + k.len()).sum();
        let tf = total as f64;
        let bits: f64 = freq.values().map(|&c| { let p = c as f64 / tf; -(c as f64) * p.log2() }).sum();
        dict + (bits / 8.0).ceil() as usize
    }
    fn read_corpus(dir: &str) -> Vec<String> {
        let strip = |t: &str| -> String {
            let s = t.find("*** START").and_then(|i| t[i..].find('\n').map(|j| i + j + 1)).unwrap_or(0);
            let e = t[s..].find("*** END").map(|j| s + j).unwrap_or(t.len()); t[s..e].to_string()
        };
        let mut out = Vec::new();
        let rd = match std::fs::read_dir(dir) { Ok(r) => r, Err(_) => return out };
        for e in rd.flatten() {
            if e.path().extension().is_some_and(|x| x == "txt") {
                let raw = std::fs::read_to_string(e.path()).unwrap_or_default();
                let body = strip(&raw);
                let mut cur = String::new();
                for ch in body.chars() {
                    let c = if ch.is_whitespace() { ' ' } else { ch };
                    if c == ' ' && cur.ends_with(' ') { continue; }
                    cur.push(c);
                    if matches!(ch, '.' | '!' | '?') {
                        let s = cur.trim().to_string();
                        if s.len() >= 30 && s.len() <= 300 && s.split_whitespace().count() >= 5 { out.push(s); }
                        cur.clear();
                    }
                }
            }
        }
        out
    }
    fn measure(label: &str, facts: &[String]) -> String {
        use crate::Neuron;
        let mut n = Neuron::new(5_000_000);
        for f in facts { n.observe(f); }
        let stored = n.episodes.len();
        let blob = n.dump();
        let raw = blob.len();
        let interned = interned_bytes(&blob);
        let comp = compress(&blob);
        let back = decompress(&comp);
        assert_eq!(back, blob, "[{label}] codec must be byte-exact");
        let floor = entropy_floor(&blob);
        // recall preserved through compress->decompress->load
        let mut n2 = Neuron::load(&back, 5_000_000);
        let mut st = 0x1357u64;
        let (mut ok, mut tot) = (0usize, 0usize);
        for _ in 0..stored.min(300) {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let i = (st >> 16) as usize % stored.max(1);
            let q = n.episodes[i].t.clone();
            if n.recall(&q).map(|r| r.value) == n2.recall(&q).map(|r| r.value) { ok += 1; } tot += 1;
        }
        assert_eq!(ok, tot, "[{label}] recall must be preserved through the codec");
        let pf = |b: usize| b as f64 / stored as f64;
        format!(
            "== {label} ({stored} facts) ==\n  raw dump()          {:>7.2} B/fact   {:>7.2} MB\n  interned+varint(b)  {:>7.2} B/fact   {:>7.2} MB\n  codec Huffman (c)   {:>7.2} B/fact   {:>7.2} MB\n  entropy floor       {:>7.2} B/fact   {:>7.2} MB\n  on-disk shrink raw->codec: {:.2}x   |  headroom to floor: {:.1}%  |  lossless: OK  recall-preserved: {}/{}\n\n",
            pf(raw), raw as f64/1e6, pf(interned), interned as f64/1e6, pf(comp.len()), comp.len() as f64/1e6,
            pf(floor), floor as f64/1e6, raw as f64/comp.len() as f64, (comp.len() as f64/floor as f64 - 1.0)*100.0, ok, tot)
    }

    #[test]
    #[ignore]
    fn report_compression() {
        let mut rep = String::from("\n### neuron-db compression: lossless squeeze of the persisted dump() blob ###\n\n");
        rep.push_str(&measure("synthetic redundant LLM-memory", &synth(50_000)));
        if let Ok(dir) = std::env::var("NDB_BOOKS") {
            let books = read_corpus(&dir);
            if !books.is_empty() { rep.push_str(&measure("real prose (Gutenberg books) = (a)", &books)); }
        }
        let path = std::env::temp_dir().join("ndb_compression_report.txt");
        std::fs::write(&path, &rep).expect("write report");
        eprint!("{rep}");
    }
}
