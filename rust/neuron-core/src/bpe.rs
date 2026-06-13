//! Byte-level BPE (GPT-2 style) in pure Rust std — no regex crate. The pretokenizer is
//! hand-rolled to match the HuggingFace ByteLevelBPETokenizer the cortex was trained with.
use std::collections::HashMap;

fn bytes_to_unicode() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::new();
    for i in 33..=126 { bs.push(i); }
    for i in 161..=172 { bs.push(i); }
    for i in 174..=255 { bs.push(i); }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 { if !bs.contains(&b) { bs.push(b); cs.push(256 + n); n += 1; } }
    let mut b2u = HashMap::new(); let mut u2b = HashMap::new();
    for (i, &b) in bs.iter().enumerate() {
        let ch = char::from_u32(cs[i]).unwrap();
        b2u.insert(b as u8, ch); u2b.insert(ch, b as u8);
    }
    (b2u, u2b)
}

fn is_letter(c: char) -> bool { c.is_alphabetic() }
fn is_num(c: char) -> bool { c.is_numeric() }
fn is_ws(c: char) -> bool { c.is_whitespace() }

/// GPT-2 pretokenizer: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
fn pretokenize(text: &str) -> Vec<String> {
    let ch: Vec<char> = text.chars().collect();
    let n = ch.len(); let mut i = 0; let mut out = Vec::new();
    let run = |start: usize, pred: &dyn Fn(char) -> bool| -> usize {
        let mut j = start; while j < n && pred(ch[j]) { j += 1; } j
    };
    while i < n {
        let c = ch[i];
        // contractions
        if c == '\'' && i + 1 < n {
            let two: String = ch[i..(i+2).min(n)].iter().collect();
            let three: String = ch[i..(i+3).min(n)].iter().collect();
            if ["'re", "'ve", "'ll"].contains(&three.as_str()) { out.push(three); i += 3; continue; }
            if ["'s", "'t", "'m", "'d"].contains(&two.as_str()) { out.push(two); i += 2; continue; }
        }
        // " ?\p{L}+" / " ?\p{N}+" / " ?[^\s\p{L}\p{N}]+"
        let (s, after) = if c == ' ' && i + 1 < n { (1usize, i + 1) } else { (0usize, i) };
        if after < n && is_letter(ch[after]) {
            let j = run(after, &is_letter); out.push(ch[i..j].iter().collect()); i = j; continue;
        }
        if after < n && is_num(ch[after]) {
            let j = run(after, &is_num); out.push(ch[i..j].iter().collect()); i = j; continue;
        }
        if after < n && !is_ws(ch[after]) && !is_letter(ch[after]) && !is_num(ch[after]) {
            let j = run(after, &|x| !is_ws(x) && !is_letter(x) && !is_num(x));
            out.push(ch[i..j].iter().collect()); i = j; continue;
        }
        // whitespace: \s+(?!\S) leaves a trailing space for the next word
        if is_ws(c) {
            let j = run(i, &is_ws);
            let end = if j < n { j - 1 } else { j };       // leave last space if a token follows
            let end = if end <= i { j } else { end };
            out.push(ch[i..end].iter().collect()); i = end; continue;
        }
        out.push(c.to_string()); i += 1;                    // safety
    }
    out
}

pub struct Bpe {
    enc: HashMap<String, u32>,
    dec: HashMap<u32, String>,
    ranks: HashMap<String, u32>,
    b2u: HashMap<u8, char>,
    u2b: HashMap<char, u8>,
}
impl Bpe {
    /// vocab_tsv lines "id<TAB>token"; merges = merges.txt ("a b" per line)
    pub fn load(vocab_tsv: &str, merges: &str) -> Bpe {
        let (b2u, u2b) = bytes_to_unicode();
        let mut enc = HashMap::new(); let mut dec = HashMap::new();
        for line in vocab_tsv.lines() {
            if line.is_empty() { continue; }
            let mut it = line.splitn(2, '\t');
            let id: u32 = it.next().unwrap().parse().unwrap();
            let tok = it.next().unwrap_or("").to_string();
            enc.insert(tok.clone(), id); dec.insert(id, tok);
        }
        let mut ranks = HashMap::new(); let mut r = 0u32;
        for line in merges.lines() {
            if line.is_empty() || line.starts_with("#") { continue; }
            ranks.insert(line.trim().to_string(), r); r += 1;
        }
        Bpe { enc, dec, ranks, b2u, u2b }
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.len() < 2 { return word; }
        loop {
            let mut best = u32::MAX; let mut bi = usize::MAX;
            for i in 0..word.len()-1 {
                if let Some(&rk) = self.ranks.get(&format!("{} {}", word[i], word[i+1])) {
                    if rk < best { best = rk; bi = i; }
                }
            }
            if bi == usize::MAX { break; }
            let merged = format!("{}{}", word[bi], word[bi+1]);
            let mut nw = Vec::with_capacity(word.len()-1);
            nw.extend_from_slice(&word[..bi]); nw.push(merged); nw.extend_from_slice(&word[bi+2..]);
            word = nw;
        }
        word
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for piece in pretokenize(text) {
            let mut chars = String::new();
            for b in piece.bytes() { chars.push(self.b2u[&b]); }
            for sym in self.bpe(&chars) {
                if let Some(&id) = self.enc.get(&sym) { ids.push(id); }
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut chars = String::new();
        for id in ids { if let Some(s) = self.dec.get(id) { chars.push_str(s); } }
        let bytes: Vec<u8> = chars.chars().filter_map(|c| self.u2b.get(&c).copied()).collect();
        String::from_utf8_lossy(&bytes).to_string()
    }
}
