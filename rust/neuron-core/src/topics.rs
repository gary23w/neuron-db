//! topics.rs — latent Dirichlet allocation over the store's own word-bags, by collapsed Gibbs
//! sampling. Finds the abstract topics a corpus of facts is drawn from and assigns every fact
//! (and every query) a sparse topic mixture — the coarse index that makes fuzzy recall
//! scope-wide, and the inspection surface that answers "what is this scope about".
//!
//! The standard conditional drives every draw:
//!
//! ```text
//! P(z_i = k | rest)  ∝  (n_dk + α) · (n_kw + β) / (n_k + Vβ)
//! ```
//!
//! Three entry points, one sampler:
//!  - `absorb`  — streaming learning at observe time: fold one fact in against the current
//!                counts, then COMMIT its assignments (the Random-Indexing posture: accumulate
//!                forever, no refit required).
//!  - `fold_in` — frozen inference for queries and posting rebuilds: same sweeps, no commit.
//!  - `refit`   — batch (re)build over pseudo-documents, for tests, backfill, and a store that
//!                wants clean batch statistics; df-gates the vocabulary (hapax and hub words
//!                carry no topical signal — the same >25% hub rule `candidates()` applies).
//!
//! DETERMINISM IS NON-NEGOTIABLE (the whole store is): every draw is seeded from the CONTENT
//! being sampled — splitmix over an fnv chain of the document's tokens, xor position and sweep —
//! so the same corpus always reaches the same state, the same fact always folds to the same
//! topics, and tests assert exact count tables. No wall clock, no thread order, no unseeded RNG.
//!
//! Ranking-and-inspection only: the model never gates truth and never mints a fact. Facts store
//! zero new bytes — assignments live in caller-side caches/postings keyed by content.
//! Std-only; feature `topics`.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

/// Gibbs sweeps when absorbing a new fact (short docs converge fast against frozen counts).
const ABSORB_SWEEPS: usize = 3;
/// Sweeps when folding in a query / rebuilding postings (frozen counts, slightly deeper).
const FOLD_SWEEPS: usize = 5;
/// A word present in more than a quarter of documents is a hub — no topical signal. The floor
/// keeps small corpora ungated, mirroring the candidates()/spread df gate.
const HUB_FLOOR: u64 = 64;
/// Hard cap on the online vocabulary so a long-lived store cannot grow it without bound; past
/// it, unseen words simply stop entering (known words keep learning).
const VOCAB_CAP: usize = 262_144;
/// How many topics a mixture reports (facts are sentences; more than a few is noise).
const TOP_T: usize = 3;

fn fnv_chain(seed: u64, s: &str) -> u64 {
    let mut h = seed;
    for b in s.bytes() { h ^= b as u64; h = h.wrapping_mul(1099511628211); }
    h ^= 0xFF; h.wrapping_mul(1099511628211)
}
fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
/// deterministic uniform in [0,1) from a content-derived seed
fn unit(seed: u64) -> f64 { (splitmix(seed) >> 11) as f64 / (1u64 << 53) as f64 }

pub struct TopicModel {
    k: usize,
    alpha: f32,   // doc-topic smoothing; small, because a fact is one sentence (sparse mixtures)
    beta: f32,    // topic-word smoothing
    vocab: HashMap<String, u32>,
    words: Vec<String>,   // id -> word
    df: Vec<u32>,         // id -> documents containing the word (drives the hub gate)
    docs_seen: u64,
    tokens_absorbed: u64,
    nkw: Vec<u32>,        // WORD-MAJOR topic-word counts: [word_id * k + topic] — vocab growth
                          // appends k contiguous slots, and a token's k-loop is one cache line run
    nk: Vec<u64>,         // per-topic totals
}

impl TopicModel {
    pub fn new(k: usize) -> Self { Self::with_params(k, 0.2, 0.01) }
    pub fn with_params(k: usize, alpha: f32, beta: f32) -> Self {
        TopicModel {
            k: k.max(2), alpha: alpha.max(1e-4), beta: beta.max(1e-4),
            vocab: HashMap::new(), words: Vec::new(), df: Vec::new(),
            docs_seen: 0, tokens_absorbed: 0, nkw: Vec::new(), nk: vec![0; k.max(2)],
        }
    }
    pub fn k(&self) -> usize { self.k }
    pub fn vocab_len(&self) -> usize { self.words.len() }
    pub fn docs(&self) -> u64 { self.docs_seen }
    pub fn tokens(&self) -> u64 { self.tokens_absorbed }
    /// Approximate resident bytes (counts + vocab strings).
    pub fn bytes(&self) -> usize {
        self.nkw.len() * 4 + self.nk.len() * 8 + self.words.iter().map(|w| w.len() + 56).sum::<usize>()
    }

    fn hub_cap(&self) -> u64 { (self.docs_seen / 4).max(HUB_FLOOR) }
    fn is_hub(&self, id: u32) -> bool { self.df[id as usize] as u64 > self.hub_cap() }

    /// One conditional draw for token `w` given local doc counts, seeded by content. Frozen
    /// global counts; the caller owns any commit and lends the cumulative scratch buffer (the
    /// draw runs once per token per sweep — a per-call allocation here would dominate absorb).
    fn draw(&self, w: u32, ndk: &[u32], seed: u64, cum: &mut [f64]) -> usize {
        let (k, vb) = (self.k, self.words.len() as f64 * self.beta as f64);
        let base = w as usize * k;
        let mut total = 0f64;
        for t in 0..k {
            let p = (ndk[t] as f64 + self.alpha as f64)
                * (self.nkw[base + t] as f64 + self.beta as f64)
                / (self.nk[t] as f64 + vb);
            total += p;
            cum[t] = total;
        }
        if !(total > 0.0) { return (splitmix(seed) % k as u64) as usize; }
        let u = unit(seed) * total;
        cum[..k].iter().position(|&c| u < c).unwrap_or(k - 1)
    }

    /// Sample assignments for one document's known, non-hub word ids against FROZEN global
    /// counts: seeded init pass, then `sweeps` reassignment passes. Returns (z, ndk).
    fn gibbs_frozen(&self, ids: &[u32], base_seed: u64, sweeps: usize) -> (Vec<usize>, Vec<u32>) {
        let mut ndk = vec![0u32; self.k];
        let mut cum = vec![0f64; self.k];
        let mut z = Vec::with_capacity(ids.len());
        for (pos, &w) in ids.iter().enumerate() {
            let t = self.draw(w, &ndk, splitmix(base_seed ^ pos as u64), &mut cum);
            ndk[t] += 1;
            z.push(t);
        }
        for sweep in 1..=sweeps {
            for (pos, &w) in ids.iter().enumerate() {
                ndk[z[pos]] -= 1;
                let t = self.draw(w, &ndk, splitmix(base_seed ^ pos as u64 ^ ((sweep as u64) << 40)), &mut cum);
                ndk[t] += 1;
                z[pos] = t;
            }
        }
        (z, ndk)
    }

    /// The sparse mixture θ for local counts: top TOP_T topics by (n_dk + α), weights normalized.
    fn mixture(&self, ndk: &[u32], len: usize) -> Vec<(usize, f32)> {
        if len == 0 { return Vec::new(); }
        let denom = len as f32 + self.k as f32 * self.alpha;
        let mut m: Vec<(usize, f32)> = ndk.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(t, &c)| (t, (c as f32 + self.alpha) / denom))
            .collect();
        m.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        m.truncate(TOP_T);
        m
    }

    /// STREAMING learning: fold one fact's tokens in against the current counts, commit the
    /// assignments, return the fact's mixture (top topics, best first; [0] is the primary —
    /// what a posting records). New words enter the vocabulary here; hub words are tracked in
    /// df but never assigned. Deterministic per (model state, tokens).
    pub fn absorb<S: AsRef<str>>(&mut self, tokens: &[S]) -> Vec<(usize, f32)> {
        // vocab + df first (df counts a word once per document)
        let mut ids: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut base_seed = 0x70D1C5u64;
        for t in tokens {
            let t = t.as_ref();
            if t.is_empty() { continue; }
            base_seed = fnv_chain(base_seed, t);
            let id = match self.vocab.get(t) {
                Some(&id) => id,
                None => {
                    if self.words.len() >= VOCAB_CAP { continue; }
                    let id = self.words.len() as u32;
                    self.vocab.insert(t.to_string(), id);
                    self.words.push(t.to_string());
                    self.df.push(0);
                    self.nkw.extend(std::iter::repeat(0).take(self.k));
                    id
                }
            };
            ids.push(id);
        }
        if ids.is_empty() { return Vec::new(); }
        self.docs_seen += 1;
        let mut seen = ids.clone();
        seen.sort_unstable(); seen.dedup();
        for &id in &seen { self.df[id as usize] += 1; }
        let active: Vec<u32> = ids.into_iter().filter(|&id| !self.is_hub(id)).collect();
        if active.is_empty() { return Vec::new(); }
        let (z, ndk) = self.gibbs_frozen(&active, base_seed, ABSORB_SWEEPS);
        for (pos, &w) in active.iter().enumerate() {
            self.nkw[w as usize * self.k + z[pos]] += 1;
            self.nk[z[pos]] += 1;
        }
        self.tokens_absorbed += active.len() as u64;
        self.mixture(&ndk, active.len())
    }

    /// FROZEN inference: the mixture a document of `tokens` folds to under the current counts.
    /// Unknown and hub words are skipped; nothing is learned. Deterministic per (state, tokens).
    pub fn fold_in<S: AsRef<str>>(&self, tokens: &[S]) -> Vec<(usize, f32)> {
        let mut ids: Vec<u32> = Vec::new();
        let mut base_seed = 0xF01Du64;
        for t in tokens {
            let t = t.as_ref();
            if t.is_empty() { continue; }
            base_seed = fnv_chain(base_seed, t);
            if let Some(&id) = self.vocab.get(t) {
                if !self.is_hub(id) { ids.push(id); }
            }
        }
        if ids.is_empty() { return Vec::new(); }
        let (_, ndk) = self.gibbs_frozen(&ids, base_seed, FOLD_SWEEPS);
        self.mixture(&ndk, ids.len())
    }

    /// BATCH (re)build from pseudo-documents: df-gated vocabulary (a word must appear in ≥2
    /// documents and in at most a quarter of them, floored at HUB_FLOOR), then full collapsed
    /// Gibbs — counts update live across `sweeps`. Replaces all state. Deterministic.
    pub fn refit<S: AsRef<str>>(&mut self, docs: &[Vec<S>], sweeps: usize) {
        let k = self.k;
        self.vocab.clear(); self.words.clear(); self.df.clear();
        self.nkw.clear(); self.nk = vec![0; k];
        self.docs_seen = docs.len() as u64;
        self.tokens_absorbed = 0;
        // document frequency over the batch
        let mut dfm: HashMap<&str, u32> = HashMap::new();
        for doc in docs {
            let mut seen: Vec<&str> = doc.iter().map(|t| t.as_ref()).filter(|t| !t.is_empty()).collect();
            seen.sort_unstable(); seen.dedup();
            for t in seen { *dfm.entry(t).or_insert(0) += 1; }
        }
        let cap = self.hub_cap();
        let mut kept: Vec<(&str, u32)> = dfm.iter().map(|(w, c)| (*w, *c)).filter(|(_, c)| *c >= 2 && (*c as u64) <= cap).collect();
        kept.sort_unstable();   // deterministic id assignment regardless of hash order
        for (w, c) in kept {
            let id = self.words.len() as u32;
            self.vocab.insert(w.to_string(), id);
            self.words.push(w.to_string());
            self.df.push(c);
            self.nkw.extend(std::iter::repeat(0).take(k));
        }
        // documents as id lists, with per-doc content seeds
        let idocs: Vec<(Vec<u32>, u64)> = docs.iter().map(|doc| {
            let mut seed = 0x2EF17u64;
            let ids: Vec<u32> = doc.iter().filter_map(|t| {
                let t = t.as_ref();
                if t.is_empty() { return None; }
                seed = fnv_chain(seed, t);
                self.vocab.get(t).copied()
            }).collect();
            (ids, seed)
        }).collect();
        // live-count Gibbs: init pass assigns and adds; sweeps decrement/resample/increment
        let mut zs: Vec<Vec<usize>> = Vec::with_capacity(idocs.len());
        let mut ndks: Vec<Vec<u32>> = Vec::with_capacity(idocs.len());
        let mut cum = vec![0f64; k];
        for (ids, seed) in &idocs {
            let mut ndk = vec![0u32; k];
            let mut z = Vec::with_capacity(ids.len());
            for (pos, &w) in ids.iter().enumerate() {
                let t = self.draw(w, &ndk, splitmix(*seed ^ pos as u64), &mut cum);
                ndk[t] += 1;
                self.nkw[w as usize * k + t] += 1;
                self.nk[t] += 1;
                z.push(t);
            }
            self.tokens_absorbed += ids.len() as u64;
            zs.push(z);
            ndks.push(ndk);
        }
        for sweep in 1..=sweeps {
            for (d, (ids, seed)) in idocs.iter().enumerate() {
                for (pos, &w) in ids.iter().enumerate() {
                    let old = zs[d][pos];
                    ndks[d][old] -= 1;
                    self.nkw[w as usize * k + old] -= 1;
                    self.nk[old] -= 1;
                    let t = self.draw(w, &ndks[d], splitmix(*seed ^ pos as u64 ^ ((sweep as u64) << 40)), &mut cum);
                    ndks[d][t] += 1;
                    self.nkw[w as usize * k + t] += 1;
                    self.nk[t] += 1;
                    zs[d][pos] = t;
                }
            }
        }
    }

    /// The strongest topic for a word currently in the vocabulary (inspection / tests).
    pub fn word_topic(&self, word: &str) -> Option<usize> {
        let &id = self.vocab.get(word)?;
        let base = id as usize * self.k;
        let (mut best, mut bc) = (0usize, 0u32);
        for t in 0..self.k {
            if self.nkw[base + t] > bc { bc = self.nkw[base + t]; best = t; }
        }
        if bc == 0 { None } else { Some(best) }
    }

    /// Top `m` words of a topic by count, with φ = (n_kw + β)/(n_k + Vβ). Inspection surface.
    pub fn top_words(&self, topic: usize, m: usize) -> Vec<(String, f32)> {
        if topic >= self.k { return Vec::new(); }
        let vb = self.words.len() as f64 * self.beta as f64;
        let denom = self.nk[topic] as f64 + vb;
        let mut out: Vec<(String, f32, u32)> = (0..self.words.len()).filter_map(|w| {
            let c = self.nkw[w * self.k + topic];
            if c == 0 { return None; }
            Some((self.words[w].clone(), ((c as f64 + self.beta as f64) / denom) as f32, c))
        }).collect();
        out.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        out.truncate(m);
        out.into_iter().map(|(w, p, _)| (w, p)).collect()
    }

    /// Each topic's share of all assigned tokens (inspection: "what is this corpus about").
    pub fn shares(&self) -> Vec<f32> {
        let total: u64 = self.nk.iter().sum();
        if total == 0 { return vec![0.0; self.k]; }
        self.nk.iter().map(|&c| c as f32 / total as f32).collect()
    }

    /// Persistence in the store's tab-line convention; counts are sparse per word. Exact reload.
    pub fn dump(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = write!(out, "topics\t{}\t{}\t{}\t{}\t{}", self.k, self.alpha, self.beta, self.docs_seen, self.tokens_absorbed);
        for w in 0..self.words.len() {
            let key = self.words[w].replace(['\t', '\n'], " ");
            let _ = write!(out, "\nw\t{}\t{}\t", key, self.df[w]);
            let mut first = true;
            for t in 0..self.k {
                let c = self.nkw[w * self.k + t];
                if c > 0 {
                    if !first { out.push(' '); }
                    first = false;
                    let _ = write!(out, "{}:{}", t, c);
                }
            }
        }
        out
    }

    pub fn load(blob: &str) -> Option<Self> {
        let mut lines = blob.split('\n');
        let head = lines.next()?;
        let mut h = head.split('\t');
        if h.next()? != "topics" { return None; }
        let k: usize = h.next()?.parse().ok()?;
        let alpha: f32 = h.next()?.parse().ok()?;
        let beta: f32 = h.next()?.parse().ok()?;
        let docs_seen: u64 = h.next()?.parse().ok()?;
        let tokens_absorbed: u64 = h.next()?.parse().ok()?;
        let mut m = TopicModel::with_params(k, alpha, beta);
        m.docs_seen = docs_seen;
        m.tokens_absorbed = tokens_absorbed;
        for line in lines {
            if line.is_empty() { continue; }
            let mut f = line.splitn(4, '\t');
            if f.next()? != "w" { continue; }
            let word = f.next()?.to_string();
            let df: u32 = f.next()?.parse().ok()?;
            let counts = f.next().unwrap_or("");
            let id = m.words.len() as u32;
            m.vocab.insert(word.clone(), id);
            m.words.push(word);
            m.df.push(df);
            m.nkw.extend(std::iter::repeat(0).take(k));
            if !counts.is_empty() {
                for pair in counts.split(' ') {
                    let mut p = pair.splitn(2, ':');
                    let (t, c): (usize, u32) = match (p.next().and_then(|x| x.parse().ok()), p.next().and_then(|x| x.parse().ok())) {
                        (Some(t), Some(c)) => (t, c),
                        _ => continue,
                    };
                    if t < k { m.nkw[id as usize * k + t] = c; m.nk[t] += c as u64; }
                }
            }
        }
        Some(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the two-cluster corpus the semantic tier's tests use: networking vs cooking. LDA must
    // find the same structure as TOPICS rather than nearest-neighbour geometry.
    const CORPUS: &[&str] = &[
        "I use wifi to get online. The wifi connects my laptop to the internet.",
        "Being online means you are connected to the internet through wifi or a router.",
        "The router broadcasts wifi so devices can reach the web and browse the internet.",
        "We browse the web online using the wireless wifi network from the router.",
        "Meanwhile the chef chopped onions and garlic for the soup.",
        "The recipe needs onions, garlic, salt, and fresh basil simmered in the pot.",
        "Cooking the soup, the chef stirred garlic and basil into the simmering pot.",
        "A good recipe balances salt and basil while the soup simmers on the stove.",
    ];

    // test-local tokenizer: lowercase alnum runs, len >= 2, tiny stop list — the shape of the
    // store's own content() output (the model itself only ever sees token slices).
    fn toks(s: &str) -> Vec<String> {
        const STOP: &[&str] = &["the","a","an","and","or","of","to","in","on","at","is","are",
            "was","be","it","its","this","that","so","can","we","you","my","for","into","from",
            "while","means","use","get","being","using"];
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 2)
            .map(|t| t.to_lowercase())
            .filter(|t| !STOP.contains(&t.as_str()))
            .collect()
    }
    fn docs() -> Vec<Vec<String>> { CORPUS.iter().map(|s| toks(s)).collect() }

    fn fitted() -> TopicModel {
        let mut m = TopicModel::new(4);
        m.refit(&docs(), 400);
        m
    }

    // The acceptance bar from the proposal: networking and cooking land in DIFFERENT topics
    // with the expected top words. Asserted as cluster PURITY — with K larger than the true
    // cluster count, LDA may legitimately split one cluster across two topics (that is still a
    // correct model); what it must never do is mix the clusters inside one topic.
    #[test]
    fn separates_the_two_clusters() {
        const NET: &[&str] = &["wifi","internet","router","web","online","wireless","laptop","devices","browse","network"];
        const COOK: &[&str] = &["garlic","basil","soup","recipe","onions","salt","chef","pot","stove","cooking"];
        let m = fitted();
        for anchor in ["wifi", "internet", "router"] {
            let t = m.word_topic(anchor).unwrap_or_else(|| panic!("{anchor} must be in vocab"));
            let top: Vec<String> = m.top_words(t, 6).into_iter().map(|(w, _)| w).collect();
            assert!(top.iter().any(|w| NET.contains(&w.as_str())),
                    "{anchor}'s topic should look like networking, top words {top:?}");
            assert!(!top.iter().any(|w| COOK.contains(&w.as_str())),
                    "cooking words leaked into {anchor}'s topic: {top:?}");
        }
        for anchor in ["garlic", "basil", "soup"] {
            let t = m.word_topic(anchor).unwrap_or_else(|| panic!("{anchor} must be in vocab"));
            let top: Vec<String> = m.top_words(t, 6).into_iter().map(|(w, _)| w).collect();
            assert!(top.iter().any(|w| COOK.contains(&w.as_str())),
                    "{anchor}'s topic should look like cooking, top words {top:?}");
            assert!(!top.iter().any(|w| NET.contains(&w.as_str())),
                    "networking words leaked into {anchor}'s topic: {top:?}");
        }
        assert_ne!(m.word_topic("wifi"), m.word_topic("garlic"),
                   "the two clusters must occupy different topics");
    }

    // Same corpus, same code -> identical count tables. The store's determinism contract.
    #[test]
    fn refit_is_deterministic() {
        let (a, b) = (fitted(), fitted());
        assert_eq!(a.nkw, b.nkw);
        assert_eq!(a.nk, b.nk);
        assert_eq!(a.words, b.words);
    }

    #[test]
    fn absorb_is_deterministic_and_learns() {
        let mut a = TopicModel::new(4);
        let mut b = TopicModel::new(4);
        for d in docs() { a.absorb(&d); }
        for d in docs() { b.absorb(&d); }
        assert_eq!(a.nkw, b.nkw);
        assert!(a.tokens() > 0 && a.vocab_len() > 0);
        // streaming absorb of the same corpus a few times over must still separate the clusters
        let mut m = TopicModel::new(4);
        for _ in 0..6 { for d in docs() { m.absorb(&d); } }
        let (wifi, garlic) = (m.word_topic("wifi").unwrap(), m.word_topic("garlic").unwrap());
        assert_ne!(wifi, garlic, "streaming absorb must separate networking from cooking");
    }

    // A paraphrase query folds into the networking topic — the gate for topic-gated recall.
    #[test]
    fn fold_in_lands_in_the_right_topic() {
        let m = fitted();
        let wifi = m.word_topic("wifi").unwrap();
        let q = toks("browse the web online with the wifi network");
        let mix = m.fold_in(&q);
        assert!(!mix.is_empty(), "known words must fold to a mixture");
        assert_eq!(mix[0].0, wifi, "query mixture was {mix:?}, expected topic {wifi}");
        // an unknown-word query folds to nothing (the gate fails open, never guesses)
        assert!(m.fold_in(&toks("zzz qqq xyzzy")).is_empty());
        assert!(m.fold_in::<&str>(&[]).is_empty());
    }

    // Hapax words and hub words never enter a refit vocabulary: no topical signal, no slot.
    #[test]
    fn refit_gates_hapax_and_hubs() {
        // 100 docs, every one carrying the schema word "unit"; two real clusters underneath
        let mut d: Vec<Vec<String>> = Vec::new();
        for i in 0..100 {
            let body = if i % 2 == 0 { "alpha beta gamma" } else { "delta epsilon zeta" };
            d.push(toks(&format!("unit {body} once{i}")));   // once{i} is a hapax every time
        }
        let mut m = TopicModel::new(4);
        m.refit(&d, 100);
        assert!(m.word_topic("unit").is_none(), "a word in 100/100 docs is a hub (cap 64)");
        assert!(m.word_topic("once3").is_none(), "hapax words carry no signal");
        assert!(m.word_topic("alpha").is_some() && m.word_topic("delta").is_some());
        assert_ne!(m.word_topic("alpha"), m.word_topic("delta"), "the two blocks must separate");
    }

    // The model survives a restart exactly: counts, vocabulary, and fold behavior.
    #[test]
    fn dump_load_round_trips() {
        let m = fitted();
        let blob = m.dump();
        let r = TopicModel::load(&blob).expect("dump must reload");
        assert_eq!(m.nkw, r.nkw);
        assert_eq!(m.nk, r.nk);
        assert_eq!(m.words, r.words);
        assert_eq!(m.docs(), r.docs());
        let q = toks("the router broadcasts the wifi signal");
        assert_eq!(m.fold_in(&q), r.fold_in(&q), "a reloaded model must fold identically");
        assert!(TopicModel::load("nope").is_none());
        assert!(TopicModel::load("").is_none());
    }
}
