//! A continuous **semantic space** built by corpus-distributional learning — no model, no
//! external dependency, std-only. This is how meaning is grounded in a brain: not by a
//! dictionary, but by the company a word keeps. We use *Random Indexing* (a cheap,
//! incremental alternative to word2vec/LSA): every word owns a fixed sparse random "index
//! vector", and a word's dense **context vector** is the running sum of the index vectors of
//! the words it co-occurs with. Words used in similar contexts end up near each other in the
//! space, so paraphrases that share no characters ("get online" ↔ "wifi") can still match.
//!
//! Feature-gated behind `semantic`; pure std (HashMap + f32 vectors).
use std::collections::HashMap;

const DIM: usize = 256;       // dimensionality of the semantic space
const NONZ: usize = 12;       // nonzeros in each word's sparse random index vector
const WINDOW: usize = 5;      // co-occurrence window (each side)

fn fnv(s: &str) -> u64 {
    let mut h = 1469598103934665603u64;
    for b in s.bytes() { h ^= b as u64; h = h.wrapping_mul(1099511628211); }
    h
}
fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Stopwords carry little distributional signal and add noise; skip them when training.
fn is_stop(w: &str) -> bool {
    matches!(w,
        "the"|"a"|"an"|"and"|"or"|"of"|"to"|"in"|"on"|"at"|"is"|"are"|"was"|"were"|"be"|"been"|
        "it"|"its"|"this"|"that"|"these"|"those"|"as"|"by"|"for"|"with"|"from"|"but"|"not"|"no"|
        "so"|"if"|"then"|"than"|"too"|"very"|"can"|"will"|"would"|"could"|"should"|"i"|"you"|"he"|
        "she"|"we"|"they"|"him"|"her"|"his"|"them"|"my"|"your"|"our"|"their"|"me"|"us"|"do"|"did"|
        "does"|"have"|"has"|"had"|"there"|"here"|"what"|"which"|"who"|"when"|"where"|"how"|"all"|
        "any"|"some"|"such"|"only"|"own"|"same"|"out"|"up"|"down"|"off"|"over"|"under"|"into"|"upon")
}
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_lowercase())
        .filter(|t| !is_stop(t))
        .collect()
}

pub struct SemanticSpace {
    ctx: HashMap<String, Vec<f32>>,   // word -> dense context vector
    tokens_seen: u64,
}
impl Default for SemanticSpace { fn default() -> Self { Self::new() } }

impl SemanticSpace {
    pub fn new() -> Self { SemanticSpace { ctx: HashMap::new(), tokens_seen: 0 } }
    pub fn vocab(&self) -> usize { self.ctx.len() }
    pub fn tokens(&self) -> u64 { self.tokens_seen }
    pub fn dim(&self) -> usize { DIM }
    /// approximate resident bytes of the space (DIM f32 per word + key)
    pub fn bytes(&self) -> usize { self.ctx.iter().map(|(k, _)| k.len() + DIM * 4 + 48).sum() }

    /// add the sparse random index vector of `word` into `target`
    fn add_index(target: &mut [f32], word: &str) {
        let seed = fnv(word);
        for k in 0..NONZ {
            let r = splitmix(seed ^ (k as u64).wrapping_mul(0x100000001B3));
            let pos = (r % DIM as u64) as usize;
            let sign = if (r >> 33) & 1 == 0 { 1.0 } else { -1.0 };
            target[pos] += sign;
        }
    }

    /// Learn from a span of text: each word accumulates the index vectors of its neighbours.
    pub fn train(&mut self, text: &str) {
        let toks = tokenize(text);
        self.tokens_seen += toks.len() as u64;
        for i in 0..toks.len() {
            let lo = i.saturating_sub(WINDOW);
            let hi = (i + WINDOW + 1).min(toks.len());
            let v = self.ctx.entry(toks[i].clone()).or_insert_with(|| vec![0.0; DIM]);
            for j in lo..hi {
                if j != i { Self::add_index(v, &toks[j]); }
            }
        }
    }

    fn norm(v: &[f32]) -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() }

    /// Embed text into the space: the L2-normalized sum of its known words' context vectors
    /// (each normalized first so frequent words don't dominate). None if no word is known.
    pub fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let mut acc = vec![0.0f32; DIM];
        let mut any = false;
        for t in tokenize(text) {
            if let Some(cv) = self.ctx.get(&t) {
                let n = Self::norm(cv);
                if n > 0.0 { for d in 0..DIM { acc[d] += cv[d] / n; } any = true; }
            }
        }
        if !any { return None; }
        let n = Self::norm(&acc);
        if n == 0.0 { return None; }
        for d in 0..DIM { acc[d] /= n; }
        Some(acc)
    }

    /// Cosine similarity of two texts in the space (-1..1), or None if either is unknown.
    pub fn similarity(&self, a: &str, b: &str) -> Option<f32> {
        let (x, y) = (self.embed(a)?, self.embed(b)?);
        Some(x.iter().zip(&y).map(|(p, q)| p * q).sum())
    }

    /// Rank candidates by semantic similarity to the query (highest first), with scores.
    pub fn rank(&self, query: &str, cands: &[String]) -> Vec<(usize, f32)> {
        let q = match self.embed(query) { Some(q) => q, None => return Vec::new() };
        let mut scored: Vec<(usize, f32)> = cands.iter().enumerate().filter_map(|(i, c)| {
            self.embed(c).map(|e| (i, e.iter().zip(&q).map(|(p, r)| p * r).sum::<f32>()))
        }).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// The k nearest words to a given word, by cosine (for inspecting the space).
    pub fn nearest(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        let base = match self.ctx.get(&word.to_lowercase()) { Some(v) => v, None => return Vec::new() };
        let bn = Self::norm(base);
        if bn == 0.0 { return Vec::new(); }
        let mut out: Vec<(String, f32)> = self.ctx.iter()
            .filter(|(w, _)| w.as_str() != word)
            .filter_map(|(w, v)| {
                let n = Self::norm(v);
                if n == 0.0 { return None; }
                let dot: f32 = base.iter().zip(v).map(|(a, b)| a * b).sum();
                Some((w.clone(), dot / (bn * n)))
            }).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a small corpus where "online", "internet", "wifi", "web", "router" co-occur, and a
    // separate cluster about cooking, so the space must separate the two senses.
    const CORPUS: &str = "
        I use wifi to get online. The wifi connects my laptop to the internet.
        Being online means you are connected to the internet through wifi or a router.
        The router broadcasts wifi so devices can reach the web and browse the internet.
        We browse the web online using the wireless wifi network from the router.
        Meanwhile the chef chopped onions and garlic for the soup.
        The recipe needs onions, garlic, salt, and fresh basil simmered in the pot.
        Cooking the soup, the chef stirred garlic and basil into the simmering pot.
        A good recipe balances salt and basil while the soup simmers on the stove.
    ";

    fn trained() -> SemanticSpace {
        let mut s = SemanticSpace::new();
        for _ in 0..30 { s.train(CORPUS); }   // repeat to strengthen co-occurrence stats
        s
    }

    #[test]
    fn clusters_co_occurring_words() {
        let s = trained();
        let near = s.nearest("online", 5);
        let words: Vec<&str> = near.iter().map(|(w, _)| w.as_str()).collect();
        // "online" should be near internet/wifi/web, not onions/garlic
        assert!(words.iter().any(|w| ["internet","wifi","web","router","wireless","connected"].contains(w)),
                "online neighbours were {:?}", words);
        assert!(!words.iter().take(3).any(|w| ["onions","garlic","basil","soup"].contains(w)),
                "cooking words leaked into 'online' neighbours: {:?}", words);
    }

    #[test]
    fn paraphrase_outranks_unrelated() {
        // the query shares NO content words with the target fact, only meaning
        let s = trained();
        let facts = vec![
            "the recipe needs garlic and basil".to_string(),
            "the wifi network reaches the web".to_string(),
        ];
        let ranked = s.rank("the thing I use to get online", &facts);
        assert_eq!(ranked[0].0, 1, "expected the wifi fact to rank first, got {:?}", ranked);
    }

    #[test]
    fn unknown_query_is_none() {
        let s = trained();
        assert!(s.embed("zzzqqq xyzzy").is_none());
        assert!(s.similarity("online", "zzzqqq").is_none());
    }
}
