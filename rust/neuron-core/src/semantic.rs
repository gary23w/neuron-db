// Hand-written vector kernels below index parallel coordinate arrays by position; the index-based
// loops mirror the linear-algebra math and read clearer than enumerate/zip chains.
#![allow(clippy::needless_range_loop)]
//! A continuous semantic space built from word co-occurrence: no model, no external dependency,
//! std-only. Uses Random Indexing (a cheap, incremental alternative to word2vec/LSA): every word
//! owns a fixed sparse random index vector, and a word's dense context vector is the running sum
//! of the index vectors of the words it co-occurs with. Words used in similar contexts end up near
//! each other, so paraphrases that share no characters ("get online" / "wifi") can still match.
//!
//! Feature-gated behind `semantic`; pure std (HashMap + f32 vectors).
use std::collections::HashMap;

const DEFAULT_DIM: usize = 256;   // default dimensionality (configurable per-space via with_dim)
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

/// Cosine k-means over unit vectors (deterministic seeding). Returns a cluster id per row.
/// Used to colour the visualization by TRUE 256-D meaning, not the 3-D projection.
fn kmeans_unit(data: &[Vec<f32>], k: usize, iters: usize) -> Vec<u8> {
    let n = data.len();
    if n == 0 { return Vec::new(); }
    let k = k.min(n);
    let dim = data[0].len();
    let mut cent: Vec<Vec<f32>> = (0..k).map(|c| data[(n * c / k).min(n - 1)].clone()).collect();
    let mut assign = vec![0u8; n];
    for _ in 0..iters {
        for i in 0..n {
            let (mut best, mut bd) = (0usize, f32::MIN);
            for c in 0..k {
                let dot: f32 = data[i].iter().zip(&cent[c]).map(|(a, b)| a * b).sum();
                if dot > bd { bd = dot; best = c; }
            }
            assign[i] = best as u8;
        }
        let mut sums = vec![vec![0f32; dim]; k];
        let mut cnt = vec![0u32; k];
        for i in 0..n { let c = assign[i] as usize; cnt[c] += 1; for d in 0..dim { sums[c][d] += data[i][d]; } }
        for c in 0..k {
            if cnt[c] == 0 { continue; }
            let mut nrm = 0f32;
            for d in 0..dim { sums[c][d] /= cnt[c] as f32; nrm += sums[c][d] * sums[c][d]; }
            nrm = nrm.sqrt().max(1e-9);
            for d in 0..dim { sums[c][d] /= nrm; }
            cent[c] = std::mem::take(&mut sums[c]);
        }
    }
    assign
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
    dim: usize,                       // this space's dimensionality (DEFAULT_DIM unless with_dim)
    ctx: HashMap<String, Vec<f32>>,   // word -> dense context vector
    cnt: HashMap<String, u32>,        // word -> occurrence count (for picking top words)
    tokens_seen: u64,
    emb_cache: HashMap<String, (Vec<i8>, f32, u64)>, // fact -> (int8 unit embedding, scale, epoch)
    ctx_q: HashMap<String, (Vec<i8>, f32)>,          // int8 context vectors after compact() (serving mode)
}
impl Default for SemanticSpace { fn default() -> Self { Self::new() } }

/// A low-dimensional projection of the space (PCA): the chosen words, their coordinates on
/// the top-K principal components, and how much variance each component explains.
pub struct Projection {
    pub words: Vec<String>,
    pub coords: Vec<Vec<f32>>,    // per word: K coordinates
    pub clusters: Vec<u8>,        // per word: k-means cluster id, computed in TRUE 256-D
    pub variance: Vec<f32>,       // per component: explained variance
    pub total_variance: f32,
}

impl SemanticSpace {
    pub fn new() -> Self { Self::with_dim(DEFAULT_DIM) }
    /// Build a space with a chosen dimensionality — the DIM knob. All vectors within one space
    /// share this dim (mixing spaces of different dim is not meaningful). Default (new) is 256.
    pub fn with_dim(dim: usize) -> Self { SemanticSpace { dim, ctx: HashMap::new(), cnt: HashMap::new(), tokens_seen: 0, emb_cache: HashMap::new(), ctx_q: HashMap::new() } }
    pub fn vocab(&self) -> usize { self.ctx.len() + self.ctx_q.len() }
    pub fn count(&self, w: &str) -> u32 { self.cnt.get(w).copied().unwrap_or(0) }
    pub fn tokens(&self) -> u64 { self.tokens_seen }
    pub fn dim(&self) -> usize { self.dim }
    /// Approximate resident bytes: the f32 context vectors plus the int8 embedding cache.
    pub fn bytes(&self) -> usize {
        self.ctx.keys().map(|k| k.len() + self.dim * 4 + 48).sum::<usize>()
            + self.ctx_q.keys().map(|k| k.len() + self.dim + 16).sum::<usize>()
            + self.emb_cache.keys().map(|k| k.len() + self.dim + 16).sum::<usize>()
    }
    pub fn cached_embeddings(&self) -> usize { self.emb_cache.len() }

    /// add the sparse random index vector of `word` into `target`
    fn add_index(target: &mut [f32], word: &str, dim: usize) {
        let seed = fnv(word);
        for k in 0..NONZ {
            let r = splitmix(seed ^ (k as u64).wrapping_mul(0x100000001B3));
            let pos = (r % dim as u64) as usize;
            let sign = if (r >> 33) & 1 == 0 { 1.0 } else { -1.0 };
            target[pos] += sign;
        }
    }

    /// Compact the context vectors to int8 for read-mostly serving (~4x smaller). Recall still
    /// works (embed dequantizes inline); a later train() transparently re-expands to f32.
    pub fn compact(&mut self) {
        for (w, v) in self.ctx.drain().collect::<Vec<_>>() {
            self.ctx_q.insert(w, Self::quantize(&v));
        }
    }
    fn expand(&mut self) {
        for (w, (q, s)) in self.ctx_q.drain().collect::<Vec<_>>() {
            self.ctx.insert(w, q.iter().map(|x| *x as f32 * s).collect());
        }
    }

    /// Learn from a span of text: each word accumulates the index vectors of its neighbours.
    pub fn train(&mut self, text: &str) {
        if !self.ctx_q.is_empty() { self.expand(); } // resume full-precision training if compacted
        let dim = self.dim;
        let toks = tokenize(text);
        self.tokens_seen += toks.len() as u64;
        for i in 0..toks.len() {
            let lo = i.saturating_sub(WINDOW);
            let hi = (i + WINDOW + 1).min(toks.len());
            *self.cnt.entry(toks[i].clone()).or_insert(0) += 1;
            let v = self.ctx.entry(toks[i].clone()).or_insert_with(|| vec![0.0; dim]);
            for j in lo..hi {
                if j != i { Self::add_index(v, &toks[j], dim); }
            }
        }
    }

    /// PCA projection of the `top_n` most frequent words onto the top `k` principal
    /// components, by power iteration + deflation (pure std, no linalg crate). Each word's
    /// context vector is L2-normalized first so meaning (direction), not frequency, drives
    /// the layout.
    pub fn project(&self, top_n: usize, k: usize) -> Projection {
        // pick the most frequent words that have a context vector
        let mut words: Vec<(&String, u32)> = self.cnt.iter().map(|(w, c)| (w, *c)).collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let words: Vec<String> = words.iter().map(|(w, _)| (*w).clone()).take(top_n)
            .filter(|w| self.ctx.contains_key(w)).collect();
        let n = words.len();
        if n == 0 || k == 0 {
            return Projection { words, coords: Vec::new(), clusters: Vec::new(), variance: Vec::new(), total_variance: 0.0 };
        }
        let dim = self.dim;
        // build the centered, direction-normalized data matrix (n x DIM)
        let mut x: Vec<Vec<f32>> = words.iter().map(|w| {
            let cv = &self.ctx[w];
            let nrm = Self::norm(cv).max(1e-9);
            cv.iter().map(|v| v / nrm).collect::<Vec<f32>>()
        }).collect();
        // cluster in TRUE 256-D (on the unit sphere = the cosine geometry the model scores on),
        // BEFORE mean-centering — so colour reflects real meaning, not the 3-D projection.
        let clusters = kmeans_unit(&x, 6, 14);
        let mut mean = vec![0.0f32; dim];
        for row in &x { for d in 0..dim { mean[d] += row[d]; } }
        for d in 0..dim { mean[d] /= n as f32; }
        for row in &mut x { for d in 0..dim { row[d] -= mean[d]; } }
        let centered = x.clone();
        let total_variance: f32 = centered.iter().map(|r| r.iter().map(|v| v * v).sum::<f32>()).sum::<f32>() / n as f32;

        // find K principal components on a working copy (deflated as we go)
        let mut comps: Vec<Vec<f32>> = Vec::new();
        let mut variance: Vec<f32> = Vec::new();
        for c in 0..k {
            // deterministic random unit init
            let mut v = vec![0.0f32; dim];
            let mut seed = 0x5EED_C0DEu64 ^ (c as u64).wrapping_mul(0x9E37_79B9);
            for d in 0..dim { seed = splitmix(seed); v[d] = ((seed >> 11) as f32 / (1u64 << 53) as f32) - 0.5; }
            let mut nv = Self::norm(&v).max(1e-9); for d in 0..dim { v[d] /= nv; }
            // power iteration on the covariance of the (deflated) data
            for _ in 0..40 {
                let u: Vec<f32> = x.iter().map(|row| row.iter().zip(&v).map(|(a, b)| a * b).sum::<f32>()).collect();
                let mut w = vec![0.0f32; dim];
                for (i, row) in x.iter().enumerate() { let ui = u[i]; for d in 0..dim { w[d] += ui * row[d]; } }
                nv = Self::norm(&w).max(1e-9); for d in 0..dim { v[d] = w[d] / nv; }
            }
            // canonicalize sign (power iteration converges to +-v): largest loading positive,
            // so reloads and axis tweens never randomly mirror the layout.
            let mut piv = 0usize;
            for d in 1..dim { if v[d].abs() > v[piv].abs() { piv = d; } }
            if v[piv] < 0.0 { for d in 0..dim { v[d] = -v[d]; } }
            // explained variance along v, then deflate
            let proj: Vec<f32> = x.iter().map(|row| row.iter().zip(&v).map(|(a, b)| a * b).sum::<f32>()).collect();
            variance.push(proj.iter().map(|p| p * p).sum::<f32>() / n as f32);
            for (i, row) in x.iter_mut().enumerate() { let p = proj[i]; for d in 0..dim { row[d] -= p * v[d]; } }
            comps.push(v);
        }
        // project the ORIGINAL centered rows onto each component
        let coords: Vec<Vec<f32>> = centered.iter().map(|row|
            comps.iter().map(|comp| row.iter().zip(comp).map(|(a, b)| a * b).sum::<f32>()).collect()
        ).collect();
        Projection { words, coords, clusters, variance, total_variance }
    }

    fn norm(v: &[f32]) -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() }

    /// Symmetric int8 quantization of a vector: one f32 scale per vector, values in [-127, 127].
    /// Cuts a cached embedding from DIM*4 bytes to DIM+4, with negligible cosine-ranking loss.
    fn quantize(v: &[f32]) -> (Vec<i8>, f32) {
        let max = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        if max == 0.0 { return (vec![0i8; v.len()], 0.0); }
        let scale = max / 127.0;
        (v.iter().map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8).collect(), scale)
    }

    /// Embed text into the space: the L2-normalized sum of its known words' context vectors
    /// (each normalized first so frequent words don't dominate). None if no word is known.
    pub fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let dim = self.dim;
        let mut acc = vec![0.0f32; dim];
        let mut any = false;
        for t in tokenize(text) {
            if let Some(cv) = self.ctx.get(&t) {
                let n = Self::norm(cv);
                if n > 0.0 { for d in 0..dim { acc[d] += cv[d] / n; } any = true; }
            } else if let Some((q, s)) = self.ctx_q.get(&t) {
                let mut nrm = 0f32; for x in q { let f = *x as f32 * s; nrm += f * f; }
                let nrm = nrm.sqrt();
                if nrm > 0.0 { for d in 0..dim { acc[d] += (q[d] as f32 * s) / nrm; } any = true; }
            }
        }
        if !any { return None; }
        let n = Self::norm(&acc);
        if n == 0.0 { return None; }
        for d in 0..dim { acc[d] /= n; }
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

    /// Precompute and cache a fact's int8 embedding so later fuzzy recall is dot products, not
    /// re-embeddings. Cheap at write time.
    pub fn cache_embed(&mut self, text: &str) {
        if let Some(e) = self.embed(text) {
            let (q, s) = Self::quantize(&e);
            self.emb_cache.insert(text.to_string(), (q, s, self.tokens_seen));
        }
    }
    /// Drop the embedding cache so it rebuilds against the current space.
    pub fn clear_cache(&mut self) { self.emb_cache.clear(); }

    const EMB_CACHE_CAP: usize = 50_000;   // bound the fact-embedding cache so a long-lived server can't grow it without limit
    /// Ensure every candidate has a fresh cached int8 embedding: re-embed when uncached or the
    /// space has more than doubled since (the drift bound rank_cached always used).
    fn ensure_cached(&mut self, cands: &[&str]) {
        let ts = self.tokens_seen;
        for &c in cands {
            let need = match self.emb_cache.get(c) { Some((_, _, ep)) => ts >= ep.saturating_mul(2), None => true };
            if need {
                if let Some(e) = self.embed(c) { let (qz, s) = Self::quantize(&e); self.emb_cache.insert(c.to_string(), (qz, s, ts)); }
            }
        }
    }
    /// Rank candidates by cosine to the query, reusing cached int8 embeddings. The query is
    /// embedded fresh each call; candidates refresh through the drift bound in ensure_cached.
    /// After warm-up this is O(N) dot products.
    pub fn rank_cached(&mut self, query: &str, cands: &[&str]) -> Vec<(usize, f32)> {
        let q = match self.embed(query) { Some(q) => q, None => return Vec::new() };
        self.ensure_cached(cands);
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(cands.len());
        for (i, &c) in cands.iter().enumerate() {
            if let Some((e, s, _)) = self.emb_cache.get(c) {
                let dot = e.iter().zip(&q).map(|(c8, qf)| *c8 as f32 * qf).sum::<f32>() * s;
                scored.push((i, dot));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if self.emb_cache.len() > Self::EMB_CACHE_CAP { self.emb_cache.clear(); }   // hard memory bound (re-embeds after a flush)
        scored
    }
    /// Score candidates along a LINEAR AXIS (w, c) in this space — z = w·x − c per candidate,
    /// highest first — reusing the same cached int8 embeddings rank_cached uses. This is how a
    /// Fisher discriminant head reads the store's facts without a second embedding pass; the
    /// axis is plain numbers here, so semantic.rs stays independent of the fisher feature.
    pub fn project_cached(&mut self, w: &[f32], c: f32, cands: &[&str]) -> Vec<(usize, f32)> {
        self.ensure_cached(cands);
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(cands.len());
        for (i, &t) in cands.iter().enumerate() {
            if let Some((e, s, _)) = self.emb_cache.get(t) {
                let dot = e.iter().zip(w).map(|(c8, wf)| *c8 as f32 * wf).sum::<f32>() * s;
                scored.push((i, dot - c));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if self.emb_cache.len() > Self::EMB_CACHE_CAP { self.emb_cache.clear(); }
        scored
    }
    /// The k nearest words to an arbitrary DIRECTION in the space, by cosine — how a learned
    /// axis becomes readable ("what does the helpful direction look like"). Mirrors nearest().
    pub fn nearest_vec(&self, v: &[f32], k: usize) -> Vec<(String, f32)> {
        let bn = Self::norm(v);
        if bn == 0.0 { return Vec::new(); }
        let mut out: Vec<(String, f32)> = self.ctx.iter()
            .filter_map(|(w, cv)| {
                let n = Self::norm(cv);
                if n == 0.0 { return None; }
                let dot: f32 = v.iter().zip(cv).map(|(a, b)| a * b).sum();
                Some((w.clone(), dot / (bn * n)))
            }).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        out
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

    #[test]
    fn projection_separates_clusters() {
        let s = trained();
        let p = s.project(40, 3);
        assert!(!p.coords.is_empty() && p.coords[0].len() == 3);
        assert!(p.total_variance > 0.0 && p.variance.len() == 3);
        let pos = |w: &str| p.words.iter().position(|x| x == w).map(|i| p.coords[i].clone());
        let dist = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        if let (Some(internet), Some(wifi), Some(garlic)) = (pos("internet"), pos("wifi"), pos("garlic")) {
            // in the projected space, the two networking words are nearer each other than to cooking
            assert!(dist(&internet, &wifi) < dist(&internet, &garlic),
                "internet-wifi {:.3} should be < internet-garlic {:.3}", dist(&internet, &wifi), dist(&internet, &garlic));
        }
    }
}
