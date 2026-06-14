//! NeuronDB: a database of neurons in one SQLite file (rusqlite, bundled). Durable,
//! thread-safe (one connection + an in-memory LRU cache behind a Mutex). Feature-gated
//! behind `sqlite`. The cache avoids re-parsing a scope blob on every op (the large-scope
//! write cost); writes still persist immediately. Batch ingest amortizes the per-write save.
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::{Neuron, Recall};
use crate::turn::turn;

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS neurons (id TEXT PRIMARY KEY, facts TEXT NOT NULL DEFAULT '[]', created INTEGER NOT NULL, updated INTEGER NOT NULL, turns INTEGER NOT NULL DEFAULT 0);";
fn now_ms() -> i64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64 }

#[derive(Debug, Clone)]
pub struct TurnOut { pub reply: String, pub kind: String, pub wrote: usize, pub facts: usize, pub capacity_reached: bool }
#[derive(Debug, Clone)]
pub struct Stats { pub facts: usize, pub max_facts: usize, pub created: i64, pub updated: i64, pub turns: i64 }

struct Entry { n: Neuron, created: i64, turns: i64, used: u64 }
struct Inner { conn: Connection, cache: HashMap<String, Entry>, tick: u64 }
pub struct NeuronDB {
    inner: Mutex<Inner>, max_facts: usize, cap: usize,
    #[cfg(feature = "semantic")] sem: Mutex<crate::semantic::SemanticSpace>,
    #[cfg(feature = "semantic")] sem_threshold: f32,
}

impl NeuronDB {
    pub fn open(path: &str, max_facts: usize) -> Self {
        let conn = Connection::open(path).expect("open sqlite");
        let _ = conn.execute_batch("PRAGMA journal_mode=WAL;");
        conn.execute(SCHEMA, []).expect("schema");
        NeuronDB {
            inner: Mutex::new(Inner { conn, cache: HashMap::new(), tick: 0 }), max_facts, cap: 256,
            #[cfg(feature = "semantic")] sem: Mutex::new(crate::semantic::SemanticSpace::new()),
            #[cfg(feature = "semantic")] sem_threshold: 0.20,
        }
    }

    /// Train the semantic space on arbitrary background text (e.g. a book/corpus), so recall
    /// can fall back to meaning when lexical cues miss. No-op unless built with `semantic`.
    #[cfg(feature = "semantic")]
    pub fn train_semantic(&self, text: &str) { self.sem.lock().unwrap().train(text); }
    /// (vocab words, tokens seen, approx bytes) of the semantic space.
    #[cfg(feature = "semantic")]
    pub fn semantic_stats(&self) -> (usize, u64, usize) {
        let s = self.sem.lock().unwrap(); (s.vocab(), s.tokens(), s.bytes())
    }
    /// The k nearest words to `word` in the learned semantic space (for inspection).
    #[cfg(feature = "semantic")]
    pub fn semantic_neighbors(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        self.sem.lock().unwrap().nearest(word, k)
    }
    /// Compact the semantic space to int8 for read-mostly serving (~4x smaller; recall intact,
    /// a later observe transparently re-expands it).
    #[cfg(feature = "semantic")]
    pub fn compact_semantic(&self) { self.sem.lock().unwrap().compact(); }

    /// Ensure `nid` is in the cache (load from sqlite on miss; evict LRU when over cap).
    fn ensure(inner: &mut Inner, nid: &str, max_facts: usize, cap: usize) {
        inner.tick += 1; let tick = inner.tick;
        if let Some(e) = inner.cache.get_mut(nid) { e.used = tick; return; }
        let row = inner.conn.query_row("SELECT facts,created,updated,turns FROM neurons WHERE id=?1", params![nid],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?))).ok();
        let entry = match row {
            Some((blob, c, _u, t)) => Entry { n: Neuron::load(&blob, max_facts), created: c, turns: t, used: tick },
            None => { let n = now_ms(); Entry { n: Neuron::new(max_facts), created: n, turns: 0, used: tick } }
        };
        if inner.cache.len() >= cap {
            if let Some(k) = inner.cache.iter().min_by_key(|(_, e)| e.used).map(|(k, _)| k.clone()) { inner.cache.remove(&k); }
        }
        inner.cache.insert(nid.to_string(), entry);
    }
    fn persist(conn: &Connection, nid: &str, e: &Entry) {
        conn.execute("INSERT INTO neurons(id,facts,created,updated,turns) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(id) DO UPDATE SET facts=excluded.facts,updated=excluded.updated,turns=excluded.turns",
            params![nid, e.n.dump(), e.created, now_ms(), e.turns]).expect("save");
    }

    pub fn observe(&self, nid: &str, text: &str) -> usize {
        let w;
        {
            let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(nid).unwrap();
            // exact-text dedup on the single-observe path (a model re-stating a fact across turns);
            // the batch path stays un-deduped so bulk ingest stays O(n).
            if e.n.episodes.iter().any(|ep| ep.t == text) { return 0; }
            w = e.n.observe(text);
            Self::persist(conn, nid, e);
        }
        #[cfg(feature = "semantic")] self.sem.lock().unwrap().train(text);
        w
    }
    /// Batch ingest: one load, many appends, one save. Amortizes the per-write commit.
    pub fn observe_many(&self, nid: &str, texts: &[String]) -> usize {
        let w;
        {
            let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(nid).unwrap();
            let mut acc = 0; for t in texts { acc += e.n.observe(t); }
            Self::persist(conn, nid, e); w = acc;
        }
        #[cfg(feature = "semantic")] { let mut s = self.sem.lock().unwrap(); for t in texts { s.train(t); } }
        w
    }
    pub fn recall(&self, nid: &str, query: &str) -> Option<Recall> {
        let lex = {
            let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            inner.cache.get_mut(nid).unwrap().n.recall(query)
        };
        #[cfg(feature = "semantic")]
        if lex.is_none() {
            if let Some(r) = self.recall_semantic(nid, query) { return Some(r); }
        }
        lex
    }
    /// Semantic fallback: when lexical recall misses, rank the scope's facts in the semantic
    /// space and return the best if it clears the similarity threshold. Resolves paraphrase
    /// that shares no words with the stored fact ("the thing I use to get online" -> wifi).
    #[cfg(feature = "semantic")]
    pub fn recall_semantic(&self, nid: &str, query: &str) -> Option<Recall> {
        let facts: Vec<(String, String)> = {
            let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            inner.cache.get(nid).unwrap().n.episodes.iter().map(|e| (e.t.clone(), e.v.clone())).collect()
        };
        if facts.is_empty() { return None; }
        let texts: Vec<String> = facts.iter().map(|(t, _)| t.clone()).collect();
        let mut s = self.sem.lock().unwrap();
        let ranked = s.rank_cached(query, &texts);
        match ranked.first() {
            Some(&(i, score)) if score >= self.sem_threshold => {
                let (fact, value) = facts[i].clone();
                Some(Recall { fact, value, coverage: score as f64, overlap: 0, exact: 0, echo: false })
            }
            _ => None,
        }
    }
    /// Hybrid block recall: rank a scope's facts by semantic similarity (cached space) plus a
    /// lexical-overlap boost, returning the top-k. Engages the semantic layer for EVERY query, so
    /// broad/narrative questions return topically-coherent facts instead of scattered keyword hits.
    #[cfg(feature = "semantic")]
    pub fn recall_blended(&self, nid: &str, query: &str, k: usize) -> Vec<Recall> {
        let facts: Vec<(String, String)> = {
            let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            inner.cache.get(nid).unwrap().n.episodes.iter().map(|e| (e.t.clone(), e.v.clone())).collect()
        };
        if facts.is_empty() { return Vec::new(); }
        let texts: Vec<String> = facts.iter().map(|(t, _)| t.clone()).collect();
        let ranked = { let mut s = self.sem.lock().unwrap(); s.rank_cached(query, &texts) };
        if ranked.is_empty() { return self.recall_many(nid, query, k); } // no semantic signal -> lexical
        // lexical boost: fraction of the query's content words (>=3 chars) present in the fact
        let qw: Vec<String> = query.to_lowercase().split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3).map(|w| w.to_string()).collect();
        let mut scored: Vec<(f32, usize)> = ranked.iter().map(|&(i, cos)| {
            let lt = texts[i].to_lowercase();
            let boost = if qw.is_empty() { 0.0 } else {
                qw.iter().filter(|w| lt.contains(w.as_str())).count() as f32 / qw.len() as f32
            };
            (cos + 0.25 * boost, i)
        }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored.into_iter().map(|(score, i)| {
            let (fact, value) = facts[i].clone();
            Recall { fact, value, coverage: score as f64, overlap: 0, exact: 0, echo: false }
        }).collect()
    }
    pub fn recall_many(&self, nid: &str, query: &str, k: usize) -> Vec<Recall> {
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        inner.cache.get_mut(nid).unwrap().n.recall_many(query, k)
    }
    pub fn get(&self, nid: &str, query: &str) -> Option<String> { self.recall(nid, query).map(|h| h.value) }
    /// Multi-hop traversal, server-side: start at `start` and follow each relation in `path`,
    /// resolving "<current> <relation>" by recall at every step. The whole chain fires in
    /// microseconds with no model round-trips, so hop count costs the LLM nothing. Returns the
    /// final value (None if the chain breaks) and the trail of resolved values for transparency.
    pub fn recall_chain(&self, nid: &str, start: &str, path: &[String]) -> (Option<String>, Vec<String>) {
        let mut current = start.trim().to_string();
        let mut trail = vec![current.clone()];
        for rel in path {
            // a relation may be multiple words ("depends on"); match on any of its content
            // words (length >= 3 to skip stopwords like "on"/"of"). rel_matches tolerates
            // morphological + stem variants (owner/owned, dependency/depends).
            let rel_words: Vec<&str> = rel.split_whitespace().filter(|w| w.len() >= 3).collect();
            match self.recall(nid, &format!("{} {}", current, rel)) {
                // only advance if the relation actually appears in the recalled fact; otherwise
                // recall's best-effort (entity overlap alone) would let a broken chain continue.
                Some(h) if rel_words.is_empty()
                    || rel_words.iter().any(|rw| h.fact.split_whitespace().any(|w| crate::rel_matches(w, rw))) => {
                    current = h.value.clone();
                    trail.push(h.value);
                }
                _ => return (None, trail),
            }
        }
        (Some(current), trail)
    }
    pub fn turn(&self, nid: &str, msg: &str) -> TurnOut {
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        let max = self.max_facts;
        let Inner { conn, cache, .. } = inner;
        let e = cache.get_mut(nid).unwrap();
        let at_cap = e.n.fact_count() >= max;
        let r = turn(&mut e.n, msg);
        if at_cap && r.wrote > 0 { e.n.episodes.truncate(max); }
        e.turns += 1;
        Self::persist(conn, nid, e);
        TurnOut { reply: r.reply, kind: r.kind, wrote: r.wrote, facts: e.n.fact_count(), capacity_reached: at_cap && r.wrote > 0 }
    }
    pub fn forget(&self, nid: &str, m: Option<&str>) -> (usize, usize) {
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        let Inner { conn, cache, .. } = inner;
        let e = cache.get_mut(nid).unwrap();
        let before = e.n.fact_count();
        match m { Some(s) => { let s = s.to_lowercase(); e.n.episodes.retain(|ep| !ep.t.to_lowercase().contains(&s)); }, None => e.n.episodes.clear() }
        e.n.invalidate_index(); // removal shifts episode indices -> force a rebuild on next recall
        let after = e.n.fact_count();
        Self::persist(conn, nid, e);
        (before - after, after)
    }
    pub fn stats(&self, nid: &str) -> Stats {
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        let e = inner.cache.get(nid).unwrap();
        Stats { facts: e.n.fact_count(), max_facts: self.max_facts, created: e.created, updated: now_ms(), turns: e.turns }
    }
    pub fn neurons(&self) -> Vec<String> {
        let g = self.inner.lock().unwrap();
        let mut st = g.conn.prepare("SELECT id FROM neurons ORDER BY updated DESC").unwrap();
        let rows = st.query_map([], |r| r.get::<_, String>(0)).unwrap();
        rows.filter_map(|x| x.ok()).collect()
    }
}
