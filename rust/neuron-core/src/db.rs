//! NeuronDB: a database of neurons in one SQLite file (rusqlite, bundled). Durable,
//! thread-safe (one connection + an in-memory LRU cache behind a Mutex). Feature-gated
//! behind `sqlite`. The cache avoids re-parsing a scope blob on every op (the large-scope
//! write cost); writes still persist immediately. Batch ingest amortizes the per-write save.
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::{Neuron, Recall, Spread};
use crate::turn::turn;

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS neurons (id TEXT PRIMARY KEY, facts TEXT NOT NULL DEFAULT '[]', created INTEGER NOT NULL, updated INTEGER NOT NULL, turns INTEGER NOT NULL DEFAULT 0);";
fn now_ms() -> i64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64 }

pub use crate::{Stats, TurnOut};   // defined at the crate root so a no-sqlite wasm build can name them too

// NeuronDB speaks the shared op vocabulary: each method delegates to its inherent counterpart, with
// the rank choice and the recall_value cross-scope fallback (the durable-store-specific semantics)
// living here so apply() stays a thin generic dispatcher.
impl crate::op::Store for NeuronDB {
    fn observe(&self, scope: &str, text: &str) -> usize { NeuronDB::observe(self, scope, text) }
    fn observe_bulk(&self, scope: &str, texts: &[String]) -> usize { self.observe_many(scope, texts) }
    fn recall_one(&self, scope: &str, query: &str) -> Option<crate::Recall> { self.recall(scope, query) }
    fn recall_block(&self, scope: &str, query: &str, k: usize, semantic: bool, across: bool) -> Vec<crate::Recall> {
        #[cfg(feature = "semantic")]
        { if across { self.recall_many_across(scope, query, k) } else if semantic { self.recall_blended(scope, query, k) } else { self.recall_many(scope, query, k) } }
        #[cfg(not(feature = "semantic"))]
        { let _ = semantic; if across { self.recall_many_across(scope, query, k) } else { self.recall_many(scope, query, k) } }
    }
    fn recall_value(&self, scope: &str, query: &str) -> Option<String> {
        self.get(scope, query).or_else(|| self.recall_many_across(scope, query, 1).into_iter().next().map(|h| h.value))
    }
    fn recall_assoc(&self, scope: &str, query: &str, k: usize, hops: usize) -> Vec<crate::Spread> { self.recall_associative(scope, query, k, hops) }
    fn recall_chain(&self, scope: &str, start: &str, path: &[String]) -> (Option<String>, Vec<String>) { NeuronDB::recall_chain(self, scope, start, path) }
    fn var_set(&self, scope: &str, key: &str, value: &str) -> usize { NeuronDB::var_set(self, scope, key, value) }
    fn var_get(&self, scope: &str, key: &str) -> Option<String> { NeuronDB::var_get(self, scope, key) }
    fn note_stance(&self, scope: &str, topic: &str, feeling: &str) -> (f32, bool) { NeuronDB::note_stance(self, scope, topic, feeling) }
    fn set_mood(&self, scope: &str, emotion: &str) { NeuronDB::set_mood(self, scope, emotion) }
    fn affect(&self, scope: &str, asked_topic: Option<&str>) -> String { NeuronDB::affect(self, scope, asked_topic) }
    fn turn(&self, scope: &str, message: &str) -> TurnOut { NeuronDB::turn(self, scope, message) }
    fn forget(&self, scope: &str, matching: Option<&str>) -> (usize, usize) { NeuronDB::forget(self, scope, matching) }
    fn stats(&self, scope: &str) -> Stats { NeuronDB::stats(self, scope) }
    fn scopes(&self) -> Vec<String> { self.neurons() }
}

struct Entry { n: Neuron, created: i64, turns: i64, used: u64, dirty: bool, writes: u32 }
struct Inner { conn: Connection, cache: HashMap<String, Entry>, tick: u64 }
pub struct NeuronDB {
    inner: Mutex<Inner>, max_facts: usize, cap: usize,
    flush_every: usize,   // 1 = persist every single observe (immediate durability); >1 = write-behind
    #[cfg(feature = "semantic")] sem: Mutex<crate::semantic::SemanticSpace>,
    #[cfg(feature = "semantic")] sem_threshold: f32,
}

impl Drop for NeuronDB {
    /// Flush any write-behind buffers on shutdown so a clean exit never loses deferred writes.
    fn drop(&mut self) {
        if let Ok(mut g) = self.inner.lock() {
            let inner = &mut *g; let Inner { conn, cache, .. } = inner;
            for (k, e) in cache.iter_mut() { if e.dirty { Self::persist(conn, k, e); e.dirty = false; } }
        }
    }
}

impl NeuronDB {
    /// Open with immediate per-write durability (every observe is persisted). The default.
    pub fn open(path: &str, max_facts: usize) -> Self { Self::open_with_flush(path, max_facts, 1) }

    /// Open with write-behind: a single observe defers the (O(scope)) SQLite blob rewrite, persisting
    /// only every `flush_every` writes to a scope (and always on eviction, flush_all(), and Drop).
    /// flush_every=1 keeps immediate durability; larger values trade up to `flush_every` facts of
    /// crash-loss per scope for far higher single-observe throughput. Recall is unaffected (it reads
    /// the in-memory cache); only on-disk durability is deferred.
    pub fn open_with_flush(path: &str, max_facts: usize, flush_every: usize) -> Self {
        let conn = Connection::open(path).expect("open sqlite");
        let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
        conn.execute(SCHEMA, []).expect("schema");
        NeuronDB {
            inner: Mutex::new(Inner { conn, cache: HashMap::new(), tick: 0 }), max_facts, cap: 256,
            flush_every: flush_every.max(1),
            #[cfg(feature = "semantic")] sem: Mutex::new(crate::semantic::SemanticSpace::new()),
            #[cfg(feature = "semantic")] sem_threshold: 0.20,
        }
    }

    /// Persist all scopes with unsaved (write-behind) changes. Call before shutdown for durability;
    /// also run automatically on Drop and on LRU eviction.
    pub fn flush_all(&self) {
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        let Inner { conn, cache, .. } = inner;
        for (k, e) in cache.iter_mut() { if e.dirty { Self::persist(conn, k, e); e.dirty = false; e.writes = 0; } }
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
            Some((blob, c, _u, t)) => Entry { n: Neuron::load(&blob, max_facts), created: c, turns: t, used: tick, dirty: false, writes: 0 },
            None => { let n = now_ms(); Entry { n: Neuron::new(max_facts), created: n, turns: 0, used: tick, dirty: false, writes: 0 } }
        };
        if inner.cache.len() >= cap {
            if let Some(k) = inner.cache.iter().min_by_key(|(_, e)| e.used).map(|(k, _)| k.clone()) {
                // persist-on-evict: under write-behind the LRU victim may hold unsaved writes —
                // dropping it without persisting would be data loss.
                if let Some(e) = inner.cache.get(&k) { if e.dirty { Self::persist(&inner.conn, &k, e); } }
                inner.cache.remove(&k);
            }
        }
        inner.cache.insert(nid.to_string(), entry);
    }
    fn persist(conn: &Connection, nid: &str, e: &Entry) {
        // prepare_cached: the INSERT is parsed once and reused across writes instead of re-parsed each call
        let mut stmt = conn.prepare_cached("INSERT INTO neurons(id,facts,created,updated,turns) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(id) DO UPDATE SET facts=excluded.facts,updated=excluded.updated,turns=excluded.turns").expect("prepare");
        stmt.execute(params![nid, e.n.dump(), e.created, now_ms(), e.turns]).expect("save");
    }
    /// Persist now and clear the dirty/write-behind state (used by the immediate-durability paths).
    fn persist_now(conn: &Connection, nid: &str, e: &mut Entry) { Self::persist(conn, nid, e); e.dirty = false; e.writes = 0; }
    /// Mark a single-observe write: persist immediately when flush_every<=1, else defer until the
    /// per-scope write count reaches the threshold (eviction/flush_all/Drop also flush).
    fn touch(conn: &Connection, nid: &str, e: &mut Entry, flush_every: usize) {
        e.dirty = true; e.writes = e.writes.saturating_add(1);
        if flush_every <= 1 || (e.writes as usize) >= flush_every { Self::persist_now(conn, nid, e); }
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
            Self::touch(conn, nid, e, self.flush_every);   // write-behind aware (immediate when flush_every=1)
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
    /// BOUNDED: only the most-recent SEM_FALLBACK_CAP facts are ranked, so a lexical MISS stays
    /// cheap as a scope grows across many chats (otherwise every miss is an O(N) embedding scan).
    #[cfg(feature = "semantic")]
    pub fn recall_semantic(&self, nid: &str, query: &str) -> Option<Recall> {
        const SEM_FALLBACK_CAP: usize = 4000;
        let facts: Vec<(String, String)> = {
            let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let eps = &inner.cache.get(nid).unwrap().n.episodes;
            let start = eps.len().saturating_sub(SEM_FALLBACK_CAP);   // most-recent window only
            eps[start..].iter().map(|e| (e.t.clone(), e.v.clone())).collect()
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
    /// Recall across a base scope AND its document sub-scopes (`base`, `base__doc1`, …), merging the
    /// top-k by the same (exact, overlap, coverage) key recall uses. Typed sub-scopes (`base::var`,
    /// `base::stance`, …) and other users' scopes are deliberately excluded. Lets "tell me about X"
    /// reach a document the user filed under its own scope without the caller knowing the scope name.
    pub fn recall_many_across(&self, base: &str, query: &str, k: usize) -> Vec<Recall> {
        let scopes: Vec<String> = self.neurons().into_iter()
            .filter(|id| id == base || (id.starts_with(base) && id[base.len()..].starts_with("__")))
            .collect();
        let mut all: Vec<Recall> = Vec::new();
        for s in &scopes { all.extend(self.recall_many(s, query, k)); }
        all.sort_by(|a, b| b.exact.cmp(&a.exact)
            .then(b.overlap.cmp(&a.overlap))
            .then(b.coverage.partial_cmp(&a.coverage).unwrap_or(std::cmp::Ordering::Equal)));
        all.truncate(k);
        all
    }
    pub fn recall_many(&self, nid: &str, query: &str, k: usize) -> Vec<Recall> {
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        inner.cache.get_mut(nid).unwrap().n.recall_many(query, k)
    }
    /// Spreading-activation recall: seeds on cue matches, then follows shared-entity links to
    /// surface associated facts (see Neuron::recall_spreading). Association-based, not keyword/cosine.
    pub fn recall_associative(&self, nid: &str, query: &str, k: usize, hops: usize) -> Vec<Spread> {
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        inner.cache.get_mut(nid).unwrap().n.recall_spreading(query, k, hops)
    }
    /// Upsert a named variable: anchored removal of any prior "{key} is …" (so distinct keys never
    /// clobber each other — "region" must not delete "deployRegion"), then store "{key} is {value}".
    /// Returns the number of facts written (0 if the value was too short to encode).
    pub fn var_set(&self, nid: &str, key: &str, value: &str) -> usize {
        let line = format!("{} is {}", key, value);
        // ATOMIC: probe with the real writer (observe -> sentences -> encode) and only forget the old
        // value if the new one will actually store, so an unencodable update never destroys the old.
        if Neuron::new(self.max_facts).observe(&line) == 0 { return 0; }
        let w;
        {
            let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(nid).unwrap();
            e.n.forget_prefix(&format!("{} is ", key));
            w = e.n.observe(&line);
            Self::persist(conn, nid, e);
        }
        #[cfg(feature = "semantic")] self.sem.lock().unwrap().train(value);
        w
    }
    /// Read a named variable's FULL value (everything after the first " is "), so multi-word values
    /// and values that themselves contain " is " round-trip exactly — unlike cue-isolated recall.
    pub fn var_get(&self, nid: &str, key: &str) -> Option<String> {
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        let kl = format!("{} is ", key.to_lowercase());
        inner.cache.get(nid).unwrap().n.episodes.iter()
            .find(|e| e.t.to_lowercase().starts_with(&kl))
            .map(|e| match e.t.find(" is ") { Some(i) => e.t[i + 4..].trim().to_string(), None => e.t.clone() })
    }
    /// Set (or clear, with an empty string) the mood override for a scope — the optional affective
    /// variable. When set it takes precedence over the auto-derived disposition in `affect`.
    pub fn set_mood(&self, nid: &str, emotion: &str) {
        let sub = format!("{}::affect", nid);
        if emotion.trim().is_empty() { self.forget(&sub, None); }
        else { self.forget(&sub, Some("mood is")); self.observe(&sub, &format!("mood is {}", emotion.trim())); }
    }
    /// The humanize basis for a scope: an always-on directive to respond with genuine feeling,
    /// plus the current mood (the override variable if set) and the strongest accumulated stance
    /// (the disposition built up over time). This is how the store colors the model's tone.
    pub fn affect(&self, nid: &str, asked_topic: Option<&str>) -> String {
        let (mood, stances) = {
            let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
            let asub = format!("{}::affect", nid);
            Self::ensure(inner, &asub, self.max_facts, self.cap);
            let mood = inner.cache.get(&asub).unwrap().n.episodes.iter()
                .find_map(|e| e.t.strip_prefix("mood is ").map(|m| m.trim().to_string()));
            let ssub = format!("{}::stance", nid);
            Self::ensure(inner, &ssub, self.max_facts, self.cap);
            // each stance Episode is stored as "topic: feeling" (reinforce_prefix), strength accumulates
            let stances: Vec<(String, String, f32)> = inner.cache.get(&ssub).unwrap().n.episodes.iter()
                .map(|e| { let (t, f) = e.t.split_once(": ").unwrap_or(("", e.t.as_str())); (t.to_string(), f.to_string(), e.strength) })
                .collect();
            (mood, stances)
        };
        // db.rs shows the persona frame even when neutral (a baseline directive the harness can lean on)
        format!("{}{}", crate::affect::FRAME, crate::affect::directive_body(mood.as_deref(), &stances, asked_topic))
    }
    /// Record/intensify a stance about `topic`. Re-stating the same topic accumulates its strength
    /// (a disposition that deepens with repetition), persisted durably. Returns (new_strength, new).
    pub fn note_stance(&self, nid: &str, topic: &str, feeling: &str) -> (f32, bool) {
        let out = {
            let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(nid).unwrap();
            let r = e.n.reinforce_prefix(topic, feeling, crate::affect::STANCE_BUMP);
            // neglected dispositions fade as new feelings form, so the active "culture" can shift
            // over time rather than monotonically hardening on whatever was felt first.
            e.n.decay_prefix_others(topic, crate::affect::STANCE_DECAY, crate::affect::STANCE_FLOOR);
            Self::persist(conn, nid, e);
            r
        };
        #[cfg(feature = "semantic")] self.sem.lock().unwrap().train(feeling);
        out
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
        let (before, after) = {
            let Inner { conn, cache, .. } = &mut *inner;
            let e = cache.get_mut(nid).unwrap();
            let before = e.n.fact_count();
            match m { Some(s) => { let s = s.to_lowercase(); e.n.episodes.retain(|ep| !ep.t.to_lowercase().contains(&s)); }, None => e.n.episodes.clear() }
            e.n.invalidate_index(); // removal shifts episode indices -> force a rebuild on next recall
            let after = e.n.fact_count();
            Self::persist(conn, nid, e);
            (before, after)
        };
        // A full wipe (no match — the "forget me" path) cascades to the typed sub-scopes so stored
        // variables (incl. secrets), standing instructions, stances, and mood aren't left behind.
        // Done INLINE on the held lock — not via self.forget(), which would re-lock and deadlock.
        if m.is_none() {
            for suffix in ["::var", "::instr", "::stance", "::affect"] {
                let sub = format!("{}{}", nid, suffix);
                Self::ensure(inner, &sub, self.max_facts, self.cap);
                let Inner { conn, cache, .. } = &mut *inner;
                let e = cache.get_mut(&sub).unwrap();
                e.n.episodes.clear(); e.n.invalidate_index();
                Self::persist(conn, &sub, e);
            }
        }
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
