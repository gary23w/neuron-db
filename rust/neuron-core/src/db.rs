//! NeuronDB: a database of neurons in one SQLite file (rusqlite, bundled). Durable and
//! thread-safe. The in-memory LRU cache + its connection are SHARDED by top-level scope family
//! (see `shard_key`), each shard behind its own Mutex and its own WAL connection to the one file,
//! so independent tenants recall/observe in parallel instead of serializing on a single global lock.
//! Feature-gated behind `sqlite`. The cache avoids re-parsing a scope blob on every op (the
//! large-scope write cost); writes still persist immediately. Batch ingest amortizes the per-write save.
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::{Neuron, Recall, Spread};
use crate::turn::turn;

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS neurons (id TEXT PRIMARY KEY, facts TEXT NOT NULL DEFAULT '[]', created INTEGER NOT NULL, updated INTEGER NOT NULL, turns INTEGER NOT NULL DEFAULT 0);\n\
CREATE TABLE IF NOT EXISTS fact_log (scope TEXT NOT NULL, seq INTEGER NOT NULL, lines TEXT NOT NULL, PRIMARY KEY(scope, seq));";
// the per-scope append-log can grow to ~the snapshot size before we fold it back into a fresh snapshot,
// so a single durable observe is one small INSERT (O(new facts)) and compaction is amortized O(1)/fact.
const COMPACT_FLOOR: usize = 256;
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

// snap_count = facts in the neurons.facts snapshot blob; episodes[snap_count..] live in the append-log
// (durable). log_next = next log seq. dirty = the log has entries not yet folded into the snapshot.
struct Entry { n: Neuron, created: i64, turns: i64, used: u64, dirty: bool, snap_count: usize, log_next: i64 }
struct Inner { conn: Connection, cache: HashMap<String, Entry>, tick: u64 }
pub struct NeuronDB {
    shards: Vec<Mutex<Inner>>,   // cache+connection partitioned by scope family; len 1 for in-memory DBs
    max_facts: usize, cap: usize,
    flush_every: usize,   // append-log compaction floor: the log folds into a snapshot once it reaches ~max(snap_count, this)
    #[cfg(feature = "semantic")] sem: Mutex<crate::semantic::SemanticSpace>,
    #[cfg(feature = "semantic")] sem_threshold: f32,
}

impl Drop for NeuronDB {
    /// Flush any write-behind buffers on shutdown so a clean exit never loses deferred writes.
    fn drop(&mut self) {
        // Lock each shard directly (de-poisoning, like the old guard()) rather than `if let Ok(lock())`:
        // a prior persist panic (e.g. SQLITE_FULL) poisons the mutex, and the old form would silently
        // SKIP this shutdown flush — dropping every dirty write-behind buffer. Each per-entry persist is
        // catch_unwind-isolated so a panic on one scope (a still-full disk) neither aborts the process
        // nor skips the remaining scopes' flushes. Shards are independent, so we take one lock at a time.
        for shard in &self.shards {
            let mut g = shard.lock().unwrap_or_else(|e| e.into_inner());
            let Inner { conn, cache, .. } = &mut *g;
            for (k, e) in cache.iter_mut() {
                // snapshot() clears dirty on success; a panic (e.g. still-full disk) leaves it dirty rather
                // than falsely clearing it — the log already holds the facts, so reopen still recovers them.
                if e.dirty { let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Self::snapshot(conn, k, e))); }
            }
        }
    }
}

impl NeuronDB {
    /// Open with immediate per-write durability (every observe is persisted). The default.
    pub fn open(path: &str, max_facts: usize) -> Self { Self::open_with_flush(path, max_facts, 1) }

    /// Open with a custom compaction floor. Every observe is durable immediately — one append-log INSERT,
    /// O(new facts), NOT a whole-scope blob rewrite — regardless of `flush_every`; the log folds back into
    /// a fresh snapshot once it reaches ~max(snapshot size, `flush_every`). A larger `flush_every` lets the
    /// log grow further between snapshots (fewer, larger compactions). Recall reads the in-memory cache.
    /// Kept for API compatibility — `open()` (flush_every=1) is the right default.
    pub fn open_with_flush(path: &str, max_facts: usize, flush_every: usize) -> Self {
        // One in-memory cache shard per hardware thread (capped), each with its OWN WAL connection to the
        // same file — independent scope families recall/observe without contending for one global lock.
        // An in-memory or anonymous DB can't share state across connections, so it stays single-shard.
        let nshards = if path == ":memory:" || path.is_empty() { 1 }
            else { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 16) };
        let mut shards = Vec::with_capacity(nshards);
        for _ in 0..nshards {
            let conn = Connection::open(path).expect("open sqlite");
            // WAL = concurrent readers + a single writer across connections; busy_timeout makes a second
            // shard's writer wait for the brief WAL write-lock instead of erroring SQLITE_BUSY.
            let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=10000;");
            conn.execute_batch(SCHEMA).expect("schema");   // CREATE IF NOT EXISTS — idempotent across shards
            shards.push(Mutex::new(Inner { conn, cache: HashMap::new(), tick: 0 }));
        }
        NeuronDB {
            // cap is PER-SHARD, so divide the ~256 global cached-scope budget across shards (with a floor so
            // a single tenant's co-located sub/cross scopes don't thrash). Eviction is per-shard; an evicted
            // scope just reloads from its durable log, so this bounds memory without risking correctness.
            shards, max_facts, cap: (256 / nshards).max(32),
            flush_every: flush_every.max(1),
            #[cfg(feature = "semantic")] sem: Mutex::new(crate::semantic::SemanticSpace::new()),
            #[cfg(feature = "semantic")] sem_threshold: 0.20,
        }
    }

    /// The top-level scope family for `nid`: the prefix before the first `::` (typed sub-scope like
    /// `::affect`/`::stance`/`::var`/`::instr`) or `__` (cross-scope child like `base__deals`). A scope
    /// and ALL of its sub-scopes and cross-children share this key, so they hash to the SAME shard — every
    /// multi-scope op (affect, a full forget, recall_many_across) stays a single-shard, single-lock path.
    fn shard_key(nid: &str) -> &str {
        let end = nid.len();
        let a = nid.find("::").unwrap_or(end);
        let b = nid.find("__").unwrap_or(end);
        &nid[..a.min(b)]
    }
    fn shard_idx(&self, nid: &str) -> usize {
        if self.shards.len() == 1 { return 0; }
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        Self::shard_key(nid).hash(&mut h);
        (h.finish() % self.shards.len() as u64) as usize
    }
    /// Lock the shard owning `nid`'s scope family, recovering from a poisoned mutex instead of panicking.
    /// A write that panics mid-persist (e.g. SQLITE_FULL on a huge import) poisons that shard's lock;
    /// de-poisoning here keeps the store usable for every later op rather than cascading one failed write
    /// into a dead store — important for the long-lived MCP server with a preload boot hook.
    fn shard(&self, nid: &str) -> std::sync::MutexGuard<'_, Inner> {
        self.shards[self.shard_idx(nid)].lock().unwrap_or_else(|e| e.into_inner())
    }
    /// Lock shard 0 for catalog reads. The `neurons`/`fact_log` tables live in the one shared DB file, so
    /// any shard's connection sees every committed row — listing ids needs only one shard's connection.
    fn catalog(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.shards[0].lock().unwrap_or_else(|e| e.into_inner())
    }
    // de-poison the semantic-space lock too: a panic inside train() (e.g. under the MCP preload
    // catch_unwind) would otherwise poison it and crash the first recall in the request loop.
    #[cfg(feature = "semantic")]
    fn sem_guard(&self) -> std::sync::MutexGuard<'_, crate::semantic::SemanticSpace> { self.sem.lock().unwrap_or_else(|e| e.into_inner()) }

    /// Persist all scopes with unsaved (write-behind) changes. Call before shutdown for durability;
    /// also run automatically on Drop and on LRU eviction.
    pub fn flush_all(&self) {
        for shard in &self.shards {
            let mut g = shard.lock().unwrap_or_else(|e| e.into_inner());
            let Inner { conn, cache, .. } = &mut *g;
            for (k, e) in cache.iter_mut() {
                // isolate each persist like Drop does: a SQLITE_FULL panic on one scope (this is the
                // pre-shutdown flush after a bulk import — the most likely spot to fill the disk) must
                // not poison the lock or skip the remaining scopes. A failed scope stays dirty for Drop.
                if e.dirty {
                    // snapshot() clears dirty on success; a SQLITE_FULL panic leaves it dirty for Drop to retry
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Self::snapshot(conn, k, e)));
                }
            }
        }
    }

    /// Train the semantic space on arbitrary background text (e.g. a book/corpus), so recall
    /// can fall back to meaning when lexical cues miss. No-op unless built with `semantic`.
    #[cfg(feature = "semantic")]
    pub fn train_semantic(&self, text: &str) { self.sem_guard().train(text); }
    /// (vocab words, tokens seen, approx bytes) of the semantic space.
    #[cfg(feature = "semantic")]
    pub fn semantic_stats(&self) -> (usize, u64, usize) {
        let s = self.sem_guard(); (s.vocab(), s.tokens(), s.bytes())
    }
    /// The k nearest words to `word` in the learned semantic space (for inspection).
    #[cfg(feature = "semantic")]
    pub fn semantic_neighbors(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        self.sem_guard().nearest(word, k)
    }
    /// Compact the semantic space to int8 for read-mostly serving (~4x smaller; recall intact,
    /// a later observe transparently re-expands it).
    #[cfg(feature = "semantic")]
    pub fn compact_semantic(&self) { self.sem_guard().compact(); }

    /// Ensure `nid` is in the cache (load from sqlite on miss; evict LRU when over cap).
    fn ensure(inner: &mut Inner, nid: &str, max_facts: usize, cap: usize) {
        inner.tick += 1; let tick = inner.tick;
        if let Some(e) = inner.cache.get_mut(nid) { e.used = tick; return; }
        let row = inner.conn.query_row("SELECT facts,created,updated,turns FROM neurons WHERE id=?1", params![nid],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?))).ok();
        let entry = match row {
            Some((blob, c, _u, t)) => {
                // snapshot facts, then replay the append-log (facts written since the last snapshot)
                let snap_count = blob.split('\n').filter(|l| !l.is_empty()).count();
                let mut combined = blob;
                let mut log_next = 0i64;
                {
                    let mut st = inner.conn.prepare_cached("SELECT seq,lines FROM fact_log WHERE scope=?1 ORDER BY seq").expect("prep log");
                    let rows = st.query_map(params![nid], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))).expect("log rows");
                    for (seq, lines) in rows.flatten() {
                        if !lines.is_empty() { if !combined.is_empty() { combined.push('\n'); } combined.push_str(&lines); }
                        log_next = seq + 1;
                    }
                }
                // dirty=false on load: a pure read session never triggers a compaction write on Drop;
                // dirty is set only when THIS session appends (the writer bounds the log via compaction).
                Entry { n: Neuron::load(&combined, max_facts), created: c, turns: t, used: tick, dirty: false, snap_count, log_next }
            }
            None => { let n = now_ms(); Entry { n: Neuron::new(max_facts), created: n, turns: 0, used: tick, dirty: false, snap_count: 0, log_next: 0 } }
        };
        if inner.cache.len() >= cap {
            if let Some(k) = inner.cache.iter().min_by_key(|(_, e)| e.used).map(|(k, _)| k.clone()) {
                // the append-log keeps every write durable, so the LRU victim is just dropped — no flush
                // needed; its uncompacted log replays on the next load.
                inner.cache.remove(&k);
            }
        }
        inner.cache.insert(nid.to_string(), entry);
    }
    /// Full snapshot: rewrite the scope blob, clear the append-log, reset the log counters. The O(scope)
    /// writer — used for compaction, for deletes/edits (which shift fact indices), and at clean shutdown.
    fn snapshot(conn: &Connection, nid: &str, e: &mut Entry) {
        // ATOMIC: the new snapshot blob and the cleared log must commit together. A crash between the
        // upsert and the DELETE would otherwise leave the new blob PLUS orphaned log rows, and reopen
        // would replay those rows on top of the blob — duplicating every just-snapshotted fact. The
        // transaction makes a mid-snapshot crash a clean rollback (old blob + log stay; replay is correct).
        let tx = conn.unchecked_transaction().expect("tx");
        {
            // prepare_cached: the upsert is parsed once and reused across writes, not re-parsed each call
            let mut stmt = tx.prepare_cached("INSERT INTO neurons(id,facts,created,updated,turns) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(id) DO UPDATE SET facts=excluded.facts,updated=excluded.updated,turns=excluded.turns").expect("prepare");
            stmt.execute(params![nid, e.n.dump(), e.created, now_ms(), e.turns]).expect("save");
        }
        tx.execute("DELETE FROM fact_log WHERE scope=?1", params![nid]).expect("clear log");
        tx.commit().expect("commit");
        e.snap_count = e.n.episodes.len(); e.log_next = 0; e.dirty = false;
    }
    /// Append the facts added since `from` as one durable log row — O(new facts), no whole-blob rewrite.
    /// When the un-snapshotted tail has grown to ~the snapshot size, fold everything into a fresh
    /// snapshot instead (amortized O(1) per fact). Caller guarantees no fact was removed/reordered.
    fn append(conn: &Connection, nid: &str, e: &mut Entry, from: usize, flush_every: usize) {
        let logged = e.n.episodes.len().saturating_sub(e.snap_count);
        if logged >= e.snap_count.max(flush_every).max(COMPACT_FLOOR) { Self::snapshot(conn, nid, e); return; }
        let block = e.n.dump_from(from);
        if block.is_empty() { return; }
        // set dirty BEFORE the write: if the INSERT panics (e.g. SQLITE_FULL), the entry stays dirty so
        // flush_all/Drop snapshots the in-memory facts rather than silently losing this append.
        e.dirty = true;
        if e.snap_count == 0 && e.log_next == 0 {
            // Brand-new scope: register its catalog row (so neurons()/scopes() lists it) AND write its first
            // facts in ONE transaction, so a crash between them can't leave a phantom empty scope — the row
            // present with facts='' but the facts in neither the blob nor the log. We never UPDATE this row
            // afterward: SQLite rewrites the whole row, and its `facts` blob can be megabytes, re-introducing
            // the O(scope) write we're eliminating; `updated` refreshes on the next snapshot() instead.
            let tx = conn.unchecked_transaction().expect("tx");
            tx.execute("INSERT OR IGNORE INTO neurons(id,facts,created,updated,turns) VALUES(?1,'',?2,?3,?4)",
                       params![nid, e.created, now_ms(), e.turns]).ok();
            tx.prepare_cached("INSERT INTO fact_log(scope,seq,lines) VALUES(?1,?2,?3)").expect("prep append")
                .execute(params![nid, e.log_next, block]).expect("append");
            tx.commit().expect("commit");
        } else {
            // steady state: a single fact_log INSERT is atomic on its own (one statement, autocommit).
            conn.prepare_cached("INSERT INTO fact_log(scope,seq,lines) VALUES(?1,?2,?3)").expect("prep append")
                .execute(params![nid, e.log_next, block]).expect("append");
        }
        e.log_next += 1;
    }

    pub fn observe(&self, nid: &str, text: &str) -> usize {
        let w;
        {
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(nid).unwrap();
            // exact-text dedup on the single-observe path (a model re-stating a fact across turns);
            // the batch path stays un-deduped so bulk ingest stays O(n). Scan only a recent window so a
            // single observe stays O(1) — not O(scope) — on a million-fact store (a re-statement that
            // matters lands within a session's worth of facts, not a million ago).
            const DEDUP_WINDOW: usize = 4096;
            let old_len = e.n.episodes.len();
            let recent = old_len.saturating_sub(DEDUP_WINDOW);
            if e.n.episodes[recent..].iter().any(|ep| ep.t == text) { return 0; }
            w = e.n.observe(text);
            // pure append (no capacity eviction shifted indices) -> log just the new facts (O(new), one
            // durable INSERT); a front-drain moved everything, so re-snapshot the whole scope instead.
            if e.n.episodes.len() == old_len + w { Self::append(conn, nid, e, old_len, self.flush_every); }
            else { Self::snapshot(conn, nid, e); }
        }
        #[cfg(feature = "semantic")] self.sem_guard().train(text);
        w
    }
    /// Batch ingest: one load, many appends, one save. Amortizes the per-write commit.
    pub fn observe_many(&self, nid: &str, texts: &[String]) -> usize {
        let w;
        {
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(nid).unwrap();
            let old_len = e.n.episodes.len();
            let mut acc = 0; for t in texts { acc += e.n.observe(t); }
            // pure append -> one durable log row for the whole batch (O(batch)); a large batch compacts
            // to a snapshot inside append(). A capacity eviction mid-batch -> re-snapshot the scope.
            if e.n.episodes.len() == old_len + acc { Self::append(conn, nid, e, old_len, self.flush_every); }
            else { Self::snapshot(conn, nid, e); }
            w = acc;
        }
        #[cfg(feature = "semantic")] { let mut s = self.sem_guard(); for t in texts { s.train(t); } }
        w
    }
    pub fn recall(&self, nid: &str, query: &str) -> Option<Recall> {
        let lex = {
            let mut g = self.shard(nid); let inner = &mut *g;
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
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let eps = &inner.cache.get(nid).unwrap().n.episodes;
            let start = eps.len().saturating_sub(SEM_FALLBACK_CAP);   // most-recent window only
            eps[start..].iter().map(|e| (e.t.clone(), e.v.clone())).collect()
        };
        if facts.is_empty() { return None; }
        let texts: Vec<&str> = facts.iter().map(|(t, _)| t.as_str()).collect();   // borrow, no second clone
        let mut s = self.sem_guard();
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
        // BOUNDED like recall_semantic: rank the most-recent window so blended recall is O(window), not
        // O(scope), as a chat's memory grows into the millions (and the embedding cache stays bounded).
        const BLENDED_CAP: usize = 4000;
        let facts: Vec<(String, String)> = {
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let eps = &inner.cache.get(nid).unwrap().n.episodes;
            let start = eps.len().saturating_sub(BLENDED_CAP);
            eps[start..].iter().map(|e| (e.t.clone(), e.v.clone())).collect()
        };
        if facts.is_empty() { return Vec::new(); }
        let texts: Vec<&str> = facts.iter().map(|(t, _)| t.as_str()).collect();   // borrow, no second clone
        let ranked = { let mut s = self.sem_guard(); s.rank_cached(query, &texts) };
        if ranked.is_empty() { return self.recall_many(nid, query, k); } // no semantic signal -> lexical
        // HYBRID FUSION via Reciprocal Rank Fusion (Cormack et al. 2009): combine the lexical ranking and
        // the semantic ranking by summing 1/(K+rank) instead of adding a cosine and a coverage score that
        // live on mismatched scales. A fact ranked high by BOTH signals wins; crucially, a strong exact
        // lexical hit is no longer demoted to rank 2 by a slightly-higher cosine on a sibling fact (the
        // hit@1 failure the coder benchmark exposed), while a pure paraphrase with no shared words still
        // surfaces through the semantic ranking. K=60 is the standard RRF constant, not a fitted weight.
        const RRF_K: f32 = 60.0;
        let qw: Vec<String> = query.to_lowercase().split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3).map(|w| w.to_string()).collect();
        // lexical ranking over the same window: facts ordered by how many query content-words they contain
        let mut lex: Vec<(usize, usize)> = (0..texts.len()).map(|i| {              // (hit_count, idx)
            let lt = texts[i].to_lowercase();
            (qw.iter().filter(|w| lt.contains(w.as_str())).count(), i)
        }).filter(|(h, _)| *h > 0).collect();
        lex.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        // `ranked` is already the semantic order (cosine desc). Fuse the two rank lists by RRF.
        let mut rrf: HashMap<usize, f32> = HashMap::new();
        for (rank, &(_, i)) in lex.iter().enumerate() { *rrf.entry(i).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0); }
        for (rank, &(i, _)) in ranked.iter().enumerate() { *rrf.entry(i).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0); }
        let mut scored: Vec<(f32, usize)> = rrf.into_iter().map(|(i, s)| (s, i)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
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
        let mut g = self.shard(nid); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        inner.cache.get_mut(nid).unwrap().n.recall_many(query, k)
    }
    /// Spreading-activation recall: seeds on cue matches, then follows shared-entity links to
    /// surface associated facts (see Neuron::recall_spreading). Association-based, not keyword/cosine.
    pub fn recall_associative(&self, nid: &str, query: &str, k: usize, hops: usize) -> Vec<Spread> {
        let mut g = self.shard(nid); let inner = &mut *g;
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
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(nid).unwrap();
            e.n.forget_prefix(&format!("{} is ", key));
            w = e.n.observe(&line);
            Self::snapshot(conn, nid, e);
        }
        #[cfg(feature = "semantic")] self.sem_guard().train(value);
        w
    }
    /// Read a named variable's FULL value (everything after the first " is "), so multi-word values
    /// and values that themselves contain " is " round-trip exactly — unlike cue-isolated recall.
    pub fn var_get(&self, nid: &str, key: &str) -> Option<String> {
        let mut g = self.shard(nid); let inner = &mut *g;
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
            let mut g = self.shard(nid); let inner = &mut *g;
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
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(nid).unwrap();
            let r = e.n.reinforce_prefix(topic, feeling, crate::affect::STANCE_BUMP);
            // neglected dispositions fade as new feelings form, so the active "culture" can shift
            // over time rather than monotonically hardening on whatever was felt first.
            e.n.decay_prefix_others(topic, crate::affect::STANCE_DECAY, crate::affect::STANCE_FLOOR);
            Self::snapshot(conn, nid, e);
            r
        };
        #[cfg(feature = "semantic")] self.sem_guard().train(feeling);
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
        let mut g = self.shard(nid); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        let max = self.max_facts;
        let Inner { conn, cache, .. } = inner;
        let e = cache.get_mut(nid).unwrap();
        let at_cap = e.n.fact_count() >= max;
        let r = turn(&mut e.n, msg);
        if at_cap && r.wrote > 0 { e.n.episodes.truncate(max); }
        e.turns += 1;
        Self::snapshot(conn, nid, e);
        TurnOut { reply: r.reply, kind: r.kind, wrote: r.wrote, facts: e.n.fact_count(), capacity_reached: at_cap && r.wrote > 0 }
    }
    pub fn forget(&self, nid: &str, m: Option<&str>) -> (usize, usize) {
        let mut g = self.shard(nid); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        let (before, after) = {
            let Inner { conn, cache, .. } = &mut *inner;
            let e = cache.get_mut(nid).unwrap();
            let before = e.n.fact_count();
            match m { Some(s) => { let s = s.to_lowercase(); e.n.episodes.retain(|ep| !ep.t.to_lowercase().contains(&s)); }, None => e.n.episodes.clear() }
            e.n.invalidate_index(); // removal shifts episode indices -> force a rebuild on next recall
            let after = e.n.fact_count();
            Self::snapshot(conn, nid, e);
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
                Self::snapshot(conn, &sub, e);
            }
        }
        (before - after, after)
    }
    pub fn stats(&self, nid: &str) -> Stats {
        let mut g = self.shard(nid); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        let e = inner.cache.get(nid).unwrap();
        Stats { facts: e.n.fact_count(), max_facts: self.max_facts, created: e.created, updated: now_ms(), turns: e.turns, dropped: e.n.dropped }
    }
    pub fn neurons(&self) -> Vec<String> {
        let g = self.catalog();
        let mut st = g.conn.prepare("SELECT id FROM neurons ORDER BY updated DESC").unwrap();
        let rows = st.query_map([], |r| r.get::<_, String>(0)).unwrap();
        rows.filter_map(|x| x.ok()).collect()
    }

    /// A scope's serialized dump() blob (`flag\ttext\tstrength` lines) — a raw read primitive
    /// (tab/newline-safe; dump escapes). The CLI export uses scope_facts() for readable packs.
    pub fn dump_scope(&self, nid: &str) -> String {
        let mut g = self.shard(nid); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        inner.cache.get(nid).unwrap().n.dump()
    }

    /// A scope's stored fact texts, in insertion order — the readable export path (`neuron export`).
    pub fn scope_facts(&self, nid: &str) -> Vec<String> {
        let mut g = self.shard(nid); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        inner.cache.get(nid).unwrap().n.episodes.iter().map(|e| e.t.clone()).collect()
    }
}
