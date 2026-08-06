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
use crate::{Neuron, Passage, Recall, Spread};
use crate::turn::turn;

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS neurons (id TEXT PRIMARY KEY, facts TEXT NOT NULL DEFAULT '[]', created INTEGER NOT NULL, updated INTEGER NOT NULL, turns INTEGER NOT NULL DEFAULT 0);\n\
CREATE TABLE IF NOT EXISTS fact_log (scope TEXT NOT NULL, seq INTEGER NOT NULL, lines TEXT NOT NULL, PRIMARY KEY(scope, seq));";
// NB: the trust "floor" (feature `trust`) creates its own trust_kv table LAZILY on the first reward,
// NOT here — so a lexical/login/KV store that never uses the floor keeps a byte-identical schema.
// the per-scope append-log can grow to ~the snapshot size before we fold it back into a fresh snapshot,
// so a single durable observe is one small INSERT (O(new facts)) and compaction is amortized O(1)/fact.
const COMPACT_FLOOR: usize = 256;
fn now_ms() -> i64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64 }

pub use crate::{Stats, TurnOut};   // defined at the crate root so a no-sqlite wasm build can name them too

/// The grounding behind a stance: the feeling, its accumulated intensity, and the facts the store holds
/// about the topic as evidence — so "why do I feel this?" has an answer, not just a free-text label.
#[derive(Clone, Debug)]
pub struct Why { pub topic: String, pub feeling: String, pub intensity: f32, pub evidence: Vec<String> }

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
    fn recall_assoc(&self, scope: &str, query: &str, k: usize, hops: usize, across: bool) -> Vec<crate::Spread> {
        if across { self.recall_assoc_across(scope, query, k, hops).into_iter().map(|(_, h)| h).collect() }
        else { self.recall_associative(scope, query, k, hops) }
    }
    fn recall_context(&self, scope: &str, query: &str, k: usize, before: usize, after: usize, across: bool) -> Vec<crate::Passage> {
        NeuronDB::recall_context(self, scope, query, k, before, after, across)
    }
    fn scope_page(&self, scope: &str, from: usize, limit: usize) -> (usize, Vec<String>) { self.scope_facts_page(scope, from, limit) }
    fn recall_chain(&self, scope: &str, start: &str, path: &[String]) -> (Option<String>, Vec<String>) { NeuronDB::recall_chain(self, scope, start, path) }
    fn var_set(&self, scope: &str, key: &str, value: &str) -> usize { NeuronDB::var_set(self, scope, key, value) }
    fn var_get(&self, scope: &str, key: &str) -> Option<String> { NeuronDB::var_get(self, scope, key) }
    fn note_stance(&self, scope: &str, topic: &str, feeling: &str) -> (f32, bool) { NeuronDB::note_stance(self, scope, topic, feeling) }
    fn strengthen(&self, scope: &str, matching: &str, bump: f32) -> usize { NeuronDB::strengthen(self, scope, matching, bump) }
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

// ---- the statistics tier (features `topics` / `fisher`+`semantic`): shared shapes + dials ----
// One topic model + one discriminant head per DB handle, resident like the semantic space.
// Per-scope postings are a SIDECAR CACHE keyed by episode index; the Neuron `gen` counter tells
// them when a removal shifted indices (pure appends extend incrementally). Everything here is
// ranking/inspection only, and everything fails open to the pre-tier behavior.
#[cfg(feature = "topics")] const TOPIC_K: usize = 64;         // topics in the streaming model
#[cfg(all(feature = "topics", feature = "semantic"))] const GATE_CAP: usize = 4000;   // gated candidate ceiling = the old window size, so a gated miss never costs more than the windowed scan it replaces
#[cfg(feature = "topics")] const BACKFILL_MAX: usize = 50_000; // most unindexed episodes a lazy postings build will fold in one call; beyond it, fail open
#[cfg(all(feature = "fisher", feature = "semantic"))] const OUTCOME_POS: &str = "outcome:+";   // strengthened facts land here
#[cfg(all(feature = "fisher", feature = "semantic"))] const OUTCOME_NEG: &str = "outcome:-";   // explicitly-forgotten facts land here
// created LAZILY on the first persist of a non-empty model — a store that never learns keeps a
// byte-identical schema (the trust_kv / quantum_kv policy).
#[cfg(any(feature = "topics", all(feature = "fisher", feature = "semantic")))]
const STATS_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS stats_kv (kind TEXT NOT NULL, scope TEXT NOT NULL, k TEXT NOT NULL, v TEXT NOT NULL, PRIMARY KEY(kind, scope, k));";
// the durable semantic space (feature `semantic-db`): one row per word — occurrence count +
// full-precision context vector as an f32-LE blob; the meta row (k='') carries tokens_seen in c.
// Loaded lazily on the FIRST op that needs meaning; saved incrementally (touched words only).
#[cfg(feature = "semantic-db")]
const SEM_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS sem_kv (k TEXT PRIMARY KEY, c INTEGER NOT NULL, v BLOB NOT NULL);";
/// PROSE GATE for the learned tiers: at least two plain word-ish tokens (2..=24 alnum chars).
/// The KV callers (auth records, sessions, vault entries) store base64 blobs — one giant token —
/// and those must never train the semantic space, absorb into topics, or trigger a space load:
/// distributional meaning lives in prose, and the KV hot path must stay spawn-cheap.
#[cfg(any(feature = "semantic-db", feature = "topics"))]
fn prose_like(text: &str) -> bool {
    let mut plain = 0usize;
    for tok in text.split(|c: char| !c.is_alphanumeric()) {
        if tok.len() >= 2 && tok.len() <= 24 {
            plain += 1;
            if plain >= 2 { return true; }
        }
    }
    false
}
/// topic -> episode indices for one scope (lists[k()] is the no-topic bucket, so facts absorbed
/// while the model was cold stay reachable through the gate). `gen`/`upto` tie it to the scope's
/// mutation state; `tokens_at` ties it to the model state (rebuilt once the model doubles).
#[cfg(feature = "topics")]
struct TopicPostings { gen: u64, upto: usize, tokens_at: u64, lists: Vec<Vec<u32>> }
/// Deterministic 1-in-16 sampling gate (fnv of the text) for scope-moment updates.
#[cfg(all(feature = "fisher", feature = "semantic"))]
fn sample16(text: &str) -> bool {
    let mut h = 1469598103934665603u64;
    for b in text.bytes() { h ^= b as u64; h = h.wrapping_mul(1099511628211); }
    h & 15 == 0
}

pub struct NeuronDB {
    shards: Vec<Mutex<Inner>>,   // cache+connection partitioned by scope family; len 1 for in-memory DBs
    max_facts: usize, cap: usize,
    flush_every: usize,   // append-log compaction floor: the log folds into a snapshot once it reaches ~max(snap_count, this)
    #[cfg(feature = "semantic")] sem: Mutex<crate::semantic::SemanticSpace>,
    #[cfg(feature = "semantic")] sem_threshold: f32,
    // cached "does any READ-AFFECTING quantum state exist" hint (0 unknown / 1 present / -1 absent),
    // so the quantum-aware read costs one atomic load — not two SQL lookups — on an ordinary store.
    // Per-process: a fresh handle re-probes once; quantum writes in this process keep it current.
    #[cfg(feature = "quantum-db")] q_hint: std::sync::atomic::AtomicI8,
    // the statistics-tier models load LAZILY (None until an op needs one) and persist only when
    // dirty — so a spawn-per-op host pays nothing for KV verbs that never touch meaning.
    #[cfg(feature = "topics")] tm: Mutex<Option<crate::topics::TopicModel>>,
    #[cfg(feature = "topics")] tm_dirty: std::sync::atomic::AtomicBool,
    #[cfg(feature = "topics")] postings: Mutex<HashMap<String, TopicPostings>>,
    #[cfg(all(feature = "fisher", feature = "semantic"))] fh: Mutex<Option<crate::fisher::FisherHead>>,
    #[cfg(all(feature = "fisher", feature = "semantic"))] fh_dirty: std::sync::atomic::AtomicBool,
    #[cfg(feature = "semantic-db")] sem_loaded: std::sync::atomic::AtomicBool,
}

impl Drop for NeuronDB {
    /// Flush any write-behind buffers on shutdown so a clean exit never loses deferred writes.
    fn drop(&mut self) {
        // the statistics tier's learned state persists first (isolated: a full disk must not
        // block the fact flush below)
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.stats_persist()));
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
            // secure_delete=FAST zeroes freed content when it costs no extra I/O, so a forgotten fact does
            // not linger as readable bytes in a free page (right-to-erasure); forget() also truncates the WAL.
            let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=10000; PRAGMA secure_delete=FAST;");
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
            #[cfg(feature = "quantum-db")] q_hint: std::sync::atomic::AtomicI8::new(0),
            #[cfg(feature = "topics")] tm: Mutex::new(None),
            #[cfg(feature = "topics")] tm_dirty: std::sync::atomic::AtomicBool::new(false),
            #[cfg(feature = "topics")] postings: Mutex::new(HashMap::new()),
            #[cfg(all(feature = "fisher", feature = "semantic"))] fh: Mutex::new(None),
            #[cfg(all(feature = "fisher", feature = "semantic"))] fh_dirty: std::sync::atomic::AtomicBool::new(false),
            #[cfg(feature = "semantic-db")] sem_loaded: std::sync::atomic::AtomicBool::new(false),
        }
    }

    // --- the learned trust "floor" (see trust.rs). Feature-gated (`trust`) and OPT-IN: a lexical /
    // login / KV store that never enables the feature, and any consumer that never calls reward(),
    // is byte-identical — trust_kv is created LAZILY on the first reward, never on plain open. ONE
    // global ledger per DB (catalog shard). The engine drives reward() from the acceptance-oracle
    // Δscore; recall reads weight(). Nothing here privileges any class — the ledger only carries what
    // outcomes have taught it.
    /// Load the persisted trust ledger (empty / all-neutral if none yet, incl. before the table exists).
    #[cfg(feature = "trust")]
    pub fn trust_ledger(&self) -> crate::trust::TrustLedger {
        let g = self.catalog();
        let blob: String = g.conn.query_row("SELECT v FROM trust_kv WHERE k='trust'", [], |r| r.get(0)).unwrap_or_default();
        crate::trust::TrustLedger::load(&blob)
    }
    /// Apply a grounded outcome to the classes recalled this round, persist, return the updated ledger.
    /// Creates trust_kv lazily here (and ONLY here), so non-floor stores never grow the table.
    #[cfg(feature = "trust")]
    pub fn trust_reward(&self, classes: &[String], delta: f32) -> crate::trust::TrustLedger {
        let g = self.catalog();
        let _ = g.conn.execute_batch("CREATE TABLE IF NOT EXISTS trust_kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);");
        let blob: String = g.conn.query_row("SELECT v FROM trust_kv WHERE k='trust'", [], |r| r.get(0)).unwrap_or_default();
        let mut l = crate::trust::TrustLedger::load(&blob);
        let refs: Vec<&str> = classes.iter().map(|s| s.as_str()).collect();
        l.reward(&refs, delta);
        let _ = g.conn.execute("INSERT INTO trust_kv(k,v) VALUES('trust',?1) ON CONFLICT(k) DO UPDATE SET v=?1", params![l.dump()]);
        l
    }
    /// The learned trust weight for a tag-class (NEUTRAL if it has never been rewarded).
    #[cfg(feature = "trust")]
    pub fn trust_weight(&self, class: &str) -> f32 { self.trust_ledger().weight(class) }
    /// The FULL learned trust for a class — weight plus the grounded reward/penalty counts (its confidence).
    /// None-safe: an unseen class reports NEUTRAL with zero counts, so a caller can rank AND weigh certainty
    /// (e.g. a UCB source-selection: weight + exploration_bonus(rewards + penalties)). A read-only query.
    #[cfg(feature = "trust")]
    pub fn trust_stats(&self, class: &str) -> crate::trust::ClassTrust {
        self.trust_ledger().stats(class).cloned().unwrap_or_default()
    }
    /// The whole ledger serialized ("<class>\t<weight>\t<rewards>\t<penalties>" per line).
    #[cfg(feature = "trust")]
    pub fn trust_dump(&self) -> String { self.trust_ledger().dump() }
    /// Spreading-activation recall, re-ranked by activation × learned trust of each fact's tag-class
    /// (trust.rs::class_of). Widens the candidate pool first so a sparse-but-trusted fact can rise
    /// above dense-but-untrusted ones, then truncates to k. This is the floor made load-bearing:
    /// density alone no longer decides what the model sees. Falls back to plain order when nothing
    /// has been learned yet (every class neutral -> act × 1.0 == act).
    #[cfg(feature = "trust")]
    pub fn recall_assoc_trusted(&self, scope: &str, query: &str, k: usize, hops: usize) -> Vec<crate::Spread> {
        let pool = k.saturating_mul(4).clamp(k.max(1), 64);
        let mut hits = self.recall_associative(scope, query, pool, hops);   // 0 = until it settles
        let ledger = self.trust_ledger();
        hits.sort_by(|a, b| {
            let sa = a.act * ledger.weight(&crate::trust::class_of(&a.fact, scope)) as f64;
            let sb = b.act * ledger.weight(&crate::trust::class_of(&b.fact, scope)) as f64;
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        hits
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
    // The statistics tier's locks follow a single-lock discipline: no path holds two of
    // {shard, sem, tm, postings, fh} at once (data is snapshotted between acquisitions), except
    // the harmless shard->postings freshness peek — so no ordering cycle can form. The loaders
    // below read their stats_kv/sem_kv blob BETWEEN lock acquisitions; a racing double-read
    // wastes one parse, never state.
    /// The topic model, loaded from stats_kv on FIRST touch (a KV-only spawn never pays this).
    #[cfg(feature = "topics")]
    fn tm_loaded(&self) -> std::sync::MutexGuard<'_, Option<crate::topics::TopicModel>> {
        {
            let g = self.tm.lock().unwrap_or_else(|e| e.into_inner());
            if g.is_some() { return g; }
        }
        let blob: String = {
            let c = self.catalog();
            c.conn.query_row("SELECT v FROM stats_kv WHERE kind='topics' AND scope='' AND k='model'", [], |r| r.get(0)).unwrap_or_default()
        };
        let m = crate::topics::TopicModel::load(&blob).unwrap_or_else(|| crate::topics::TopicModel::new(TOPIC_K));
        let mut g = self.tm.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_none() { *g = Some(m); }
        g
    }
    #[cfg(feature = "topics")]
    fn postings_guard(&self) -> std::sync::MutexGuard<'_, HashMap<String, TopicPostings>> { self.postings.lock().unwrap_or_else(|e| e.into_inner()) }
    /// The discriminant head, loaded from stats_kv on first touch (dim-drifted dumps discarded).
    #[cfg(all(feature = "fisher", feature = "semantic"))]
    fn fh_loaded(&self) -> std::sync::MutexGuard<'_, Option<crate::fisher::FisherHead>> {
        {
            let g = self.fh.lock().unwrap_or_else(|e| e.into_inner());
            if g.is_some() { return g; }
        }
        let dim = { self.sem_guard().dim() };
        let blob: String = {
            let c = self.catalog();
            c.conn.query_row("SELECT v FROM stats_kv WHERE kind='fisher' AND scope='' AND k='head'", [], |r| r.get(0)).unwrap_or_default()
        };
        let h = crate::fisher::FisherHead::load(&blob).filter(|h| h.dim() == dim)
            .unwrap_or_else(|| crate::fisher::FisherHead::new(dim));
        let mut g = self.fh.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_none() { *g = Some(h); }
        g
    }
    /// Load the durable semantic space once per handle, on the first op that needs meaning.
    /// Tolerates a missing sem_kv (nothing persisted yet). Rows are read under the catalog lock,
    /// imported under the sem lock with the flag re-checked inside, so a race serializes there.
    #[cfg(feature = "semantic-db")]
    fn sem_ensure_loaded(&self) {
        use std::sync::atomic::Ordering::SeqCst;
        if self.sem_loaded.load(SeqCst) { return; }
        let rows: Vec<(String, i64, Vec<u8>)> = {
            let c = self.catalog();
            let mut st = match c.conn.prepare_cached("SELECT k,c,v FROM sem_kv") {
                Ok(s) => s,
                Err(_) => { self.sem_loaded.store(true, SeqCst); return; }   // table absent: nothing durable yet
            };
            let collected: Vec<(String, i64, Vec<u8>)> = match st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, Vec<u8>>(2)?))) {
                Ok(it) => it.flatten().collect(),
                Err(_) => Vec::new(),
            };
            collected
        };
        let mut s = self.sem_guard();
        if self.sem_loaded.swap(true, SeqCst) { return; }   // another thread imported while we read
        for (k, cnum, blob) in rows {
            if k.is_empty() { s.set_tokens(cnum.max(0) as u64); continue; }
            let mut v = Vec::with_capacity(blob.len() / 4);
            for ch in blob.chunks_exact(4) { v.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]])); }
            s.import_word(&k, cnum.max(0) as u32, v);
        }
    }

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
        self.stats_persist();
    }

    /// Train the semantic space on arbitrary background text (e.g. a book/corpus), so recall
    /// can fall back to meaning when lexical cues miss. No-op unless built with `semantic`.
    #[cfg(feature = "semantic")]
    pub fn train_semantic(&self, text: &str) {
        #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
        self.sem_guard().train(text);
    }
    /// (vocab words, tokens seen, approx bytes) of the semantic space.
    #[cfg(feature = "semantic")]
    pub fn semantic_stats(&self) -> (usize, u64, usize) {
        #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
        let s = self.sem_guard(); (s.vocab(), s.tokens(), s.bytes())
    }
    /// The k nearest words to `word` in the learned semantic space (for inspection).
    #[cfg(feature = "semantic")]
    pub fn semantic_neighbors(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
        self.sem_guard().nearest(word, k)
    }
    /// Compact the semantic space to int8 for read-mostly serving (~4x smaller; recall intact,
    /// a later observe transparently re-expands it).
    #[cfg(feature = "semantic")]
    pub fn compact_semantic(&self) {
        #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
        self.sem_guard().compact();
    }

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
        #[cfg(feature = "topics")] let mut bags: Vec<Vec<std::sync::Arc<str>>> = Vec::new();
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
            // the new episodes' word bags, cloned under the lock (Arc<str> clones — cheap) so the
            // topic absorb below runs OUTSIDE it. Typed `::` sub-scopes (vars/stances/moods) are
            // bookkeeping, not prose — they stay out of the topic space; so does anything the
            // prose gate rejects (KV/base64 blobs), and blob-shaped tokens are filtered from the
            // bag so a stray hash inside real prose never enters the topic vocabulary.
            #[cfg(feature = "topics")]
            if w > 0 && !nid.contains("::") && prose_like(text) {
                let start = e.n.episodes.len().saturating_sub(w);
                for ep in &e.n.episodes[start..] {
                    let bag: Vec<std::sync::Arc<str>> = ep.raw.iter().filter(|t| t.len() <= 24).cloned().collect();
                    if bag.len() >= 2 { bags.push(bag); }
                }
            }
        }
        // the semantic space trains on PROSE only under semantic-db (a KV blob must not load or
        // grow the durable space); a resident-only build keeps the historical train-everything.
        #[cfg(all(feature = "semantic", not(feature = "semantic-db")))]
        self.sem_guard().train(text);
        #[cfg(feature = "semantic-db")]
        if prose_like(text) { self.sem_ensure_loaded(); self.sem_guard().train(text); }
        // streaming topic learning: each new fact folds in against the current counts and commits
        // its assignments (the Random-Indexing posture — accumulate forever, no refit required)
        #[cfg(feature = "topics")]
        if !bags.is_empty() {
            let mut g = self.tm_loaded();
            if let Some(tm) = g.as_mut() {
                for b in &bags { tm.absorb(b); }
                self.tm_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        // sampled scope moment (1-in-16, content-gated): feeds the scope-vs-rest side of the
        // discriminant head so "does this fact even look like this scope" is answerable on demand
        #[cfg(all(feature = "fisher", feature = "semantic"))]
        if w > 0 && !nid.contains("::") && sample16(text) && prose_like(text) {
            #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
            let x = { self.sem_guard().embed(text) };
            if let Some(x) = x {
                let class = format!("scope:{}", Self::shard_key(nid));
                if let Some(fh) = self.fh_loaded().as_mut() {
                    fh.observe_labeled(&class, &x);
                    self.fh_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        w
    }
    /// Batch ingest: one load, many appends, one save. Amortizes the per-write commit.
    pub fn observe_many(&self, nid: &str, texts: &[String]) -> usize {
        let w;
        #[cfg(feature = "topics")] let mut bags: Vec<Vec<std::sync::Arc<str>>> = Vec::new();
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
            #[cfg(feature = "topics")]
            if acc > 0 && !nid.contains("::") {
                let start = e.n.episodes.len().saturating_sub(acc);
                for ep in &e.n.episodes[start..] {
                    let bag: Vec<std::sync::Arc<str>> = ep.raw.iter().filter(|t| t.len() <= 24).cloned().collect();
                    if bag.len() >= 2 { bags.push(bag); }
                }
            }
        }
        #[cfg(all(feature = "semantic", not(feature = "semantic-db")))]
        { let mut s = self.sem_guard(); for t in texts { s.train(t); } }
        #[cfg(feature = "semantic-db")]
        if texts.iter().any(|t| prose_like(t)) {
            self.sem_ensure_loaded();
            let mut s = self.sem_guard();
            for t in texts.iter().filter(|t| prose_like(t)) { s.train(t); }
        }
        #[cfg(feature = "topics")]
        if !bags.is_empty() {
            let mut g = self.tm_loaded();
            if let Some(tm) = g.as_mut() {
                for b in &bags { tm.absorb(b); }
                self.tm_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        #[cfg(all(feature = "fisher", feature = "semantic"))]
        if w > 0 && !nid.contains("::") {
            for t in texts.iter().filter(|t| sample16(t) && prose_like(t)) {
                #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
                let x = { self.sem_guard().embed(t) };
                if let Some(x) = x {
                    let class = format!("scope:{}", Self::shard_key(nid));
                    if let Some(fh) = self.fh_loaded().as_mut() {
                        fh.observe_labeled(&class, &x);
                        self.fh_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        }
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
        // topic gate first: SCOPE-WIDE topical candidates when the topics tier can offer them
        // (a paraphrase of a fact 100k episodes back becomes reachable); fail-open to the
        // recent window otherwise — byte-identical to the pre-topics behavior.
        #[cfg(all(feature = "topics", feature = "semantic"))] let gated = self.topic_gate(nid, query);
        #[cfg(not(all(feature = "topics", feature = "semantic")))] let gated: Option<Vec<u32>> = None;
        let (idxs, facts): (Vec<usize>, Vec<(String, String)>) = {
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let eps = &inner.cache.get(nid).unwrap().n.episodes;
            match gated {
                Some(list) => {
                    let idxs: Vec<usize> = list.into_iter().map(|i| i as usize).filter(|&i| i < eps.len()).collect();
                    let facts = idxs.iter().map(|&i| (eps[i].t.clone(), eps[i].v.clone())).collect();
                    (idxs, facts)
                }
                None => {
                    let start = eps.len().saturating_sub(SEM_FALLBACK_CAP);   // most-recent window only
                    ((start..eps.len()).collect(), eps[start..].iter().map(|e| (e.t.clone(), e.v.clone())).collect())
                }
            }
        };
        if facts.is_empty() { return None; }
        #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
        let texts: Vec<&str> = facts.iter().map(|(t, _)| t.as_str()).collect();   // borrow, no second clone
        let mut s = self.sem_guard();
        let ranked = s.rank_cached(query, &texts);
        match ranked.first() {
            Some(&(i, score)) if score >= self.sem_threshold => {
                // SPECIFICITY gate: with enough candidates, the winner must stand OUT of the
                // field, not merely clear the absolute bar. A templated scope ("record recN maps
                // to dataN") ranks EVERY fact ~equal on shared frame words — visible since the
                // space became durable (a fresh process used to know nothing) — and the honest
                // answer to "which one" there is none-of-them. A real paraphrase hit towers over
                // the field median; frame similarity does not. Small candidate sets keep the
                // absolute-threshold behavior (a median needs a field to be meaningful).
                const SPEC_MIN_N: usize = 3;
                if ranked.len() >= SPEC_MIN_N {
                    let median = ranked[ranked.len() / 2].1;
                    if score - median < 0.10 { return None; }
                }
                let (fact, value) = facts[i].clone();
                // i indexes the CANDIDATE list; idxs[i] is the true episode index in the scope
                Some(Recall { fact, value, coverage: score as f64, overlap: 0, exact: 0, echo: false, idx: idxs[i] })
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
        // The topic gate upgrades the window to SCOPE-WIDE topical candidates at the same ceiling.
        const BLENDED_CAP: usize = 4000;
        #[cfg(all(feature = "topics", feature = "semantic"))] let gated = self.topic_gate(nid, query);
        #[cfg(not(all(feature = "topics", feature = "semantic")))] let gated: Option<Vec<u32>> = None;
        let (idxs, facts): (Vec<usize>, Vec<(String, String)>) = {
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let eps = &inner.cache.get(nid).unwrap().n.episodes;
            match gated {
                Some(list) => {
                    let idxs: Vec<usize> = list.into_iter().map(|i| i as usize).filter(|&i| i < eps.len()).collect();
                    let facts = idxs.iter().map(|&i| (eps[i].t.clone(), eps[i].v.clone())).collect();
                    (idxs, facts)
                }
                None => {
                    let start = eps.len().saturating_sub(BLENDED_CAP);
                    ((start..eps.len()).collect(), eps[start..].iter().map(|e| (e.t.clone(), e.v.clone())).collect())
                }
            }
        };
        if facts.is_empty() { return Vec::new(); }
        #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
        let texts: Vec<&str> = facts.iter().map(|(t, _)| t.as_str()).collect();   // borrow, no second clone
        let ranked = { let mut s = self.sem_guard(); s.rank_cached(query, &texts) };
        if ranked.is_empty() { return self.recall_many(nid, query, k); } // no semantic signal -> lexical
        // the discriminant's rank list: candidates scoring positive along the learned outcome
        // axis (helped-vs-hurt), best first — absent (and costless) while the head is inert.
        #[cfg(all(feature = "fisher", feature = "semantic"))]
        let fisher_list: Vec<(usize, f32)> = {
            let ax = { self.fh_loaded().as_mut().and_then(|h| h.axis(OUTCOME_POS, OUTCOME_NEG)) };
            match ax {
                Some(ax) => { let mut s = self.sem_guard(); s.project_cached(&ax.w, ax.c, &texts).into_iter().filter(|&(_, z)| z > 0.0).collect() }
                None => Vec::new(),
            }
        };
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
        // `ranked` is already the semantic order (cosine desc). Fuse the rank lists by RRF —
        // and because RRF composes over any number of lists, the discriminant's evidence joins
        // as a THIRD list when it exists: rank evidence, never a raw score on a foreign scale.
        let mut rrf: HashMap<usize, f32> = HashMap::new();
        for (rank, &(_, i)) in lex.iter().enumerate() { *rrf.entry(i).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0); }
        for (rank, &(i, _)) in ranked.iter().enumerate() { *rrf.entry(i).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0); }
        #[cfg(all(feature = "fisher", feature = "semantic"))]
        for (rank, &(i, _)) in fisher_list.iter().enumerate() { *rrf.entry(i).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0); }
        let mut scored: Vec<(f32, usize)> = rrf.into_iter().map(|(i, s)| (s, i)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        scored.truncate(k);
        scored.into_iter().map(|(score, i)| {
            let (fact, value) = facts[i].clone();
            // i indexes the CANDIDATE list; idxs[i] is the true episode index in the scope
            Recall { fact, value, coverage: score as f64, overlap: 0, exact: 0, echo: false, idx: idxs[i] }
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
    /// Spreading recall across a base scope AND its document sub-scopes (`base`, `base__doc1`, …),
    /// mirroring recall_many_across's scope filter (typed `::` sub-scopes and other tenants excluded).
    /// Each scope spreads independently with the same k, then the hits merge by activation. Returns
    /// (scope, hit) pairs so a caller can expand a hit in ITS OWN scope (Spread.idx is scope-local).
    /// This is what lets "what does the hive know about X" reach a document absorbed under its own
    /// sub-scope without the caller knowing document names.
    pub fn recall_assoc_across(&self, base: &str, query: &str, k: usize, hops: usize) -> Vec<(String, Spread)> {
        let scopes: Vec<String> = self.neurons().into_iter()
            .filter(|id| id == base || (id.starts_with(base) && id[base.len()..].starts_with("__")))
            .collect();
        let mut all: Vec<(String, Spread)> = Vec::new();
        for s in &scopes {
            for h in self.recall_associative(s, query, k, hops) { all.push((s.clone(), h)); }
        }
        // seeds (direct query matches) outrank pure associates regardless of which scope is denser —
        // activation magnitudes aren't comparable across scopes of different sizes, seed-ness is.
        all.sort_by(|a, b| b.1.seed.cmp(&a.1.seed)
            .then(b.1.act.partial_cmp(&a.1.act).unwrap_or(std::cmp::Ordering::Equal)));
        all.truncate(k);
        all
    }
    /// The insertion-order window around episode `idx` of a scope (see Neuron::neighbors).
    pub fn neighbors(&self, nid: &str, idx: usize, before: usize, after: usize) -> (usize, Vec<String>) {
        let mut g = self.shard(nid); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        inner.cache.get(nid).unwrap().n.neighbors(idx, before, after)
    }
    /// STITCHED recall: top-k hits, each expanded into its surrounding episodes in insertion
    /// (= document) order — coherent passages instead of isolated sentences. `across` widens the
    /// search over `base__*` document sub-scopes (same filter as recall_many_across), and each hit
    /// expands in the scope it came from. Overlapping windows in the same scope dedupe: a hit that
    /// falls inside an already-emitted passage is skipped rather than re-quoted.
    pub fn recall_context(&self, base: &str, query: &str, k: usize, before: usize, after: usize, across: bool) -> Vec<Passage> {
        let mut tagged: Vec<(String, Recall)> = if across {
            let scopes: Vec<String> = self.neurons().into_iter()
                .filter(|id| id == base || (id.starts_with(base) && id[base.len()..].starts_with("__")))
                .collect();
            let mut all: Vec<(String, Recall)> = Vec::new();
            for s in &scopes {
                for h in self.recall_many(s, query, k) { all.push((s.clone(), h)); }
            }
            all
        } else {
            self.recall_many(base, query, k).into_iter().map(|h| (base.to_string(), h)).collect()
        };
        tagged.sort_by(|a, b| b.1.exact.cmp(&a.1.exact)
            .then(b.1.overlap.cmp(&a.1.overlap))
            .then(b.1.coverage.partial_cmp(&a.1.coverage).unwrap_or(std::cmp::Ordering::Equal)));
        tagged.truncate(k);
        let mut out: Vec<Passage> = Vec::new();
        for (scope, hit) in tagged {
            if out.iter().any(|p| p.scope == scope && hit.idx >= p.start && hit.idx < p.start + p.facts.len()) { continue; }
            let (start, facts) = self.neighbors(&scope, hit.idx, before, after);
            if facts.is_empty() { continue; }
            out.push(Passage { scope, start, hit_pos: hit.idx - start, facts });
        }
        out
    }
    /// One page of a scope's facts in insertion (= document) order: (total_facts, facts[from..from+limit]).
    /// The full-document read path — a summary walks a document scope page by page instead of asking
    /// top-k recall to reconstruct a whole book from fragments. Past-the-end `from` returns an empty page.
    pub fn scope_facts_page(&self, nid: &str, from: usize, limit: usize) -> (usize, Vec<String>) {
        let mut g = self.shard(nid); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        let eps = &inner.cache.get(nid).unwrap().n.episodes;
        let total = eps.len();
        if from >= total || limit == 0 { return (total, Vec::new()); }
        let end = (from + limit).min(total);
        (total, eps[from..end].iter().map(|e| e.t.clone()).collect())
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
        #[cfg(all(feature = "semantic", not(feature = "semantic-db")))]
        self.sem_guard().train(value);
        #[cfg(feature = "semantic-db")]
        if prose_like(value) { self.sem_ensure_loaded(); self.sem_guard().train(value); }
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
        let (mood, stances) = self.affect_state(nid);
        // db.rs shows the persona frame even when neutral (a baseline directive the harness can lean on)
        format!("{}{}", crate::affect::FRAME, crate::affect::directive_body(mood.as_deref(), &stances, asked_topic))
    }
    /// Load the current mood + accumulated stances for a scope (the raw material both the neutral and the
    /// persona-colored directive build from).
    fn affect_state(&self, nid: &str) -> (Option<String>, Vec<(String, String, f32)>) {
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
    }
    /// The affect directive COLORED by an explicitly-attached persona: OCEAN styles the voice and modulates
    /// the threshold at which a budding stance hardens. Opt-in — the neutral `affect` is untouched, and a
    /// neutral persona produces the same text.
    #[cfg(feature = "personality")]
    pub fn affect_with(&self, nid: &str, asked_topic: Option<&str>, persona: &crate::persona::Persona) -> String {
        let (mood, stances) = self.affect_state(nid);
        let style = persona.style();
        let threshold = persona.stance_threshold(crate::affect::STANCE_THRESHOLD);
        format!("{}{}", crate::affect::FRAME,
            crate::affect::directive_body_styled(mood.as_deref(), &stances, asked_topic, Some(&style), threshold))
    }
    /// Record/intensify a stance about `topic`. Re-stating the same topic accumulates its strength
    /// (a disposition that deepens with repetition), persisted durably. Returns (new_strength, new).
    pub fn note_stance(&self, nid: &str, topic: &str, feeling: &str) -> (f32, bool) {
        self.note_stance_tuned(nid, topic, feeling, crate::affect::STANCE_BUMP, crate::affect::STANCE_DECAY, crate::affect::STANCE_FLOOR)
    }
    /// Same as `note_stance` but the reinforcement bump, others-decay, and floor come from the dials —
    /// `note_stance` passes the base constants; `note_stance_with` passes a persona's modulated values.
    fn note_stance_tuned(&self, nid: &str, topic: &str, feeling: &str, bump: f32, decay: f32, floor: f32) -> (f32, bool) {
        // canonicalize the topic (collapse whitespace) so spacing variants reinforce ONE disposition
        // instead of fragmenting into separate stances; reinforce_prefix already case-folds the key.
        let topic: String = topic.split_whitespace().collect::<Vec<_>>().join(" ");
        let out = {
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(nid).unwrap();
            let r = e.n.reinforce_prefix(&topic, feeling, bump);
            // neglected dispositions fade as new feelings form, so the active "culture" can shift
            // over time rather than monotonically hardening on whatever was felt first.
            e.n.decay_prefix_others(&topic, decay, floor);
            Self::snapshot(conn, nid, e);
            r
        };
        #[cfg(all(feature = "semantic", not(feature = "semantic-db")))]
        self.sem_guard().train(feeling);
        #[cfg(feature = "semantic-db")]
        if prose_like(feeling) { self.sem_ensure_loaded(); self.sem_guard().train(feeling); }
        out
    }
    /// Record a stance with the reactivity of an explicitly-attached persona: high Neuroticism (and a hot
    /// temperament) makes the disposition spike harder and linger; opt-in, `note_stance` is unchanged.
    #[cfg(feature = "personality")]
    pub fn note_stance_with(&self, nid: &str, topic: &str, feeling: &str, persona: &crate::persona::Persona) -> (f32, bool) {
        self.note_stance_tuned(nid, topic, feeling,
            persona.stance_bump(crate::affect::STANCE_BUMP),
            persona.stance_decay(crate::affect::STANCE_DECAY),
            persona.stance_floor(crate::affect::STANCE_FLOOR))
    }
    /// Attach a persona to a scope, persisted in the `<scope>::persona` sub-scope (one var per trait). The
    /// store is otherwise neutral; nothing reads this unless a caller asks for a persona-aware op.
    #[cfg(feature = "personality")]
    pub fn set_persona(&self, nid: &str, persona: &crate::persona::Persona) {
        let sub = format!("{}::persona", nid);
        for (k, v) in persona.to_pairs() { self.var_set(&sub, &k, &v); }
    }
    /// Load a scope's attached persona, or None if it was never set.
    #[cfg(feature = "personality")]
    pub fn get_persona(&self, nid: &str) -> Option<crate::persona::Persona> {
        let sub = format!("{}::persona", nid);
        let facts = self.scope_facts(&sub);
        if facts.is_empty() { return None; }
        let pairs: Vec<(String, String)> = facts.iter()
            .filter_map(|f| f.split_once(" is ").map(|(k, v)| (k.trim().to_string(), v.trim().to_string())))
            .collect();
        Some(crate::persona::Persona::from_pairs(&pairs))
    }
    /// Answer "why do I feel this about <topic>?" — the current stance plus the facts the store holds about
    /// that topic as its grounding. This connects a feeling (in `<scope>::stance`) to its likely cause (facts
    /// in the base scope), which a bare "topic: feeling" label can't. None if there is no stance on the topic.
    pub fn why(&self, nid: &str, topic: &str) -> Option<Why> {
        let topic: String = topic.split_whitespace().collect::<Vec<_>>().join(" ");
        let stances = self.affect_state(nid).1; // (topic, feeling, strength)
        let (feeling, intensity) = stances.into_iter()
            .find(|(t, _, _)| crate::affect::topic_matches(&t.to_lowercase(), &topic.to_lowercase()))
            .map(|(_, f, s)| (f, s))?;
        // evidence: the facts the base scope holds about the topic — the likely cause of the feeling
        let evidence: Vec<String> = self.recall_many(nid, &topic, 3).into_iter().map(|h| h.fact).collect();
        Some(Why { topic, feeling, intensity, evidence })
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
    /// Strengthen-only Hebbian plasticity: bump the strength of every fact in `nid` whose text
    /// contains `matching` (case-insensitive substring — the same matcher as forget, its positive
    /// mirror). Never mints and never rewrites text, unlike note_stance: outcome feedback re-ranks
    /// what the scope already learned, it cannot invent memories. Returns the touched count.
    pub fn strengthen(&self, nid: &str, matching: &str, bump: f32) -> usize {
        #[cfg(all(feature = "fisher", feature = "semantic"))] let mut touched: Vec<String> = Vec::new();
        let hit;
        {
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(nid).unwrap();
            hit = e.n.strengthen_matching(matching, bump);
            if hit > 0 {
                // a strengthen IS a grounded positive outcome on specific facts — feed their
                // embeddings to the discriminant head's "+" class (bounded, base scopes only)
                #[cfg(all(feature = "fisher", feature = "semantic"))]
                if !nid.contains("::") {
                    let nl = matching.trim().to_lowercase();
                    touched.extend(e.n.episodes.iter().filter(|ep| ep.t.to_lowercase().contains(&nl)).take(16).map(|ep| ep.t.clone()));
                }
                Self::snapshot(conn, nid, e);   // strength is persisted in the fact blob
            }
        }
        #[cfg(all(feature = "fisher", feature = "semantic"))]
        if !touched.is_empty() {
            #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
            let xs: Vec<Vec<f32>> = { let s = self.sem_guard(); touched.iter().filter_map(|t| s.embed(t)).collect() };
            if !xs.is_empty() {
                if let Some(fh) = self.fh_loaded().as_mut() {
                    for x in &xs { fh.observe_labeled(OUTCOME_POS, x); }
                    self.fh_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        hit
    }
    pub fn forget(&self, nid: &str, m: Option<&str>) -> (usize, usize) {
        let mut g = self.shard(nid); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        #[cfg(all(feature = "fisher", feature = "semantic"))] let mut removed_texts: Vec<String> = Vec::new();
        let (before, after) = {
            let Inner { conn, cache, .. } = &mut *inner;
            let e = cache.get_mut(nid).unwrap();
            let before = e.n.fact_count();
            match m {
                Some(s) => {
                    let s = s.to_lowercase();
                    // a TARGETED forget is a grounded negative outcome on specific content — the
                    // removed facts feed the discriminant's "−" class (bounded, base scopes only;
                    // a full wipe carries no per-fact signal and trains nothing)
                    #[cfg(all(feature = "fisher", feature = "semantic"))]
                    if !nid.contains("::") {
                        removed_texts.extend(e.n.episodes.iter().filter(|ep| ep.t.to_lowercase().contains(&s)).take(16).map(|ep| ep.t.clone()));
                    }
                    e.n.episodes.retain(|ep| !ep.t.to_lowercase().contains(&s));
                }
                None => e.n.episodes.clear(),
            }
            e.n.invalidate_index(); // removal shifts episode indices -> force a rebuild on next recall
            let after = e.n.fact_count();
            Self::snapshot(conn, nid, e);
            (before, after)
        };
        // A full wipe (no match — the "forget me" path) cascades to the typed sub-scopes so stored
        // variables (incl. secrets), standing instructions, stances, and mood aren't left behind.
        // Done INLINE on the held lock — not via self.forget(), which would re-lock and deadlock.
        if m.is_none() {
            // include ::persona so a forgotten subject leaves no attached personality behind either
            for suffix in ["::var", "::instr", "::stance", "::affect", "::persona"] {
                let sub = format!("{}{}", nid, suffix);
                Self::ensure(inner, &sub, self.max_facts, self.cap);
                let Inner { conn, cache, .. } = &mut *inner;
                let e = cache.get_mut(&sub).unwrap();
                e.n.episodes.clear(); e.n.invalidate_index();
                Self::snapshot(conn, &sub, e);
            }
        }
        // Truncate the WAL so the just-deleted plaintext does not survive in -wal frames after a logical
        // delete. With secure_delete=FAST the freed pages are already zeroed; this flushes + truncates the
        // log. Best-effort (a concurrent writer can defer it) and only on the cold forget path.
        let _ = inner.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
        #[cfg(all(feature = "fisher", feature = "semantic"))]
        {
            drop(g);   // release the shard before touching sem/fisher (single-lock discipline)
            if !removed_texts.is_empty() {
                #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
                let xs: Vec<Vec<f32>> = { let s = self.sem_guard(); removed_texts.iter().filter_map(|t| s.embed(t)).collect() };
                if !xs.is_empty() {
                    if let Some(fh) = self.fh_loaded().as_mut() {
                        for x in &xs { fh.observe_labeled(OUTCOME_NEG, x); }
                        self.fh_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
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

    // ---- the statistics tier: topic postings + the gate, discriminant inspection, persistence ----

    /// Bring `nid`'s topic postings up to date (lazy, incremental): fold only the episodes the
    /// postings haven't seen, rebuild from scratch when a removal shifted indices (the Neuron
    /// `gen` counter) or the model has doubled since. False = the tier can't serve this scope
    /// right now (cold model, empty scope, or too much unindexed history) — callers fail open.
    #[cfg(feature = "topics")]
    fn ensure_topic_postings(&self, nid: &str) -> bool {
        let ttok = {
            let g = self.tm_loaded();
            match g.as_ref() { Some(tm) if tm.tokens() > 0 => tm.tokens(), _ => return false }
        };
        // snapshot the scope's mutation state and the un-posted tail's word bags (Arc clones).
        // The shard->postings freshness peek is the one sanctioned nested lock pair.
        let (gen, len, from, bags): (u64, usize, usize, Vec<Vec<std::sync::Arc<str>>>) = {
            let mut g = self.shard(nid); let inner = &mut *g;
            Self::ensure(inner, nid, self.max_facts, self.cap);
            let n = &inner.cache.get(nid).unwrap().n;
            let (gen, len) = (n.gen, n.episodes.len());
            if len == 0 { return false; }
            let from = {
                let p = self.postings_guard();
                match p.get(nid) {
                    Some(tp) if tp.gen == gen && tp.upto <= len
                        && ttok < tp.tokens_at.saturating_mul(2).max(tp.tokens_at + 512) => tp.upto,
                    _ => 0,
                }
            };
            if len - from > BACKFILL_MAX { return false; }   // too much unindexed history: fail open
            (gen, len, from, n.episodes[from..].iter().map(|e| e.raw.clone()).collect())
        };
        if bags.is_empty() { return true; }   // postings already cover the scope
        // frozen fold per un-posted episode, outside every other lock. A fact posts under EVERY
        // topic in its mixture (a sentence's vocabulary can straddle topics — top-1 alone loses
        // it to whichever side won the majority); a fact the model can't place lands in the
        // no-topic bucket so it stays reachable through the gate.
        let (kk, assigns): (usize, Vec<Vec<usize>>) = {
            let g = self.tm_loaded();
            let Some(tm) = g.as_ref() else { return false };
            let kk = tm.k();
            (kk, bags.iter().map(|b| {
                let mix = tm.fold_in(b);
                if mix.is_empty() { vec![kk] } else { mix.into_iter().map(|(t, _)| t).collect() }
            }).collect())
        };
        let mut p = self.postings_guard();
        let tp = p.entry(nid.to_string()).or_insert_with(|| TopicPostings { gen, upto: 0, tokens_at: ttok, lists: vec![Vec::new(); kk + 1] });
        if tp.gen != gen || tp.upto > len || from == 0 {
            *tp = TopicPostings { gen, upto: 0, tokens_at: ttok, lists: vec![Vec::new(); kk + 1] };
        }
        if tp.upto != from { return true; }   // another thread already extended past our snapshot
        for (off, ts) in assigns.iter().enumerate() {
            for &t in ts { tp.lists[t.min(kk)].push((from + off) as u32); }
        }
        tp.upto = len;
        true
    }

    /// The topic gate: SCOPE-WIDE candidate indices for a query, selected by topical overlap —
    /// the query folds into the model, its top topics' postings union (plus the no-topic bucket),
    /// capped at GATE_CAP most-recent. None = no usable signal; callers fall back to the recent
    /// window, so the gate can only widen reach, never lose it.
    #[cfg(all(feature = "topics", feature = "semantic"))]
    fn topic_gate(&self, nid: &str, query: &str) -> Option<Vec<u32>> {
        let mut qtok: Vec<String> = crate::content(query).into_iter().collect();
        qtok.sort();   // content() is set-ordered; sorting makes the fold deterministic
        if qtok.is_empty() { return None; }
        // the query's topics: its folded mixture PLUS each query word's own strongest topic — a
        // two-word query must reach a topic its words dominate even when the joint fold tips
        // elsewhere (the word-level view is what recall needs; the doc-level view alone is not).
        let qtopics: Vec<usize> = {
            let g = self.tm_loaded();
            let Some(tm) = g.as_ref() else { return None };
            if tm.tokens() == 0 { return None; }
            let mut ts: Vec<usize> = tm.fold_in(&qtok).into_iter().map(|(t, _)| t).collect();
            for w in &qtok { if let Some(t) = tm.word_topic(w) { ts.push(t); } }
            ts.sort_unstable(); ts.dedup();
            ts
        };
        if qtopics.is_empty() { return None; }
        if !self.ensure_topic_postings(nid) { return None; }
        let (mut topical, bucket): (Vec<u32>, Vec<u32>) = {
            let p = self.postings_guard();
            let tp = p.get(nid)?;
            let kk = tp.lists.len() - 1;
            let mut u: Vec<u32> = Vec::new();
            for &t in &qtopics { if t < kk { u.extend_from_slice(&tp.lists[t]); } }
            (u, tp.lists[kk].clone())   // cold-model facts ride along, so they stay reachable
        };
        topical.sort_unstable(); topical.dedup();
        // the cap trims the BUCKET first, then the oldest topical facts — the whole point of the
        // gate is reaching old-but-topical episodes, so topical evidence outlives the fallback pool
        let mut union: Vec<u32> = if topical.len() >= GATE_CAP {
            let cut = topical.len() - GATE_CAP;
            topical.drain(0..cut);
            topical
        } else {
            let room = GATE_CAP - topical.len();
            let mut u = topical;
            let start = bucket.len().saturating_sub(room);
            u.extend_from_slice(&bucket[start..]);
            u.sort_unstable(); u.dedup();
            u
        };
        if union.is_empty() { return None; }
        union.sort_unstable();
        Some(union)
    }

    /// The top-m topics of a scope by posting share, each with its top words — "what is this
    /// scope about". Empty while the topic model has seen nothing (or the scope is empty).
    #[cfg(feature = "topics")]
    pub fn scope_topics(&self, nid: &str, m: usize, words: usize) -> Vec<(usize, f32, Vec<(String, f32)>)> {
        if !self.ensure_topic_postings(nid) { return Vec::new(); }
        let counts: Vec<(usize, usize)> = {
            let p = self.postings_guard();
            match p.get(nid) {
                Some(tp) => tp.lists[..tp.lists.len() - 1].iter().enumerate().map(|(t, l)| (t, l.len())).collect(),
                None => return Vec::new(),
            }
        };
        let total: usize = counts.iter().map(|&(_, c)| c).sum();
        if total == 0 { return Vec::new(); }
        let mut top: Vec<(usize, usize)> = counts.into_iter().filter(|&(_, c)| c > 0).collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        top.truncate(m);
        let g = self.tm_loaded();
        let Some(tm) = g.as_ref() else { return Vec::new() };
        top.into_iter().map(|(t, c)| (t, c as f32 / total as f32, tm.top_words(t, words))).collect()
    }
    /// (k, documents absorbed, tokens assigned, vocabulary) of the topic model — its stats line.
    #[cfg(feature = "topics")]
    pub fn topics_stats(&self) -> (usize, u64, u64, usize) {
        let g = self.tm_loaded();
        match g.as_ref() { Some(tm) => (tm.k(), tm.docs(), tm.tokens(), tm.vocab_len()), None => (0, 0, 0, 0) }
    }

    /// The learned outcome axis (helped-vs-hurt), or None while the head is inert.
    #[cfg(all(feature = "fisher", feature = "semantic"))]
    pub fn outcome_axis(&self) -> Option<crate::fisher::Axis> {
        self.fh_loaded().as_mut().and_then(|h| h.axis(OUTCOME_POS, OUTCOME_NEG))
    }
    /// The discriminant head's observed classes with their effective sample weights.
    #[cfg(all(feature = "fisher", feature = "semantic"))]
    pub fn fisher_classes(&self) -> Vec<(String, f64)> {
        self.fh_loaded().as_ref().map(|h| h.classes()).unwrap_or_default()
    }
    /// The outcome axis made READABLE: the nearest vocabulary words to the helpful (+) and the
    /// harmful (−) direction, plus the effective sample weight behind each side. None while the
    /// head is inert — an axis nobody earned prints nothing.
    #[cfg(all(feature = "fisher", feature = "semantic"))]
    pub fn axis_words(&self, m: usize) -> Option<(Vec<(String, f32)>, Vec<(String, f32)>, f64, f64)> {
        let ax = self.outcome_axis()?;
        #[cfg(feature = "semantic-db")] self.sem_ensure_loaded();
        let s = self.sem_guard();
        let pos = s.nearest_vec(&ax.w, m);
        let neg_w: Vec<f32> = ax.w.iter().map(|v| -v).collect();
        let neg = s.nearest_vec(&neg_w, m);
        Some((pos, neg, ax.n_pos, ax.n_neg))
    }

    /// Persist the statistics tier's learned state into the lazily-created stats_kv side table
    /// (and, under `semantic-db`, the touched words of the semantic space into sem_kv). Runs on
    /// flush_all and Drop; DIRTY-GATED, so a read-only spawn that merely loaded a model writes
    /// nothing back, and a store that never learned keeps a byte-identical schema. No-op without
    /// the features.
    fn stats_persist(&self) {
        #[cfg(feature = "topics")]
        if self.tm_dirty.swap(false, std::sync::atomic::Ordering::Relaxed) {
            let blob = {
                let g = self.tm.lock().unwrap_or_else(|e| e.into_inner());
                match g.as_ref() { Some(tm) if tm.tokens() > 0 => Some(tm.dump()), _ => None }
            };
            if let Some(blob) = blob {
                let g = self.catalog();
                let _ = g.conn.execute_batch(STATS_SCHEMA);
                let _ = g.conn.execute("INSERT INTO stats_kv(kind,scope,k,v) VALUES('topics','','model',?1) ON CONFLICT(kind,scope,k) DO UPDATE SET v=?1", params![blob]);
            }
        }
        #[cfg(all(feature = "fisher", feature = "semantic"))]
        if self.fh_dirty.swap(false, std::sync::atomic::Ordering::Relaxed) {
            let blob = {
                let mut g = self.fh.lock().unwrap_or_else(|e| e.into_inner());
                match g.as_mut() { Some(fh) if fh.updates() > 0 => Some(fh.dump()), _ => None }
            };
            if let Some(blob) = blob {
                let g = self.catalog();
                let _ = g.conn.execute_batch(STATS_SCHEMA);
                let _ = g.conn.execute("INSERT INTO stats_kv(kind,scope,k,v) VALUES('fisher','','head',?1) ON CONFLICT(kind,scope,k) DO UPDATE SET v=?1", params![blob]);
            }
        }
        #[cfg(feature = "semantic-db")]
        {
            // incremental: only the words train() touched this process, one transaction. The
            // meta row (k='') carries tokens_seen so the emb_cache drift bound survives too.
            let (rows, tokens) = { let mut s = self.sem_guard(); (s.export_touched(), s.tokens()) };
            if !rows.is_empty() {
                let g = self.catalog();
                let _ = g.conn.execute_batch(SEM_SCHEMA);
                let tx = match g.conn.unchecked_transaction() { Ok(t) => t, Err(_) => return };
                {
                    if let Ok(mut st) = tx.prepare_cached("INSERT INTO sem_kv(k,c,v) VALUES(?1,?2,?3) ON CONFLICT(k) DO UPDATE SET c=?2, v=?3") {
                        for (w, c, v) in &rows {
                            let mut blob = Vec::with_capacity(v.len() * 4);
                            for f in v { blob.extend_from_slice(&f.to_le_bytes()); }
                            let _ = st.execute(params![w, *c as i64, blob]);
                        }
                        let _ = st.execute(params!["", tokens as i64, Vec::<u8>::new()]);
                    }
                }
                let _ = tx.commit();
            }
        }
    }
}

// ---- the quantum-teleportation tier's durable state (feature `quantum-db`) ----
// Same policy as the trust ledger: the side tables are created LAZILY on the first quantum WRITE,
// on the catalog shard's connection, and reads tolerate their absence — so a store that never
// touches the tier keeps a byte-identical schema. The protocol logic itself lives in quantum/;
// these impls only give it durable storage.
#[cfg(feature = "quantum-db")]
mod quantum_db {
    use super::*;
    use crate::quantum::{EntanglementRecord, HasEntanglements, QuantumBack, QuantumSide};
    use crate::{esc, unesc};

    const Q_SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS entanglements (id INTEGER PRIMARY KEY AUTOINCREMENT, src_scope TEXT NOT NULL, src_text TEXT NOT NULL, dst_scope TEXT NOT NULL, dst_text TEXT NOT NULL, classical TEXT NOT NULL, ebits INTEGER NOT NULL, created INTEGER NOT NULL);\n\
CREATE INDEX IF NOT EXISTS idx_ent_src ON entanglements(src_scope, src_text);\n\
CREATE INDEX IF NOT EXISTS idx_ent_dst ON entanglements(dst_scope, dst_text);\n\
CREATE TABLE IF NOT EXISTS quantum_kv (kind TEXT NOT NULL, scope TEXT NOT NULL, k TEXT NOT NULL, v TEXT NOT NULL, PRIMARY KEY(kind, scope, k));";

    fn q_ensure(conn: &Connection) { let _ = conn.execute_batch(Q_SCHEMA); }

    use std::sync::atomic::Ordering::Relaxed;
    // hint maintenance: read-affecting quantum WRITES arm the hint, deletes drop it back to
    // "unknown" (the next read re-probes once). Links never gate a plain read, so the
    // entanglement ops don't touch it.
    fn q_arm(db: &NeuronDB) { db.q_hint.store(1, Relaxed); }
    fn q_unknown(db: &NeuronDB) { db.q_hint.store(0, Relaxed); }

    fn rec(r: &rusqlite::Row) -> rusqlite::Result<EntanglementRecord> {
        Ok(EntanglementRecord {
            id: r.get::<_, i64>(0)? as u64,
            source_scope: r.get(1)?, source_text: r.get(2)?,
            dest_scope: r.get(3)?, dest_text: r.get(4)?,
            classical: r.get(5)?, ebits: r.get::<_, i64>(6)?.max(0) as u32,
            created_at: r.get::<_, i64>(7)?.max(0) as u64,
        })
    }
    const REC_COLS: &str = "id,src_scope,src_text,dst_scope,dst_text,classical,ebits,created";

    impl HasEntanglements for NeuronDB {
        fn write_entanglement(&self, r: EntanglementRecord) -> u64 {
            let g = self.catalog();
            q_ensure(&g.conn);
            let _ = g.conn.execute(
                "INSERT INTO entanglements(src_scope,src_text,dst_scope,dst_text,classical,ebits,created) VALUES(?1,?2,?3,?4,?5,?6,?7)",
                params![r.source_scope, r.source_text, r.dest_scope, r.dest_text, r.classical, r.ebits as i64, r.created_at as i64]);
            g.conn.last_insert_rowid() as u64
        }
        fn read_entanglement(&self, id: u64) -> Option<EntanglementRecord> {
            let g = self.catalog();
            let mut st = g.conn.prepare_cached(&format!("SELECT {} FROM entanglements WHERE id=?1", REC_COLS)).ok()?;
            st.query_row(params![id as i64], |r| rec(r)).ok()
        }
        fn find_entanglements(&self, scope: &str, text: &str) -> Vec<EntanglementRecord> {
            // the teleport hot path: two indexed probes (idx_ent_src / idx_ent_dst) via cached
            // statements — an OR across both endpoints would defeat the per-endpoint indexes.
            let g = self.catalog();
            let mut out: Vec<EntanglementRecord> = Vec::new();
            for sql in [
                format!("SELECT {} FROM entanglements WHERE src_scope=?1 AND src_text=?2 ORDER BY id", REC_COLS),
                format!("SELECT {} FROM entanglements WHERE dst_scope=?1 AND dst_text=?2 ORDER BY id", REC_COLS),
            ] {
                let mut st = match g.conn.prepare_cached(&sql) { Ok(s) => s, Err(_) => return Vec::new() };   // table absent: nothing entangled
                let batch: Vec<EntanglementRecord> = match st.query_map(params![scope, text], |r| rec(r)) {
                    Ok(rows) => rows.flatten().collect(),
                    Err(_) => Vec::new(),
                };
                out.extend(batch);
            }
            out.sort_by_key(|l| l.id);
            out.dedup_by_key(|l| l.id);   // a same-scope self-pair would match both probes
            out
        }
        fn scope_entanglements(&self, scope: &str) -> Vec<EntanglementRecord> {
            let g = self.catalog();
            let mut st = match g.conn.prepare_cached(&format!("SELECT {} FROM entanglements WHERE src_scope=?1 OR dst_scope=?1 ORDER BY id", REC_COLS)) {
                Ok(s) => s, Err(_) => return Vec::new(),
            };
            let out = match st.query_map(params![scope], |r| rec(r)) { Ok(rows) => rows.flatten().collect(), Err(_) => Vec::new() };
            out
        }
        fn consume_ebit(&self, id: u64) -> Option<u32> {
            let g = self.catalog();
            let changed = g.conn.execute("UPDATE entanglements SET ebits=ebits-1 WHERE id=?1 AND ebits>0", params![id as i64]).unwrap_or(0);
            if changed == 0 { return None; }   // no live link (missing table included)
            let left: i64 = g.conn.query_row("SELECT ebits FROM entanglements WHERE id=?1", params![id as i64], |r| r.get(0)).unwrap_or(0);
            if left <= 0 { let _ = g.conn.execute("DELETE FROM entanglements WHERE id=?1", params![id as i64]); }
            Some(left.max(0) as u32)
        }
        fn delete_entanglement(&self, id: u64) -> bool {
            let g = self.catalog();
            g.conn.execute("DELETE FROM entanglements WHERE id=?1", params![id as i64]).unwrap_or(0) > 0
        }
        fn rebind_text(&self, scope: &str, old: &str, new: &str) {
            let g = self.catalog();
            let _ = g.conn.execute("UPDATE entanglements SET src_text=?3 WHERE src_scope=?1 AND src_text=?2", params![scope, old, new]);
            let _ = g.conn.execute("UPDATE entanglements SET dst_text=?3 WHERE dst_scope=?1 AND dst_text=?2", params![scope, old, new]);
        }
    }

    impl QuantumSide for NeuronDB {
        fn quantum_dormant(&self) -> bool {
            // one atomic load on the hot path; the probe below runs once per handle (and again
            // only after a quantum delete flips the hint back to unknown)
            match self.q_hint.load(Relaxed) {
                -1 => true,
                1 => false,
                _ => {
                    let g = self.catalog();
                    // a missing table reads as an error -> no state -> dormant
                    let any = g.conn.query_row("SELECT 1 FROM quantum_kv LIMIT 1", [], |_| Ok(())).is_ok();
                    self.q_hint.store(if any { 1 } else { -1 }, Relaxed);
                    !any
                }
            }
        }
        fn noclone_set(&self, scope: &str, text: &str, reads: u32) {
            let g = self.catalog();
            q_ensure(&g.conn);
            let _ = g.conn.execute("INSERT INTO quantum_kv(kind,scope,k,v) VALUES('noclone',?1,?2,?3) ON CONFLICT(kind,scope,k) DO UPDATE SET v=?3",
                params![scope, text, reads.max(1).to_string()]);
            q_arm(self);
        }
        fn noclone_get(&self, scope: &str, text: &str) -> Option<u32> {
            let g = self.catalog();
            let mut st = g.conn.prepare_cached("SELECT v FROM quantum_kv WHERE kind='noclone' AND scope=?1 AND k=?2").ok()?;
            st.query_row(params![scope, text], |r| r.get::<_, String>(0)).ok().and_then(|v| v.parse().ok())
        }
        fn noclone_dec(&self, scope: &str, text: &str) -> Option<u32> {
            let g = self.catalog();
            let cur: u32 = {
                let mut st = g.conn.prepare_cached("SELECT v FROM quantum_kv WHERE kind='noclone' AND scope=?1 AND k=?2").ok()?;
                st.query_row(params![scope, text], |r| r.get::<_, String>(0)).ok().and_then(|v| v.parse().ok())?
            };
            let left = cur.saturating_sub(1);
            if left == 0 {
                let _ = g.conn.execute("DELETE FROM quantum_kv WHERE kind='noclone' AND scope=?1 AND k=?2", params![scope, text]);
                q_unknown(self);   // the last read-affecting entry may be gone: re-probe next read
            } else {
                let _ = g.conn.execute("UPDATE quantum_kv SET v=?3 WHERE kind='noclone' AND scope=?1 AND k=?2", params![scope, text, left.to_string()]);
            }
            Some(left)
        }
        fn super_set(&self, scope: &str, text: &str, alts: &[(String, f64)]) {
            // one "weight\tesc(value)" line per alternative (escape-aware: a value can hold tabs/newlines)
            let v = alts.iter().map(|(a, w)| format!("{}\t{}", w, esc(a))).collect::<Vec<_>>().join("\n");
            let g = self.catalog();
            q_ensure(&g.conn);
            let _ = g.conn.execute("INSERT INTO quantum_kv(kind,scope,k,v) VALUES('super',?1,?2,?3) ON CONFLICT(kind,scope,k) DO UPDATE SET v=?3",
                params![scope, text, v]);
            q_arm(self);
        }
        fn super_get(&self, scope: &str, text: &str) -> Option<Vec<(String, f64)>> {
            let g = self.catalog();
            let mut st = g.conn.prepare_cached("SELECT v FROM quantum_kv WHERE kind='super' AND scope=?1 AND k=?2").ok()?;
            let v: String = st.query_row(params![scope, text], |r| r.get(0)).ok()?;
            Some(parse_alts(&v))
        }
        fn super_all(&self, scope: &str) -> Vec<(String, Vec<(String, f64)>)> {
            let g = self.catalog();
            let mut st = match g.conn.prepare_cached("SELECT k,v FROM quantum_kv WHERE kind='super' AND scope=?1 ORDER BY k") {
                Ok(s) => s, Err(_) => return Vec::new(),
            };
            let out = match st.query_map(params![scope], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))) {
                Ok(rows) => rows.flatten().map(|(k, v)| (k, parse_alts(&v))).collect(),
                Err(_) => Vec::new(),
            };
            out
        }
        fn super_del(&self, scope: &str, text: &str) {
            let g = self.catalog();
            let _ = g.conn.execute("DELETE FROM quantum_kv WHERE kind='super' AND scope=?1 AND k=?2", params![scope, text]);
            q_unknown(self);
        }
    }

    fn parse_alts(v: &str) -> Vec<(String, f64)> {
        v.split('\n').filter(|l| !l.is_empty()).filter_map(|l| {
            let (w, a) = l.split_once('\t')?;
            Some((unesc(a), w.parse::<f64>().ok()?))
        }).collect()
    }

    impl QuantumBack for NeuronDB {
        fn observe(&self, scope: &str, text: &str) -> usize { NeuronDB::observe(self, scope, text) }
        fn recall_one(&self, scope: &str, query: &str) -> Option<crate::Recall> { self.recall(scope, query) }
        fn has_fact(&self, scope: &str, text: &str) -> bool {
            let mut g = self.shard(scope); let inner = &mut *g;
            Self::ensure(inner, scope, self.max_facts, self.cap);
            inner.cache.get(scope).unwrap().n.episodes.iter().any(|e| e.t == text)
        }
        fn forget_exact(&self, scope: &str, text: &str) -> usize {
            let mut g = self.shard(scope); let inner = &mut *g;
            Self::ensure(inner, scope, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(scope).unwrap();
            let before = e.n.episodes.len();
            e.n.episodes.retain(|ep| ep.t != text);
            let removed = before - e.n.episodes.len();
            if removed > 0 { e.n.invalidate_index(); Self::snapshot(conn, scope, e); }
            removed
        }
        fn rewrite_fact(&self, scope: &str, old: &str, new: &str) -> bool {
            let mut g = self.shard(scope); let inner = &mut *g;
            Self::ensure(inner, scope, self.max_facts, self.cap);
            let Inner { conn, cache, .. } = inner;
            let e = cache.get_mut(scope).unwrap();
            // same semantics as the in-memory backing (quantum::rewrite_in): first exact match is
            // removed and the re-encoded text appends — teleport's swap ordering relies on this.
            let i = match e.n.episodes.iter().position(|ep| ep.t == old) { Some(i) => i, None => return false };
            let strength = e.n.episodes[i].strength;
            match crate::encode(new, None) {
                Some(mut ep) => {
                    ep.strength = strength;
                    e.n.episodes.remove(i);
                    e.n.episodes.push(ep);
                    e.n.invalidate_index();
                    Self::snapshot(conn, scope, e);
                    true
                }
                None => false,
            }
        }
    }
}
