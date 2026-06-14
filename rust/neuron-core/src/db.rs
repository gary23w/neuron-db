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
pub struct NeuronDB { inner: Mutex<Inner>, max_facts: usize, cap: usize }

impl NeuronDB {
    pub fn open(path: &str, max_facts: usize) -> Self {
        let conn = Connection::open(path).expect("open sqlite");
        let _ = conn.execute_batch("PRAGMA journal_mode=WAL;");
        conn.execute(SCHEMA, []).expect("schema");
        NeuronDB { inner: Mutex::new(Inner { conn, cache: HashMap::new(), tick: 0 }), max_facts, cap: 256 }
    }

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
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        let Inner { conn, cache, .. } = inner;
        let e = cache.get_mut(nid).unwrap();
        let w = e.n.observe(text);
        Self::persist(conn, nid, e); w
    }
    /// Batch ingest: one load, many appends, one save. Amortizes the per-write commit.
    pub fn observe_many(&self, nid: &str, texts: &[String]) -> usize {
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        let Inner { conn, cache, .. } = inner;
        let e = cache.get_mut(nid).unwrap();
        let mut w = 0; for t in texts { w += e.n.observe(t); }
        Self::persist(conn, nid, e); w
    }
    pub fn recall(&self, nid: &str, query: &str) -> Option<Recall> {
        let mut g = self.inner.lock().unwrap(); let inner = &mut *g;
        Self::ensure(inner, nid, self.max_facts, self.cap);
        inner.cache.get_mut(nid).unwrap().n.recall(query)
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
            let rel_root = crate::root_token(rel);
            match self.recall(nid, &format!("{} {}", current, rel)) {
                // only advance if the relation actually appears in the recalled fact; otherwise
                // recall's best-effort (entity overlap alone) would let a broken chain continue.
                Some(h) if h.fact.split_whitespace().any(|w| crate::root_token(w) == rel_root) => {
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
