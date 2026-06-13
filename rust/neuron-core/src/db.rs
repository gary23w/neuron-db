//! NeuronDB: a database of neurons in one SQLite file (rusqlite, bundled sqlite). Durable,
//! thread-safe (one connection behind a Mutex). Feature-gated behind `sqlite` so the wasm
//! core stays std-only. Faithful port of neuron_db/db.py (Rust-native blob format).
use rusqlite::{params, Connection};
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

pub struct NeuronDB { conn: Mutex<Connection>, max_facts: usize }

impl NeuronDB {
    pub fn open(path: &str, max_facts: usize) -> Self {
        let conn = Connection::open(path).expect("open sqlite");
        let _ = conn.execute_batch("PRAGMA journal_mode=WAL;");
        conn.execute(SCHEMA, []).expect("schema");
        NeuronDB { conn: Mutex::new(conn), max_facts }
    }
    fn load(&self, conn: &Connection, nid: &str) -> (Neuron, i64, i64, i64) {
        let row = conn.query_row("SELECT facts,created,updated,turns FROM neurons WHERE id=?1", params![nid],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?))).ok();
        match row {
            Some((blob, c, u, t)) => (Neuron::load(&blob, self.max_facts), c, u, t),
            None => { let n = now_ms(); (Neuron::new(self.max_facts), n, n, 0) }
        }
    }
    fn save(&self, conn: &Connection, nid: &str, n: &Neuron, created: i64, turns: i64) {
        conn.execute("INSERT INTO neurons(id,facts,created,updated,turns) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(id) DO UPDATE SET facts=excluded.facts,updated=excluded.updated,turns=excluded.turns",
            params![nid, n.dump(), created, now_ms(), turns]).expect("save");
    }
    pub fn observe(&self, nid: &str, text: &str) -> usize {
        let conn = self.conn.lock().unwrap();
        let (mut n, c, _u, t) = self.load(&conn, nid);
        let w = n.observe(text); self.save(&conn, nid, &n, c, t); w
    }
    pub fn recall(&self, nid: &str, query: &str) -> Option<Recall> {
        let conn = self.conn.lock().unwrap();
        let (mut n, _, _, _) = self.load(&conn, nid); n.recall(query)
    }
    pub fn get(&self, nid: &str, query: &str) -> Option<String> { self.recall(nid, query).map(|h| h.value) }
    pub fn turn(&self, nid: &str, msg: &str) -> TurnOut {
        let conn = self.conn.lock().unwrap();
        let (mut n, c, _u, mut t) = self.load(&conn, nid);
        let at_cap = n.fact_count() >= self.max_facts;
        let r = turn(&mut n, msg);
        if at_cap && r.wrote > 0 { n.episodes.truncate(self.max_facts); }
        t += 1; self.save(&conn, nid, &n, c, t);
        TurnOut { reply: r.reply, kind: r.kind, wrote: r.wrote, facts: n.fact_count(), capacity_reached: at_cap && r.wrote > 0 }
    }
    pub fn forget(&self, nid: &str, m: Option<&str>) -> (usize, usize) {
        let conn = self.conn.lock().unwrap();
        let (mut n, c, _u, t) = self.load(&conn, nid);
        let before = n.fact_count();
        match m { Some(s) => { let s = s.to_lowercase(); n.episodes.retain(|e| !e.t.to_lowercase().contains(&s)); }, None => n.episodes.clear() }
        self.save(&conn, nid, &n, c, t);
        (before - n.fact_count(), n.fact_count())
    }
    pub fn stats(&self, nid: &str) -> Stats {
        let conn = self.conn.lock().unwrap();
        let (n, c, u, t) = self.load(&conn, nid);
        Stats { facts: n.fact_count(), max_facts: self.max_facts, created: c, updated: u, turns: t }
    }
    pub fn neurons(&self) -> Vec<String> {
        let conn = self.conn.lock().unwrap();
        let mut st = conn.prepare("SELECT id FROM neurons ORDER BY updated DESC").unwrap();
        let rows = st.query_map([], |r| r.get::<_, String>(0)).unwrap();
        rows.filter_map(|x| x.ok()).collect()
    }
}
