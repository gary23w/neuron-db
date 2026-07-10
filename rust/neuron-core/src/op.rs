//! One op vocabulary for the neuron-db store, and the single function — `apply` — that every
//! transport routes its store access through. A dispatcher's job shrinks to `translate` (its wire
//! format -> NeuronOp) and `render` (OpResult -> its wire format).
//!
//! `apply` is generic over the `Store` trait, so BOTH the durable `NeuronDB` (sqlite, in db.rs) and
//! the in-browser `MemDB` (wasm, in wasm.rs) share the exact same vocabulary and dispatch. Each
//! backend owns the op SEMANTICS that differ between them — the recall rank choice, the
//! recall_value fallback — inside its own `Store` impl, so they can't drift here. This module is
//! std-only and not feature-gated, so the no-sqlite wasm build compiles it too.
use crate::{Recall, Spread, Stats, TurnOut};

/// The store surface `apply` needs. Implemented by `NeuronDB` (durable) and `MemDB` (in-browser);
/// `&self` because both stores use interior mutability (a Mutex / a RefCell), which also lets a
/// shared `&NeuronDB` (e.g. the HTTP server's Arc) drive it.
pub trait Store {
    fn observe(&self, scope: &str, text: &str) -> usize;
    fn observe_bulk(&self, scope: &str, texts: &[String]) -> usize;
    fn recall_one(&self, scope: &str, query: &str) -> Option<Recall>;
    /// Top-k block; the backend picks the ranking (NeuronDB honors semantic/across, MemDB is lexical).
    fn recall_block(&self, scope: &str, query: &str, k: usize, semantic: bool, across: bool) -> Vec<Recall>;
    /// A single value for a direct question; the backend owns any cross-scope fallback.
    fn recall_value(&self, scope: &str, query: &str) -> Option<String>;
    fn recall_assoc(&self, scope: &str, query: &str, k: usize, hops: usize) -> Vec<Spread>;
    fn recall_chain(&self, scope: &str, start: &str, path: &[String]) -> (Option<String>, Vec<String>);
    fn var_set(&self, scope: &str, key: &str, value: &str) -> usize;
    fn var_get(&self, scope: &str, key: &str) -> Option<String>;
    fn note_stance(&self, scope: &str, topic: &str, feeling: &str) -> (f32, bool);
    /// Strengthen-only plasticity: bump the strength of facts whose text contains `matching`
    /// (substring, like forget). Never mints — returns how many existing facts were touched.
    fn strengthen(&self, scope: &str, matching: &str, bump: f32) -> usize;
    fn set_mood(&self, scope: &str, emotion: &str);
    fn affect(&self, scope: &str, asked_topic: Option<&str>) -> String;
    fn turn(&self, scope: &str, message: &str) -> TurnOut;
    fn forget(&self, scope: &str, matching: Option<&str>) -> (usize, usize);
    fn stats(&self, scope: &str) -> Stats;
    fn scopes(&self) -> Vec<String>;
}

/// A single store operation, parsed from a transport's wire format. Scopes arrive already resolved:
/// a caller that uses sub-scopes — e.g. the MCP `note` tool's `::var` / `::instr` / `::stance`
/// suffixes — bakes that into `scope` at translate time, so `apply` stays a pure primitive.
#[derive(Debug, Clone)]
pub enum NeuronOp {
    Observe { scope: String, text: String },
    ObserveMany { scope: String, texts: Vec<String> },   // per-fact observe (dedups exact restatements)
    ObserveBulk { scope: String, texts: Vec<String> },   // bulk ingest, no dedup
    Recall { scope: String, query: String, k: usize, semantic: bool, across: bool },
    RecallOne { scope: String, query: String },
    RecallValue { scope: String, query: String },
    RecallAssoc { scope: String, query: String, k: usize, hops: usize },
    RecallChain { scope: String, start: String, path: Vec<String> },
    VarSet { scope: String, key: String, value: String },
    VarGet { scope: String, key: String },
    Stance { scope: String, topic: String, feeling: String },
    Strengthen { scope: String, matching: String, bump: f32 },   // strengthen-only plasticity; never mints
    Mood { scope: String, emotion: String },
    Affect { scope: String, topic: Option<String> },     // topic = optional stance to bias the persona toward
    Turn { scope: String, message: String },
    Forget { scope: String, matching: Option<String> },
    Stats { scope: String },
    List,
}

/// The outcome of an op, carrying the raw store return so each transport renders it its own way.
#[derive(Debug, Clone)]
pub enum OpResult {
    Wrote(usize),
    Hit(Option<Recall>),
    Hits(Vec<Recall>),
    Assoc(Vec<Spread>),
    Value(Option<String>),
    Chain { value: Option<String>, trail: Vec<String> },
    Stance { intensity: f32, created: bool },
    Strengthened(usize),   // how many existing facts the strengthen touched (0 = nothing matched)
    Turned(TurnOut),
    Forgot { forgot: usize, remaining: usize },
    Stats(Stats),
    Scopes(Vec<String>),
    Text(String),
    Ok,
}

impl OpResult {
    /// Convenience extractors for transports whose op fixes the result variant (recall -> Hits,
    /// remember -> Wrote, …). The op/result pairing is 1:1, so a mismatch is a programming error;
    /// these return an empty default rather than panic to keep dispatchers total.
    pub fn wrote(&self) -> usize { if let OpResult::Wrote(n) = self { *n } else { 0 } }
    pub fn hit(self) -> Option<Recall> { if let OpResult::Hit(h) = self { h } else { None } }
    pub fn hits(self) -> Vec<Recall> { if let OpResult::Hits(h) = self { h } else { Vec::new() } }
    pub fn assoc(self) -> Vec<Spread> { if let OpResult::Assoc(a) = self { a } else { Vec::new() } }
    pub fn value(self) -> Option<String> { if let OpResult::Value(v) = self { v } else { None } }
    pub fn text(self) -> String { if let OpResult::Text(t) = self { t } else { String::new() } }
}

/// Execute one op against any `Store`. The ONLY place a transport reaches the store, so the clamps
/// live here once and the per-backend semantics live in each `Store` impl.
pub fn apply<S: Store + ?Sized>(db: &S, op: NeuronOp) -> OpResult {
    match op {
        NeuronOp::Observe { scope, text } => OpResult::Wrote(db.observe(&scope, &text)),
        // per-fact observe (dedups exact restatements) — matches the MCP `remember` path
        NeuronOp::ObserveMany { scope, texts } => OpResult::Wrote(texts.iter().map(|t| db.observe(&scope, t)).sum()),
        NeuronOp::ObserveBulk { scope, texts } => OpResult::Wrote(db.observe_bulk(&scope, &texts)),
        NeuronOp::Recall { scope, query, k, semantic, across } => OpResult::Hits(db.recall_block(&scope, &query, k.clamp(1, 50), semantic, across)),
        NeuronOp::RecallOne { scope, query } => OpResult::Hit(db.recall_one(&scope, &query)),
        NeuronOp::RecallValue { scope, query } => OpResult::Value(db.recall_value(&scope, &query)),
        NeuronOp::RecallAssoc { scope, query, k, hops } => OpResult::Assoc(db.recall_assoc(&scope, &query, k.clamp(1, 64), hops.clamp(1, 32))),
        NeuronOp::RecallChain { scope, start, path } => { let (value, trail) = db.recall_chain(&scope, &start, &path); OpResult::Chain { value, trail } }
        NeuronOp::VarSet { scope, key, value } => OpResult::Wrote(db.var_set(&scope, &key, &value)),
        NeuronOp::VarGet { scope, key } => OpResult::Value(db.var_get(&scope, &key)),
        NeuronOp::Stance { scope, topic, feeling } => { let (intensity, created) = db.note_stance(&scope, &topic, &feeling); OpResult::Stance { intensity, created } }
        NeuronOp::Strengthen { scope, matching, bump } => OpResult::Strengthened(db.strengthen(&scope, &matching, bump.clamp(0.05, 4.0))),
        NeuronOp::Mood { scope, emotion } => { db.set_mood(&scope, &emotion); OpResult::Ok }
        NeuronOp::Affect { scope, topic } => OpResult::Text(db.affect(&scope, topic.as_deref())),
        NeuronOp::Turn { scope, message } => OpResult::Turned(db.turn(&scope, &message)),
        NeuronOp::Forget { scope, matching } => { let (forgot, remaining) = db.forget(&scope, matching.as_deref()); OpResult::Forgot { forgot, remaining } }
        NeuronOp::Stats { scope } => OpResult::Stats(db.stats(&scope)),
        NeuronOp::List => OpResult::Scopes(db.scopes()),
    }
}
