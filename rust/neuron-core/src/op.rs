//! One op vocabulary for the neuron-db store, and the single function — `apply` — that every
//! transport routes its store access through. A dispatcher's job shrinks to `translate` (its wire
//! format -> NeuronOp) and `render` (OpResult -> its wire format); the op SEMANTICS — which db
//! method, the clamps, the recall-rank choice, the recall_value cross-scope fallback — live here,
//! once, instead of being re-derived in each of the CLI / MCP / HTTP dispatchers.
//!
//! Step 1 is concrete over the durable `NeuronDB` (sqlite). A later step generalizes `apply` over a
//! `Store` trait so the in-browser `MemDB` (wasm) shares the exact same vocabulary and the affective
//! layer stops being implemented twice.
use crate::db::{NeuronDB, Stats, TurnOut};
use crate::{Recall, Spread};

/// A single store operation, parsed from a transport's wire format. Scopes arrive already resolved:
/// a caller that uses sub-scopes — e.g. the MCP `note` tool's `::var` / `::instr` / `::stance`
/// suffixes — bakes that into `scope` at translate time, so `apply` stays a pure primitive.
#[derive(Debug, Clone)]
pub enum NeuronOp {
    Observe { scope: String, text: String },
    ObserveMany { scope: String, texts: Vec<String> },
    Recall { scope: String, query: String, k: usize, semantic: bool, across: bool },
    RecallOne { scope: String, query: String },
    RecallValue { scope: String, query: String },
    RecallAssoc { scope: String, query: String, k: usize, hops: usize },
    RecallChain { scope: String, start: String, path: Vec<String> },
    VarSet { scope: String, key: String, value: String },
    VarGet { scope: String, key: String },
    Stance { scope: String, topic: String, feeling: String },
    Mood { scope: String, emotion: String },
    Affect { scope: String },
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
    Turned(TurnOut),
    Forgot { forgot: usize, remaining: usize },
    Stats(Stats),
    Scopes(Vec<String>),
    Text(String),
    Ok,
}

/// Execute one op against the durable store — the ONLY place a transport reaches `db.rs` store
/// methods, so the rank choice, the cross-scope value fallback, and the clamps can't drift between
/// the CLI and the MCP server.
pub fn apply(db: &NeuronDB, op: NeuronOp) -> OpResult {
    match op {
        NeuronOp::Observe { scope, text } => OpResult::Wrote(db.observe(&scope, &text)),
        // observe() per fact (dedups exact restatements) — matches the MCP `remember` path; bulk
        // ingest that wants speed over dedup calls db.observe_many directly, not this.
        NeuronOp::ObserveMany { scope, texts } => OpResult::Wrote(texts.iter().map(|t| db.observe(&scope, t)).sum()),
        NeuronOp::Recall { scope, query, k, semantic, across } => {
            let k = k.clamp(1, 50);
            #[cfg(feature = "semantic")]
            let hits = if across { db.recall_many_across(&scope, &query, k) }
                       else if semantic { db.recall_blended(&scope, &query, k) }
                       else { db.recall_many(&scope, &query, k) };
            #[cfg(not(feature = "semantic"))]
            let hits = { let _ = semantic; if across { db.recall_many_across(&scope, &query, k) } else { db.recall_many(&scope, &query, k) } };
            OpResult::Hits(hits)
        }
        NeuronOp::RecallOne { scope, query } => OpResult::Hit(db.recall(&scope, &query)),
        // main scope first; on a miss, fall back across the user's document sub-scopes so a direct
        // question still finds a value the user filed inside a shared document.
        NeuronOp::RecallValue { scope, query } => OpResult::Value(
            db.get(&scope, &query).or_else(|| db.recall_many_across(&scope, &query, 1).into_iter().next().map(|h| h.value))
        ),
        NeuronOp::RecallAssoc { scope, query, k, hops } =>
            OpResult::Assoc(db.recall_associative(&scope, &query, k.clamp(1, 50), hops.clamp(1, 4))),
        NeuronOp::RecallChain { scope, start, path } => {
            let (value, trail) = db.recall_chain(&scope, &start, &path);
            OpResult::Chain { value, trail }
        }
        NeuronOp::VarSet { scope, key, value } => OpResult::Wrote(db.var_set(&scope, &key, &value)),
        NeuronOp::VarGet { scope, key } => OpResult::Value(db.var_get(&scope, &key)),
        NeuronOp::Stance { scope, topic, feeling } => {
            let (intensity, created) = db.note_stance(&scope, &topic, &feeling);
            OpResult::Stance { intensity, created }
        }
        NeuronOp::Mood { scope, emotion } => { db.set_mood(&scope, &emotion); OpResult::Ok }
        NeuronOp::Affect { scope } => OpResult::Text(db.affect(&scope)),
        NeuronOp::Turn { scope, message } => OpResult::Turned(db.turn(&scope, &message)),
        NeuronOp::Forget { scope, matching } => {
            let (forgot, remaining) = db.forget(&scope, matching.as_deref());
            OpResult::Forgot { forgot, remaining }
        }
        NeuronOp::Stats { scope } => OpResult::Stats(db.stats(&scope)),
        NeuronOp::List => OpResult::Scopes(db.neurons()),
    }
}
