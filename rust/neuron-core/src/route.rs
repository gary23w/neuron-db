//! The cortex route — the one shared "recall then dispatch" path every transport uses, so gary-neuron
//! is the front gate on the CLI, MCP, and WASM alike and is never bypassed. It recalls a working set
//! from the scope, then lets the cortex route the turn:
//!   answer   — memory answers it; `value` is the recalled answer
//!   escalate — memory can't; hand the turn (with `facts` as context) to the host model
//!   fetch    — a live-world lookup; `value` is the topic to search
//!   store    — a declarative to remember; `value` is the canonical fact
//! gary-neuron is only built with the `cortex` feature (which bakes the weights), so this module is
//! feature-gated; the CLI bin and the `mcp` feature both require `cortex`, so the route is always there.
use crate::op::{apply, NeuronOp, Store};
use crate::model::{Dispatch, GaryModel};
use std::sync::OnceLock;

static CORTEX: OnceLock<GaryModel> = OnceLock::new();
/// Load the baked cortex once per process and reuse it (it is ~7 MB of int8 weights).
fn cortex() -> &'static GaryModel { CORTEX.get_or_init(GaryModel::embedded) }

/// A routed turn: the cortex's decision and the working set it saw.
pub struct Routed {
    /// one of "answer" | "escalate" | "fetch" | "store"
    pub kind: &'static str,
    /// the answer / topic / fact (empty for escalate)
    pub value: String,
    /// the facts recalled and handed to the cortex (the host passes these on escalate)
    pub facts: Vec<String>,
}

/// Recall `k` facts for `query` from `scope`, then dispatch the turn through the cortex.
pub fn route<S: Store + ?Sized>(db: &S, scope: &str, query: &str, k: usize) -> Routed {
    let facts: Vec<String> = apply(db, NeuronOp::Recall {
        scope: scope.to_string(), query: query.to_string(), k, semantic: false, across: false,
    }).hits().into_iter().map(|r| r.fact).collect();
    let (kind, value) = match cortex().dispatch(&facts, query) {
        Dispatch::Answer(v) => ("answer", v),
        Dispatch::Escalate  => ("escalate", String::new()),
        Dispatch::Fetch(t)  => ("fetch", t),
        Dispatch::Store(f)  => ("store", f),
    };
    Routed { kind, value, facts }
}
