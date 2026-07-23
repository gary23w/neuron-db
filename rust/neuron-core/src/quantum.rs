//! Quantum-teleportation tier: ephemeral, spooky, one-shot memory behaviors mapped onto the
//! associative store. **This is not literal quantum computing** — no hardware, no QPU, no
//! superposition of physical states. It is a faithful structural analogy running on classical
//! code: entanglement is a link record, teleportation is a joint recall that consumes the link,
//! collapse is a decrement, the no-cloning theorem is a read counter that burns the fact at 0.
//!
//! The map:
//!   qubit                  -> a fact's text (the store's stable identity — facts have no numeric id)
//!   entanglement           -> an `EntanglementRecord` in a side table, linking two (scope, text) facts
//!   Bell measurement       -> `teleport`: recall the source, consume one e-bit, reconstruct on the dest
//!   classical channel      -> the record's plain-text `classical` instruction (copy/swap/invert/verbatim)
//!   no-cloning theorem     -> `write_once`: a per-fact read budget; the read that spends it deletes the fact
//!   superposition          -> `store_super`: one cue holding weighted alternatives; recall measures one
//!   collapse               -> the e-bit decrement / the alternative decay on measurement
//!
//! Layering mirrors the other tiers: the protocol logic here is std-only and generic over three
//! narrow storage traits, so the same teleport/burn/collapse code runs over the in-memory
//! [`MemBack`] (feature `quantum`) and over the durable `NeuronDB` (feature `quantum-db`, whose
//! trait impls live in db.rs beside the trust ledger's, with lazily-created side tables — a store
//! that never uses the tier keeps a byte-identical schema).
//!
//! Reads are quantum-aware only through this tier's surface (`recall_once`, `teleport`, the CLI /
//! HTTP arms compiled under `quantum-db`); a transport built without the feature reads the same
//! facts without consuming anything. That boundary is deliberate: the base store's recall stays a
//! pure read.

mod entangle;
mod noclone;
mod superposition;
mod teleport;
mod router;

pub use entangle::{disentangle, entangle, entangled_recall, scope_entanglements, EntangledHit, EntangledRecall, EntanglementRecord, HasEntanglements};
pub use noclone::{reads_remaining, write_once};
pub use superposition::{measure, recall_super, store_super, PRUNE_THRESHOLD, SUPER_DECAY, ZENO_BOOST};
pub use teleport::{invert_value, teleport, TeleportResult};
pub use router::QuantumRouter;

use crate::{Neuron, Recall};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// The minimal fact surface the tier needs from a backing store. `&self` with interior mutability,
/// matching `op::Store`, so the durable `NeuronDB` (its own Mutex shards) and the in-memory
/// [`MemBack`] (one Mutex here) both fit, and a shared `&S` (e.g. the HTTP server's Arc) can drive it.
pub trait QuantumBack {
    fn observe(&self, scope: &str, text: &str) -> usize;
    fn recall_one(&self, scope: &str, query: &str) -> Option<Recall>;
    /// Whether a fact with this EXACT text exists in the scope (identity check, not recall).
    fn has_fact(&self, scope: &str, text: &str) -> bool;
    /// Remove the fact(s) whose text equals `text` exactly (NOT the substring matcher forget uses,
    /// which could take innocent bystanders containing the text). Returns the removed count.
    fn forget_exact(&self, scope: &str, text: &str) -> usize;
    /// Replace the fact whose text equals `old` with `new` (re-encoded, strength carried over).
    /// False when `old` is absent or `new` does not encode — the dest is then left untouched.
    fn rewrite_fact(&self, scope: &str, old: &str, new: &str) -> bool;
}

/// Side-state for the no-cloning counters and the superposed values. Kept separate from
/// [`HasEntanglements`] so a backend could implement burn-after-reading without the link table.
pub trait QuantumSide {
    fn noclone_set(&self, scope: &str, text: &str, reads: u32);
    fn noclone_get(&self, scope: &str, text: &str) -> Option<u32>;
    /// Decrement the read budget, removing the marker when it reaches 0. Returns the remaining
    /// budget (Some(0) = this read spent the last one), or None when the fact was never marked.
    fn noclone_dec(&self, scope: &str, text: &str) -> Option<u32>;
    fn super_set(&self, scope: &str, text: &str, alts: &[(String, f64)]);
    fn super_get(&self, scope: &str, text: &str) -> Option<Vec<(String, f64)>>;
    /// Every superposition in the scope, as (cue text, alternatives).
    fn super_all(&self, scope: &str) -> Vec<(String, Vec<(String, f64)>)>;
    fn super_del(&self, scope: &str, text: &str);
}

/// The quantum-aware single read: superpositions measure first (a matching unresolved value
/// collapses), then a normal recall — and a hit carrying a no-clone budget spends one read,
/// deleting the fact when the budget hits 0 (the read itself still returns the value; the NEXT
/// one finds nothing). This is the read the CLI `get`/`recall` and the HTTP `/get`/`/recall`
/// routes use when compiled with `quantum-db`.
pub fn recall_once<S: QuantumBack + QuantumSide + ?Sized>(s: &S, scope: &str, query: &str) -> Option<Recall> {
    if let Some(r) = superposition::measure_matching(s, scope, query) { return Some(r); }
    let hit = s.recall_one(scope, query)?;
    if s.noclone_get(scope, &hit.fact).is_some() {
        if let Some(0) = s.noclone_dec(scope, &hit.fact) {
            s.forget_exact(scope, &hit.fact);   // the no-cloning theorem: the last read burns the fact
        }
    }
    Some(hit)
}

/// The tier as an owned wrapper, mirroring the other tiers' shape (`PlasticNeuron` wraps `Neuron`):
/// `EntangledStore::new(MemBack::new())` is the pure in-memory tier; `EntangledStore::new(db)`
/// wraps a durable `NeuronDB` when built with `quantum-db`. Every method is thin sugar over the
/// free functions, which a borrowed caller (the CLI / HTTP arms) invokes directly on `&S`.
pub struct EntangledStore<S> {
    pub inner: S,
}

impl<S> EntangledStore<S> {
    pub fn new(inner: S) -> Self { EntangledStore { inner } }
}

impl<S: QuantumBack> EntangledStore<S> {
    pub fn observe(&self, scope: &str, text: &str) -> usize { self.inner.observe(scope, text) }
    pub fn recall(&self, scope: &str, query: &str) -> Option<Recall> { self.inner.recall_one(scope, query) }
}

impl<S: QuantumBack + HasEntanglements> EntangledStore<S> {
    /// Link two facts (observing either side first if absent) so recalling one can reach the other.
    pub fn entangle(&self, scope_a: &str, text_a: &str, scope_b: &str, text_b: &str, classical: &str, ebits: u32) -> u64 {
        entangle(&self.inner, scope_a, text_a, scope_b, text_b, classical, ebits)
    }
    pub fn disentangle(&self, id: u64) -> bool { disentangle(&self.inner, id) }
    pub fn entangled_recall(&self, scope: &str, query: &str) -> Option<EntangledRecall> { entangled_recall(&self.inner, scope, query) }
    pub fn teleport(&self, scope: &str, cue: &str) -> Option<TeleportResult> { teleport(&self.inner, scope, cue) }
}

impl<S: QuantumBack + QuantumSide> EntangledStore<S> {
    pub fn write_once(&self, scope: &str, text: &str, max_reads: u32) -> usize { write_once(&self.inner, scope, text, max_reads) }
    pub fn reads_remaining(&self, scope: &str, fact_text: &str) -> Option<u32> { reads_remaining(&self.inner, scope, fact_text) }
    pub fn store_super(&self, scope: &str, text: &str, alternatives: &[&str]) { store_super(&self.inner, scope, text, alternatives) }
    pub fn recall_super(&self, scope: &str, cue: &str) -> Option<String> { recall_super(&self.inner, scope, cue) }
    pub fn recall_once(&self, scope: &str, query: &str) -> Option<Recall> { recall_once(&self.inner, scope, query) }
}

// ---- the in-memory backing (feature `quantum`, no sqlite) ----

const BIG: usize = usize::MAX / 2;   // scopes never self-truncate here; this tier is for small, deliberate state

struct MemInner {
    scopes: HashMap<String, Neuron>,
    links: Vec<EntanglementRecord>,
    next_id: u64,
    noclone: HashMap<(String, String), u32>,
    supers: HashMap<(String, String), Vec<(String, f64)>>,
}

/// In-memory backing for the base `quantum` feature: a scope→Neuron map plus the quantum side
/// state, all behind one Mutex. No persistence — the "in-memory only" half of the tier; the
/// durable half is `NeuronDB` under `quantum-db`.
pub struct MemBack {
    inner: Mutex<MemInner>,
}

impl Default for MemBack {
    fn default() -> Self { Self::new() }
}

impl MemBack {
    pub fn new() -> Self {
        MemBack { inner: Mutex::new(MemInner { scopes: HashMap::new(), links: Vec::new(), next_id: 1, noclone: HashMap::new(), supers: HashMap::new() }) }
    }
    fn lock(&self) -> std::sync::MutexGuard<'_, MemInner> { self.inner.lock().unwrap_or_else(|e| e.into_inner()) }
    pub fn fact_count(&self, scope: &str) -> usize { self.lock().scopes.get(scope).map(|n| n.fact_count()).unwrap_or(0) }
}

impl QuantumBack for MemBack {
    fn observe(&self, scope: &str, text: &str) -> usize {
        self.lock().scopes.entry(scope.to_string()).or_insert_with(|| Neuron::new(BIG)).observe(text)
    }
    fn recall_one(&self, scope: &str, query: &str) -> Option<Recall> {
        self.lock().scopes.get_mut(scope)?.recall(query)
    }
    fn has_fact(&self, scope: &str, text: &str) -> bool {
        self.lock().scopes.get(scope).is_some_and(|n| n.episodes.iter().any(|e| e.t == text))
    }
    fn forget_exact(&self, scope: &str, text: &str) -> usize {
        let mut g = self.lock();
        let n = match g.scopes.get_mut(scope) { Some(n) => n, None => return 0 };
        let before = n.episodes.len();
        n.episodes.retain(|e| e.t != text);
        let removed = before - n.episodes.len();
        if removed > 0 { n.invalidate_index(); }
        removed
    }
    fn rewrite_fact(&self, scope: &str, old: &str, new: &str) -> bool {
        let mut g = self.lock();
        let n = match g.scopes.get_mut(scope) { Some(n) => n, None => return false };
        rewrite_in(n, old, new)
    }
}

/// Shared rewrite primitive over a raw `Neuron`: swap the episode whose text is exactly `old`
/// for a re-encoded `new`, carrying its learned strength. Used by MemBack and QuantumRouter.
pub(crate) fn rewrite_in(n: &mut Neuron, old: &str, new: &str) -> bool {
    let i = match n.episodes.iter().position(|e| e.t == old) { Some(i) => i, None => return false };
    let strength = n.episodes[i].strength;
    match crate::encode(new, None) {
        Some(mut e) => {
            e.strength = strength;
            n.episodes.remove(i);
            n.episodes.push(e);
            n.invalidate_index();   // removal shifts indices -> rebuild on next recall
            true
        }
        None => false,   // an unencodable reconstruction leaves the dest untouched
    }
}

impl HasEntanglements for MemBack {
    fn write_entanglement(&self, mut rec: EntanglementRecord) -> u64 {
        let mut g = self.lock();
        rec.id = g.next_id;
        g.next_id += 1;
        let id = rec.id;
        g.links.push(rec);
        id
    }
    fn read_entanglement(&self, id: u64) -> Option<EntanglementRecord> {
        self.lock().links.iter().find(|l| l.id == id).cloned()
    }
    fn find_entanglements(&self, scope: &str, text: &str) -> Vec<EntanglementRecord> {
        self.lock().links.iter()
            .filter(|l| (l.source_scope == scope && l.source_text == text) || (l.dest_scope == scope && l.dest_text == text))
            .cloned().collect()
    }
    fn scope_entanglements(&self, scope: &str) -> Vec<EntanglementRecord> {
        self.lock().links.iter().filter(|l| l.source_scope == scope || l.dest_scope == scope).cloned().collect()
    }
    fn consume_ebit(&self, id: u64) -> Option<u32> {
        let mut g = self.lock();
        let i = g.links.iter().position(|l| l.id == id && l.ebits > 0)?;
        g.links[i].ebits -= 1;
        let left = g.links[i].ebits;
        if left == 0 { g.links.remove(i); }   // fully consumed -> the pair is disentangled
        Some(left)
    }
    fn delete_entanglement(&self, id: u64) -> bool {
        let mut g = self.lock();
        let before = g.links.len();
        g.links.retain(|l| l.id != id);
        g.links.len() < before
    }
    fn rebind_text(&self, scope: &str, old: &str, new: &str) {
        for l in self.lock().links.iter_mut() {
            if l.source_scope == scope && l.source_text == old { l.source_text = new.to_string(); }
            if l.dest_scope == scope && l.dest_text == old { l.dest_text = new.to_string(); }
        }
    }
}

impl QuantumSide for MemBack {
    fn noclone_set(&self, scope: &str, text: &str, reads: u32) {
        self.lock().noclone.insert((scope.to_string(), text.to_string()), reads.max(1));
    }
    fn noclone_get(&self, scope: &str, text: &str) -> Option<u32> {
        self.lock().noclone.get(&(scope.to_string(), text.to_string())).copied()
    }
    fn noclone_dec(&self, scope: &str, text: &str) -> Option<u32> {
        let mut g = self.lock();
        let key = (scope.to_string(), text.to_string());
        let left = g.noclone.get(&key).copied()?.saturating_sub(1);
        if left == 0 { g.noclone.remove(&key); } else { g.noclone.insert(key, left); }
        Some(left)
    }
    fn super_set(&self, scope: &str, text: &str, alts: &[(String, f64)]) {
        self.lock().supers.insert((scope.to_string(), text.to_string()), alts.to_vec());
    }
    fn super_get(&self, scope: &str, text: &str) -> Option<Vec<(String, f64)>> {
        self.lock().supers.get(&(scope.to_string(), text.to_string())).cloned()
    }
    fn super_all(&self, scope: &str) -> Vec<(String, Vec<(String, f64)>)> {
        self.lock().supers.iter().filter(|((s, _), _)| s == scope)
            .map(|((_, t), a)| (t.clone(), a.clone())).collect()
    }
    fn super_del(&self, scope: &str, text: &str) {
        self.lock().supers.remove(&(scope.to_string(), text.to_string()));
    }
}
