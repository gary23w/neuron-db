//! Entanglement: the link record, its storage trait, and the non-consuming correlated read.
//! A link ties two (scope, exact-text) facts together with a plain-text classical instruction
//! and a budget of e-bits; `teleport` (teleport.rs) is the consuming op, `entangled_recall`
//! here is the read that surfaces a hit's partners without spending anything.

use super::{now_ms, QuantumBack};
use crate::Recall;

/// One entanglement: two facts, the classical channel, and the remaining e-bits. Fact identity
/// is (scope, exact text) — the store's stable identity; base facts carry no numeric id.
#[derive(Clone, Debug)]
pub struct EntanglementRecord {
    pub id: u64,
    pub source_scope: String,
    pub source_text: String,
    pub dest_scope: String,
    pub dest_text: String,
    /// The classical channel: a plain-text instruction written at entangle time and applied at
    /// teleport time — "copy" | "swap" | "invert" | anything else (stored verbatim on the dest).
    pub classical: String,
    /// Remaining entanglement units; each teleport consumes one, and 0 deletes the link.
    pub ebits: u32,
    pub created_at: u64,
}

/// The entanglement side table. `&self` interior mutability, like the other storage traits.
pub trait HasEntanglements {
    /// Store a link (the record's `id` field is assigned by the store). Returns the new id.
    fn write_entanglement(&self, rec: EntanglementRecord) -> u64;
    fn read_entanglement(&self, id: u64) -> Option<EntanglementRecord>;
    /// Every link touching this exact (scope, fact text), as source or dest.
    fn find_entanglements(&self, scope: &str, text: &str) -> Vec<EntanglementRecord>;
    /// Every link with either endpoint in the scope (the CLI listing).
    fn scope_entanglements(&self, scope: &str) -> Vec<EntanglementRecord>;
    /// Spend one e-bit: decrement, deleting the link at 0 (disentangled). Returns the remaining
    /// budget (Some(0) = this was the last teleport over the pair); None when no live link exists.
    fn consume_ebit(&self, id: u64) -> Option<u32>;
    fn delete_entanglement(&self, id: u64) -> bool;
    /// A teleport that rewrites a fact's text must re-point the surviving links at the new text
    /// (text IS identity here — a link left on the old text would dangle).
    fn rebind_text(&self, scope: &str, old: &str, new: &str);
}

/// Link two facts so a recall on one can reach (or teleport onto) the other. Either side that is
/// not already stored verbatim is observed first, so `entangle` can introduce the placeholder
/// dest ("the gate code is ----") in the same breath. `ebits` floors at 1. Returns the link id.
pub fn entangle<S: QuantumBack + HasEntanglements + ?Sized>(
    s: &S, scope_a: &str, text_a: &str, scope_b: &str, text_b: &str, classical: &str, ebits: u32,
) -> u64 {
    if !s.has_fact(scope_a, text_a) { s.observe(scope_a, text_a); }
    if !s.has_fact(scope_b, text_b) { s.observe(scope_b, text_b); }
    s.write_entanglement(EntanglementRecord {
        id: 0,
        source_scope: scope_a.to_string(), source_text: text_a.to_string(),
        dest_scope: scope_b.to_string(), dest_text: text_b.to_string(),
        classical: classical.to_string(), ebits: ebits.max(1), created_at: now_ms(),
    })
}

/// Unglue a pair. The facts themselves stay; only the link (and its side effects) is gone.
pub fn disentangle<S: HasEntanglements + ?Sized>(s: &S, id: u64) -> bool {
    s.delete_entanglement(id)
}

/// One partner surfaced by an entangled recall: the link, the far side's scope + current text,
/// and that fact's own recalled value (None when the partner fact no longer exists).
#[derive(Clone, Debug)]
pub struct EntangledHit {
    pub link: EntanglementRecord,
    pub partner_scope: String,
    pub partner_fact: String,
    pub partner_value: Option<String>,
}

/// A recall hit together with everything it is entangled with.
#[derive(Clone, Debug)]
pub struct EntangledRecall {
    pub hit: Recall,
    pub entangled: Vec<EntangledHit>,
}

/// The correlated read: recall as normal, then surface every entangled partner of the hit —
/// symmetric (a hit on either end sees the other) and NON-consuming (no e-bit is spent; this is
/// looking at the pair, not measuring it — `teleport` is the op that collapses).
pub fn entangled_recall<S: QuantumBack + HasEntanglements + ?Sized>(s: &S, scope: &str, query: &str) -> Option<EntangledRecall> {
    let hit = s.recall_one(scope, query)?;
    let entangled = s.find_entanglements(scope, &hit.fact).into_iter().map(|link| {
        let (p_scope, p_text) = if link.source_scope == scope && link.source_text == hit.fact {
            (link.dest_scope.clone(), link.dest_text.clone())
        } else {
            (link.source_scope.clone(), link.source_text.clone())
        };
        // the partner fact's own value: recall it by its own text (max-overlap self hit)
        let partner_value = s.recall_one(&p_scope, &p_text).map(|h| h.value);
        EntangledHit { link, partner_scope: p_scope, partner_fact: p_text, partner_value }
    }).collect();
    Some(EntangledRecall { hit, entangled })
}

/// Every link with an endpoint in `scope` (sugar over the trait, for the CLI listing).
pub fn scope_entanglements<S: HasEntanglements + ?Sized>(s: &S, scope: &str) -> Vec<EntanglementRecord> {
    HasEntanglements::scope_entanglements(s, scope)
}
