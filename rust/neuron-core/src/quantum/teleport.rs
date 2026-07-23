//! Teleportation: the consuming joint op. Measure (recall the source), collapse (spend one
//! e-bit), send the classical instruction, reconstruct on the entangled partner. The protocol is
//! atomic from the caller's view — one call either runs all of it or returns None — which is the
//! whole point over "store two facts and delete one by hand": there is no window where the
//! association exists on both sides un-accounted (the no-cloning spirit).

use super::entangle::{EntanglementRecord, HasEntanglements};
use super::QuantumBack;

/// What a teleport did: the reconstructed value, where it came from / arrived, the classical
/// instruction that guided the reconstruction, and the e-bits the link has left (0 = the pair
/// is now disentangled and the link record is gone).
#[derive(Clone, Debug)]
pub struct TeleportResult {
    pub value: String,
    pub source_scope: String,
    pub source_fact: String,
    pub dest_scope: String,
    /// The dest fact's NEW text after reconstruction.
    pub dest_fact: String,
    pub classical_used: String,
    pub ebits_remaining: u32,
}

/// Invert a value: negate a number, otherwise reverse the string. ("4491" -> "-4491",
/// "-3.5" -> "3.5", "falcon" -> "noclaf".)
pub fn invert_value(v: &str) -> String {
    let t = v.trim();
    if let Ok(n) = t.parse::<i64>() { return (-n).to_string(); }
    if let Ok(f) = t.parse::<f64>() {
        let neg = -f;
        return if neg == neg.trunc() && neg.abs() < 1e15 { format!("{}", neg as i64) } else { format!("{}", neg) };
    }
    t.chars().rev().collect()
}

/// The classical channel applied: given the instruction and the measured source, produce the
/// dest fact's new text and the value the teleport reports. "swap" is handled by the caller
/// (it rewrites both sides); this covers copy / invert / verbatim.
pub(crate) fn reconstruct(classical: &str, src_text: &str, src_value: &str) -> (String, String) {
    match classical {
        "copy" => (src_text.to_string(), src_value.to_string()),
        "invert" => {
            let inv = invert_value(src_value);
            let new = src_text.replacen(src_value, &inv, 1);
            // the value is always a substring of its fact in practice (pick_value extracts it);
            // if a clip/expansion ever breaks that, fall back to a fact that still encodes.
            let new = if new == src_text { format!("the inverted value is {}", inv) } else { new };
            (new, inv)
        }
        other => (other.to_string(), other.to_string()),   // any other plain text: stored verbatim on the dest
    }
}

/// Teleport: find the best match for `cue` in `scope`; if that fact is the SOURCE of a live
/// entanglement, consume one e-bit, apply the link's classical instruction to the dest fact,
/// and report what moved. None when the cue misses or the hit has no live source-side link —
/// callers then fall back to a normal recall. Directional on purpose: the dest is the receiving
/// end; the correlated peek in both directions is `entangled_recall`.
///
/// "swap" exchanges the two facts' texts. Both backings rewrite "first exact match, re-append",
/// so the source-first order below stays correct even when both facts share a scope; what a
/// SAME-scope swap does NOT do is re-point third-party links on the two facts (the two rebinds
/// would collide) — split-knowledge pairs are cross-scope, which rebinds fully.
pub fn teleport<S: QuantumBack + HasEntanglements + ?Sized>(s: &S, scope: &str, cue: &str) -> Option<TeleportResult> {
    let hit = s.recall_one(scope, cue)?;                                    // 1. measure the source
    let link = s.find_entanglements(scope, &hit.fact).into_iter()
        .find(|l| l.source_scope == scope && l.source_text == hit.fact && l.ebits > 0)?;   // 2-3. a live source-side link
    let remaining = s.consume_ebit(link.id)?;                               // 4. collapse (0 deletes the link)
    let cross_scope = link.source_scope != link.dest_scope;
    let (new_dest, value) = if link.classical == "swap" {
        // exchange: the source takes the dest's old text first (its old text is the recall hit,
        // so no information is lost), then the common dest rewrite below gives the dest the
        // source's old text. First-match rewrite finds the ORIGINAL dest, not the re-appended
        // source, so this order is safe even in one scope.
        if s.rewrite_fact(scope, &hit.fact, &link.dest_text) && cross_scope {
            s.rebind_text(scope, &hit.fact, &link.dest_text);               // third-party links follow the source
        }
        (hit.fact.clone(), hit.value.clone())
    } else {
        reconstruct(&link.classical, &hit.fact, &hit.value)                 // 5-6. classical channel
    };
    if s.rewrite_fact(&link.dest_scope, &link.dest_text, &new_dest) {       // 7. reconstruct on the dest
        if link.classical == "swap" {
            if remaining > 0 {
                // the surviving link's endpoints exchanged: re-issue it (fresh id) — a field-wise
                // rebind of both sides would collide when the texts trade places.
                s.delete_entanglement(link.id);
                s.write_entanglement(EntanglementRecord {
                    id: 0,
                    source_scope: link.source_scope.clone(), source_text: link.dest_text.clone(),
                    dest_scope: link.dest_scope.clone(), dest_text: hit.fact.clone(),
                    classical: link.classical.clone(), ebits: remaining, created_at: link.created_at,
                });
            }
            if cross_scope { s.rebind_text(&link.dest_scope, &link.dest_text, &new_dest); }
        } else {
            s.rebind_text(&link.dest_scope, &link.dest_text, &new_dest);    // surviving links follow the fact
        }
    }
    Some(TeleportResult {                                                    // 8. report
        value,
        source_scope: link.source_scope,
        source_fact: hit.fact,
        dest_scope: link.dest_scope,
        dest_fact: new_dest,
        classical_used: link.classical,
        ebits_remaining: remaining,
    })
}

/// Relay teleportation: keep teleporting along the entanglement graph until it SETTLES — the
/// arriving scope has no live source-side link the cue still measures. There is no hop cap, by
/// design (hops are unbounded everywhere): every hop consumes exactly one e-bit from a finite
/// budget, so even a CYCLIC entanglement graph drains and the cascade terminates — conservation
/// ends the relay, not a limit. Returns the full hop trail (empty when nothing teleported).
pub fn teleport_cascade<S: QuantumBack + HasEntanglements + ?Sized>(s: &S, scope: &str, cue: &str) -> Vec<TeleportResult> {
    let mut trail = Vec::new();
    let mut here = scope.to_string();
    while let Some(t) = teleport(s, &here, cue) {
        here = t.dest_scope.clone();
        trail.push(t);
    }
    trail
}
