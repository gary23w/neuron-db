//! Superposition: one cue holding several weighted alternatives, none of them yet a fact.
//! A quantum-aware recall that matches the cue MEASURES the superposition: the highest-amplitude
//! candidate is returned, the losers decay (collapse), the winner is reinforced (the quantum
//! Zeno effect — repeatedly observing the same outcome pins it), and candidates that decay
//! below threshold are pruned. When a single candidate remains, the superposition resolves into
//! an ordinary stored fact ("<cue text> <winner>") and the side entry is deleted — measurement
//! has produced a classical state.
//!
//! Measurement is deterministic (argmax; ties -> first stored) so the behavior is testable and
//! repeatable: amplitude decides the outcome, not a dice roll.

use super::{QuantumBack, QuantumSide};
use crate::{content, stems_s, Recall};

/// Loser decay per measurement (collapse).
pub const SUPER_DECAY: f64 = 0.5;
/// Winner reinforcement per measurement (Zeno).
pub const ZENO_BOOST: f64 = 1.1;
/// Below this amplitude a candidate is removed from the superposition.
pub const PRUNE_THRESHOLD: f64 = 0.1;

/// Store alternatives for a cue, all at amplitude 1.0. Re-storing the same cue text replaces the
/// superposition (a fresh preparation). Empty/whitespace alternatives are dropped.
pub fn store_super<S: QuantumSide + ?Sized>(s: &S, scope: &str, text: &str, alternatives: &[&str]) {
    let alts: Vec<(String, f64)> = alternatives.iter()
        .map(|a| a.trim()).filter(|a| !a.is_empty())
        .map(|a| (a.to_string(), 1.0)).collect();
    if alts.is_empty() { return; }
    s.super_set(scope, text.trim(), &alts);
}

/// One measurement over an alternatives list, in place: returns the winner, boosts it, decays
/// the rest, prunes the faded. Pure math — the storage round-trip lives in the callers.
pub fn measure(alts: &mut Vec<(String, f64)>) -> Option<String> {
    if alts.is_empty() { return None; }
    let mut bi = 0;
    for (i, (_, w)) in alts.iter().enumerate() { if *w > alts[bi].1 { bi = i; } }
    let chosen = alts[bi].0.clone();
    for (i, (_, w)) in alts.iter_mut().enumerate() {
        if i == bi { *w *= ZENO_BOOST; } else { *w *= SUPER_DECAY; }
    }
    alts.retain(|(_, w)| *w >= PRUNE_THRESHOLD);
    Some(chosen)
}

/// The best-matching superposition for a cue: stem overlap between the cue and each entry's
/// text (>= 1 required), most overlap wins. Returns (entry text, overlap).
fn best_entry<S: QuantumSide + ?Sized>(s: &S, scope: &str, cue: &str) -> Option<(String, usize)> {
    let cue_stems = stems_s(&content(cue));
    if cue_stems.is_empty() { return None; }
    let mut best: Option<(String, usize)> = None;
    for (text, _) in s.super_all(scope) {
        let ts = stems_s(&content(&text));
        let ov = cue_stems.intersection(&ts).count();
        if ov >= 1 && best.as_ref().is_none_or(|(_, b)| ov > *b) { best = Some((text, ov)); }
    }
    best
}

/// Measure the superposition matching `cue` (if any): collapse, persist the survivors, resolve
/// to an ordinary fact when one candidate remains. Returns the measured value.
pub fn recall_super<S: QuantumBack + QuantumSide + ?Sized>(s: &S, scope: &str, cue: &str) -> Option<String> {
    measure_matching(s, scope, cue).map(|r| r.value)
}

/// The Recall-shaped measurement used by `recall_once`: the synthetic fact reads
/// "<cue text> <chosen>" so a CLI/HTTP hit renders naturally. `idx` is 0 — the hit is not an
/// episode (yet); once resolved it becomes one.
pub(crate) fn measure_matching<S: QuantumBack + QuantumSide + ?Sized>(s: &S, scope: &str, cue: &str) -> Option<Recall> {
    let (text, ov) = best_entry(s, scope, cue)?;
    let mut alts = s.super_get(scope, &text)?;
    let chosen = measure(&mut alts)?;
    if alts.len() <= 1 {
        // collapse complete: the superposition becomes a classical fact and the entry is gone
        s.super_del(scope, &text);
        s.observe(scope, &format!("{} {}", text, chosen));
    } else {
        s.super_set(scope, &text, &alts);
    }
    Some(Recall { fact: format!("{} {}", text, chosen), value: chosen, coverage: 1.0, overlap: ov, exact: 0, echo: false, idx: 0 })
}
