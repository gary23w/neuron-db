//! The no-cloning theorem as a read budget: `write_once` stores a fact with a number of reads
//! remaining; each quantum-aware recall that returns it spends one; the read that spends the
//! last one deletes the fact from the store (the caller still gets the value that one time —
//! the NEXT reader finds nothing). Normal facts in the same scope are untouched.
//!
//! The budget lives in the tier's side state, not on the Episode itself, so the base store's
//! format and the dump()/load() round-trip stay byte-identical when the feature is off.

use super::{QuantumBack, QuantumSide};

/// Store `text` with a read budget. Multi-sentence text is split by the store's own sentence
/// writer, and every sentence that actually landed is marked. Returns the facts written (0 when
/// the text was already stored verbatim — the budget is then just re-armed on the existing fact,
/// which is also how a budget is refreshed).
pub fn write_once<S: QuantumBack + QuantumSide + ?Sized>(s: &S, scope: &str, text: &str, max_reads: u32) -> usize {
    let reads = max_reads.max(1);
    let wrote = if s.has_fact(scope, text.trim()) { 0 } else { s.observe(scope, text) };
    // mark exactly what landed: the store writes per-sentence, so probe each sentence's presence
    let mut marked = 0;
    for sent in crate::sentences(text, 400) {
        if s.has_fact(scope, &sent) { s.noclone_set(scope, &sent, reads); marked += 1; }
    }
    if marked == 0 && s.has_fact(scope, text.trim()) { s.noclone_set(scope, text.trim(), reads); }
    wrote
}

/// How many reads remain on a marked fact (by its exact text). None = not marked (an ordinary
/// fact, or already burned).
pub fn reads_remaining<S: QuantumSide + ?Sized>(s: &S, scope: &str, fact_text: &str) -> Option<u32> {
    s.noclone_get(scope, fact_text)
}
