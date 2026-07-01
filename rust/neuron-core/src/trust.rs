//! trust.rs — a learned, outcome-reinforced trust signal over fact tag-classes: the "floor".
//!
//! Recall is weighted by relevance x learned-trust. Trust is earned ONLY from externally-grounded
//! outcomes (a score delta supplied by the engine's acceptance oracle) — never from restatement or
//! mere recall. No tag-class is privileged a priori: every class — including one a mind mints at
//! runtime — starts NEUTRAL and earns or loses the floor by whether recalling it precedes real gain.
//!
//! What is fixed here is the LEARNING RULE (a bounded update + relax-to-neutral), which is the
//! safety-floor mechanism — analogous to a learning rate. What is NOT fixed, and is never coded, is
//! WHICH classes are trusted or what the taxonomy is. Those are discovered. (rsi-control-boundary)

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

// --- the fixed learning rule's hyperparameters (mechanism, not content) ---
const NEUTRAL: f32 = 1.0; // every class — known or newly minted — starts here
const W_MIN: f32 = 0.05; // a class can fall this far but never to zero, so it stays re-learnable
const W_MAX: f32 = 8.0; // and rise only this far, so one lucky round can't make a class dominate
const LR: f32 = 0.25; // learning rate: fraction of the (clamped) reward applied per round
const STALL_DECAY: f32 = 0.97; // recalled but produced no gain -> gentle erosion (self-noise fades)
const RELAX: f32 = 0.01; // each update nudges a hair toward NEUTRAL, so trust is never locked in

#[derive(Clone, Debug)]
pub struct ClassTrust {
    pub weight: f32,
    pub rewards: u32,
    pub penalties: u32,
}
impl Default for ClassTrust {
    fn default() -> Self { ClassTrust { weight: NEUTRAL, rewards: 0, penalties: 0 } }
}

/// A learned trust weight per tag-class. Persisted alongside the store; it IS the accumulated,
/// externally-grounded experience of which provenance/scope classes are worth standing on.
#[derive(Clone, Debug, Default)]
pub struct TrustLedger {
    classes: HashMap<String, ClassTrust>,
}

impl TrustLedger {
    pub fn new() -> Self { TrustLedger { classes: HashMap::new() } }

    /// Trust weight for a class. Unknown / newly-minted classes are NEUTRAL — the taxonomy is open,
    /// nothing is special-cased, and a fresh tag is neither trusted nor distrusted until it earns it.
    pub fn weight(&self, class: &str) -> f32 {
        self.classes.get(class).map(|c| c.weight).unwrap_or(NEUTRAL)
    }

    /// relevance x learned-trust — the score a candidate of `class` would carry.
    pub fn score(&self, relevance: f32, class: &str) -> f32 { relevance * self.weight(class) }

    fn entry(&mut self, class: &str) -> &mut ClassTrust {
        self.classes.entry(class.to_string()).or_default()
    }

    /// Credit / penalize the classes that were RECALLED in a round, by that round's grounded
    /// outcome (`delta_score`, the change in the oracle's fitness — expected roughly in [-1, 1]):
    ///   > 0  reward, proportional to the gain
    ///   < 0  penalize, proportional to the loss
    ///   == 0 used but moved nothing -> gentle stall-decay
    /// Each class is credited once per round regardless of how many times it was recalled, so
    /// volume cannot inflate trust. There is deliberately NO path that raises trust from restatement
    /// or repetition — only grounded gain does. This is the inversion of the old `strength` signal.
    pub fn reward(&mut self, recalled_classes: &[&str], delta_score: f32) {
        let unique: HashSet<&str> = recalled_classes.iter().copied().collect();
        for c in unique {
            let e = self.entry(c);
            if delta_score > 0.0 {
                e.weight *= 1.0 + LR * delta_score.clamp(0.0, 1.0);
                e.rewards += 1;
            } else if delta_score < 0.0 {
                e.weight *= 1.0 - LR * (-delta_score).clamp(0.0, 1.0);
                e.penalties += 1;
            } else {
                e.weight *= STALL_DECAY;
            }
            // relax toward neutral a hair so trust relaxes if the world changes (no permanent lock-in)
            e.weight += (NEUTRAL - e.weight) * RELAX;
            e.weight = e.weight.clamp(W_MIN, W_MAX);
        }
    }

    /// Rank recall candidates `(relevance, class)` by relevance x learned-trust, best first.
    /// Returns indices into `candidates`. Raw density no longer wins alone: a sparse-but-trusted
    /// class can outrank a dense-but-untrusted one. Stable for equal scores (preserves input order).
    pub fn rank<S: AsRef<str>>(&self, candidates: &[(f32, S)]) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..candidates.len()).collect();
        idx.sort_by(|&a, &b| {
            let sa = candidates[a].0 * self.weight(candidates[a].1.as_ref());
            let sb = candidates[b].0 * self.weight(candidates[b].1.as_ref());
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx
    }

    pub fn len(&self) -> usize { self.classes.len() }
    pub fn is_empty(&self) -> bool { self.classes.is_empty() }
    pub fn stats(&self, class: &str) -> Option<&ClassTrust> { self.classes.get(class) }

    /// Persistence: "<class>\t<weight>\t<rewards>\t<penalties>" per line, matching the store's
    /// tab-line convention. f32 Display is shortest-round-trippable, so reload is exact.
    pub fn dump(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(self.classes.len() * 24);
        let mut first = true;
        for (k, v) in &self.classes {
            if !first { out.push('\n'); }
            first = false;
            let key = k.replace(['\t', '\n'], " ");
            let _ = write!(out, "{}\t{}\t{}\t{}", key, v.weight, v.rewards, v.penalties);
        }
        out
    }

    pub fn load(blob: &str) -> Self {
        let mut l = TrustLedger::new();
        for line in blob.split('\n') {
            if line.is_empty() { continue; }
            let mut it = line.splitn(4, '\t');
            let class = match it.next() {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };
            let weight = it.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(NEUTRAL);
            let rewards = it.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
            let penalties = it.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
            l.classes.insert(
                class.to_string(),
                ClassTrust { weight: weight.clamp(W_MIN, W_MAX), rewards, penalties },
            );
        }
        l
    }
}

/// The tag-class of a fact, for trust lookup. Keyed on PROVENANCE (the source type), not author:
/// - `"[src:corpus] ..."` -> `"src:corpus"`, `"[verified x.py] ..."` -> `"verified"` — the first token.
/// - `"[vega r12] ..."` / `"[orion r4] ..."` -> `"self"` — a round-stamped authored observation is the
///   "self" class regardless of WHICH mind wrote it (the floor distinguishes grounded-vs-self, not minds).
/// - otherwise the scope.
/// Generic over any tag a mind mints at runtime — no fixed set of classes, and this never decides whether
/// a class is good, only what it is. NOTE: kept in sync with the engine's sampleClassesAlloc (memory.zig).
pub fn class_of(text: &str, scope: &str) -> String {
    let t = text.trim_start();
    if let Some(rest) = t.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            let inside = &rest[..end];
            let mut toks = inside.split_whitespace();
            if let Some(first) = toks.next() {
                // "[<author> r<NN>]" (round-stamped authored observation) collapses to "self".
                if let Some(last) = inside.split_whitespace().last() {
                    if is_round_stamp(last) {
                        return "self".to_string();
                    }
                }
                let tok = first.trim_matches(|c: char| c == '"' || c == '\'');
                if !tok.is_empty() {
                    return tok.to_lowercase();
                }
            }
        }
    }
    scope.to_lowercase()
}

/// True for a round stamp like `r12` / `r1622` — the letter 'r' followed by ≥1 digits.
fn is_round_stamp(s: &str) -> bool {
    s.len() >= 2 && s.starts_with('r') && s[1..].bytes().all(|b| b.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_extraction_is_generic() {
        assert_eq!(class_of("[src:corpus] import graph is a DAG", "knowledge"), "src:corpus");
        assert_eq!(class_of("[verified handlers.py] def f()", "verified"), "verified");
        assert_eq!(class_of("plain untagged fact", "knowledge"), "knowledge");
        assert_eq!(class_of("  [SRC:Corpus] x", "k"), "src:corpus"); // trimmed + lowercased
        assert_eq!(class_of("[a_minted_tag] y", "k"), "a_minted_tag"); // a class invented at runtime
    }

    #[test]
    fn authored_observations_collapse_to_self() {
        // round-stamped mind observations share the "self" class regardless of author,
        // so the floor learns grounded-vs-self, not which mind is "better".
        assert_eq!(class_of("[vega r1622] i made a file", "knowledge"), "self");
        assert_eq!(class_of("[orion r4] note", "knowledge"), "self");
        assert_eq!(class_of("[lyra r12] x", "knowledge"), "self");
        // but a non-round-stamped grounded tag stays its own class
        assert_ne!(class_of("[src:corpus] x", "knowledge"), "self");
        assert_ne!(class_of("[verified a.py] x", "knowledge"), "self");
    }

    // An unseen / newly-minted class is NEUTRAL — nothing is privileged a priori.
    #[test]
    fn unseen_classes_start_neutral() {
        let t = TrustLedger::new();
        assert!(t.is_empty());
        assert_eq!(t.weight("verified"), NEUTRAL);
        assert_eq!(t.weight("some_tag_a_mind_just_invented"), NEUTRAL);
    }

    // A grounded gain raises trust; a grounded loss lowers it.
    #[test]
    fn reward_raises_penalty_lowers() {
        let mut t = TrustLedger::new();
        t.reward(&["a"], 0.5);
        t.reward(&["b"], -0.5);
        assert!(t.weight("a") > NEUTRAL, "gain should raise trust: {}", t.weight("a"));
        assert!(t.weight("b") < NEUTRAL, "loss should lower trust: {}", t.weight("b"));
    }

    // stats() (behind the `trust_get` CLI verb) exposes weight AND the grounded reward/penalty counts, so a
    // caller can UCB-rank sources by earned trust and still explore the never-tried.
    #[test]
    fn stats_expose_weight_and_grounded_counts() {
        let mut t = TrustLedger::new();
        assert!(t.stats("source:mdn").is_none()); // unseen -> no entry (wrapper reports NEUTRAL default)
        t.reward(&["source:mdn"], 0.8); // one grounded win
        t.reward(&["source:mdn"], -0.5); // one grounded loss
        let s = t.stats("source:mdn").expect("class now seen");
        assert_eq!(s.rewards, 1);
        assert_eq!(s.penalties, 1);
        assert!(s.weight > W_MIN && s.weight < W_MAX);
    }

    // THE anti-amplifier property: recall without grounded gain can only ERODE trust. There is no
    // path by which restating / re-recalling a fact raises it. (Old `strength` did the opposite.)
    #[test]
    fn restatement_without_gain_cannot_raise_trust() {
        let mut t = TrustLedger::new();
        // "restate / re-recall the same self-note 50 rounds" == recalled, zero grounded gain each time
        for _ in 0..50 {
            t.reward(&["selfnote"], 0.0);
        }
        assert!(
            t.weight("selfnote") < NEUTRAL,
            "repetition without grounded gain must erode, got {}",
            t.weight("selfnote")
        );
    }

    // Volume within a round cannot inflate trust: a class recalled many times in one round is
    // credited once for that round.
    #[test]
    fn intra_round_volume_is_deduped() {
        let mut a = TrustLedger::new();
        let mut b = TrustLedger::new();
        a.reward(&["x"], 0.4);
        b.reward(&["x", "x", "x", "x", "x"], 0.4);
        assert_eq!(a.weight("x"), b.weight("x"));
    }

    // THE floor emerges: with everything seeded neutral, a class that keeps preceding grounded gain
    // outranks a much denser class that only ever precedes stalls — trust overrides raw density, and
    // nothing in the code ever named which class is "good".
    #[test]
    fn trust_overrides_density_emergently() {
        let mut t = TrustLedger::new();
        for _ in 0..40 {
            t.reward(&["grounded"], 0.5); // recalling this kept producing real gains
            t.reward(&["selfnote"], 0.0); // this got recalled but never moved the needle
        }
        assert!(t.weight("grounded") > t.weight("selfnote"));

        // a SPARSE grounded candidate (low relevance) beats a DENSE self-note (high relevance)
        let candidates = [(0.20f32, "grounded"), (0.95f32, "selfnote")];
        let order = t.rank(&candidates);
        assert_eq!(order[0], 0, "grounded must rank first despite lower relevance");

        // and below-floor noise is downweighted hard
        assert!(t.score(0.95, "selfnote") < t.score(0.20, "grounded"));
    }

    // No permanent lock-in: a once-trusted class that starts preceding regressions decays back down.
    // This is what makes the floor recursively self-correcting rather than a one-way ratchet.
    #[test]
    fn trust_is_relearnable_not_locked() {
        let mut t = TrustLedger::new();
        for _ in 0..40 {
            t.reward(&["grounded"], 0.5);
        }
        let peak = t.weight("grounded");
        assert!(peak > 2.0, "should have climbed, got {peak}");
        // the world changes: now recalling it precedes regressions
        for _ in 0..60 {
            t.reward(&["grounded"], -0.5);
        }
        assert!(
            t.weight("grounded") < NEUTRAL,
            "trust must decay when it stops paying off, got {}",
            t.weight("grounded")
        );
    }

    // Weights stay bounded and finite under sustained extreme reward/penalty.
    #[test]
    fn weights_stay_bounded() {
        let mut t = TrustLedger::new();
        for _ in 0..500 {
            t.reward(&["hi"], 1.0);
        }
        for _ in 0..500 {
            t.reward(&["lo"], -1.0);
        }
        assert!(t.weight("hi").is_finite() && t.weight("hi") <= W_MAX);
        assert!(t.weight("lo") >= W_MIN && t.weight("lo") > 0.0);
    }

    // The learned floor survives a restart.
    #[test]
    fn dump_load_round_trips() {
        let mut t = TrustLedger::new();
        for _ in 0..10 {
            t.reward(&["corpus"], 0.6);
        }
        t.reward(&["junk"], -0.3);
        let blob = t.dump();
        let r = TrustLedger::load(&blob);
        assert!((r.weight("corpus") - t.weight("corpus")).abs() < 1e-6);
        assert!((r.weight("junk") - t.weight("junk")).abs() < 1e-6);
        assert_eq!(r.stats("corpus").unwrap().rewards, 10);
        assert_eq!(r.stats("junk").unwrap().penalties, 1);
    }
}
