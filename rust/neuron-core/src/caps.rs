//! The capability manifest — the spine of the polymorphism layer (CLI_ROADMAP §7). It declares what
//! neuron-db can do and, for each capability, whether neuron OWNS it (**grounded** — it reads or
//! writes the store, so a host without the store would hallucinate the answer) or would YIELD it to
//! a richer host (**deferrable** — store-free work a real model or tool does better).
//!
//! The whole point is the inverse-guard, **"grounded beats tier":** when a host advertises a better
//! tool, neuron cedes only its *deferrable* capabilities; the grounded ones (`recall`/`chain`/
//! `assess`/`var`/`stance`/…) always stay local, so mounting into a smarter host never demotes
//! neuron to a dumb cache. std-only and not feature-gated, so every transport can advertise it.

/// Who should own a capability when neuron is mounted alongside a host that has its own tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ownership {
    /// Always neuron's — it touches the grounded store; a host without it can't do it correctly.
    Grounded,
    /// neuron can do it in-core, but a host with a real model/tool does it better; cede if offered.
    Deferrable,
}
impl Ownership {
    pub fn tag(self) -> &'static str {
        match self { Ownership::Grounded => "grounded", Ownership::Deferrable => "deferrable" }
    }
}

/// One advertised capability: a stable name, who owns it, and a one-line description.
#[derive(Debug, Clone, Copy)]
pub struct Capability {
    pub name: &'static str,
    pub ownership: Ownership,
    pub about: &'static str,
}

use Ownership::{Deferrable, Grounded};
/// neuron-db's capability surface. Grounded entries touch the store and are never ceded; deferrable
/// entries are store-free, so a richer host may own them (and neuron composes with it — see §7's
/// `recall_then_summarize`).
pub const CAPABILITIES: &[Capability] = &[
    Capability { name: "recall",    ownership: Grounded,   about: "associative recall over a scope" },
    Capability { name: "chain",     ownership: Grounded,   about: "multi-hop relation walk, server-side" },
    Capability { name: "assoc",     ownership: Grounded,   about: "spreading-activation recall" },
    Capability { name: "assess",    ownership: Grounded,   about: "knowledge-gap (coverage) signal for a query" },
    Capability { name: "store",     ownership: Grounded,   about: "remember a fact" },
    Capability { name: "var",       ownership: Grounded,   about: "exact named values" },
    Capability { name: "stance",    ownership: Grounded,   about: "accumulating affective disposition" },
    Capability { name: "summarize", ownership: Deferrable, about: "condense a recalled block to prose" },
    Capability { name: "embed",     ownership: Deferrable, about: "dense vector for semantic ranking" },
    Capability { name: "normalize", ownership: Deferrable, about: "clean / canonicalize text before storing" },
    Capability { name: "fetch",     ownership: Deferrable, about: "retrieve from the web" },
];

/// Whether `name` is a capability neuron will cede to a host that advertises a better tool for it.
/// An unknown name is treated as grounded (never ceded) — the safe default.
pub fn is_deferrable(name: &str) -> bool {
    CAPABILITIES.iter().any(|c| c.name == name && c.ownership == Ownership::Deferrable)
}

/// The manifest as tab-delimited `name\towner\tabout` lines — the wire form a host reads to learn
/// what neuron can do and which capabilities it may own.
pub fn manifest() -> String {
    CAPABILITIES.iter()
        .map(|c| format!("{}\t{}\t{}", c.name, c.ownership.tag(), c.about))
        .collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grounded_caps_are_never_deferrable() {
        // the inverse-guard: store-touching capabilities must never be ceded to a host
        for name in ["recall", "chain", "assoc", "assess", "store", "var", "stance"] {
            assert!(!is_deferrable(name), "{name} is grounded and must never be deferrable");
        }
    }
    #[test]
    fn store_free_caps_are_deferrable() {
        for name in ["summarize", "embed", "normalize", "fetch"] {
            assert!(is_deferrable(name), "{name} is store-free and should be deferrable");
        }
    }
    #[test]
    fn unknown_is_grounded_by_default() { assert!(!is_deferrable("definitely-not-a-cap")); }
    #[test]
    fn manifest_lists_every_capability() {
        let m = manifest();
        assert!(m.contains("recall\tgrounded") && m.contains("summarize\tdeferrable"), "{m}");
        assert_eq!(m.lines().count(), CAPABILITIES.len());
    }
}
