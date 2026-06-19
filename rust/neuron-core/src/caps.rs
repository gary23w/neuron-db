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

/// The always-local (grounded) capability names — never ceded to any host, whatever it advertises.
/// Transports advertise this as the floor of what neuron always owns. Order matches `CAPABILITIES`.
pub fn grounded_names() -> Vec<&'static str> {
    CAPABILITIES.iter().filter(|c| c.ownership == Ownership::Grounded).map(|c| c.name).collect()
}

/// The capability names neuron will cede to a host advertising `host_has` — the `Defer` set of
/// `resolve`. A grounded name never appears here even if the host claims it, so this is the exact
/// surface a transport hands off in a handshake. Order matches `CAPABILITIES`.
pub fn deferred_for(host_has: &[&str]) -> Vec<&'static str> {
    resolve(host_has).into_iter()
        .filter(|(_, d)| *d == Disposition::Defer)
        .map(|(n, _)| n)
        .collect()
}

/// What neuron does with a capability after a host advertises its own tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// neuron keeps doing it (it's grounded, or the host can't).
    Keep,
    /// the host has a better tool and it's store-free, so neuron yields it.
    Defer,
}
impl Disposition {
    pub fn tag(self) -> &'static str { match self { Disposition::Keep => "keep", Disposition::Defer => "defer" } }
}

/// Resolve neuron's effective surface against a host advertising the capabilities in `host_has`.
/// Grounded capabilities are ALWAYS kept — a host without the store would hallucinate them, so even
/// a host that *claims* to recall must not take it. A deferrable capability is yielded only if the
/// host actually advertises it, else kept (neuron still does it in-core). This is "grounded beats
/// tier" as a pure decision — the negotiation brain, with no wire and no async needed.
pub fn resolve(host_has: &[&str]) -> Vec<(&'static str, Disposition)> {
    CAPABILITIES.iter().map(|c| {
        let d = match c.ownership {
            Ownership::Grounded => Disposition::Keep,
            Ownership::Deferrable if host_has.contains(&c.name) => Disposition::Defer,
            Ownership::Deferrable => Disposition::Keep,
        };
        (c.name, d)
    }).collect()
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
    fn disp(r: &[(&'static str, Disposition)], name: &str) -> Disposition {
        r.iter().find(|(n, _)| *n == name).unwrap().1
    }
    #[test]
    fn a_lying_host_cannot_take_a_grounded_capability() {
        // the whole point: a host claiming it can recall/chain must NOT be ceded them — it has no store
        let r = resolve(&["recall", "chain", "assess", "stance", "summarize"]);
        for grounded in ["recall", "chain", "assess", "stance"] {
            assert_eq!(disp(&r, grounded), Disposition::Keep, "{grounded} must stay local even if the host claims it");
        }
        // but a deferrable the host genuinely has IS yielded
        assert_eq!(disp(&r, "summarize"), Disposition::Defer);
    }
    #[test]
    fn no_host_keeps_everything() {
        // a bare host (advertises nothing) -> neuron does it all in-core
        let r = resolve(&[]);
        assert!(r.iter().all(|(_, d)| *d == Disposition::Keep));
    }
    #[test]
    fn richer_host_yields_only_store_free_work() {
        let r = resolve(&["summarize", "embed", "normalize", "fetch", "recall"]);
        for deferred in ["summarize", "embed", "normalize", "fetch"] {
            assert_eq!(disp(&r, deferred), Disposition::Defer);
        }
        assert_eq!(disp(&r, "recall"), Disposition::Keep);   // grounded, never yielded
    }
    #[test]
    fn grounded_names_are_exactly_the_grounded_caps() {
        let g = grounded_names();
        assert!(g.iter().all(|n| !is_deferrable(n)), "grounded_names must contain no deferrable cap");
        assert_eq!(g.len(), CAPABILITIES.iter().filter(|c| c.ownership == Ownership::Grounded).count());
        assert!(g.contains(&"recall") && g.contains(&"stance") && !g.contains(&"summarize"));
    }
    #[test]
    fn deferred_for_is_the_defer_set_and_never_grounded() {
        // a bare host cedes nothing; a sampling-class host cedes only the store-free names it offered
        assert!(deferred_for(&[]).is_empty());
        let d = deferred_for(&["summarize", "embed", "recall", "chain"]);
        assert_eq!(d, vec!["summarize", "embed"]);            // grounded names dropped, source order kept
        assert!(d.iter().all(|n| is_deferrable(n)));
    }
}
