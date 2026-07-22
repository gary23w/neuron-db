//! Coverage + regression tests added during the audit pass. Direct unit tests for public Neuron
//! surfaces that were previously exercised only indirectly, plus the PlasticNeuron::consolidate()
//! link-hygiene fix. std-only — compiles under any feature set.

use neuron_core::{json_escape, plastic::PlasticNeuron, rel_matches, root_token, Neuron};

#[test]
fn recall_many_returns_k_bounded_topk() {
    let mut n = Neuron::new(500);
    for f in [
        "the api key is zeta-9931",
        "the deploy region is us-west-2",
        "the db host is pgmain",
        "the cache ttl is 60 seconds",
        "the log level is debug",
    ] {
        n.observe(f);
    }
    let r = n.recall_many("what is the api key", 3);
    assert!(!r.is_empty() && r.len() <= 3, "k bounds the result: got {}", r.len());
    assert!(
        r.iter().any(|x| x.value == "zeta-9931"),
        "the api key fact should be among the top results: {:?}",
        r.iter().map(|x| &x.value).collect::<Vec<_>>()
    );
    assert!(n.recall_many("the", 2).len() <= 2, "k=2 caps the output");
}

#[test]
fn recall_spreading_base_seeds_and_bounds() {
    let mut n = Neuron::new(500);
    n.observe("my dog is called Biscuit");
    n.observe("my sister is called Dana");
    let r = n.recall_spreading("what is my dog's name?", 5, 1);
    assert!(
        r.iter().any(|s| s.fact.to_lowercase().contains("biscuit")),
        "the matching fact should seed: {:?}",
        r.iter().map(|s| &s.fact).collect::<Vec<_>>()
    );
    assert!(r.len() <= 5, "k bounds the spread");
    assert!(n.recall_spreading("", 5, 1).is_empty(), "empty query -> empty");
    assert!(n.recall_spreading("quantum chromodynamics", 5, 1).is_empty(), "no-match query -> empty");
}

#[test]
fn recall_spreading_hub_prefix_does_not_drown_rare_cue() {
    // The absorbed-document shape: EVERY fact carries the same "[label]" provenance prefix, so the
    // label's stems are hubs (df == scope size), while the facts cross-link through a mid-df shared
    // vocabulary. Pre-fix, a query naming the document seeded the whole scope uniformly off the hub
    // stems and the ranking collapsed into query-independent graph centrality — the same hits came
    // back for every query, and a rare discriminative cue word ("forged cheque") was drowned.
    let pool = [
        "harbor", "lantern", "meadow", "orchard", "chapel", "willow", "granite", "saddle", "copper",
        "marsh", "cedar", "anvil", "quarry", "heather", "brook", "gable", "thicket", "furrow",
        "hollow", "spire", "moss", "shale", "bramble", "wharf", "kiln", "beacon", "fern", "crag",
        "sluice", "tarn",
    ];
    let mut n = Neuron::new(10_000);
    for i in 0..300usize {
        let (a, b, c) = (pool[i % 30], pool[(i + 7) % 30], pool[(i + 13) % 30]);
        n.observe(&format!("[Green Overcoat] the {} beside the {} near the {} stayed quiet.", a, b, c));
    }
    n.observe("[Green Overcoat] Professor Higginson forged the cheque under duress.");
    // rare cue stems (forged/cheque, df=1) must carry the seed past the hub title stems (df=301)
    let r = n.recall_spreading("Green Overcoat who forged the cheque", 5, 1);
    assert!(
        r.first().is_some_and(|s| s.fact.contains("forged the cheque") && s.seed),
        "the rare-stem fact must rank first as a direct seed: {:?}",
        r.iter().map(|s| &s.fact).collect::<Vec<_>>()
    );
    // a genuinely broad query (every cue stem is a hub) still falls back to seeding everything
    assert!(
        !n.recall_spreading("Green Overcoat", 5, 1).is_empty(),
        "an all-hub query keeps the old seed-everything behavior"
    );
}

#[test]
fn json_escape_handles_control_and_quotes() {
    assert_eq!(json_escape(r#"a"b\c"#), r#"a\"b\\c"#);
    assert_eq!(json_escape("tab\tnewline\n"), "tab\\tnewline\\n");
    assert_eq!(json_escape("\u{0001}"), "\\u0001"); // control char -> \u00XX
    assert_eq!(json_escape("\r"), "\\r");
    assert_eq!(json_escape("plain ünïcödé"), "plain ünïcödé"); // multibyte passes through
}

#[test]
fn root_token_normalizes_morphology() {
    assert_eq!(root_token("owner"), "own");
    assert_eq!(root_token("owned"), "own");
    assert_eq!(root_token("owns"), "own");
    assert_eq!(root_token("reports"), root_token("report"));
}

#[test]
fn rel_matches_bridges_relations() {
    assert!(rel_matches("owner", "owned"));
    assert!(rel_matches("owns", "owned"));
    assert!(rel_matches("reports", "report"));
    assert!(rel_matches("reports", "manager")); // synonym bridge via canon
    assert!(!rel_matches("owner", "wifi"));
}

#[test]
fn forget_prefix_removes_only_matching() {
    let mut n = Neuron::new(500);
    n.observe("stance::rust is great");
    n.observe("stance::python is fine");
    n.observe("note::remember the milk");
    let removed = n.forget_prefix("stance::");
    assert_eq!(removed, 2);
    assert_eq!(n.fact_count(), 1);
}

#[test]
fn dropped_counts_capacity_evictions() {
    let mut n = Neuron::new(10);
    for i in 0..15 {
        n.observe(&format!("item {} maps to value{}", i, i));
    }
    assert_eq!(n.fact_count(), 10, "capacity holds at max_facts");
    assert_eq!(n.dropped, 5, "the 5 oldest were evicted and counted");
}

#[test]
fn consolidate_accounts_for_every_fact_and_leaves_no_dead_links() {
    let mut n = PlasticNeuron::new(10_000_000, Some(1e9), 3);
    n.observe("the alpha key is aaa111");
    n.observe("the beta relay forwards traffic");
    n.observe("the gamma budget was approved");
    for _ in 0..5 {
        n.recall("alpha key");
        n.recall("beta relay");
        n.recall("gamma budget");
    }
    let before = n.base.episodes.len();

    // distinct, retained facts -> consolidate is a no-op
    let (m0, p0, k0) = n.consolidate(0.0);
    assert_eq!((m0, p0, k0), (0, 0, before), "nothing to merge or prune");
    assert!(n.recall("what is the alpha key?").is_some());

    // prune-all -> exercises the prune + link-cleanup path on a linked graph; must not panic and must
    // leave a consistent graph (spreading then surfaces only alive facts: none).
    let (_m1, p1, k1) = n.consolidate(f64::MAX);
    assert_eq!(k1 + p1, before, "every fact is kept or pruned");
    assert_eq!(k1, 0, "a max threshold prunes all unpinned facts");
    assert!(
        n.recall_spreading("alpha key", 2, 5, 0.6, 6).is_empty(),
        "no facts left -> no spread, no dangling-link panic"
    );
}
