#![cfg(all(feature = "sqlite", feature = "semantic", feature = "fisher", feature = "topics"))]
//! The statistics tier end-to-end: topic-gated SCOPE-WIDE blended recall (the window-blindness
//! fix), the outcome axis forming from strengthen/forget and surviving a reopen, scope topic
//! introspection, and the fail-open guarantee (tiny scopes behave exactly as before the tier).
use neuron_core::db::NeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp(tag: &str) -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_stats_{}_{}_{}.db", tag, std::process::id(), n)).to_string_lossy().into_owned()
}

const CORPUS: &str = "
    I use wifi to get online. The wifi connects my laptop to the internet.
    Being online means connected to the internet through wifi or a router.
    The router broadcasts wifi so devices reach the web and browse the internet.
    We browse the web online using the wireless wifi network from the router.
";

// The headline: a theme buried BEYOND the 4000-fact blended window stays reachable, because the
// query folds into the theme's topic and the gate serves that topic's postings scope-wide. With
// the window alone (the pre-tier behavior) the candidate set would be 4000 cooking facts and the
// top hit would be junk.
#[test]
fn blended_recall_reaches_a_theme_beyond_the_window() {
    let db = NeuronDB::open(&tmp("gate"), 20_000);
    // theme A: 80 distinct facts (episode indices 0..80)
    for i in 0..80 { db.observe("u", &format!("aurora launch note {i}: the rocket engine telemetry looked nominal today")); }
    // filler: 4400 cooking facts push theme A far beyond the blended window
    for i in 0..4400 { db.observe("u", &format!("kitchen log {i}: the chef simmered garlic and basil in the copper pot")); }
    let hits = db.recall_blended("u", "how did the rocket telemetry look?", 3);
    assert!(!hits.is_empty(), "blended recall must return candidates");
    assert!(hits[0].fact.contains("aurora launch note"), "expected an aurora fact, got: {}", hits[0].fact);
    assert!(hits[0].idx < 80, "the hit must come from beyond the window (idx {})", hits[0].idx);
}

// Grounded outcomes arm the discriminant: strengthen feeds "+", a targeted forget feeds "−";
// below the floor the axis is inert; and the learned head + topic model survive a reopen
// through the lazily-created stats_kv side table.
#[test]
fn outcome_axis_forms_from_strengthen_and_forget_and_persists() {
    let path = tmp("axis");
    {
        let db = NeuronDB::open(&path, 5000);
        for i in 0..30 { db.observe("u", &format!("winning play {i}: the alpha strategy shipped the release cleanly")); }
        for i in 0..30 { db.observe("u", &format!("losing play {i}: the beta shortcut zzjunk broke the deploy badly")); }
        assert!(db.outcome_axis().is_none(), "axis must be inert before any outcome");
        assert!(db.strengthen("u", "alpha strategy", 0.5) >= 8, "strengthen must touch the winning plays");
        let (removed, _) = db.forget("u", Some("zzjunk"));
        assert!(removed >= 8, "forget must remove the junk plays");
        let ax = db.outcome_axis().expect("8+ outcomes per side must arm the axis");
        assert!(ax.n_pos >= 8.0 && ax.n_neg >= 8.0, "n+ {} n- {}", ax.n_pos, ax.n_neg);
        let (posw, _negw, _, _) = db.axis_words(12).expect("an armed axis is readable");
        assert!(!posw.is_empty(), "the helpful direction must have nearest words");
        db.flush_all();
    }
    let db2 = NeuronDB::open(&path, 5000);
    let ax = db2.outcome_axis().expect("the outcome axis must survive a reopen (stats_kv)");
    assert!(ax.n_pos >= 8.0);
    let (_k, docs, tokens, vocab) = db2.topics_stats();
    assert!(docs > 0 && tokens > 0 && vocab > 0, "the topic model must survive a reopen");
}

// Scope introspection: "what is this scope about" surfaces both themes with their own words.
#[test]
fn scope_topics_reports_the_scopes_themes() {
    let db = NeuronDB::open(&tmp("topics"), 20_000);
    for i in 0..60 { db.observe("u", &format!("garden log {i}: the tomato seedlings and the basil bed got water")); }
    for i in 0..60 { db.observe("u", &format!("server log {i}: the api latency and the cache ratio stayed healthy")); }
    let tops = db.scope_topics("u", 4, 6);
    assert!(!tops.is_empty(), "a 120-fact scope must report topics");
    let all_words: Vec<String> = tops.iter().flat_map(|(_, _, ws)| ws.iter().map(|(w, _)| w.clone())).collect();
    assert!(all_words.iter().any(|w| ["tomato", "basil", "seedlings"].contains(&w.as_str())), "garden theme missing: {all_words:?}");
    assert!(all_words.iter().any(|w| ["latency", "cache", "api"].contains(&w.as_str())), "server theme missing: {all_words:?}");
}

// The fail-open guarantee: on a tiny scope with an un-foldable paraphrase the gate abstains and
// everything behaves exactly as the semantic tier always has — the classic wifi paraphrase still
// resolves, the unrelated query still abstains.
#[test]
fn tiny_scope_behavior_is_unchanged() {
    let db = NeuronDB::open(&tmp("open"), 500);
    for _ in 0..30 { db.train_semantic(CORPUS); }
    db.observe("u", "the wifi password is vekam73");
    let r = db.recall("u", "what is the thing I use to get online?");
    assert!(r.is_some(), "the classic paraphrase must still resolve with the tier compiled in");
    assert_eq!(r.unwrap().value, "vekam73");
    assert!(db.get("u", "what is the boiling point of magma?").is_none(), "abstention must survive too");
}
