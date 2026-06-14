//! Comprehensive unit tests for the NeuronDB (sqlite) tier.
//! Covers observe/observe_many, recall/recall_many/get, turn, forget, stats, neurons,
//! durability across reopen, LRU cache eviction + reload, max_facts capacity eviction,
//! scope isolation, concurrency/thread-safety, and edge cases (unicode, special chars,
//! empty input, the `?`-drop and short-value limitations).
#![cfg(feature = "sqlite")]
use neuron_core::db::NeuronDB;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir()
        .join(format!("ndb_cx_{}_{}.db", std::process::id(), n))
        .to_string_lossy()
        .into_owned()
}

// ---------- observe / recall / get basics ----------

#[test]
fn value_isolation_picks_value_near_cue() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "the deploy region is us-west-2");
    assert_eq!(db.get("u", "what is the deploy region?").as_deref(), Some("us-west-2"));
    db.observe("u", "the gate code is 4471");
    assert_eq!(db.get("u", "what is the gate code?").as_deref(), Some("4471"));
}

#[test]
fn observe_returns_fact_count() {
    let db = NeuronDB::open(&tmp(), 500);
    assert_eq!(db.observe("u", "my name is Sam"), 1);
    // three sentences -> three facts
    assert_eq!(db.observe("v", "my name is Sam. my plan is pro. my city is Halifax"), 3);
    assert_eq!(db.stats("v").facts, 3);
}

#[test]
fn recall_abstains_when_nothing_matches() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "my plan is pro");
    assert_eq!(db.get("u", "what is the temperature outside?"), None);
    // recall on a never-written scope is also None
    assert!(db.recall("ghost", "what is anything?").is_none());
}

#[test]
fn get_matches_recall_value() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "the api endpoint is https://api.example.com");
    let r = db.recall("u", "what is the api endpoint?").unwrap();
    assert_eq!(db.get("u", "what is the api endpoint?"), Some(r.value));
}

// ---------- numeric recall ----------

#[test]
fn numeric_how_many_isolates_the_number() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "the team has 12 engineers");
    assert_eq!(db.get("u", "how many engineers are there?").as_deref(), Some("12"));
}

#[test]
fn short_numeric_value_is_retrievable() {
    // digits bypass the 3-char minimum-length floor
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "the gate code is 17");
    assert_eq!(db.get("u", "what is the gate code?").as_deref(), Some("17"));
}

// ---------- KNOWN LIMITATIONS (asserted so regressions/changes are visible) ----------

#[test]
fn limitation_observe_drops_text_with_question_mark() {
    // observe() rejects any text containing '?' (so questions aren't stored as facts).
    // Side effect: a URL/value with a query string never lands.
    let db = NeuronDB::open(&tmp(), 500);
    assert_eq!(db.observe("u", "the tracking url is site.com?utm_source=x"), 0);
    assert_eq!(db.get("u", "what is the tracking url?"), None);
}

#[test]
fn limitation_two_char_alpha_value_not_retrievable() {
    // values < 3 chars that aren't all digits are dropped as candidates
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "my state code is CA");
    assert_ne!(db.get("u", "what is my state code?").as_deref(), Some("CA"));
}

#[test]
fn limitation_single_content_word_fact_dropped() {
    // a fact with only one content word and a stopword value is not stored at all
    let db = NeuronDB::open(&tmp(), 500);
    assert_eq!(db.observe("u", "my country is US"), 0); // "us" is a stopword, "country" lone content word
}

// ---------- morphological fallback (fixes the lexical owner/owned/owns gap) ----------

#[test]
fn morphological_fallback_bridges_owner_owned_owns() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "project Aurora is owned by Marisol");
    // queries phrased with different morphology all recall Marisol via the root fallback
    assert_eq!(db.get("u", "who is the owner of Aurora?").as_deref(), Some("Marisol"));
    assert_eq!(db.get("u", "who owns Aurora?").as_deref(), Some("Marisol"));
}

#[test]
fn fallback_does_not_fabricate_on_true_miss() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "my plan is pro");
    assert_eq!(db.get("u", "what is the weather in Tokyo?"), None);
}

// ---------- recall_chain: server-side multi-hop ----------

#[test]
fn recall_chain_three_hops() {
    let db = NeuronDB::open(&tmp(), 5000);
    db.observe_many("o", &[
        "project Aurora owner is Marisol".into(),
        "Marisol manager is Dana".into(),
        "Dana timezone is WET".into(),
    ]);
    let (val, trail) = db.recall_chain("o", "Aurora", &["owner".into(), "manager".into(), "timezone".into()]);
    assert_eq!(val.as_deref(), Some("WET"));
    assert_eq!(trail, vec!["Aurora", "Marisol", "Dana", "WET"]);
}

#[test]
fn recall_chain_reports_break() {
    let db = NeuronDB::open(&tmp(), 5000);
    db.observe("o", "project Aurora owner is Marisol");
    let (val, trail) = db.recall_chain("o", "Aurora", &["owner".into(), "manager".into()]);
    assert_eq!(val, None);                       // Marisol's manager was never stored
    assert_eq!(trail, vec!["Aurora", "Marisol"]); // trail shows where it stopped
}

// ---------- stemming precision ----------

#[test]
fn stemming_does_not_overmatch_planet_to_plan() {
    // "planet" must not collapse onto the stored "plan" stem (5-6 char words keep 5 chars)
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "my plan is pro");
    assert_ne!(db.get("u", "what is my favorite planet?").as_deref(), Some("pro"));
    // the real cue still works
    assert_eq!(db.get("u", "what plan am i on?").as_deref(), Some("pro"));
}

// ---------- empty / whitespace / too-short ----------

#[test]
fn empty_and_short_inputs_store_nothing() {
    let db = NeuronDB::open(&tmp(), 500);
    assert_eq!(db.observe("u", ""), 0);
    assert_eq!(db.observe("u", "   \t  "), 0);
    assert_eq!(db.observe("u", "hi"), 0); // < 3 words, no digit
    assert_eq!(db.stats("u").facts, 0);
}

// ---------- unicode / special chars / long values ----------

#[test]
fn unicode_cjk_value_roundtrips() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "my city is 東京");
    assert_eq!(db.get("u", "what is my city?").as_deref(), Some("東京"));
}

#[test]
fn curly_apostrophe_does_not_panic() {
    // smart-quote possessive (\u{2019}s) must not panic the stemmer
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "Sarah\u{2019}s favorite drink is espresso");
    assert_eq!(db.get("u", "what is the favorite drink?").as_deref(), Some("espresso"));
}

#[test]
fn emoji_and_cjk_input_do_not_panic() {
    // arbitrary unicode must never crash observe/recall (DoS safety)
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "my reaction is 🎉🎉🎉 pure joy");
    db.observe("u", "the slogan is 世界 こんにちは 안녕하세요");
    let _ = db.get("u", "what is my reaction?");
    let _ = db.get("u", "what is the slogan?");
}

#[test]
fn accented_value_roundtrips() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "the destination is Kraków");
    assert_eq!(db.get("u", "what is the destination?").as_deref(), Some("Kraków"));
}

#[test]
fn hyphenated_alphanumeric_value() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "my license key is a1b2-c3d4-e5f6");
    assert_eq!(db.get("u", "what is my license key?").as_deref(), Some("a1b2-c3d4-e5f6"));
}

#[test]
fn long_value_roundtrips() {
    let db = NeuronDB::open(&tmp(), 500);
    let long: String = std::iter::repeat('x').take(300).collect();
    db.observe("u", &format!("the token is {}", long));
    assert_eq!(db.get("u", "what is the token?").as_deref(), Some(long.as_str()));
}

// ---------- observe_many (batch) ----------

#[test]
fn observe_many_equals_individual_observes() {
    let a = NeuronDB::open(&tmp(), 10_000);
    let b = NeuronDB::open(&tmp(), 10_000);
    let facts: Vec<String> = (0..200).map(|i| format!("the metric{} reading is v{}", i, i)).collect();
    let na = a.observe_many("s", &facts);
    let mut nb = 0;
    for f in &facts { nb += b.observe("s", f); }
    assert_eq!(na, nb);
    assert_eq!(a.stats("s").facts, b.stats("s").facts);
    assert_eq!(a.get("s", "what is the metric137 reading?").as_deref(), Some("v137"));
    assert_eq!(b.get("s", "what is the metric137 reading?").as_deref(), Some("v137"));
}

#[test]
fn observe_many_empty_slice_is_noop() {
    let db = NeuronDB::open(&tmp(), 500);
    assert_eq!(db.observe_many("s", &[]), 0);
    assert_eq!(db.stats("s").facts, 0);
}

// ---------- recall_many (top-k) ----------

#[test]
fn recall_many_returns_multiple_and_respects_k() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "my plan is pro");
    db.observe("u", "my city is Halifax");
    db.observe("u", "my editor is neovim");
    let hits = db.recall_many("u", "my plan and city and editor", 3);
    assert!(hits.len() >= 2, "top-k should surface several facts, got {}", hits.len());
    let capped = db.recall_many("u", "my plan and city and editor", 1);
    assert!(capped.len() <= 1);
}

// ---------- forget ----------

#[test]
fn forget_by_substring_is_case_insensitive() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "the Gate code is 4471");
    db.observe("u", "the vault code is 9920");
    let (forgot, remaining) = db.forget("u", Some("GATE"));
    assert_eq!(forgot, 1);
    assert_eq!(remaining, 1);
    assert_eq!(db.get("u", "what is the vault code?").as_deref(), Some("9920"));
}

#[test]
fn forget_all_clears_scope() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "fact one is alpha");
    db.observe("u", "fact two is bravo");
    let (forgot, remaining) = db.forget("u", None);
    assert_eq!(forgot, 2);
    assert_eq!(remaining, 0);
    assert_eq!(db.stats("u").facts, 0);
}

#[test]
fn forget_nonexistent_substring_removes_nothing() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "the gate code is 4471");
    let (forgot, remaining) = db.forget("u", Some("zzzzz"));
    assert_eq!(forgot, 0);
    assert_eq!(remaining, 1);
}

// ---------- stats / neurons ----------

#[test]
fn stats_reports_capacity_and_turns() {
    let db = NeuronDB::open(&tmp(), 42);
    db.observe("u", "my plan is pro");
    let s = db.stats("u");
    assert_eq!(s.facts, 1);
    assert_eq!(s.max_facts, 42);
    assert_eq!(s.turns, 0);
    db.turn("u", "my city is Halifax");
    assert_eq!(db.stats("u").turns, 1);
}

#[test]
fn neurons_lists_every_written_scope() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("a", "alpha fact is one");
    db.observe("b", "bravo fact is two");
    db.observe("c", "charlie fact is three");
    let mut ids = db.neurons();
    ids.sort();
    assert_eq!(ids, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
}

// ---------- durability across reopen (WAL) ----------

#[test]
fn durable_across_reopen() {
    let path = tmp();
    {
        let db = NeuronDB::open(&path, 500);
        db.observe("a", "my color is teal");
        db.observe("b", "my color is amber");
        db.turn("a", "my plan is pro");
    }
    let db = NeuronDB::open(&path, 500); // reopen
    assert_eq!(db.get("a", "what is my color?").as_deref(), Some("teal"));
    assert_eq!(db.get("b", "what is my color?").as_deref(), Some("amber"));
    assert_eq!(db.get("a", "what plan am i on?").as_deref(), Some("pro"));
    assert_eq!(db.stats("a").turns, 1); // turn counter persisted
}

// ---------- LRU cache eviction + reload (cap is 256 scopes) ----------

#[test]
fn data_survives_cache_eviction_and_reload() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("keep", "the deploy region is us-west-2");
    // touch far more than the 256-scope cache cap to force "keep" out of cache
    for i in 0..400 {
        db.observe(&format!("filler{}", i), "the filler value is noise");
    }
    // recall must reload "keep" from sqlite and still be correct
    assert_eq!(db.get("keep", "what is the deploy region?").as_deref(), Some("us-west-2"));
}

#[test]
fn writes_after_eviction_append_not_replace() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("keep", "the alpha value is 111");
    for i in 0..400 {
        db.observe(&format!("filler{}", i), "the filler value is noise");
    }
    // re-touch keep (reloads), then append a second fact
    db.observe("keep", "the bravo value is 222");
    assert_eq!(db.stats("keep").facts, 2);
    assert_eq!(db.get("keep", "what is the alpha value?").as_deref(), Some("111"));
    assert_eq!(db.get("keep", "what is the bravo value?").as_deref(), Some("222"));
}

// ---------- max_facts capacity eviction (oldest dropped, newest kept) ----------

#[test]
fn capacity_evicts_oldest_keeps_newest() {
    let db = NeuronDB::open(&tmp(), 3);
    db.observe("u", "the alpha reading is 111");
    db.observe("u", "the bravo signal is 222");
    db.observe("u", "the charlie level is 333");
    db.observe("u", "the delta count is 444");
    db.observe("u", "the echo mark is 555");
    assert_eq!(db.stats("u").facts, 3, "capacity should cap at max_facts");
    // newest survive
    assert_eq!(db.get("u", "what is the echo mark?").as_deref(), Some("555"));
    assert_eq!(db.get("u", "what is the delta count?").as_deref(), Some("444"));
    // oldest (alpha) fully evicted -> no overlap -> abstain
    assert_eq!(db.get("u", "what is the alpha reading?"), None);
}

// ---------- turn brain over the db ----------

#[test]
fn turn_statement_then_question() {
    let db = NeuronDB::open(&tmp(), 500);
    let t = db.turn("u", "my plan is pro");
    assert_eq!(t.kind, "ack");
    assert!(t.wrote >= 1);
    let q = db.turn("u", "what plan am i on?");
    assert_eq!(q.kind, "recall");
    assert!(q.reply.contains("pro"));
}

#[test]
fn turn_arithmetic() {
    let db = NeuronDB::open(&tmp(), 500);
    let t = db.turn("u", "what is 12 * 11?");
    assert_eq!(t.kind, "math");
    assert!(t.reply.contains("132"), "got: {}", t.reply);
    assert_eq!(t.wrote, 0);
}

#[test]
fn turn_yes_no_question() {
    let db = NeuronDB::open(&tmp(), 500);
    db.turn("u", "my plan is pro");
    let y = db.turn("u", "is my plan pro?");
    assert_eq!(y.kind, "recall");
    assert!(y.reply.to_lowercase().contains("yes"), "got: {}", y.reply);
}

#[test]
fn turn_idk_when_unknown() {
    let db = NeuronDB::open(&tmp(), 500);
    let t = db.turn("u", "what is the stock price?");
    assert_eq!(t.kind, "idk");
}

#[test]
fn turn_smalltalk_greeting_writes_nothing() {
    let db = NeuronDB::open(&tmp(), 500);
    let t = db.turn("u", "hello");
    assert_eq!(t.kind, "smalltalk");
    assert_eq!(t.wrote, 0);
    assert_eq!(db.stats("u").facts, 0);
}

// ---------- scope isolation ----------

#[test]
fn scopes_do_not_leak() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("alice", "my secret is alpha123");
    db.observe("bob", "my secret is bravo456");
    assert_eq!(db.get("alice", "what is my secret?").as_deref(), Some("alpha123"));
    assert_eq!(db.get("bob", "what is my secret?").as_deref(), Some("bravo456"));
}

// ---------- concurrency / thread-safety ----------

#[test]
fn concurrent_distinct_scopes() {
    let db = Arc::new(NeuronDB::open(&tmp(), 10_000));
    let mut handles = Vec::new();
    for t in 0..16 {
        let db = db.clone();
        handles.push(std::thread::spawn(move || {
            db.observe(&format!("user{}", t), &format!("the secret token is tok{}", t));
        }));
    }
    for h in handles { h.join().unwrap(); }
    for t in 0..16 {
        assert_eq!(
            db.get(&format!("user{}", t), "what is the secret token?").as_deref(),
            Some(format!("tok{}", t).as_str())
        );
    }
}

#[test]
fn concurrent_same_scope_no_lost_writes() {
    let db = Arc::new(NeuronDB::open(&tmp(), 100_000));
    let threads = 8;
    let per = 50;
    let mut handles = Vec::new();
    for t in 0..threads {
        let db = db.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..per {
                db.observe("hot", &format!("the item {}-{} value is v{}-{}", t, i, t, i));
            }
        }));
    }
    for h in handles { h.join().unwrap(); }
    assert_eq!(db.stats("hot").facts, threads * per, "no writes may be lost under contention");
}

#[test]
fn concurrent_readers_and_writer_no_panic() {
    let db = Arc::new(NeuronDB::open(&tmp(), 10_000));
    db.observe("s", "the deploy region is us-west-2");
    let mut handles = Vec::new();
    let writer = {
        let db = db.clone();
        std::thread::spawn(move || {
            for i in 0..200 { db.observe("s", &format!("the metric{} value is v{}", i, i)); }
        })
    };
    for _ in 0..8 {
        let db = db.clone();
        handles.push(std::thread::spawn(move || {
            let mut ok = 0;
            for _ in 0..200 {
                if db.get("s", "what is the deploy region?").as_deref() == Some("us-west-2") { ok += 1; }
            }
            ok
        }));
    }
    writer.join().unwrap();
    for h in handles { let ok: i32 = h.join().unwrap(); assert!(ok > 0, "readers should keep seeing a stable value"); }
}

// ---------- tricky-payload durability (dump/load uses \t and \n delimiters) ----------

#[test]
fn tab_in_text_survives_reopen() {
    // a literal tab in a fact collides with dump's "<flag>\t<text>" delimiter; reload
    // must not corrupt the record. (cue "region" is distinct from the value to avoid
    // the stem-collision quirk where a value sharing the cue's stem is dropped.)
    let path = tmp();
    let q = "what is the deploy region?";
    let before;
    {
        let db = NeuronDB::open(&path, 500);
        db.observe("u", "the deploy region is us-west-2\tinternal note");
        before = db.get("u", q);
    }
    let db = NeuronDB::open(&path, 500);
    assert_eq!(before.as_deref(), Some("us-west-2"));
    assert_eq!(db.get("u", q), before, "reopen must not change the answer");
    assert_eq!(db.stats("u").facts, 1);
}

#[test]
fn many_facts_persist_across_reopen() {
    let path = tmp();
    { let db = NeuronDB::open(&path, 10_000);
      let facts: Vec<String> = (0..1000).map(|i| format!("the metric{} reading is v{}", i, i)).collect();
      db.observe_many("s", &facts); }
    let db = NeuronDB::open(&path, 10_000);
    assert_eq!(db.stats("s").facts, 1000);
    assert_eq!(db.get("s", "what is the metric777 reading?").as_deref(), Some("v777"));
}

#[test]
fn turn_reports_capacity_reached() {
    let db = NeuronDB::open(&tmp(), 2);
    db.turn("u", "the alpha value is 111");
    db.turn("u", "the bravo value is 222");
    let t = db.turn("u", "the charlie value is 333");
    assert!(t.capacity_reached, "third write past max_facts=2 should flag capacity");
    assert_eq!(db.stats("u").facts, 2);
}

// ---------- concurrency stress with cache eviction ----------

#[test]
fn concurrent_writes_force_eviction_no_corruption() {
    // many threads writing to far more than the 256 cache slots, then verify each persisted
    let db = Arc::new(NeuronDB::open(&tmp(), 10_000));
    let mut handles = Vec::new();
    for t in 0..8 {
        let db = db.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..100 {
                let scope = format!("u{}_{}", t, i);
                db.observe(&scope, &format!("the deploy region is region-{}-{}", t, i));
            }
        }));
    }
    for h in handles { h.join().unwrap(); }
    // spot-check 40 scopes across the range (most evicted from cache, reloaded from sqlite)
    for t in 0..8 { for i in (0..100).step_by(20) {
        let scope = format!("u{}_{}", t, i);
        assert_eq!(db.get(&scope, "what is the deploy region?").as_deref(),
                   Some(format!("region-{}-{}", t, i).as_str()), "scope {} corrupted", scope);
    }}
    assert_eq!(db.neurons().len(), 800);
}

// ---------- duplicate facts ----------

#[test]
fn duplicate_facts_both_stored() {
    let db = NeuronDB::open(&tmp(), 500);
    db.observe("u", "the gate code is 4471");
    db.observe("u", "the gate code is 4471");
    assert_eq!(db.stats("u").facts, 2);
    assert_eq!(db.get("u", "what is the gate code?").as_deref(), Some("4471"));
}
