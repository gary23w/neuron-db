//! Keyed facts: upsert replaces by exact key, forget_key removes by exact key, and the key
//! survives dump/load. std-only — compiles under any feature set.
use neuron_core::{wasm, Neuron};

#[test]
fn upsert_replaces_only_its_own_key() {
    let mut n = Neuron::new(500);
    // the first record's text mentions "rec12": a substring forget of "rec12 " would take it out
    assert_eq!(n.upsert_keyed("customer:1", "customer Dana Reyes of Oakville note see rec12 file"), 1);
    assert_eq!(n.upsert_keyed("customer:12", "customer Omar Haddad of Milton"), 1);
    assert_eq!(n.upsert_keyed("customer:12", "customer Omar Haddad of Burlington"), 1);
    assert_eq!(n.fact_count(), 2, "an upsert on a live key replaces, never duplicates");
    let hits: Vec<String> = n.recall_many("Omar Haddad", 5).into_iter().map(|r| r.fact).collect();
    assert!(hits.iter().any(|f| f.contains("Burlington")) && !hits.iter().any(|f| f.contains("Milton")), "{hits:?}");
    assert!(n.recall_many("Dana Reyes", 5).iter().any(|r| r.fact.contains("Oakville")), "the other key is untouched");
    assert!(n.episodes.iter().all(|e| !e.t.contains("customer:")), "the key never enters the fact text");
}

#[test]
fn forget_key_is_exact() {
    let mut n = Neuron::new(500);
    n.upsert_keyed("customer:1", "customer Dana Reyes of Oakville");
    n.upsert_keyed("customer:12", "customer Omar Haddad of Milton");
    n.observe("the office is in Toronto on King Street");
    assert_eq!(n.forget_key("customer:1"), 1, "customer:1 must not match customer:12");
    assert_eq!(n.forget_key("customer:99"), 0);
    assert_eq!(n.forget_key(""), 0, "an empty key removes nothing");
    assert_eq!(n.fact_count(), 2);
    assert!(n.recall_many("Omar Haddad", 5).iter().any(|r| r.fact.contains("Milton")));
}

#[test]
fn upsert_that_stores_nothing_keeps_the_old_fact() {
    let mut n = Neuron::new(500);
    n.upsert_keyed("customer:1", "customer Dana Reyes of Oakville");
    assert_eq!(n.upsert_keyed("customer:1", "ok"), 0, "unencodable text stores nothing");
    assert_eq!(n.upsert_keyed("", "customer Nobody of Nowhere town"), 0, "an empty key stores nothing");
    assert_eq!(n.fact_count(), 1);
    assert!(n.recall_many("Dana Reyes", 5).iter().any(|r| r.fact.contains("Oakville")));
}

#[test]
fn multi_sentence_upsert_is_one_unit() {
    let mut n = Neuron::new(500);
    assert_eq!(n.upsert_keyed("job:7", "the job is a kitchen remodel. the installer is Marco Bell."), 2);
    assert_eq!(n.upsert_keyed("job:7", "the job is a bathroom remodel."), 1);
    assert_eq!(n.fact_count(), 1, "every sentence the key held is replaced");
    assert_eq!(n.upsert_keyed("job:7", "the job is a deck build. the installer is Priya Nair."), 2);
    assert_eq!(n.forget_key("job:7"), 2);
}

#[test]
fn keys_survive_dump_load_and_legacy_blobs_still_load() {
    let mut n = Neuron::new(500);
    n.upsert_keyed("a:b\tc", "customer Dana Reyes of Oakville");   // ':' and a tab inside the key
    n.observe("the office is in Toronto on King Street");
    let mut m = Neuron::load(&n.dump(), 500);
    assert_eq!(m.fact_count(), 2);
    assert_eq!(m.upsert_keyed("a:b\tc", "customer Dana Reyes of Hamilton"), 1);
    assert_eq!(m.fact_count(), 2, "the reloaded key is still the one the upsert replaces");
    assert_eq!(m.forget_key("a:b\tc"), 1);
    let mut legacy = Neuron::load("0\tthe meeting is at 5\n1\tmy name is Sam Okafor\t2", 500);
    assert_eq!(legacy.fact_count(), 2);
    assert!(legacy.episodes.iter().all(|e| e.key.is_none()));
    assert_eq!(legacy.forget_key("0"), 0);
}

fn call(req: &str) -> String {
    let b = req.as_bytes();
    let p = wasm::alloc(b.len());
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), p, b.len()); }
    let n = wasm::mem(p, b.len());
    let out = unsafe { std::slice::from_raw_parts(wasm::answer_ptr(), n) };
    String::from_utf8_lossy(out).to_string()
}
macro_rules! m { ($($a:expr),*) => { call(&[$($a),*].join("\t")) } }

// the wasm module keeps process-global state, so its surface is driven from this one test only
#[test]
fn wasm_upsert_and_forgetkey() {
    wasm::mem_reset();
    let s = "crm";
    assert!(m!("ops", "_").contains("upsert") && m!("ops", "_").contains("forgetkey"));
    assert_eq!(m!("upsert", s, "customer:1", "customer Dana Reyes of Oakville note see rec12 file"), "1");
    assert_eq!(m!("upsert", s, "customer:12", "customer Omar Haddad of Milton"), "1");
    assert_eq!(m!("upsert", s, "customer:12", "customer Omar Haddad of Burlington"), "1");
    assert_eq!(m!("stats", s), "2");
    assert!(!m!("recall", s, "Omar Haddad", "5").contains("Milton"));
    // recallkeyed: recallscored's three fields plus the key; empty key for an unkeyed fact
    let line = m!("recallkeyed", s, "Omar Haddad", "1");
    let f: Vec<&str> = line.split('\t').collect();
    assert_eq!(f.len(), 4, "{line:?}");
    assert!(f[0].contains("Burlington") && f[1].parse::<f64>().is_ok() && f[2].parse::<usize>().is_ok());
    assert_eq!(f[3], "customer:12");
    assert_eq!(m!("recallscored", s, "Omar Haddad", "1").split('\t').count(), 3, "recallscored stays three fields");
    assert_eq!(m!("observe", "plain", "the office is in Toronto on King Street"), "1");
    assert!(m!("recallkeyed", "plain", "office Toronto", "1").ends_with('\t'), "an unkeyed hit has an empty key field");
    let blob = m!("dump", s);
    assert_eq!(m!("load", "crm2", &blob), "2");
    assert_eq!(m!("forgetkey", "crm2", "customer:12"), "1");
    assert_eq!(m!("stats", "crm2"), "1");
    assert_eq!(m!("forgetkey", s, "customer:1"), "1");
    assert_eq!(m!("forgetkey", s, "customer:1"), "0");
    assert!(m!("recall", s, "Omar Haddad", "5").contains("Burlington"));
}
