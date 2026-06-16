//! WASM-tier surface test. Drives the in-browser `mem()` dispatch through its real C-ABI
//! (alloc → write bytes → mem → read answer_ptr) exactly as the browser lab does, and asserts the
//! full neuron-db surface — including the new multi-hop `chain` and batch `obsmany` ops.
//!
//! One sequential test fn on purpose: the wasm module keeps process-global `static mut` state
//! (the single instance a wasm isolate has), so the surface is exercised in order, like a session.
use neuron_core::wasm;

fn call(req: &str) -> String {
    let b = req.as_bytes();
    let p = wasm::alloc(b.len());
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), p, b.len()); }
    let n = wasm::mem(p, b.len());
    let out = unsafe { std::slice::from_raw_parts(wasm::answer_ptr(), n) };
    String::from_utf8_lossy(out).to_string()
}
macro_rules! m { ($($a:expr),*) => { call(&[$($a),*].join("\t")) } }

#[test]
fn wasm_full_surface() {
    wasm::mem_reset();
    let s = "user:test";

    // ---- observe + recall + exact value ----
    assert_eq!(m!("observe", s, "the wifi password is hunter2"), "1");
    assert!(m!("recall", s, "wifi password", "5").contains("hunter2"));
    assert_eq!(m!("value", s, "what is the wifi password"), "hunter2");

    // ---- obsmany: a whole document in ONE wasm crossing ----
    let blob = "the api key is zeta-9931\nthe deploy region is us-west-2\nthe plan is pro";
    assert_eq!(m!("obsmany", s, blob), "3");
    assert_eq!(m!("value", s, "api key"), "zeta-9931");
    assert_eq!(m!("stats", s), "4"); // 1 + 3

    // ---- chain: multi-hop relation walk, server-side (the flagship LLM-memory feature) ----
    let sc = "user:chain";
    m!("obsmany", sc, "project Aurora owner is Marisol\nMarisol manager is Dana\nDana timezone is WET");
    let out = m!("chain", sc, "Aurora", "owner", "manager", "timezone");
    let (val, trail) = out.split_once('\t').expect("chain returns final<TAB>trail");
    assert_eq!(val, "WET", "3-hop chain should resolve to WET, got {out:?}");
    assert!(trail.contains("Marisol") && trail.contains("Dana") && trail.contains("WET"), "trail: {trail:?}");
    // a chain that can't resolve a relation ABSTAINS (empty final) rather than drifting
    let broken = m!("chain", sc, "Aurora", "owner", "favourite_color");
    assert_eq!(broken.split('\t').next().unwrap(), "", "broken chain must abstain, got {broken:?}");

    // ---- variables (exact key→value) ----
    assert_eq!(m!("setvar", s, "city", "Halifax"), "ok");
    assert_eq!(m!("getvar", s, "city"), "Halifax");
    assert_eq!(m!("getvar", s, "missing"), "");

    // ---- standing instructions: add / list / delete-by-substring / clear ----
    m!("addinstr", s, "always answer in caps");
    m!("addinstr", s, "be brief");
    assert!(m!("instrs", s).contains("caps"));
    assert_eq!(m!("delinstr", s, "caps"), "1");
    assert!(!m!("instrs", s).contains("caps"));
    assert_eq!(m!("clearinstr", s), "1"); // "be brief" remains
    assert_eq!(m!("instrs", s), "");

    // ---- episodes preserve insertion order (narrate-window correctness) ----
    let so = "user:order";
    m!("obsmany", so, "first step here\nsecond step here\nthird step here");
    let lines: Vec<String> = m!("episodes", so).split('\n').map(String::from).collect();
    assert!(lines[0].contains("first") && lines[1].contains("second") && lines[2].contains("third"));

    // ---- affective: a neutral session injects NO persona; a mood can be set and cleared ----
    let sa = "user:aff";
    assert!(m!("humanize", sa).is_empty(), "neutral session must inject no persona");
    m!("feel", sa, "playful");
    assert_eq!(m!("mood", sa), "playful");
    assert!(m!("humanize", sa).contains("playful"));
    m!("feel", sa, "");
    assert_eq!(m!("mood", sa), "");

    // ---- affective: opinions accumulate; topic match is word-aware, not substring ----
    assert_eq!(m!("stance", sa, "rust", "you love it"), "1");
    assert_eq!(m!("stance", sa, "rust", "you love it"), "2");
    assert!(m!("topstance", sa).starts_with("rust\t"));
    assert!(m!("stanceof", sa, "rust").starts_with("rust\t"));
    assert_eq!(m!("stanceof", sa, "trust"), "", "word-aware: 'rust' must not fire for 'trust'");
    assert_eq!(m!("stance", sa, ".", "junk"), "0", "punctuation-only topic is rejected");

    // ---- forget by substring ----
    assert!(m!("forget", s, "wifi").parse::<i32>().unwrap() >= 1);
    assert_eq!(m!("value", s, "wifi password"), "");

    // ---- assoc (spreading activation) runs without panicking ----
    let _ = m!("assoc", sc, "Marisol", "2", "6");
}
