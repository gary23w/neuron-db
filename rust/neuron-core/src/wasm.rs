//! C-ABI exports so neuron-core runs as a raw WebAssembly module in a Cloudflare Worker
//! (no wasm-bindgen, no external crates). JS instantiates this and calls the exports.
// A single `static mut` buffer is the intentional shape for this no-bindgen C ABI (JS reads the
// pointer/len); access is single-threaded inside the wasm sandbox, so the static_mut_refs lint
// (a Rust 2024 future-compat warning) is acknowledged and allowed for this module.
#![allow(static_mut_refs)]
// The exports take raw (ptr, len) pairs because that IS the C ABI the JS host calls them with;
// the host guarantees a valid buffer for the given length, so the not_unsafe_ptr_arg_deref lint
// (which would have us mark every export `unsafe`, changing the exported signature) is allowed here.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
use crate::Neuron;
#[cfg(feature = "cortex")]
use crate::model::GaryModel;
use std::collections::HashMap;
#[cfg(feature = "cortex")]
use std::sync::OnceLock;

/// The cortex + tokenizer, built ONCE and reused across calls. Rebuilding it per call
/// (dequantizing ~6.9M int8 weights + reloading the BPE every run()/selftest()) was the
/// dominant cost of the in-browser inference path.
#[cfg(feature = "cortex")]
fn cortex() -> &'static GaryModel {
    static M: OnceLock<GaryModel> = OnceLock::new();
    M.get_or_init(GaryModel::embedded)
}

// 1 MiB result buffer — large enough that `dump`/`episodes`/big recalls of a realistic scope (~20k facts)
// don't truncate. Zero-initialized static, so it costs linear memory at runtime, not module size.
static mut BUF: [u8; 1048576] = [0u8; 1048576];
static mut BUFLEN: usize = 0;

// ---- persistent in-browser store, for the live "synapse" visualization ----
static mut STORE: Option<Neuron> = None;
#[allow(static_mut_refs)]
fn store() -> &'static mut Neuron {
    unsafe { STORE.get_or_insert_with(|| Neuron::new(100_000)) }
}
#[allow(static_mut_refs)]
fn put(s: &str) -> usize {
    let b = s.as_bytes();
    let n = b.len().min(unsafe { BUF.len() });
    unsafe { BUF[..n].copy_from_slice(&b[..n]); BUFLEN = n; }
    n
}
fn input(ptr: *const u8, len: usize) -> String {
    String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(ptr, len) }).into_owned()
}

/// Reset the in-browser store. Call before seeding a fresh graph.
#[no_mangle] pub extern "C" fn syn_reset() { unsafe { STORE = Some(Neuron::new(100_000)); } }

/// Observe one fact. Returns the new neuron's index, or -1 if it was not stored.
#[no_mangle]
pub extern "C" fn syn_add(ptr: *const u8, len: usize) -> i32 {
    let s = store();
    let before = s.fact_count();
    s.observe(&input(ptr, len));
    if s.fact_count() > before { (s.fact_count() - 1) as i32 } else { -1 }
}

/// Fire a single recall. Writes "value\tbestIdx\tc1,c2,..." to BUF; returns its length.
/// bestIdx = the winning neuron index, the c-list = every candidate neuron that fired.
#[no_mangle]
pub extern "C" fn syn_fire(ptr: *const u8, len: usize) -> usize {
    let q = input(ptr, len);
    let s = store();
    let mut cue = crate::stems_s(&crate::content(&q));
    crate::expand_cue(&q, &mut cue);
    let pet = cue.contains(&crate::stem1("pet")) || cue.contains(&crate::stem1("animal"));
    let cands = s.candidates(&cue, pet);
    let (best_idx, value) = match s.recall(&q) {
        Some(r) => (s.episodes.iter().position(|e| e.t == r.fact).map(|x| x as i64).unwrap_or(-1), r.value),
        None => (-1, String::new()),
    };
    let clist: Vec<String> = cands.iter().map(|i| i.to_string()).collect();
    put(&format!("{}\t{}\t{}", value, best_idx, clist.join(",")))
}

/// Fire a multi-hop chain. Input: "start\nrel1\nrel2\n...". Walks the chain server-side
/// (each hop a recall, the relation must appear in the resolved fact). Writes one line per
/// resolved hop, "neuronIdx\tvalue", to BUF; returns its length. This is the synapse linking
/// neuron to neuron with no extra cost per hop.
#[no_mangle]
pub extern "C" fn syn_chain(ptr: *const u8, len: usize) -> usize {
    let text = input(ptr, len);
    let mut lines = text.split('\n').map(|x| x.trim()).filter(|x| !x.is_empty());
    let mut current = match lines.next() { Some(s) => s.to_string(), None => return put("") };
    let s = store();
    let mut out: Vec<String> = Vec::new();
    for rel in lines {
        let rel_words: Vec<&str> = rel.split_whitespace().filter(|w| w.len() >= 3).collect();
        match s.recall(&format!("{} {}", current, rel)) {
            Some(r) if rel_words.is_empty()
                || rel_words.iter().any(|rw| r.fact.split_whitespace().any(|w| crate::rel_matches(w, rw))) => {
                let idx = s.episodes.iter().position(|e| e.t == r.fact).map(|x| x as i64).unwrap_or(-1);
                out.push(format!("{}\t{}", idx, r.value));
                current = r.value;
            }
            _ => break,
        }
    }
    put(&out.join("\n"))
}

/// Full self-test inside the wasm sandbox: store recall + emergence cortex generation.
/// Returns a bitmask: 1 = store recall correct, 2 = cortex copied the value from context.
#[cfg(feature = "cortex")]
#[no_mangle]
pub extern "C" fn selftest() -> i32 {
    let mut s = Neuron::new(500);
    s.observe("the wifi password is vekam73");
    s.observe("the launch is on Friday");
    let mut code = 0;
    if let Some(r) = s.recall("what is the wifi password?") {
        if r.value == "vekam73" { code |= 1; }
    }
    let m = cortex();
    let facts: Vec<String> = s.recall("what is the wifi password?").map(|r| vec![r.fact]).unwrap_or_default();
    let ans = m.think(&facts, "what is the wifi password?", 8);
    if ans.contains("vekam73") { code |= 2; }
    let b = ans.as_bytes(); let n = b.len().min(256);
    unsafe { BUF[..n].copy_from_slice(&b[..n]); BUFLEN = n; }
    code
}
#[no_mangle] pub extern "C" fn answer_ptr() -> *const u8 { unsafe { BUF.as_ptr() } }
#[no_mangle] pub extern "C" fn answer_len() -> usize { unsafe { BUFLEN } }

/// Allocate `n` bytes in wasm memory and return a pointer the host writes into.
#[no_mangle]
pub extern "C" fn alloc(n: usize) -> *mut u8 {
    let mut v = Vec::with_capacity(n); let p = v.as_mut_ptr(); std::mem::forget(v); p
}

/// Free a buffer previously handed out by `alloc` (host calls this once it has written + the
/// guest has consumed the input). Without it every `alloc` leaks — the Vec was `mem::forget`'d.
/// `n` is the original allocation size; the buffer was capacity `n`, length 0. The result `BUF`
/// is a separate static and is NOT freed here.
#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, n: usize) {
    if !ptr.is_null() && n > 0 { unsafe { drop(Vec::from_raw_parts(ptr, 0, n)); } }
}

/// Request protocol (UTF-8 at in_ptr..in_ptr+in_len): first line = query, each remaining
/// line = a fact. Builds a store from the facts, recalls + lets the cortex answer the query,
/// writes the answer into BUF, returns its length (read via answer_ptr/answer_len).
#[cfg(feature = "cortex")]
#[no_mangle]
pub extern "C" fn run(in_ptr: *const u8, in_len: usize) -> usize {
    let bytes = unsafe { std::slice::from_raw_parts(in_ptr, in_len) };
    let text = String::from_utf8_lossy(bytes);
    let mut lines = text.split('\n');
    let query = lines.next().unwrap_or("").to_string();
    let mut store = Neuron::new(500);
    for f in lines { if !f.trim().is_empty() { store.observe(f); } }
    let facts: Vec<String> = store.recall(&query).map(|r| vec![r.fact]).unwrap_or_default();
    let ans = cortex().think(&facts, &query, 10);
    let b = ans.as_bytes(); let n = b.len().min(256);
    unsafe { BUF[..n].copy_from_slice(&b[..n]); BUFLEN = n; }
    n
}

/// gary-neuron as the lab's in-browser model: the host has ALREADY recalled the working set, so
/// pass it through verbatim (no internal re-recall) and let the cortex reason over ALL of it —
/// first line = the query, each remaining non-empty line = one recalled fact. Answer -> BUF.
#[cfg(feature = "cortex")]
#[no_mangle]
pub extern "C" fn gary(in_ptr: *const u8, in_len: usize) -> usize {
    let bytes = unsafe { std::slice::from_raw_parts(in_ptr, in_len) };
    let text = String::from_utf8_lossy(bytes);
    let mut lines = text.split('\n');
    let query = lines.next().unwrap_or("").to_string();
    let facts: Vec<String> = lines.filter(|l| !l.trim().is_empty()).map(|s| s.to_string()).collect();
    let ans = cortex().think(&facts, &query, 24);
    let b = ans.as_bytes(); let n = b.len().min(256);
    unsafe { BUF[..n].copy_from_slice(&b[..n]); BUFLEN = n; }
    n
}

// ---- in-browser NeuronDB-equivalent: scopes + typed neurons (var, instruction) + recall, so the
// browser lab has the SAME public surface the MCP server exposes (recall / recall_value /
// recall_associative / remember / note(var|instruction) / recall_var / forget / stats) — all
// in-memory, no SQLite (which doesn't target wasm). Vars and instructions are exact key→value /
// ordered lists; free-text memory is a Neuron per scope. Driven by the tab-delimited mem() call.
struct MemDB {
    scopes: HashMap<String, Neuron>,
    vars: HashMap<String, HashMap<String, String>>,
    instrs: HashMap<String, Vec<String>>,
    moods: HashMap<String, String>,
    stances: HashMap<String, Vec<(String, String, f32)>>,   // scope -> [(topic, feeling, strength)]
}
impl MemDB {
    fn new() -> Self {
        MemDB { scopes: HashMap::new(), vars: HashMap::new(), instrs: HashMap::new(),
                moods: HashMap::new(), stances: HashMap::new() }
    }
    fn n(&mut self, scope: &str) -> &mut Neuron {
        self.scopes.entry(scope.to_string()).or_insert_with(|| Neuron::new(1_000_000))
    }
}
static mut MEM: Option<MemDB> = None;
fn memdb() -> &'static mut MemDB { unsafe { MEM.get_or_insert_with(MemDB::new) } }

use crate::affect::topic_matches;   // the one shared whole-word topic matcher
use crate::op::{apply, NeuronOp, OpResult};

/// Drives the in-browser MemDB through the shared op::apply vocabulary. A zero-sized handle: the
/// real store is the single-threaded `memdb()` global, reached inside each method (the same access
/// the raw mem() arms use), so no interior-mutability wrapper is needed. Each method makes exactly
/// ONE memdb() call so two `&mut MemDB` are never live at once.
struct MemStore;
impl crate::op::Store for MemStore {
    fn observe(&self, scope: &str, text: &str) -> usize { memdb().n(scope).observe(text) }
    fn observe_bulk(&self, scope: &str, texts: &[String]) -> usize {
        let n = memdb().n(scope);
        texts.iter().map(|t| t.trim()).filter(|t| !t.is_empty()).map(|t| n.observe(t)).sum()
    }
    fn recall_one(&self, scope: &str, query: &str) -> Option<crate::Recall> { memdb().n(scope).recall(query) }
    // the in-browser store is lexical only — semantic/across are durable-store concepts
    fn recall_block(&self, scope: &str, query: &str, k: usize, _semantic: bool, _across: bool) -> Vec<crate::Recall> { memdb().n(scope).recall_many(query, k) }
    fn recall_value(&self, scope: &str, query: &str) -> Option<String> { memdb().n(scope).recall(query).map(|r| r.value) }
    fn recall_assoc(&self, scope: &str, query: &str, k: usize, hops: usize) -> Vec<crate::Spread> { memdb().n(scope).recall_spreading(query, k, hops) }
    fn recall_chain(&self, scope: &str, start: &str, path: &[String]) -> (Option<String>, Vec<String>) {
        let n = memdb().n(scope);
        let mut current = start.trim().to_string();
        let mut trail = vec![current.clone()];
        for rel in path {
            let rel_words: Vec<&str> = rel.split_whitespace().filter(|w| w.len() >= 3).collect();
            match n.recall(&format!("{} {}", current, rel)) {
                Some(h) if rel_words.is_empty()
                    || rel_words.iter().any(|rw| h.fact.split_whitespace().any(|w| crate::rel_matches(w, rw))) => {
                    current = h.value.clone(); trail.push(current.clone());
                }
                _ => return (None, trail),   // a broken relation abstains instead of drifting
            }
        }
        (Some(current), trail)
    }
    fn var_set(&self, scope: &str, key: &str, value: &str) -> usize { memdb().vars.entry(scope.to_string()).or_default().insert(key.to_string(), value.to_string()); 1 }
    fn var_get(&self, scope: &str, key: &str) -> Option<String> { memdb().vars.get(scope).and_then(|m| m.get(key)).cloned() }
    fn note_stance(&self, scope: &str, topic: &str, feeling: &str) -> (f32, bool) {
        let v = memdb().stances.entry(scope.to_string()).or_default();
        for s in v.iter_mut() { if s.0 != topic { s.2 = (s.2 * crate::affect::STANCE_DECAY).max(crate::affect::STANCE_FLOOR); } }
        match v.iter_mut().find(|s| s.0 == topic) {
            Some(s) => { s.2 += crate::affect::STANCE_BUMP; s.1 = feeling.to_string(); (s.2, false) }
            None => { v.push((topic.to_string(), feeling.to_string(), crate::affect::STANCE_BUMP)); (crate::affect::STANCE_BUMP, true) }
        }
    }
    fn set_mood(&self, scope: &str, emotion: &str) {
        if emotion.is_empty() { memdb().moods.remove(scope); } else { memdb().moods.insert(scope.to_string(), emotion.to_string()); }
    }
    fn affect(&self, scope: &str, asked_topic: Option<&str>) -> String {
        let db = memdb();
        let empty = Vec::new();
        let body = crate::affect::directive_body(db.moods.get(scope).map(|s| s.as_str()), db.stances.get(scope).unwrap_or(&empty), asked_topic);
        if body.is_empty() { String::new() } else { format!("{}{}", crate::affect::FRAME, body) }
    }
    fn turn(&self, _scope: &str, _message: &str) -> crate::TurnOut { crate::TurnOut::default() }   // the in-browser store has no turn engine
    fn forget(&self, scope: &str, matching: Option<&str>) -> (usize, usize) {
        let n = memdb().n(scope);
        let before = n.fact_count();
        match matching { None => n.episodes.clear(), Some(m) => { let needle = m.to_lowercase(); n.episodes.retain(|ep| !ep.t.to_lowercase().contains(&needle)); } }
        n.invalidate_index();
        let after = n.fact_count();
        (before - after, after)
    }
    fn stats(&self, scope: &str) -> crate::Stats {
        let (facts, dropped) = memdb().scopes.get(scope).map(|n| (n.fact_count(), n.dropped)).unwrap_or((0, 0));
        crate::Stats { facts, max_facts: 1_000_000, dropped, ..Default::default() }
    }
    fn scopes(&self) -> Vec<String> { let mut v: Vec<String> = memdb().scopes.keys().cloned().collect(); v.sort(); v }
}

/// Reset the whole in-browser database (all scopes, vars, instructions).
#[no_mangle] pub extern "C" fn mem_reset() { unsafe { MEM = Some(MemDB::new()); } }

// ---- opt-in HTTP capability (`--features http`) — the WASM owns the request, the host owns the socket.
// The guest calls `host_http` to START a fetch and gets a token; the host runs the fetch and delivers the
// body back via `http_deliver`; the guest reads it with the `fetched` op. This is the only way a
// wasm32-unknown-unknown module reaches the network in a browser: the runtime/host exposes the transport
// as an import the guest calls (per the WASI/HTTP discussion — no in-module TCP exists).
#[cfg(all(feature = "http", target_arch = "wasm32"))]
#[link(wasm_import_module = "env")]
extern "C" { fn host_http(ptr: *const u8, len: usize) -> i32; }
#[cfg(all(feature = "http", not(target_arch = "wasm32")))]
unsafe fn host_http(_ptr: *const u8, _len: usize) -> i32 { -1 }   // no host transport off-wasm (native tests)
#[cfg(feature = "http")]
static mut HTTP: Option<HashMap<i32, String>> = None;
#[cfg(feature = "http")]
fn httpmap() -> &'static mut HashMap<i32, String> { unsafe { HTTP.get_or_insert_with(HashMap::new) } }
/// The host calls this to deliver a fetched body for `token` (bytes written into wasm memory at ptr/len).
#[cfg(feature = "http")]
#[no_mangle] pub extern "C" fn http_deliver(token: i32, ptr: *const u8, len: usize) { httpmap().insert(token, input(ptr, len)); }

/// Tab-delimited request "op\tscope\targ1\targ2…"; writes the result to BUF, returns its length.
/// ops: observe | obsmany | recall | recallscored | value | assess | assoc | chain | setvar | getvar |
///      vars | delvar | addinstr | instrs | delinstr | clearinstr | forget | stats | episodes |
///      dump | load | scopes | feel | stance | humanize | mood | topstance | stanceof | fetch | fetched.
#[no_mangle]
pub extern "C" fn mem(ptr: *const u8, len: usize) -> usize {
    let req = input(ptr, len);
    let f: Vec<&str> = req.split('\t').collect();
    let op = f.first().copied().unwrap_or("");
    let scope = f.get(1).copied().unwrap_or("default").to_string();
    let arg = |i: usize| f.get(i).copied().unwrap_or("");
    let num = |i: usize, d: usize| f.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
    // routed arms go through op::apply (the one vocabulary, generic over Store); wasm-specific arms
    // reach memdb() directly. Each path touches one &mut MemDB at a time, never two at once.
    let out: String = match op {
        "observe" => apply(&MemStore, NeuronOp::Observe { scope, text: arg(2).to_string() }).wrote().to_string(),
        "recall" => apply(&MemStore, NeuronOp::Recall { scope, query: arg(2).to_string(), k: num(3, 6), semantic: false, across: false })
            .hits().into_iter().map(|r| r.fact).collect::<Vec<_>>().join("\n"),
        "value" => apply(&MemStore, NeuronOp::RecallValue { scope, query: arg(2).to_string() }).value().unwrap_or_default(),
        "assoc" => apply(&MemStore, NeuronOp::RecallAssoc { scope, query: arg(2).to_string(), k: num(4, 8), hops: num(3, 2) })
            .assoc().into_iter().map(|s| s.fact).collect::<Vec<_>>().join("\n"),
        "setvar" => { apply(&MemStore, NeuronOp::VarSet { scope, key: arg(2).to_string(), value: arg(3).to_string() }); "ok".into() }
        "getvar" => apply(&MemStore, NeuronOp::VarGet { scope, key: arg(2).to_string() }).value().unwrap_or_default(),
        "addinstr" => {
            let v = memdb().instrs.entry(scope).or_default();
            let t = arg(2).to_string();
            if !t.is_empty() && !v.contains(&t) { v.push(t); }
            "ok".into()
        }
        "instrs" => memdb().instrs.get(&scope).map(|v| v.join("\n")).unwrap_or_default(),
        // remove every standing instruction whose text contains the (case-insensitive) needle; returns count removed
        "delinstr" => {
            let needle = arg(2).trim().to_lowercase();
            match memdb().instrs.get_mut(&scope) {
                Some(v) if !needle.is_empty() => {
                    let before = v.len();
                    v.retain(|i| !i.to_lowercase().contains(&needle));
                    (before - v.len()).to_string()
                }
                _ => "0".into(),
            }
        }
        // drop all standing instructions for the scope; returns count removed
        "clearinstr" => memdb().instrs.get_mut(&scope).map(|v| { let n = v.len(); v.clear(); n.to_string() }).unwrap_or_else(|| "0".into()),
        // empty needle clears the scope; otherwise drops facts containing it (same substring semantics as the MCP/db forget)
        "forget" => match apply(&MemStore, NeuronOp::Forget { scope, matching: { let m = arg(2); if m.is_empty() { None } else { Some(m.to_string()) } } }) {
            OpResult::Forgot { forgot, .. } => forgot.to_string(), _ => "0".into(),
        },
        "stats" => match apply(&MemStore, NeuronOp::Stats { scope }) { OpResult::Stats(s) => s.facts.to_string(), _ => "0".into() },
        // the scope's stored episode texts, in insertion order — so a caller's ordered view matches
        // exactly what was stored (the JS sentence splitter and Rust's differ, e.g. on ';')
        "episodes" => memdb().scopes.get(&scope)
            .map(|n| n.episodes.iter().map(|e| e.t.clone()).collect::<Vec<_>>().join("\n"))
            .unwrap_or_default(),
        // batch ingest: arg2 is a newline-joined block. One wasm crossing for a whole document
        // instead of N — fewer boundary hops + encodes. Returns the count of newly-stored facts.
        "obsmany" => apply(&MemStore, NeuronOp::ObserveBulk { scope, texts: arg(2).split('\n').map(|l| l.to_string()).collect() }).wrote().to_string(),
        // loadmany\t<scope1>\t<facts1 \n-joined>\t<scope2>\t<facts2>… — multi-scope fact preload in ONE
        // crossing. mem() already split the request on \t, so each scope's \n-joined facts are one field
        // here (facts must be tab/newline-free — the pack reader + the JS helper guarantee that). Returns
        // "<stored>\t<lines>" so the host can compare episodes-stored against lines-submitted.
        "loadmany" => {
            let (mut total, mut lines) = (0usize, 0usize);
            let mut idx = 1;
            while idx + 1 < f.len() {
                let (sc, blob) = (f[idx], f[idx + 1]); idx += 2;
                let texts: Vec<String> = blob.split('\n').map(str::trim).filter(|t| !t.is_empty()).map(|t| { lines += 1; t.to_string() }).collect();
                if !texts.is_empty() { total += apply(&MemStore, NeuronOp::ObserveBulk { scope: sc.to_string(), texts }).wrote(); }
            }
            format!("{}\t{}", total, lines)
        }
        // multi-hop recall: start at arg2 and follow each subsequent field as one relation. Walked
        // server-side in microseconds; abstains (final empty) if a relation doesn't actually appear.
        // Returns "<final>\t<step → step → …>".
        "chain" => {
            let path: Vec<String> = f.iter().skip(3).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            match apply(&MemStore, NeuronOp::RecallChain { scope, start: arg(2).trim().to_string(), path }) {
                OpResult::Chain { value, trail } => format!("{}\t{}", value.unwrap_or_default(), trail.join(" → ")),
                _ => String::new(),
            }
        }
        // --- opt-in web fetch (feature "http"): the guest INITIATES an HTTP GET through the host
        // transport and gets a token; poll `fetched <token>` until the host delivers the body ---
        #[cfg(feature = "http")]
        "fetch" => {
            let url = arg(2);
            let token = unsafe { host_http(url.as_ptr(), url.len()) };
            if token < 0 { "http-unavailable".into() } else { format!("pending:{}", token) }
        }
        #[cfg(feature = "http")]
        "fetched" => {
            let token: i32 = arg(2).parse().unwrap_or(-1);
            httpmap().get(&token).cloned().unwrap_or_else(|| "pending".into())
        }
        // --- affective layer: mood + stance writes route through apply (its Store impl owns the
        // accumulation); the read-only accessors below stay direct ---
        "feel" => { apply(&MemStore, NeuronOp::Mood { scope, emotion: arg(2).trim().to_string() }); "ok".into() }
        "stance" => {
            let topic = arg(2).trim().to_lowercase();
            if topic.is_empty() || !topic.chars().any(|c| c.is_alphanumeric()) { "0".into() }   // no empty/punctuation topics
            else { match apply(&MemStore, NeuronOp::Stance { scope, topic, feeling: arg(3).trim().to_string() }) {
                OpResult::Stance { intensity, .. } => format!("{:.0}", intensity), _ => "0".into(),
            } }
        }
        "humanize" => {
            let topic = arg(2).trim().to_lowercase();   // optional: bias toward the asked-about topic's stance
            apply(&MemStore, NeuronOp::Affect { scope, topic: if topic.is_empty() { None } else { Some(topic) } }).text()
        }
        "mood" => memdb().moods.get(&scope).cloned().unwrap_or_default(),
        "topstance" => memdb().stances.get(&scope)
            .and_then(|v| v.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)))
            .map(|s| format!("{}\t{}\t{:.1}", s.0, s.1, s.2)).unwrap_or_default(),
        "stanceof" => {
            let topic = arg(2).trim().to_lowercase();
            memdb().stances.get(&scope)
                .and_then(|v| v.iter().find(|s| topic_matches(&s.0, &topic)))
                .map(|s| format!("{}\t{}\t{:.1}", s.0, s.1, s.2)).unwrap_or_default()
        }
        // --- the knowledge-GAP signal (reasoned-routing keystone, lifted into the core). For a query,
        // surface the recall engine's OWN confidence for the best hit — coverage/overlap/exact — plus how
        // many facts fired. A controller reads this to decide "do I know this, or must I go find out?"
        // Returns "coverage\toverlap\texact\tn_hits\thas_value\tbest_fact".
        "assess" => {
            let q = arg(2);
            let n = memdb().n(&scope);
            let n_hits = n.recall_many(q, 8).iter().filter(|r| r.overlap > 0 || r.coverage > 0.0).count();
            match n.recall(q) {
                Some(r) => format!("{:.4}\t{}\t{}\t{}\t{}\t{}", r.coverage, r.overlap, r.exact, n_hits, (!r.value.is_empty()) as u8, r.fact),
                None => format!("0.0000\t0\t0\t{}\t0\t", n_hits),
            }
        }
        // recall WITH per-hit confidence: "fact\tcoverage\toverlap" lines, best first (so the lab can rank + show how sure it is)
        "recallscored" => memdb().n(&scope).recall_many(arg(2), num(3, 6))
            .into_iter().map(|r| format!("{}\t{:.4}\t{}", r.fact, r.coverage, r.overlap)).collect::<Vec<_>>().join("\n"),
        // list every variable as "key\tvalue" lines (sorted, for a deterministic view)
        "vars" => memdb().vars.get(&scope).map(|m| { let mut v: Vec<String> = m.iter().map(|(k,val)| format!("{}\t{}", k, val)).collect(); v.sort(); v.join("\n") }).unwrap_or_default(),
        // delete one variable; returns "1" if it existed else "0"
        "delvar" => memdb().vars.get_mut(&scope).map(|m| if m.remove(arg(2)).is_some() { "1" } else { "0" }).unwrap_or("0").to_string(),
        // serialize the scope's facts to a portable blob (Neuron::dump) — the host persists it and restores
        // with `load`, so a tab can rehydrate its whole memory in ONE crossing instead of replaying every observe
        "dump" => memdb().scopes.get(&scope).map(|n| n.dump()).unwrap_or_default(),
        // restore a scope's facts from a dump blob (replaces the scope); returns the fact count loaded.
        // The blob carries its own tabs, which the protocol split — rejoin fields 2.. to reconstruct it.
        "load" => {
            let blob = f.iter().skip(2).copied().collect::<Vec<_>>().join("\t");
            let n = crate::Neuron::load(&blob, 1_000_000);
            let c = n.fact_count();
            memdb().scopes.insert(scope, n);
            c.to_string()
        }
        // overview: every live scope as "scope\tfactcount" lines (sorted) — for debugging / a memory map
        "scopes" => { let mut v: Vec<String> = memdb().scopes.iter().map(|(k,n)| format!("{}\t{}", k, n.fact_count())).collect(); v.sort(); v.join("\n") }
        // the capability manifest a host reads to learn what neuron can do and which capabilities it
        // owns: "name\tgrounded|deferrable\tabout" lines (§7 — grounded beats tier). With host cap
        // names as trailing fields, resolves the live keep/defer surface (parity with CLI/MCP);
        // grounded names stay `keep` even if the host claims them.
        "caps" => {
            let host: Vec<&str> = f.get(1..).unwrap_or(&[]).iter().copied().filter(|s| !s.is_empty()).collect();
            if host.is_empty() { crate::caps::manifest() }
            else { crate::caps::resolve(&host).into_iter().map(|(n, d)| format!("{}\t{}", n, d.tag())).collect::<Vec<_>>().join("\n") }
        }
        _ => String::new(),
    };
    put(&out)
}

// ---- semantic space exports (feature `semantic`), for the in-browser PCA visualization ----
#[cfg(feature = "semantic")]
mod sem_exports {
    use super::{input, put, BUF};
    use crate::semantic::SemanticSpace;
    static mut SEM: Option<SemanticSpace> = None;
    #[allow(static_mut_refs)]
    fn sem() -> &'static mut SemanticSpace { unsafe { SEM.get_or_insert_with(SemanticSpace::new) } }

    /// Reset the semantic space (call before learning a fresh corpus).
    #[no_mangle] pub extern "C" fn sem_reset() { unsafe { SEM = Some(SemanticSpace::new()); } }
    /// Learn co-occurrence from a span of text.
    #[no_mangle] pub extern "C" fn sem_learn(ptr: *const u8, len: usize) { sem().train(&input(ptr, len)); }
    /// vocabulary size of the learned space.
    #[no_mangle] pub extern "C" fn sem_vocab() -> usize { sem().vocab() }

    /// TRUE 256-D cosine nearest neighbours of one word (honest neighbours, not the
    /// projection). Writes up to `k` lines "word\tcosine" to BUF; returns its length.
    #[no_mangle]
    pub extern "C" fn sem_neighbors(ptr: *const u8, len: usize, k: usize) -> usize {
        let near = sem().nearest(&input(ptr, len), k);
        let mut s = String::new();
        for (w, cos) in near { s.push_str(&format!("{}\t{:.3}\n", w, cos)); }
        put(&s)
    }

    /// PCA-project the top `top_n` words onto the top `k` principal components and write a
    /// blob to BUF; returns its length. Format (tab/comma/newline text):
    ///   line 0:  "<n>\t<k>\t<var0,var1,...,var(k-1)>\t<total_variance>"
    ///   line i:  "<word>\t<count>\t<cluster>\t<c0,c1,...,c(k-1)>"   (cluster from TRUE 256-D k-means)
    #[no_mangle]
    pub extern "C" fn sem_project(top_n: usize, k: usize) -> usize {
        let p = sem().project(top_n, k);
        let vars: Vec<String> = p.variance.iter().map(|v| format!("{:.5}", v)).collect();
        let mut s = format!("{}\t{}\t{}\t{:.5}\n", p.words.len(), k, vars.join(","), p.total_variance);
        for (i, w) in p.words.iter().enumerate() {
            let coords: Vec<String> = p.coords[i].iter().map(|c| format!("{:.4}", c)).collect();
            s.push_str(&format!("{}\t{}\t{}\t{}\n", w, sem().count(w), p.clusters[i], coords.join(",")));
            if s.len() > unsafe { BUF.len() } - 256 { break; }
        }
        put(&s)
    }
}
