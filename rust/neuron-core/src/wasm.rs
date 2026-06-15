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
use crate::model::GaryModel;

static mut BUF: [u8; 262144] = [0u8; 262144];
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
#[no_mangle]
pub extern "C" fn selftest() -> i32 {
    let mut s = Neuron::new(500);
    s.observe("the wifi password is vekam73");
    s.observe("the launch is on Friday");
    let mut code = 0;
    if let Some(r) = s.recall("what is the wifi password?") {
        if r.value == "vekam73" { code |= 1; }
    }
    let m = GaryModel::embedded();
    let facts: Vec<String> = s.recall("what is the wifi password?").map(|r| vec![r.fact]).unwrap_or_default();
    let ans = m.think(&facts, "what is the wifi password?", 8);
    if ans.contains("vekam73") { code |= 2; }
    let b = ans.as_bytes(); let n = b.len().min(256);
    unsafe { for i in 0..n { BUF[i] = b[i]; } BUFLEN = n; }
    code
}
#[no_mangle] pub extern "C" fn answer_ptr() -> *const u8 { unsafe { BUF.as_ptr() } }
#[no_mangle] pub extern "C" fn answer_len() -> usize { unsafe { BUFLEN } }

/// Allocate `n` bytes in wasm memory and return a pointer the host writes into.
#[no_mangle]
pub extern "C" fn alloc(n: usize) -> *mut u8 {
    let mut v = Vec::with_capacity(n); let p = v.as_mut_ptr(); std::mem::forget(v); p
}

/// Request protocol (UTF-8 at in_ptr..in_ptr+in_len): first line = query, each remaining
/// line = a fact. Builds a store from the facts, recalls + lets the cortex answer the query,
/// writes the answer into BUF, returns its length (read via answer_ptr/answer_len).
#[no_mangle]
pub extern "C" fn run(in_ptr: *const u8, in_len: usize) -> usize {
    let bytes = unsafe { std::slice::from_raw_parts(in_ptr, in_len) };
    let text = String::from_utf8_lossy(bytes);
    let mut lines = text.split('\n');
    let query = lines.next().unwrap_or("").to_string();
    let mut store = Neuron::new(500);
    for f in lines { if !f.trim().is_empty() { store.observe(f); } }
    let facts: Vec<String> = store.recall(&query).map(|r| vec![r.fact]).unwrap_or_default();
    let ans = GaryModel::embedded().think(&facts, &query, 10);
    let b = ans.as_bytes(); let n = b.len().min(256);
    unsafe { for i in 0..n { BUF[i] = b[i]; } BUFLEN = n; }
    n
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
