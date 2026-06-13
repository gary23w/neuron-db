//! C-ABI exports so neuron-core runs as a raw WebAssembly module in a Cloudflare Worker
//! (no wasm-bindgen, no external crates). JS instantiates this and calls the exports.
use crate::Neuron;
use crate::model::GaryModel;

static mut BUF: [u8; 256] = [0u8; 256];
static mut BUFLEN: usize = 0;

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
