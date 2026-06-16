//! MCP ↔ WASM parity. The same facts and the same tool calls are run through BOTH engines — the
//! native MCP server (`handle_line` over JSON-RPC, backed by SQLite + semantic) and the in-browser
//! WASM `mem()` dispatch — and we assert both surface the same answer for every tool. This is the
//! "everything the MCP does, so does the WASM" guarantee, checked mechanically.
//!
//! Compiled only with `--features mcp` (which pulls in the MCP server). Without it the file is empty.
#![cfg(feature = "mcp")]
use neuron_core::db::NeuronDB;
use neuron_core::mcp::handle_line;
use neuron_core::wasm;

fn tmp() -> String {
    let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_parity_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}
// drive the WASM mem() through its real C-ABI, exactly like the browser
fn wmem(req: &str) -> String {
    let b = req.as_bytes();
    let p = wasm::alloc(b.len());
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), p, b.len()); }
    let n = wasm::mem(p, b.len());
    let out = unsafe { std::slice::from_raw_parts(wasm::answer_ptr(), n) };
    String::from_utf8_lossy(out).to_string()
}
macro_rules! w { ($($a:expr),*) => { wmem(&[$($a),*].join("\t")) } }
fn call(db: &NeuronDB, name: &str, args: &str) -> String {
    handle_line(db, &format!("{{\"id\":1,\"method\":\"tools/call\",\"params\":{{\"name\":\"{}\",\"arguments\":{}}}}}", name, args)).unwrap_or_default()
}

#[test]
fn mcp_and_wasm_agree() {
    let db = NeuronDB::open(&tmp(), 5000);
    wasm::mem_reset();
    let s = "u";
    let facts = ["the wifi password is hunter2", "project Aurora owner is Marisol", "Marisol manager is Dana", "Dana timezone is WET"];

    // ---- store the same facts in both engines ----
    call(&db, "remember", "{\"scope\":\"u\",\"facts\":[\"the wifi password is hunter2\",\"project Aurora owner is Marisol\",\"Marisol manager is Dana\",\"Dana timezone is WET\"]}");
    for f in facts { w!("observe", s, f); }

    // ---- recall_value: both isolate the value ----
    let m = call(&db, "recall_value", "{\"scope\":\"u\",\"query\":\"what is the wifi password\"}");
    let x = w!("value", s, "what is the wifi password");
    assert!(m.contains("hunter2"), "MCP recall_value: {m}");
    assert!(x.contains("hunter2"), "WASM value: {x}");

    // ---- recall: both surface the matching fact ----
    let m = call(&db, "recall", "{\"scope\":\"u\",\"query\":\"wifi password\",\"k\":5}");
    let x = w!("recall", s, "wifi password", "5");
    assert!(m.contains("hunter2") && x.contains("hunter2"), "recall parity: mcp={m} wasm={x}");

    // ---- recall_chain: both walk the 3-hop relation chain to WET ----
    let m = call(&db, "recall_chain", "{\"scope\":\"u\",\"start\":\"Aurora\",\"path\":[\"owner\",\"manager\",\"timezone\"]}");
    let x = w!("chain", s, "Aurora", "owner", "manager", "timezone");
    assert!(m.contains("WET"), "MCP recall_chain: {m}");
    assert!(x.contains("WET"), "WASM chain: {x}");

    // ---- named variable: note(kind=var) / setvar, then recall_var / getvar ----
    call(&db, "note", "{\"scope\":\"u\",\"kind\":\"var\",\"key\":\"region\",\"text\":\"us-west-2\"}");
    w!("setvar", s, "region", "us-west-2");
    let m = call(&db, "recall_var", "{\"scope\":\"u\",\"key\":\"region\"}");
    let x = w!("getvar", s, "region");
    assert!(m.contains("us-west-2"), "MCP recall_var: {m}");
    assert_eq!(x, "us-west-2", "WASM getvar");

    // ---- standing instruction: note(kind=instruction) / addinstr, then read it back ----
    call(&db, "note", "{\"scope\":\"u\",\"kind\":\"instruction\",\"text\":\"always answer in caps\"}");
    w!("addinstr", s, "always answer in caps");
    let m = call(&db, "recall", "{\"scope\":\"u::instr\",\"query\":\"caps\",\"k\":5}");   // MCP stores instructions in a typed sub-scope
    let x = w!("instrs", s);
    assert!(m.contains("caps"), "MCP instruction recall: {m}");
    assert!(x.contains("caps"), "WASM instrs: {x}");

    // ---- forget by substring: both drop the wifi fact ----
    call(&db, "forget", "{\"scope\":\"u\",\"match\":\"wifi\"}");
    w!("forget", s, "wifi");
    let m = call(&db, "recall_value", "{\"scope\":\"u\",\"query\":\"what is the wifi password\"}");
    let x = w!("value", s, "what is the wifi password");
    assert!(!m.contains("hunter2"), "MCP forget left wifi: {m}");
    assert!(!x.contains("hunter2"), "WASM forget left wifi: {x}");

    // ---- the chain still resolves after the unrelated forget (parity on a follow-up) ----
    let m = call(&db, "recall_chain", "{\"scope\":\"u\",\"start\":\"Aurora\",\"path\":[\"owner\",\"manager\",\"timezone\"]}");
    let x = w!("chain", s, "Aurora", "owner", "manager", "timezone");
    assert!(m.contains("WET") && x.contains("WET"), "post-forget chain parity: mcp={m} wasm={x}");
}
