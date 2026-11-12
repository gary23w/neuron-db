//! A wasm result larger than the 1 MiB static buffer must come back whole, not cut off.
//! One test fn: the wasm module keeps process-global state.
use neuron_core::wasm;

fn call(req: &str) -> String {
    let b = req.as_bytes();
    let p = wasm::alloc(b.len());
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), p, b.len()); }
    let n = wasm::mem(p, b.len());
    assert_eq!(n, wasm::answer_len());
    let out = unsafe { std::slice::from_raw_parts(wasm::answer_ptr(), n) };
    String::from_utf8_lossy(out).to_string()
}
macro_rules! m { ($($a:expr),*) => { call(&[$($a),*].join("\t")) } }

#[test]
fn big_dump_is_whole_and_small_results_follow() {
    wasm::mem_reset();
    let facts: Vec<String> = (0..15000).map(|i| format!("customer Person{i} Example{i} of Oakville at {} King Street phone 905-555-{:04}", 100 + i, i % 10000)).collect();
    for chunk in facts.chunks(500) { m!("obsmany", "big", &chunk.join("\n")); }
    assert_eq!(m!("stats", "big"), "15000");
    let blob = m!("dump", "big");
    assert!(blob.len() > 1048576, "the fixture must exceed the static buffer: {}", blob.len());
    assert_eq!(blob.split('\n').count(), 15000, "every fact is in the dump");
    assert!(blob.ends_with("\t1"), "the last line is complete");
    assert_eq!(m!("load", "big2", &blob), "15000");
    // a small result after a big one is read from the static buffer again
    assert_eq!(m!("stats", "big2"), "15000");
    assert!(m!("recall", "big2", "Person14999", "1").contains("Person14999"));
    let eps = m!("episodes", "big");
    assert_eq!(eps.split('\n').count(), 15000);
}
