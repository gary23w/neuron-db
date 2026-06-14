//! LLM-memory benchmark: needle-in-a-haystack at scale. An LLM's context window is
//! small; NeuronDB is the memory that lives *outside* it. So the question is not "can
//! it hold a big blob" but "with a huge store, does recall still surface the right
//! fact, accurately and fast, so we can inject just the top-k into the prompt?".
//!
//! For each haystack size N we plant M known needles among N diverse distractors, then:
//!   - single recall  : ask each needle's question, expect its exact value
//!   - top-k block     : recall_many(k); is the needle in the injectable memory block?
//!   - broad query     : a cue shared by every distractor (worst case, O(N)) for contrast
//!
//! Run: cargo run --release --features sqlite --example llm_memory_bench
use neuron_core::db::NeuronDB;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_llm_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}
// unique 4-char alpha key (no stem collision); distinct for distinct ids < 26^4.
fn code(mut x: usize) -> String {
    let mut s = String::new();
    for _ in 0..4 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; }
    s
}

fn main() {
    println!("== NeuronDB LLM-memory benchmark (needle-in-haystack) ==\n");
    let needles = 50usize;
    let k = 8usize; // memory-block size injected into the prompt

    println!("{:>7} | single-recall | top-{} block | needle p50/p95 | broad cue (O(N))", "haystack", k);
    println!("{}", "-".repeat(86));

    for &n in &[1_000usize, 5_000, 20_000, 50_000] {
        let db = NeuronDB::open(&tmp(), 5_000_000);

        // distractors: diverse, each with a unique subject key (offset so it never
        // collides with a needle key), plus a shared word "reading" for the broad test.
        let bg: Vec<String> = (0..n).map(|i| format!("the {} reading is r{}", code(100_000 + i), i)).collect();
        db.observe_many("history", &bg);
        // needles: a distinct subject + a secret value, sharing the word "passphrase".
        let nd: Vec<String> = (0..needles).map(|j| format!("the {} passphrase is secret{}", code(j), j)).collect();
        db.observe_many("history", &nd);
        let _ = db.get("history", &format!("what is the {} passphrase?", code(0))); // warm index

        // 1) single recall accuracy + latency
        let mut hit = 0usize;
        let mut lat: Vec<u128> = Vec::with_capacity(needles);
        for j in 0..needles {
            let q = format!("what is the {} passphrase?", code(j));
            let t = Instant::now();
            let r = db.get("history", &q);
            lat.push(t.elapsed().as_nanos());
            if r.as_deref() == Some(format!("secret{}", j).as_str()) { hit += 1; }
        }
        lat.sort_unstable();
        let p = |q: f64| lat[((lat.len() as f64 * q) as usize).min(lat.len() - 1)] as f64 / 1000.0;

        // 2) top-k memory block: is the needle fact present in recall_many(k)?
        let mut block_hit = 0usize;
        for j in 0..needles {
            let q = format!("what is the {} passphrase?", code(j));
            let hits = db.recall_many("history", &q, k);
            if hits.iter().any(|h| h.value == format!("secret{}", j)) { block_hit += 1; }
        }

        // 3) broad cue shared by all N distractors -> candidate set = whole scope
        let t = Instant::now();
        let _ = db.get("history", "what is the reading?");
        let broad = t.elapsed().as_micros() as f64;

        println!("{:>7} | {:>4}/{:<3} {:>3.0}% | {:>4}/{:<3} {:>3.0}% | {:>5.1}/{:>6.1} us | {:>8.0} us",
                 n, hit, needles, 100.0 * hit as f64 / needles as f64,
                 block_hit, needles, 100.0 * block_hit as f64 / needles as f64,
                 p(0.50), p(0.95), broad);
    }

    println!("\nReading: with {} needles hidden among up to 50,000 facts, selective recall stays", needles);
    println!("flat and accurate — the LLM only ever sees the top-{} block, not the whole store.", k);
    println!("The 'broad cue' column is the worst case (a word in every fact) and grows with N;");
    println!("avoid it by giving each memory a distinct subject. == done ==");
}
