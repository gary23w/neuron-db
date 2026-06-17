//! "How many dimensions does a single neuron fill?" — the byte anatomy of a stored fact, and
//! where (if anywhere) dense dimensions actually live.
//!
//! The premise worth checking: a vector-DB item IS a dense D-dimensional coordinate (e.g. 1536
//! f32 = 6 KB), and you wonder if a neuron-db fact is similarly a partly-filled D-dim vector with
//! room to "fill more dimensions" and pack more in. It is not. dump() serializes a fact as
//! `flag \t text \t strength` — the stems/positions/content-word vectors are REBUILT from the text
//! on load, never stored. So a fact stores ZERO coordinates. This bench shows that, and measures
//! the one place real dimensions exist: the shared semantic space (DIM = 256), amortized over the
//! whole vocabulary rather than charged per fact.
//!
//! Run: cargo run --release --features "sqlite semantic" --example dimension_anatomy
use neuron_core::db::NeuronDB;
use neuron_core::Neuron;
use std::time::{SystemTime, UNIX_EPOCH};

const DIM: usize = 256; // semantic.rs const DIM — the only dense dimensionality in the system

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_dim_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}
fn rm(p: &str) { for e in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{}{}", p, e)); } }

fn main() {
    // A realistic small-fact corpus (the LLM-memory case): short declaratives, ~39 bytes each.
    let n = 20_000usize;
    let facts: Vec<String> = (0..n).map(|i| format!("the server{} holds value alpha{}", i, i)).collect();
    let text_bytes: usize = facts.iter().map(|f| f.len()).sum();

    // ---- 1) store anatomy: what a fact actually serializes to ----
    let mut neuron = Neuron::new(5_000_000);
    for f in &facts { neuron.observe(f); }
    let _ = neuron.recall("warm the derived index once"); // build stems/positions in memory
    let blob = neuron.dump();
    let serialized = blob.len();
    let avg_stems: f64 = neuron.episodes.iter().map(|e| e.s.len()).sum::<usize>() as f64 / n as f64;

    println!("== a single neuron's byte anatomy ({} short facts) ==\n", n);
    println!("-- on disk, per fact (what dump() persists) --");
    println!("  serialized total ......... {:>6.1} bytes/fact", serialized as f64 / n as f64);
    println!("    of which: text ......... {:>6.1} bytes/fact  ({:.0}%)", text_bytes as f64 / n as f64,
             100.0 * text_bytes as f64 / serialized as f64);
    println!("    of which: flag+strength+delims  {:>6.1} bytes/fact  ({:.0}%)",
             (serialized - text_bytes) as f64 / n as f64, 100.0 * (serialized - text_bytes) as f64 / serialized as f64);
    println!("    of which: DENSE COORDINATES ...... {:>6} bytes/fact   <-- there is no vector stored", 0);
    println!("\n-- the fact's 'dimensions' are SPARSE SYMBOLS, derived from the text, not stored --");
    println!("  active cue stems / fact .. {:.1} sparse stems (rebuilt on load; 0 disk bytes)", avg_stems);
    println!("  a vector-DB item, for contrast: {} dense f32 = {} bytes of pure coordinate", 1536, 1536 * 4);

    // ---- 2) where real dimensions live: the shared semantic space ----
    let path = tmp();
    let db = NeuronDB::open(&path, 5_000_000);
    for f in &facts { db.train_semantic(f); }
    let (vocab, _tokens, sem_bytes) = db.semantic_stats();

    println!("\n== the only dense dimensions in the system: the shared semantic space ==\n");
    println!("  dimensionality (DIM) ..... {}", DIM);
    println!("  vocabulary ............... {} unique words", vocab);
    println!("  resident bytes ........... {:.2} MB total", sem_bytes as f64 / 1e6);
    println!("  -> it is SHARED + vocabulary-bound, so the per-FACT share shrinks as facts grow:");
    println!("       amortized over {} facts: {:.1} bytes/fact (vs {} for a per-item embedding)",
             n, sem_bytes as f64 / n as f64, DIM * 4);
    println!("  per-word cost: f32 {} B  |  int8 (compact_semantic) {} B  ({}x)", DIM * 4, DIM, 4);

    println!("\n-- 'extend the dimensions' cost (linear in DIM), per vocabulary word --");
    for d in [64usize, 128, 256, 512, 1024] {
        println!("    DIM={:>4}:  f32 {:>5} B/word   int8 {:>4} B/word   (space scales x{:.2} vs 256)",
                 d, d * 4, d, d as f64 / 256.0);
    }
    println!("\n  Adding dense dimensions ADDS bytes (that's what a vector DB is). neuron-db is dense");
    println!("  precisely BECAUSE a fact carries no per-item vector. The way to fit more in a smaller");
    println!("  device is fewer bytes per fact (compress text / dedup / int8 / lower DIM), not more.");
    println!("  The legitimate 'pack many items into one vector' idea is SUPERPOSITION/HDC — separate,");
    println!("  lossy, and capacity-limited (~DIM/ln(DIM) items before crosstalk). See chat notes.");

    drop(db); rm(&path);
    println!("\n== done ==");
}
