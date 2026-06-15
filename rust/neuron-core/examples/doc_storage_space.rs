//! Space breakdown of the per-document storage pattern (each pasted document in its own scope,
//! one shared semantic space). Ingests N synthetic documents and splits the footprint three ways:
//!   1. SQLite store    — the sentence text (~text size, linear)
//!   2. semantic space  — context vectors, SHARED across docs, vocabulary-bound
//!   3. embedding cache — int8 fuzzy-recall vectors, lazily ~400 B per recalled fact
//! Then compact_semantic() int8-quantizes the space to show the read-mostly serving footprint.
//! Run: cargo run --release --features "sqlite semantic" --example doc_storage_space
use neuron_core::db::NeuronDB;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_docspace_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}
fn rm(p: &str) { let _ = std::fs::remove_file(p); let _ = std::fs::remove_file(format!("{}-wal", p)); let _ = std::fs::remove_file(format!("{}-shm", p)); }
fn code(mut x: usize) -> String { let mut s = String::new(); for _ in 0..6 { s.push((b'a' + (x % 26) as u8) as char); x /= 26; } s }

/// a synthetic document: `sents` sentences with a realistic mix of shared + unique vocabulary
/// (each sentence introduces a couple of distinctive tokens, so the vocabulary grows like real prose)
fn document(d: usize, sents: usize) -> Vec<String> {
    let topic = ["auth", "billing", "search", "cache", "queue", "graph", "render", "sync", "audit", "deploy"][d % 10];
    (0..sents).map(|i| format!(
        "The {topic} service handles request {} by validating the {} payload and appending ledger entry {} to the {topic} store.",
        code(d * 1000 + i), code(d * 7 + i * 13 + 5), d * 100 + i)).collect()
}

fn main() {
    let path = tmp();
    let db = NeuronDB::open(&path, 5_000_000);
    let ndocs = 100usize;
    let sents = 40usize;
    println!("== per-document storage footprint: {} documents x {} sentences ==\n", ndocs, sents);

    let docs: Vec<Vec<String>> = (0..ndocs).map(|d| document(d, sents)).collect();
    let in_bytes: usize = docs.iter().flatten().map(|s| s.len()).sum();   // raw text crossing into the db

    let t0 = Instant::now();
    let mut facts = 0usize;
    for (d, doc) in docs.iter().enumerate() {
        facts += db.observe_many(&format!("doc{}", d), doc);             // own scope; trains the shared space
    }
    let ingest = t0.elapsed();

    let db_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
        + std::fs::metadata(format!("{}-wal", path)).map(|m| m.len()).unwrap_or(0);
    let (vocab, tokens, sem_pre) = db.semantic_stats();              // cache still empty here

    // worst case for the cache: recall once from every document (caches that doc's fact vectors)
    let q = "how does the service handle the request payload";
    let t1 = Instant::now();
    let mut out_bytes = 0usize;
    for d in 0..ndocs {
        let hits = db.recall_blended(&format!("doc{}", d), q, 10);
        out_bytes += hits.iter().map(|h| h.fact.len() + h.value.len()).sum::<usize>(); // block sent to the LLM
    }
    let recall = t1.elapsed();
    let (_, _, sem_post) = db.semantic_stats();                      // now includes the embedding cache

    println!("throughput & transfer (large-block ingest, after int8):");
    println!("  ingest ... {:>6.1} MB/s   {:>8.0} facts/s   ({:.2} MB of text in {} ms)",
             in_bytes as f64 / 1e6 / ingest.as_secs_f64(), facts as f64 / ingest.as_secs_f64(),
             in_bytes as f64 / 1e6, ingest.as_millis());
    println!("  recall ... {:>6.1} us/query   {:>6.0} B/query transferred to the LLM   (top-10 block)",
             recall.as_micros() as f64 / ndocs as f64, out_bytes as f64 / ndocs as f64);
    println!();
    let cache_bytes = sem_post.saturating_sub(sem_pre);
    let total = db_bytes + sem_post as u64;

    println!("facts stored ......... {}", facts);
    println!();
    println!("1) SQLite store ...... {:>7.2} MB   {:>5.0} B/fact   {:>6.1} KB/doc   (just the sentence text)",
             db_bytes as f64 / 1e6, db_bytes as f64 / facts.max(1) as f64, db_bytes as f64 / ndocs as f64 / 1e3);
    println!("2) semantic space .... {:>7.2} MB   {} vocab words, {} tokens   (SHARED across all {} docs)",
             sem_pre as f64 / 1e6, vocab, tokens, ndocs);
    println!("3) embedding cache ... {:>7.2} MB   {:>5.0} B/fact   (lazy: only facts that were recalled)",
             cache_bytes as f64 / 1e6, cache_bytes as f64 / facts.max(1) as f64);
    println!("   ------------------------------------------------------------");
    println!("   TOTAL resident .... {:>7.2} MB   {:>6.1} KB/doc   {:>5.0} B/fact",
             total as f64 / 1e6, total as f64 / ndocs as f64 / 1e3, total as f64 / facts.max(1) as f64);

    // production "serve" mode: int8-compact the context vectors (a later observe re-expands)
    db.compact_semantic();
    let (_, _, sem_compact) = db.semantic_stats();
    let total_compact = db_bytes + sem_compact as u64;
    println!("\nafter compact_semantic() (int8 context vectors, read-mostly serving):");
    println!("   semantic + cache .. {:>7.2} MB  (was {:.2} MB)", sem_compact as f64 / 1e6, sem_post as f64 / 1e6);
    println!("   TOTAL resident .... {:>7.2} MB  (was {:.2} MB)   {:>6.1} KB/doc",
             total_compact as f64 / 1e6, total as f64 / 1e6, total_compact as f64 / ndocs as f64 / 1e3);

    println!("\nwhere the space goes:");
    println!("  - the STORE is text: ~450 B/fact, linear; per-doc scopes cost ~nothing vs one scope.");
    println!("  - the SEMANTIC SPACE (context vectors) is the big cost; int8 via compact() cuts it ~4x.");
    println!("  - the EMBEDDING CACHE is int8, ~400 B per recalled fact (lazy; clear_cache() to drop).");
    rm(&path);
    println!("\n== done ==");
}
