//! How much space does the per-document storage pattern actually use? (The chat-lab stores each
//! pasted document in its own scope and trains the shared semantic space on it.) This ingests N
//! synthetic documents, then breaks the footprint into the THREE places space goes:
//!   1. SQLite store    — the sentence text (cheap, ~text size)
//!   2. semantic space  — context vectors, SHARED across docs, ~1 KB per vocabulary word
//!   3. embedding cache — fuzzy-recall vectors, lazily ~1 KB per fact that gets recalled
//! Run: cargo run --release --features "sqlite semantic" --example doc_storage_space
use neuron_core::db::NeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};

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

    let mut facts = 0usize;
    for d in 0..ndocs {
        facts += db.observe_many(&format!("doc{}", d), &document(d, sents)); // own scope; trains the shared space
    }
    let db_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
        + std::fs::metadata(format!("{}-wal", path)).map(|m| m.len()).unwrap_or(0);
    let (vocab, tokens, sem_pre) = db.semantic_stats();              // cache still empty here

    // worst case for the cache: recall once from every document (caches that doc's fact vectors)
    for d in 0..ndocs { let _ = db.recall_blended(&format!("doc{}", d), "how does the service handle the request payload", 10); }
    let (_, _, sem_post) = db.semantic_stats();                      // now includes the embedding cache
    let cache_bytes = sem_post.saturating_sub(sem_pre);
    let total = db_bytes + sem_post as u64;

    println!("facts stored ......... {}", facts);
    println!("");
    println!("1) SQLite store ...... {:>7.2} MB   {:>5.0} B/fact   {:>6.1} KB/doc   (just the sentence text)",
             db_bytes as f64 / 1e6, db_bytes as f64 / facts.max(1) as f64, db_bytes as f64 / ndocs as f64 / 1e3);
    println!("2) semantic space .... {:>7.2} MB   {} vocab words, {} tokens   (SHARED across all {} docs)",
             sem_pre as f64 / 1e6, vocab, tokens, ndocs);
    println!("3) embedding cache ... {:>7.2} MB   {:>5.0} B/fact   (lazy: only facts that were recalled)",
             cache_bytes as f64 / 1e6, cache_bytes as f64 / facts.max(1) as f64);
    println!("   ------------------------------------------------------------");
    println!("   TOTAL resident .... {:>7.2} MB   {:>6.1} KB/doc   {:>5.0} B/fact",
             total as f64 / 1e6, total as f64 / ndocs as f64 / 1e3, total as f64 / facts.max(1) as f64);

    println!("\nwhere the space goes:");
    println!("  - the STORE is text — ~300 B/fact, linear, cheap; per-doc scopes add ~nothing vs one scope.");
    println!("  - the SEMANTIC SPACE is the big cost: ~{} B/word, shared, grows with VOCABULARY not doc count.", 256 * 4);
    println!("  - the EMBEDDING CACHE is ~1 KB per recalled fact (lazy; drop with clear_cache()).");
    println!("  mitigations: int8-quantize the space (4x), lower DIM, or persist+evict the space/cache.");
    rm(&path);
    println!("\n== done ==");
}
