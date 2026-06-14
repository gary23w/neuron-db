//! The "ultimate test": ingest several whole books, build the knowledge base + semantic
//! space, and measure storage, ingestion throughput, recall latency at scale, and the
//! quality of the distributional semantic space learned from the corpus.
//!
//! Run: cargo run --release --features "sqlite semantic" --example book_ingest -- <dir-of-.txt>
//! (or set NDB_BOOKS to the directory). Books: plain-text Project Gutenberg files.
use neuron_core::db::NeuronDB;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_book_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}
fn rm(p: &str) { let _ = std::fs::remove_file(p); let _ = std::fs::remove_file(format!("{}-wal", p)); let _ = std::fs::remove_file(format!("{}-shm", p)); }

/// strip the Project Gutenberg header/footer
fn strip(text: &str) -> &str {
    let s = text.find("*** START").and_then(|i| text[i..].find('\n').map(|j| i + j + 1)).unwrap_or(0);
    let e = text[s..].find("*** END").map(|j| s + j).unwrap_or(text.len());
    &text[s..e]
}
/// split prose into sentence-like facts (30..300 chars), whitespace-normalized
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        let c = if ch.is_whitespace() { ' ' } else { ch };
        if c == ' ' && cur.ends_with(' ') { continue; }
        cur.push(c);
        if matches!(ch, '.' | '!' | '?') {
            let s = cur.trim().to_string();
            if s.len() >= 30 && s.len() <= 300 && s.split_whitespace().count() >= 5 { out.push(s); }
            cur.clear();
        }
    }
    out
}
fn words(text: &str) -> usize { text.split_whitespace().count() }

fn main() {
    let dir = std::env::args().nth(1).or_else(|| std::env::var("NDB_BOOKS").ok())
        .unwrap_or_else(|| "books".into());
    let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap_or_else(|_| panic!("cannot read dir {}", dir))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |x| x == "txt")).collect();
    files.sort();
    if files.is_empty() { eprintln!("no .txt books in {}", dir); return; }

    let path = tmp();
    let db = NeuronDB::open(&path, 5_000_000);
    println!("== neuron-db book ingestion ({} books) ==\n", files.len());

    // ---- ingest + learn ----
    let (mut tot_words, mut tot_sents) = (0usize, 0usize);
    let mut train_s = 0f64; let mut obs_s = 0f64;
    for f in &files {
        let name = f.file_stem().unwrap().to_string_lossy().to_string();
        let raw = std::fs::read_to_string(f).unwrap_or_default();
        let body = strip(&raw);
        let w = words(body);
        let sents = sentences(body);
        let t = Instant::now(); db.train_semantic(body); train_s += t.elapsed().as_secs_f64();
        let t = Instant::now(); let stored = db.observe_many("library", &sents); obs_s += t.elapsed().as_secs_f64();
        tot_words += w; tot_sents += stored;
        println!("  {:<22} {:>8} words  {:>6} sentences", name, w, stored);
    }
    let (vocab, tokens, sem_bytes) = db.semantic_stats();
    let db_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
        + std::fs::metadata(format!("{}-wal", path)).map(|m| m.len()).unwrap_or(0);

    println!("\n-- corpus --");
    println!("  words ingested ...... {}", tot_words);
    println!("  facts (sentences) ... {}", tot_sents);
    println!("  ingest throughput ... {:.0} sentences/s ({:.0}k words/s), train {:.1}s + store {:.1}s",
             tot_sents as f64 / (train_s + obs_s), tot_words as f64 / (train_s + obs_s) / 1000.0, train_s, obs_s);
    println!("\n-- storage / provisioning --");
    println!("  sqlite on disk ...... {:.1} MB  ({:.0} bytes / sentence)", db_bytes as f64 / 1e6, db_bytes as f64 / tot_sents.max(1) as f64);
    println!("  semantic space ...... {:.1} MB  ({} words, {} tokens, dim 256)", sem_bytes as f64 / 1e6, vocab, tokens);
    println!("  total resident ...... ~{:.1} MB (store blob ~{:.1} MB + semantic {:.1} MB)",
             (db_bytes as f64 + sem_bytes as f64) / 1e6, db_bytes as f64 / 1e6, sem_bytes as f64 / 1e6);

    // ---- semantic space learned from the books (distributional knowledge) ----
    println!("\n-- semantic space: nearest words (learned only from the text) --");
    for w in ["whale", "ship", "love", "fear", "london", "murder", "science", "beauty"] {
        let near = db.semantic_neighbors(w, 6);
        if !near.is_empty() {
            let ws: Vec<String> = near.iter().map(|(x, s)| format!("{}({:.2})", x, s)).collect();
            println!("  {:<9} -> {}", w, ws.join(", "));
        } else { println!("  {:<9} -> (not in corpus)", w); }
    }

    // ---- recall over the whole library (one scope, 29k facts) ----
    println!("\n-- recall over {} facts in ONE scope --", tot_sents);
    let _ = db.get("library", "the white whale Moby Dick"); // warm the index
    let selective = ["Ahab and the white whale", "Sherlock Holmes deduction", "Victor Frankenstein creature",
                     "Elizabeth Bennet and Darcy", "Dorian Gray portrait"];
    let broad = ["the man and the woman", "the night and the day", "love and the heart",
                 "fear and the soul", "the great house"];
    let mut sl = Vec::new(); for _ in 0..4 { for p in &selective { let t = Instant::now(); let _ = db.recall("library", p); sl.push(t.elapsed().as_micros()); } }
    let mut bl = Vec::new(); for _ in 0..4 { for p in &broad { let t = Instant::now(); let _ = db.recall("library", p); bl.push(t.elapsed().as_micros()); } }
    sl.sort_unstable(); bl.sort_unstable();
    println!("  lexical, selective cue (proper nouns): p50 {} us", sl[sl.len()/2]);
    println!("  lexical, broad cue (common words):     p50 {} us", bl[bl.len()/2]);

    // ---- fuzzy / paraphrase recall via the semantic fallback (force a true paraphrase) ----
    println!("\n-- fuzzy recall (semantic fallback over the library) --");
    for q in ["a gigantic sea creature of the ocean", "a horrifying dreadful fiend", "a happy joyful matrimony"] {
        let t = Instant::now();
        let r = db.recall("library", q);
        let ms = t.elapsed().as_millis();
        match r { Some(h) => println!("  {:<38} -> \"{}\"  ({} ms)", q, h.fact.chars().take(64).collect::<String>(), ms),
                  None => println!("  {:<38} -> (no memory)  ({} ms)", q, ms) }
    }

    drop(db);
    // ---- reopen latency (note: the semantic space is in-memory; reopen reloads the store
    //      blob but must re-train the space) ----
    let db2 = NeuronDB::open(&path, 5_000_000);
    let t = Instant::now(); let _ = db2.get("library", "the white whale");
    println!("\n-- durability -- reopen + first recall of the {}-fact store: {:.2}s", tot_sents, t.elapsed().as_secs_f64());
    println!("   (the semantic space is resident-only; reopen reloads the store, re-train to restore meaning)");
    drop(db2); rm(&path);
    println!("\n== done ==");
}
