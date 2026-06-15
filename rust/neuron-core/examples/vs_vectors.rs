//! Head-to-head harness (neuron-db side). Scored IDENTICALLY to the dense-vector side
//! (vs_vectors_vec.py) by RETRIEVAL IDENTITY: each fact has an id (its line number); a query's
//! gold_id names the one correct fact; a hit = gold_id is in the returned top-k. This is
//! system-output-agnostic and fair to both engines. We additionally report value-exactness
//! (does neuron-db hand back the literal stored value char-for-char) as a SEPARATE bonus column,
//! never folded into the head-to-head hit, and an abstention flag (recall() returned None) so the
//! no-answer class can be scored symmetrically against the vector side.
//!
//! Modes:
//!   lexical : recall_many(k)     — the default inverted-index path (no semantic)
//!   blended : recall_blended(k)  — lexical + the std-only Random-Indexing semantic re-rank
//! Optional corpus.txt trains the semantic space first (blended only) so open-vocabulary
//! paraphrase has grounded co-occurrence. Without it, blended only knows the fact text itself —
//! reported honestly as a separate run.
//!
//! Run: cargo run --release --features "sqlite semantic" --example vs_vectors -- \
//!        <facts.tsv> <queries.tsv> <lexical|blended> [corpus.txt]
//! facts.tsv   : id<TAB>fact_text   (id = 0-based; must match the file the vector side reads)
//! queries.tsv : query<TAB>gold_id<TAB>gold_value<TAB>class   (gold_id = NONE for no-answer)
//! stdout      : "#facts ... #db_bytes ..." header, then per query:
//!               class<TAB>mode<TAB>lat_ns<TAB>hit1<TAB>hit3<TAB>value_exact<TAB>abstain
use neuron_core::db::NeuronDB;
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_vs_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 { eprintln!("usage: vs_vectors <facts.tsv> <queries.tsv> <lexical|blended> [corpus.txt]"); std::process::exit(2); }
    let (facts_path, queries_path, mode) = (&a[1], &a[2], a[3].as_str());
    let corpus = a.get(4).cloned();
    let k = 3usize;

    // id<TAB>fact ; preserve order, build text->id for retrieval-identity scoring.
    let mut facts: Vec<(usize, String)> = Vec::new();
    let mut id_of: HashMap<String, usize> = HashMap::new();
    for line in std::fs::read_to_string(facts_path).unwrap().lines() {
        let l = line.trim_end();
        if l.is_empty() { continue; }
        let mut it = l.splitn(2, '\t');
        let id: usize = it.next().unwrap().parse().unwrap();
        let text = it.next().unwrap_or("").to_string();
        id_of.insert(text.clone(), id);
        facts.push((id, text));
    }
    let fact_texts: Vec<String> = facts.iter().map(|(_, t)| t.clone()).collect();

    let queries: Vec<(String, String, String, String)> = std::fs::read_to_string(queries_path).unwrap()
        .lines().filter(|l| !l.trim().is_empty()).map(|l| {
            let p: Vec<&str> = l.split('\t').collect();
            (p[0].to_string(), p.get(1).unwrap_or(&"NONE").to_string(),
             p.get(2).unwrap_or(&"").to_string(), p.get(3).unwrap_or(&"?").to_string())
        }).collect();

    let dbpath = tmp();
    let db = NeuronDB::open(&dbpath, 10_000_000);
    if mode == "blended" {
        if let Some(cp) = &corpus { if let Ok(t) = std::fs::read_to_string(cp) { db.train_semantic(&t); } }
    }
    db.observe_many("bench", &fact_texts);
    db.compact_semantic();
    db.flush_all();
    let _ = db.get("bench", &queries[0].0); // warm

    // Real on-disk SQLite footprint = main db + WAL + shm (data lives in the WAL until checkpoint).
    let sz = |p: String| std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
    let bytes = sz(dbpath.clone()) + sz(format!("{}-wal", dbpath)) + sz(format!("{}-shm", dbpath));
    let src_text: usize = fact_texts.iter().map(|t| t.len()).sum(); // shared source text, counted on both sides
    let (sv, _st, sb) = db.semantic_stats();
    println!("#facts {} #db_bytes {} #bytes_per_fact {:.1} #src_text_bytes {} #sem_vocab {} #sem_bytes {} #mode {}",
             facts.len(), bytes, bytes as f64 / facts.len() as f64, src_text, sv, sb, mode);

    for (q, gold_id, gold_val, class) in &queries {
        let t = Instant::now();
        let hits = if mode == "blended" { db.recall_blended("bench", q, k) } else { db.recall_many("bench", q, k) };
        let ns = t.elapsed().as_nanos();
        let ids: Vec<i64> = hits.iter().map(|h| id_of.get(&h.fact).map(|&x| x as i64).unwrap_or(-1)).collect();
        let g: i64 = gold_id.parse().unwrap_or(-1);
        let hit1 = (g >= 0 && ids.first() == Some(&g)) as u8;
        let hit3 = (g >= 0 && ids.contains(&g)) as u8;
        let value_exact = (!gold_val.is_empty() && hits.first().map(|h| h.value == *gold_val).unwrap_or(false)) as u8;
        let abstain = db.recall("bench", q).is_none() as u8; // for the no-answer class
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{}", class, mode, ns, hit1, hit3, value_exact, abstain);
    }
    let _ = std::fs::remove_file(&dbpath);
}
