//! Coder-memory A/B benchmark: how well does neuron-db serve the memory of a developer who codes with an
//! AI assistant? We build a realistic, DETERMINISTIC "codebase memory" corpus (file locations, commands,
//! configs, conventions, dependency graphs) and a ground-truthed query set across five slices that each
//! honestly favor a DIFFERENT approach — so the comparison can't be dismissed as rigged:
//!
//!   direct      — query shares words with the stored fact            (everyone should do well)
//!   exact       — query is a literal substring: a path or command    (favors grep)
//!   paraphrase  — query uses a SYNONYM not in the stored fact         (favors vector / semantic)
//!   disambig    — two near-identical facts differ by one key term     (favors ranked overlap)
//!   chain       — a multi-hop "what does X ultimately depend on?"     (favors associative recall)
//!
//! Approaches A/B'd, all given the SAME facts (and the semantic ones the SAME background doc corpus to
//! train on, as a real deployment would have):
//!   grep            — word-overlap substring search over the fact texts (a fair grep, not a strawman)
//!   vector          — pure cosine over a Random-Indexing embedding of every fact ("just use a vector DB")
//!   neuron-lexical  — NeuronDB::recall_many (inverted-index cue overlap + df gating + specificity)
//!   neuron-blended  — NeuronDB::recall_blended (lexical + semantic)
//!   neuron-assoc    — NeuronDB::recall_associative (spreading activation over shared entities)
//!
//! Accuracy-first scoring: hit@1, hit@3, MRR per slice + overall, plus latency and the context-cost
//! contrast vs dumping the whole memory into the prompt.
//!
//! Run: cargo run --release --features "sqlite semantic" --example coder_bench

use neuron_core::db::NeuronDB;
use neuron_core::semantic::SemanticSpace;
use std::time::Instant;

struct Fact { text: String }
struct Query { slice: &'static str, target: String, query: String }

fn norm(s: &str) -> String { s.trim().to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ") }

/// The synthetic project: modules, where they live, what they do, and a synonym a developer might use.
/// (module, path, role-phrase, synonym-for-the-module-word)
const MODULES: &[(&str, &str, &str, &str)] = &[
    ("authentication", "src/auth/login.rs",        "verifies user credentials",     "sign-in"),
    ("billing",        "src/billing/mod.rs",       "creates and charges invoices",  "payments"),
    ("scheduler",      "src/jobs/scheduler.rs",    "runs background jobs",          "cron"),
    ("router",         "src/http/router.rs",       "maps urls to handlers",         "routing"),
    ("cache",          "src/store/cache.rs",       "memoizes hot lookups",          "memoization"),
    ("telemetry",      "src/obs/telemetry.rs",     "emits metrics and traces",      "observability"),
    ("migrations",     "src/db/migrate.rs",        "evolves the database schema",   "schema"),
    ("notifications",  "src/notify/email.rs",      "sends user emails",             "messaging"),
    ("search",         "src/search/index.rs",      "indexes and queries documents", "retrieval"),
    ("uploads",        "src/media/upload.rs",      "stores user files",             "attachments"),
];

/// (action-phrase, command, synonym-action)
const COMMANDS: &[(&str, &str, &str)] = &[
    ("deploy to production",  "make deploy-prod",      "ship to production"),
    ("run the test suite",    "cargo test --all",      "execute the tests"),
    ("start the dev server",  "npm run dev",           "launch the local server"),
    ("apply database migrations", "make migrate-up",   "update the database schema"),
    ("build the release",     "cargo build --release", "compile for release"),
    ("lint the codebase",     "make lint",             "check code style"),
];

/// (service, setting, value) — two services share each setting name, so the only disambiguator is the service.
const CONFIGS: &[(&str, &str, &str)] = &[
    ("staging", "api base url", "https://api.staging.acme.dev"),
    ("production", "api base url", "https://api.acme.com"),
    ("staging", "database url", "postgres://stg-db.acme.dev:5432/app"),
    ("production", "database url", "postgres://prod-db.acme.com:5432/app"),
    ("staging", "redis url", "redis://stg-cache.acme.dev:6379"),
    ("production", "redis url", "redis://prod-cache.acme.com:6379"),
];

const CONVENTIONS: &[(&str, &str)] = &[
    ("database columns use snake_case naming", "how should i name a new database column"),
    ("all recoverable errors use the AppError type", "what error type do we return on failure"),
    ("integration tests live in the tests directory", "where do new integration tests go"),
    ("public api responses are camelCase json", "what casing do api responses use"),
    ("secrets are read from the environment, never committed", "where should an api key be stored"),
    ("feature work happens on branches off develop", "which branch do i start a feature from"),
];

/// Dependency chains a -> b -> c (the LAST element is the ultimate dependency the chain query asks for).
const CHAINS: &[(&str, &str, &str)] = &[
    ("checkout", "billing", "the stripe client"),
    ("dashboard", "search", "the elasticsearch cluster"),
    ("onboarding", "notifications", "the email provider"),
];

fn build_corpus() -> (Vec<Fact>, Vec<Query>, Vec<String>) {
    let mut facts: Vec<Fact> = Vec::new();
    let mut queries: Vec<Query> = Vec::new();
    let mut background: Vec<String> = Vec::new(); // doc corpus the semantic approaches train on

    let mut push_fact = |facts: &mut Vec<Fact>, t: String| { facts.push(Fact { text: t }); };

    // modules: location facts + direct/paraphrase queries
    for (m, path, role, syn) in MODULES {
        let f = format!("the {m} module lives in {path} and {role}");
        push_fact(&mut facts, f.clone());
        queries.push(Query { slice: "direct", target: f.clone(), query: format!("where does the {m} module live") });
        // paraphrase: use the synonym, which is NOT in the stored fact
        queries.push(Query { slice: "paraphrase", target: f.clone(), query: format!("which file holds the {syn} code") });
        // exact: the path itself is a literal substring of the fact
        queries.push(Query { slice: "exact", target: f.clone(), query: (*path).to_string() });
        // background doc bridges the module word and its synonym so the space can learn the link
        background.push(format!("the {m} system is the {syn} component; {m} and {syn} refer to the same {role} feature"));
    }

    // commands
    for (action, cmd, syn) in COMMANDS {
        let f = format!("to {action} run {cmd}");
        push_fact(&mut facts, f.clone());
        queries.push(Query { slice: "direct", target: f.clone(), query: format!("what command do i run to {action}") });
        queries.push(Query { slice: "paraphrase", target: f.clone(), query: format!("how do i {syn}") });
        queries.push(Query { slice: "exact", target: f.clone(), query: (*cmd).to_string() });
        background.push(format!("to {action} is to {syn}; both mean running {cmd}"));
    }

    // configs: disambiguation — two facts share the setting name, differ only by environment
    for (svc, setting, val) in CONFIGS {
        let f = format!("the {svc} {setting} is {val}");
        push_fact(&mut facts, f.clone());
        queries.push(Query { slice: "disambig", target: f.clone(), query: format!("what is the {svc} {setting}") });
    }

    // conventions
    for (statement, q) in CONVENTIONS {
        let f = format!("convention: {statement}");
        push_fact(&mut facts, f.clone());
        queries.push(Query { slice: "paraphrase", target: f.clone(), query: (*q).to_string() });
    }

    // dependency chains: a -> b -> c. Store the hop facts; chain query asks for the terminal dependency.
    for (a, b, c) in CHAINS {
        let hop1 = format!("the {a} module depends on the {b} module");
        let hop2 = format!("the {b} module depends on {c}");
        push_fact(&mut facts, hop1.clone());
        push_fact(&mut facts, hop2.clone());
        // the answer fact is the terminal hop (it names c); a good approach surfaces it from an a-rooted question
        queries.push(Query { slice: "chain", target: hop2.clone(), query: format!("what does the {a} module ultimately depend on") });
        background.push(format!("{a} relies on {b}, and {b} relies on {c}, so {a} transitively needs {c}"));
    }

    (facts, queries, background)
}

// ---- approaches: each returns a ranked Vec of fact texts for a query ----

fn grep_rank<'a>(facts: &'a [Fact], query: &str, k: usize) -> Vec<&'a str> {
    let qwords: Vec<String> = query.to_lowercase().split_whitespace().filter(|w| w.len() >= 3).map(|w| w.to_string()).collect();
    let mut scored: Vec<(usize, &str)> = facts.iter().map(|f| {
        let lf = f.text.to_lowercase();
        let hits = qwords.iter().filter(|w| lf.contains(w.as_str())).count();
        (hits, f.text.as_str())
    }).filter(|(h, _)| *h > 0).collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0)); // most query-words-matched first; stable keeps corpus order on ties
    scored.into_iter().take(k).map(|(_, t)| t).collect()
}

fn vector_rank<'a>(sp: &SemanticSpace, fact_texts: &'a [String], query: &str, k: usize) -> Vec<&'a str> {
    sp.rank(query, fact_texts).into_iter().take(k).map(|(i, _)| fact_texts[i].as_str()).collect()
}

/// rank of `target` (1-based) within a ranked list, or 0 if absent.
fn rank_of(ranked: &[&str], target: &str) -> usize {
    let t = norm(target);
    ranked.iter().position(|r| norm(r) == t).map(|p| p + 1).unwrap_or(0)
}

#[derive(Default, Clone)]
struct Tally { n: usize, hit1: usize, hit3: usize, mrr: f64, micros: u128 }
impl Tally {
    fn add(&mut self, rank: usize, micros: u128) {
        self.n += 1;
        if rank == 1 { self.hit1 += 1; }
        if rank >= 1 && rank <= 3 { self.hit3 += 1; }
        if rank >= 1 { self.mrr += 1.0 / rank as f64; }
        self.micros += micros;
    }
    fn row(&self) -> String {
        let n = self.n.max(1) as f64;
        format!("{:>6.1}% {:>6.1}% {:>6.3} {:>7.0}us",
            100.0 * self.hit1 as f64 / n, 100.0 * self.hit3 as f64 / n, self.mrr / n, self.micros as f64 / n)
    }
}

fn main() {
    let k = 3usize;
    let (facts, queries, background) = build_corpus();
    let fact_texts: Vec<String> = facts.iter().map(|f| f.text.clone()).collect();
    let total_bytes: usize = fact_texts.iter().map(|s| s.len() + 1).sum();

    println!("== neuron-db coder-memory A/B benchmark ==");
    println!("{} facts, {} queries, {} background docs; scoring hit@1 / hit@3 / MRR @k={}\n",
        facts.len(), queries.len(), background.len(), k);

    // neuron store: observe every fact, train its semantic space on the background docs
    let path = std::env::temp_dir().join("ndb_coder_bench.sqlite");
    let p = path.to_string_lossy().to_string();
    for suffix in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{p}{suffix}")); }
    let db = NeuronDB::open(&p, 100_000);
    db.observe_many("proj", &fact_texts);
    for d in &background { db.train_semantic(d); }

    // standalone semantic space for the pure-vector baseline (same training material: background + facts)
    let mut sp = SemanticSpace::new();
    for d in &background { sp.train(d); }
    for f in &fact_texts { sp.train(f); }

    let slices = ["direct", "exact", "paraphrase", "disambig", "chain"];
    let approaches = ["grep", "vector", "neuron-lexical", "neuron-blended", "neuron-assoc"];
    // tallies[approach][slice]
    let mut tallies: Vec<Vec<Tally>> = vec![vec![Tally::default(); slices.len()]; approaches.len()];
    let mut overall: Vec<Tally> = vec![Tally::default(); approaches.len()];

    let slice_idx = |s: &str| slices.iter().position(|x| *x == s).unwrap();

    for q in &queries {
        let si = slice_idx(q.slice);
        // grep
        { let t = Instant::now(); let r = grep_rank(&facts, &q.query, k); let us = t.elapsed().as_micros();
          let rk = rank_of(&r, &q.target); tallies[0][si].add(rk, us); overall[0].add(rk, us); }
        // vector
        { let t = Instant::now(); let r = vector_rank(&sp, &fact_texts, &q.query, k); let us = t.elapsed().as_micros();
          let rk = rank_of(&r, &q.target); tallies[1][si].add(rk, us); overall[1].add(rk, us); }
        // neuron-lexical
        { let t = Instant::now(); let r = db.recall_many("proj", &q.query, k); let us = t.elapsed().as_micros();
          let v: Vec<&str> = r.iter().map(|h| h.fact.as_str()).collect();
          let rk = rank_of(&v, &q.target); tallies[2][si].add(rk, us); overall[2].add(rk, us); }
        // neuron-blended
        { let t = Instant::now(); let r = db.recall_blended("proj", &q.query, k); let us = t.elapsed().as_micros();
          let v: Vec<&str> = r.iter().map(|h| h.fact.as_str()).collect();
          let rk = rank_of(&v, &q.target); tallies[3][si].add(rk, us); overall[3].add(rk, us); }
        // neuron-assoc (spreading activation, 2 hops)
        { let t = Instant::now(); let r = db.recall_associative("proj", &q.query, k, 2); let us = t.elapsed().as_micros();
          let v: Vec<&str> = r.iter().map(|h| h.fact.as_str()).collect();
          let rk = rank_of(&v, &q.target); tallies[4][si].add(rk, us); overall[4].add(rk, us); }
    }

    // per-slice tables
    for (si, slice) in slices.iter().enumerate() {
        let n = queries.iter().filter(|q| q.slice == *slice).count();
        println!("--- slice: {slice}  ({n} queries) ---");
        println!("  {:<16} {:>7} {:>7} {:>7} {:>9}", "approach", "hit@1", "hit@3", "MRR", "latency");
        for (ai, a) in approaches.iter().enumerate() {
            println!("  {:<16} {}", a, tallies[ai][si].row());
        }
        println!();
    }

    println!("=== OVERALL (all {} queries) ===", queries.len());
    println!("  {:<16} {:>7} {:>7} {:>7} {:>9}", "approach", "hit@1", "hit@3", "MRR", "latency");
    for (ai, a) in approaches.iter().enumerate() {
        println!("  {:<16} {}", a, overall[ai].row());
    }

    // context-cost contrast vs the "dump everything into the prompt" habit
    let avg_recall_bytes = {
        let mut b = 0usize; let mut n = 0usize;
        for q in &queries { if let Some(h) = db.recall_many("proj", &q.query, 1).first() { b += h.fact.len(); n += 1; } }
        if n > 0 { b / n } else { 0 }
    };
    println!("\n--- context cost ---");
    println!("  full-context dump : ships ALL {} facts = {} bytes (~{} tokens) every query",
        facts.len(), total_bytes, total_bytes / 4);
    println!("  neuron targeted   : ships ~1 fact = ~{} bytes (~{} tokens) per query  ({}x less)",
        avg_recall_bytes, avg_recall_bytes / 4, if avg_recall_bytes > 0 { total_bytes / avg_recall_bytes.max(1) } else { 0 });

    for suffix in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{p}{suffix}")); }
}
