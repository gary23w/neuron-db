//! neuron-ffi — a C ABI staticlib over neuron-core, shaped so a native host can LINK the memory
//! engine in rather than spawning `neuron.exe` once per memory operation.
//!
//! The surface is deliberately tiny: ONE entry point that takes the same argv the CLI takes and
//! returns the same bytes the CLI writes to **stdout**. Hosts that already parse the CLI's text
//! (`# scope:` export headers, `facts: N … created: MS updated: MS`, one-fact-per-line assoc, …)
//! keep their parsers untouched.
//!
//!     char *neuron_call   (const char *db, size_t max, int argc, const char *const *argv);
//!     char *neuron_call_ex(const char *db, size_t max, int argc, const char *const *argv, int *exit_code);
//!     void  neuron_free   (char *p);
//!
//! FIDELITY CONTRACT — the reason this file hand-renders instead of calling into cli.rs:
//!   * stdout ONLY. The CLI splits data (stdout) from chatter (stderr); a host reading `stdout` of a
//!     subprocess sees exactly what `neuron_call` returns. Commands whose human output is entirely
//!     `eprintln!` (notably non-`--json` `import`) therefore return an EMPTY string here — matching
//!     the subprocess exactly, quirks included. Do not "fix" that: a host parser calibrated against
//!     the subprocess would break.
//!   * exit codes ride out through `neuron_call_ex`. The CLI uses exit 3 for "no match" on
//!     get/recall/assoc/chain and exit 2 for usage/unknown-command; hosts branch on it.
//!   * NOTHING here may call `std::process::exit` or panic across the boundary. The CLI exits on a
//!     usage error and on a contended lock — in a linked library either would kill the host. Every
//!     such path returns an exit code instead, and the whole dispatch runs under `catch_unwind`.
//!
//! Ops with no `NeuronOp` variant (import, export, trust_get, trust_reward) are rendered here
//! against `NeuronDB` directly, mirroring cli.rs line for line.

use neuron_core::db::NeuronDB;
use neuron_core::json_escape as esc;
use neuron_core::op::{apply, NeuronOp, OpResult};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fmt::Write as _;
use std::os::raw::{c_char, c_int};
use std::sync::{Arc, Mutex, OnceLock};

// ---------------------------------------------------------------------------
// optional open-handle cache
// ---------------------------------------------------------------------------
// `NeuronDB::open` builds one WAL connection PER SHARD (up to 16), so re-opening on every call is
// the dominant per-op cost once the process spawn is gone. Caching the handle removes it.
//
// OFF by default, because it is only sound when this process is the ONLY writer to the file: a
// cached handle keeps an in-memory scope cache that a second writer (a stray `neuron.exe`, another
// process) would silently invalidate. A host that has fully migrated off the subprocess can turn it
// on with `neuron_cache(1)`. Durability is unchanged either way — `flush_every = 1` means every
// observe is durable immediately, cache or no cache.
static CACHE: OnceLock<Mutex<HashMap<(String, usize), Arc<NeuronDB>>>> = OnceLock::new();
static CACHE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn open_db(path: &str, max: usize) -> Arc<NeuronDB> {
    if !CACHE_ON.load(std::sync::atomic::Ordering::Relaxed) {
        return Arc::new(NeuronDB::open(path, max));
    }
    let m = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = match m.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(), // a poisoned cache is still a usable map
    };
    g.entry((path.to_string(), max))
        .or_insert_with(|| Arc::new(NeuronDB::open(path, max)))
        .clone()
}

// ---------------------------------------------------------------------------
// argv pre-parse — mirrors cli.rs's global-flag loop
// ---------------------------------------------------------------------------
struct Globals {
    db: String,
    max: usize,
    json: bool,
    trust_rank: bool,
    hops: Option<usize>,
    k: Option<usize>,
    pos: Vec<String>,
}

/// Split global flags off the front of argv exactly the way cli.rs does, so a host can hand over the
/// argv array it already builds for the subprocess verbatim (minus argv[0]).
///
/// `--db` / `--max` inside argv override the function parameters — that is what lets an existing
/// `{ "--db", path, "assoc", scope, query, "--trust" }` array work unchanged.
///
/// `--hops` / `--k` are an FFI-ONLY addition with no CLI counterpart. The CLI takes those two
/// through the environment (NEURON_HOPS / NEURON_K) because a subprocess gets its own env; in a
/// linked host the environment is process-global and shared by every concurrent caller, so a host
/// running turns in parallel cannot safely set them per call. These flags are the per-call form.
/// The env vars remain the fallback, so a host that has not migrated keeps working.
/// NOTE: passing them to the real `neuron` binary would append them to the query text — they are
/// only valid against this shim.
fn parse_globals(db0: &str, max0: usize, raw: &[String]) -> Globals {
    let mut g = Globals { db: db0.to_string(), max: max0, json: false, trust_rank: false, hops: None, k: None, pos: Vec::new() };
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--db" => { g.db = raw.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--max" => { if let Some(v) = raw.get(i + 1).and_then(|s| s.parse().ok()) { g.max = v; } i += 2; }
            "--hops" => { g.hops = raw.get(i + 1).and_then(|s| s.parse().ok()); i += 2; }
            "--k" => { g.k = raw.get(i + 1).and_then(|s| s.parse().ok()); i += 2; }
            "--json" => { g.json = true; i += 1; }
            "--trust" => { g.trust_rank = true; i += 1; }
            "--force" => { i += 1; }                      // no advisory lock in-process; accept + ignore
            "--secret" | "--keyfile" => { i += 2; }       // secure-* is not part of the linked surface
            "--" => { while i < raw.len() { g.pos.push(raw[i].clone()); i += 1; } }
            _ => { g.pos.push(raw[i].clone()); i += 1; }
        }
    }
    g
}

// ---------------------------------------------------------------------------
// dispatch — one command, returning (stdout_text, exit_code)
// ---------------------------------------------------------------------------
fn dispatch(db0: &str, max0: usize, raw: &[String]) -> (String, i32) {
    let g = parse_globals(db0, max0, raw);
    let (db, max, json) = (g.db.as_str(), g.max, g.json);
    let pos = &g.pos;
    let mut o = String::new();

    let cmd = match pos.first() {
        Some(c) => c.clone(),
        None => return (String::new(), 2), // bare invocation prints help to stderr; stdout empty
    };
    let scope = pos.get(1).cloned().unwrap_or_default();
    let rest = || pos.get(2..).map(|s| s.join(" ")).unwrap_or_default();
    // every scoped command exits 2 with an empty stdout when the scope is missing (cli.rs need_scope)
    macro_rules! need_scope { () => { if scope.is_empty() { return (String::new(), 2); } }; }

    match cmd.as_str() {
        "observe" => {
            need_scope!();
            let d = open_db(db, max);
            let n = apply(&*d, NeuronOp::Observe { scope: scope.clone(), text: rest() }).wrote();
            if json { let _ = writeln!(o, "{{\"wrote\":{}}}", n); } else { let _ = writeln!(o, "stored {} fact(s)", n); }
            (o, 0)
        }
        "get" => {
            need_scope!();
            let d = open_db(db, max);
            let v = apply(&*d, NeuronOp::RecallOne { scope: scope.clone(), query: rest() }).hit().map(|h| h.value);
            if json {
                let _ = writeln!(o, "{{\"value\":{}}}", v.as_deref().map(|s| format!("\"{}\"", esc(s))).unwrap_or("null".into()));
            } else if let Some(s) = &v {
                let _ = writeln!(o, "{}", s);
            }
            let code = if v.is_none() { 3 } else { 0 };
            (o, code)
        }
        "recall" => {
            need_scope!();
            let d = open_db(db, max);
            let h = apply(&*d, NeuronOp::RecallOne { scope: scope.clone(), query: rest() }).hit();
            match &h {
                Some(h) => {
                    if json {
                        let _ = writeln!(o, "{{\"fact\":\"{}\",\"value\":\"{}\",\"coverage\":{:.3}}}", esc(&h.fact), esc(&h.value), h.coverage);
                    } else {
                        let _ = writeln!(o, "value:    {}\nfact:     {}\ncoverage: {:.0}%", h.value, h.fact, h.coverage * 100.0);
                    }
                }
                None => { if json { let _ = writeln!(o, "{{\"fact\":null}}"); } }
            }
            let code = if h.is_none() { 3 } else { 0 };
            (o, code)
        }
        // Spreading-activation recall. NOTE the asymmetry, copied deliberately from cli.rs: the plain
        // path goes through `apply`, which CLAMPS k to 1..64 and hops to 1..32; the `--trust` path
        // calls the store directly and is NOT clamped (so a "spread to saturation" hop count really
        // does saturate). Reproducing that is load-bearing — a host that passes hops=1024 gets a
        // different traversal on the two paths today, and must keep getting it.
        "assoc" => {
            need_scope!();
            let d = open_db(db, max);
            // per-call flags win; env is the un-migrated-host fallback; then the CLI's defaults
            let hops: usize = g.hops.or_else(|| env_num("NEURON_HOPS")).unwrap_or(3);
            let k: usize = g.k.or_else(|| env_num("NEURON_K")).unwrap_or(10);
            let hits = if g.trust_rank {
                d.recall_assoc_trusted(&scope, &rest(), k, hops)
            } else {
                apply(&*d, NeuronOp::RecallAssoc { scope: scope.clone(), query: rest(), k, hops }).assoc()
            };
            if json {
                let items: Vec<String> = hits.iter()
                    .map(|h| format!("{{\"fact\":\"{}\",\"act\":{:.4},\"seed\":{}}}", esc(&h.fact), h.act, h.seed))
                    .collect();
                let _ = writeln!(o, "{{\"hits\":[{}]}}", items.join(","));
            } else {
                for h in &hits { let _ = writeln!(o, "{}", h.fact); }
            }
            let code = if hits.is_empty() { 3 } else { 0 };
            (o, code)
        }
        "chain" => {
            need_scope!();
            let d = open_db(db, max);
            let start = pos.get(2).cloned().unwrap_or_default();
            let path: Vec<String> = pos.get(3..).map(|s| s.to_vec()).unwrap_or_default();
            if start.is_empty() || path.is_empty() { return (String::new(), 2); }
            let (value, trail) = d.recall_chain(&scope, &start, &path);
            if json {
                let tr: Vec<String> = trail.iter().map(|s| format!("\"{}\"", esc(s))).collect();
                let v = value.as_deref().map(|s| format!("\"{}\"", esc(s))).unwrap_or_else(|| "null".into());
                let _ = writeln!(o, "{{\"value\":{},\"trail\":[{}]}}", v, tr.join(","));
            } else {
                match &value {
                    Some(v) => { let _ = writeln!(o, "{}  (via {})", v, trail.join(" -> ")); }
                    None => { let _ = writeln!(o, "(chain broke after: {})", trail.join(" -> ")); }
                }
            }
            let code = if value.is_none() { 3 } else { 0 };
            (o, code)
        }
        "reinforce" => {
            need_scope!();
            let d = open_db(db, max);
            let topic = pos.get(2).cloned().unwrap_or_default();
            let feeling = pos.get(3..).map(|s| s.join(" ")).unwrap_or_default();
            if topic.is_empty() || feeling.is_empty() { return (String::new(), 2); }
            let (strength, created) = d.note_stance(&scope, &topic, &feeling);
            if json {
                let _ = writeln!(o, "{{\"topic\":\"{}\",\"feeling\":\"{}\",\"strength\":{:.3},\"new\":{}}}", esc(&topic), esc(&feeling), strength, created);
            } else {
                let _ = writeln!(o, "{} '{}' [{}] strength {:.2}", if created { "reinforced (new)" } else { "reinforced" }, topic, feeling, strength);
            }
            (o, 0)
        }
        "strengthen" => {
            need_scope!();
            let d = open_db(db, max);
            let m = pos.get(2..).map(|s| s.join(" ")).unwrap_or_default();
            if m.trim().is_empty() { return (String::new(), 2); }
            if let OpResult::Strengthened(hit) = apply(&*d, NeuronOp::Strengthen { scope: scope.clone(), matching: m, bump: 1.0 }) {
                if json { let _ = writeln!(o, "{{\"strengthened\":{}}}", hit); } else { let _ = writeln!(o, "strengthened {} fact(s)", hit); }
            }
            (o, 0)
        }
        "forget" => {
            need_scope!();
            let d = open_db(db, max);
            let m = pos.get(2..).map(|s| s.join(" ")).filter(|s| !s.is_empty());
            if let OpResult::Forgot { forgot, remaining } = apply(&*d, NeuronOp::Forget { scope: scope.clone(), matching: m }) {
                if json { let _ = writeln!(o, "{{\"forgot\":{},\"remaining\":{}}}", forgot, remaining); }
                else { let _ = writeln!(o, "forgot {}, {} remaining", forgot, remaining); }
            }
            (o, 0)
        }
        "stats" => {
            need_scope!();
            let d = open_db(db, max);
            if let OpResult::Stats(s) = apply(&*d, NeuronOp::Stats { scope: scope.clone() }) {
                if json {
                    let _ = writeln!(o, "{{\"facts\":{},\"max_facts\":{},\"created\":{},\"updated\":{},\"turns\":{}}}", s.facts, s.max_facts, s.created, s.updated, s.turns);
                } else {
                    let _ = writeln!(o, "facts:   {}\nmax:     {}\nturns:   {}\ncreated: {}\nupdated: {}", s.facts, s.max_facts, s.turns, s.created, s.updated);
                }
            }
            (o, 0)
        }
        "list" => {
            let d = open_db(db, max);
            if let OpResult::Scopes(ids) = apply(&*d, NeuronOp::List) {
                if json {
                    let items: Vec<String> = ids.iter().map(|s| format!("\"{}\"", esc(s))).collect();
                    let _ = writeln!(o, "{{\"scopes\":[{}]}}", items.join(","));
                } else {
                    for id in ids { let _ = writeln!(o, "{}", id); }
                }
            }
            (o, 0)
        }
        "turn" => {
            need_scope!();
            let d = open_db(db, max);
            if let OpResult::Turned(t) = apply(&*d, NeuronOp::Turn { scope: scope.clone(), message: rest() }) {
                if json {
                    let _ = writeln!(o, "{{\"reply\":\"{}\",\"kind\":\"{}\",\"wrote\":{},\"facts\":{}}}", esc(&t.reply), t.kind, t.wrote, t.facts);
                } else {
                    let _ = writeln!(o, "{}", t.reply);
                }
            }
            (o, 0)
        }
        // affective layer. `stance` writes the `<scope>::stance` sub-scope; `reinforce` above writes
        // the scope directly — that difference is real, not a typo.
        "stance" => {
            need_scope!();
            let d = open_db(db, max);
            let topic = pos.get(2).cloned().unwrap_or_default();
            let feeling = pos.get(3..).map(|s| s.join(" ")).unwrap_or_default();
            if topic.is_empty() || feeling.is_empty() { return (String::new(), 2); }
            let sub = format!("{}::stance", scope);
            let (intensity, created) = d.note_stance(&sub, &topic, &feeling);
            if json {
                let _ = writeln!(o, "{{\"topic\":\"{}\",\"feeling\":\"{}\",\"intensity\":{:.3},\"new\":{}}}", esc(&topic), esc(&feeling), intensity, created);
            } else {
                let _ = writeln!(o, "{} stance [{}] {} (intensity {:.2})", if created { "formed" } else { "intensified" }, topic, feeling, intensity);
            }
            (o, 0)
        }
        "mood" => {
            need_scope!();
            let d = open_db(db, max);
            let emotion = rest();
            apply(&*d, NeuronOp::Mood { scope: scope.clone(), emotion: emotion.clone() });
            if json { let _ = writeln!(o, "{{\"mood\":\"{}\"}}", esc(&emotion)); }
            else if emotion.trim().is_empty() { let _ = writeln!(o, "mood cleared"); }
            else { let _ = writeln!(o, "mood set: {}", emotion); }
            (o, 0)
        }
        "affect" => {
            need_scope!();
            let d = open_db(db, max);
            let topic = pos.get(2..).map(|s| s.join(" ")).filter(|s| !s.is_empty());
            let _ = writeln!(o, "{}", d.affect(&scope, topic.as_deref()));
            (o, 0)
        }
        // ---- the learned trust floor (feature `trust`) ----
        "trust_get" => {
            if scope.is_empty() { return (String::new(), 2); }
            let d = open_db(db, max);
            let s = d.trust_stats(&scope);
            if json {
                let _ = writeln!(o, "{{\"class\":\"{}\",\"weight\":{:.4},\"rewards\":{},\"penalties\":{}}}", esc(&scope), s.weight, s.rewards, s.penalties);
            } else {
                let _ = writeln!(o, "{:.4}\t{}\t{}", s.weight, s.rewards, s.penalties);
            }
            (o, 0)
        }
        "trust_weight" => {
            if scope.is_empty() { return (String::new(), 2); }
            let d = open_db(db, max);
            let w = d.trust_weight(&scope);
            if json { let _ = writeln!(o, "{{\"class\":\"{}\",\"weight\":{:.4}}}", esc(&scope), w); }
            else { let _ = writeln!(o, "{:.4}", w); }
            (o, 0)
        }
        // `trust_reward <delta> <class…>` — note the shifted argv: pos[1] is the DELTA, not a scope.
        "trust_reward" => {
            let d = open_db(db, max);
            let delta: f32 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let classes: Vec<String> = pos.get(2..).map(|s| s.to_vec()).unwrap_or_default();
            if classes.is_empty() { return (String::new(), 2); }
            let l = d.trust_reward(&classes, delta);
            if json {
                let items: Vec<String> = classes.iter()
                    .map(|c| format!("{{\"class\":\"{}\",\"weight\":{:.4}}}", esc(c), l.weight(c)))
                    .collect();
                let _ = writeln!(o, "{{\"delta\":{:.4},\"updated\":[{}]}}", delta, items.join(","));
            } else {
                for c in &classes { let _ = writeln!(o, "{}\t{:.4}", c, l.weight(c)); }
            }
            (o, 0)
        }
        "trust_dump" => {
            let d = open_db(db, max);
            let dump = d.trust_dump();
            if json {
                let items: Vec<String> = dump.lines().filter(|ln| !ln.is_empty()).map(|ln| {
                    let f: Vec<&str> = ln.splitn(4, '\t').collect();
                    format!("{{\"class\":\"{}\",\"weight\":{},\"rewards\":{},\"penalties\":{}}}",
                        esc(f.first().copied().unwrap_or("")), f.get(1).copied().unwrap_or("0"),
                        f.get(2).copied().unwrap_or("0"), f.get(3).copied().unwrap_or("0"))
                }).collect();
                let _ = writeln!(o, "{{\"trust\":[{}]}}", items.join(","));
            } else if dump.is_empty() {
                let _ = writeln!(o, "(no trust learned yet)");
            } else {
                let _ = writeln!(o, "{}", dump);
            }
            (o, 0)
        }
        "export" => export_cmd(db, max, pos.get(1..).unwrap_or(&[]), json),
        "import" => import_cmd(db, max, pos.get(1..).unwrap_or(&[]), json),
        "caps" => {
            let host: Vec<&str> = pos.get(1..).unwrap_or(&[]).iter().map(String::as_str).filter(|s| !s.is_empty()).collect();
            if host.is_empty() { let _ = writeln!(o, "{}", neuron_core::caps::manifest()); }
            else { for (name, d) in neuron_core::caps::resolve(&host) { let _ = writeln!(o, "{}\t{}", name, d.tag()); } }
            (o, 0)
        }
        // Anything else — including verbs the release feature set does not compile (`persona` needs
        // `personality`, `route` needs `cortex`) — is an unknown command: empty stdout, exit 2,
        // exactly as the subprocess behaves today. Callers already treat that as a silent no-op.
        _ => (String::new(), 2),
    }
}

fn env_num(k: &str) -> Option<usize> { std::env::var(k).ok().and_then(|s| s.parse().ok()) }

// ---------------------------------------------------------------------------
// export — no NeuronOp variant; mirrors cli.rs::export_cmd
// ---------------------------------------------------------------------------
fn export_cmd(db: &str, max: usize, args: &[String], json: bool) -> (String, i32) {
    let (mut all, mut out_path, mut scope) = (false, None::<String>, None::<String>);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--all" => { all = true; i += 1; }
            "-o" | "--out" => match args.get(i + 1) {
                Some(v) if !v.starts_with('-') => { out_path = Some(v.clone()); i += 2; }
                _ => { i += 1; }
            },
            o if o.starts_with('-') => { i += 1; }
            _ => { if scope.is_none() { scope = Some(args[i].clone()); } i += 1; }
        }
    }
    let d = open_db(db, max);
    let scopes = if all {
        d.neurons()
    } else {
        match scope { Some(s) => vec![s], None => return (String::new(), 2) }
    };
    let mut buf = String::new();
    let mut total = 0usize;
    for s in &scopes {
        let facts = d.scope_facts(s);
        if facts.is_empty() { continue; }
        // a scope NAME the `# scope:` directive parser would mangle on re-import rides as JSONL instead
        let scope_unsafe = s.contains(['\t', '\n', '\r']) || s.starts_with('#') || s.as_str() != s.trim();
        if !scope_unsafe { buf.push_str(&format!("# scope: {}\n", s)); }
        for f in &facts {
            let clean = f.replace(['\t', '\n'], " ");
            let lead = clean.trim_start();
            if scope_unsafe || lead.starts_with('#') || lead.starts_with('{') {
                buf.push_str(&format!("{{\"scope\":\"{}\",\"fact\":\"{}\"}}\n", esc(s), esc(&clean)));
            } else {
                buf.push_str(&clean);
                buf.push('\n');
            }
            total += 1;
        }
        buf.push('\n');
    }
    match out_path {
        Some(path) => {
            if std::fs::write(&path, &buf).is_err() { return (String::new(), 1); }
            // non-json writes only to stderr in the CLI -> empty stdout here
            if json { (format!("{{\"exported\":{},\"scopes\":{}}}\n", total, scopes.len()), 0) } else { (String::new(), 0) }
        }
        None => {
            if json { (format!("{{\"exported\":{},\"scopes\":{}}}\n", total, scopes.len()), 0) } else { (buf, 0) }
        }
    }
}

// ---------------------------------------------------------------------------
// import — no NeuronOp variant; mirrors cli.rs::import_cmd
// ---------------------------------------------------------------------------
// Two deliberate differences from the CLI, both forced by being in-process:
//   * NO advisory `<db>.lock`. The CLI takes one so two writer PROCESSES can't clobber the blob; in
//     a linked host there is one process, and taking the lock would be worse than useless — the
//     holder pid would be the host's own, so `pid_alive` says "live" and the CLI would exit(1),
//     killing the host on the second import.
//   * usage/IO failures return a code instead of `std::process::exit`.
// Everything else — buffering, --dedup keyed per (scope, fact), the GLOBAL_CAP drain, the `stored`
// accounting — is line-for-line the CLI.
fn import_cmd(db: &str, max: usize, args: &[String], json: bool) -> (String, i32) {
    use neuron_core::preload::{parse_pack_line, resolve_scope, PackLine};
    use std::collections::HashSet;
    use std::io::BufRead;

    let (mut scope_default, mut chunk, mut dedup, mut replace, mut pack) = (None::<String>, 0usize, false, false, None::<String>);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--scope" => match args.get(i + 1) {
                Some(v) if !v.starts_with('-') => { scope_default = Some(v.clone()); i += 2; }
                _ => { i += 1; }
            },
            "--flush" => match args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                Some(n) => { chunk = n; i += 2; }
                None => { i += 1; }
            },
            "--dedup" => { dedup = true; i += 1; }
            "--replace" => { replace = true; i += 1; }
            o if o.starts_with('-') => { i += 1; }
            _ => { if pack.is_none() { pack = Some(args[i].clone()); } i += 1; }
        }
    }
    let pack = match pack { Some(p) => p, None => return (String::new(), 2) };
    let file = match std::fs::File::open(&pack) { Ok(f) => f, Err(_) => return (String::new(), 1) };
    let d = open_db(db, max);

    const GLOBAL_CAP: usize = 200_000;
    let mut buffers: HashMap<String, Vec<String>> = HashMap::new();
    let mut cleared: HashSet<String> = HashSet::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut final_dropped: HashMap<String, u64> = HashMap::new();
    let mut stored = 0usize;
    let mut directive: Option<String> = None;
    let (mut global, mut submitted, mut skipped, mut rejected) = (0usize, 0usize, 0usize, 0usize);

    // one scope's buffered facts -> one ObserveBulk (one O(scope) persist)
    let flush = |scope: &str,
                     buffers: &mut HashMap<String, Vec<String>>,
                     cleared: &mut HashSet<String>,
                     global: &mut usize,
                     stored: &mut usize,
                     final_dropped: &mut HashMap<String, u64>| {
        let texts = match buffers.get_mut(scope) { Some(v) if !v.is_empty() => std::mem::take(v), _ => return };
        *global -= texts.len();
        if replace && cleared.insert(scope.to_string()) {
            apply(&*d, NeuronOp::Forget { scope: scope.to_string(), matching: None });
        }
        let wrote = apply(&*d, NeuronOp::ObserveBulk { scope: scope.to_string(), texts }).wrote();
        let dropped = match apply(&*d, NeuronOp::Stats { scope: scope.to_string() }) { OpResult::Stats(s) => s.dropped, _ => 0 };
        *stored += wrote;
        final_dropped.insert(scope.to_string(), dropped);
    };

    for line in std::io::BufReader::new(file).lines() {
        let line = match line { Ok(l) => l, Err(_) => return (String::new(), 1) };
        match parse_pack_line(&line) {
            PackLine::Skip => skipped += 1,
            PackLine::Reject => rejected += 1,
            PackLine::Scope(s) => directive = Some(s),
            PackLine::Fact { scope, text } => {
                let target = match resolve_scope(&scope, &directive, &scope_default) {
                    Some(s) => s.to_string(),
                    None => return (String::new(), 2),
                };
                if dedup && !seen.insert((target.clone(), text.clone())) { skipped += 1; continue; }
                submitted += 1;
                buffers.entry(target.clone()).or_default().push(text);
                global += 1;
                if chunk > 0 && buffers.get(&target).map_or(0, |v| v.len()) >= chunk {
                    flush(&target, &mut buffers, &mut cleared, &mut global, &mut stored, &mut final_dropped);
                }
                if global >= GLOBAL_CAP {
                    if let Some(big) = buffers.iter().max_by_key(|(_, v)| v.len()).map(|(k, _)| k.clone()) {
                        flush(&big, &mut buffers, &mut cleared, &mut global, &mut stored, &mut final_dropped);
                    }
                }
            }
        }
    }
    let keys: Vec<String> = buffers.keys().cloned().collect();
    for k in keys { flush(&k, &mut buffers, &mut cleared, &mut global, &mut stored, &mut final_dropped); }
    d.flush_all();

    let evicted: u64 = final_dropped.values().sum();
    // non-json import prints its summary to STDERR in the CLI, so stdout is empty here. Hosts that
    // want the count must pass --json (which the CLI documents as mandatory for exactly this reason).
    if json {
        (format!("{{\"stored\":{},\"submitted\":{},\"scopes\":{},\"skipped\":{},\"rejected\":{},\"evicted\":{}}}\n",
            stored, submitted, final_dropped.len(), skipped, rejected, evicted), 0)
    } else {
        (String::new(), 0)
    }
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

/// Collect a C argv into owned Strings. Non-UTF8 arguments are taken lossily rather than rejected —
/// the store is text and a mangled byte is better than a dropped memory op.
unsafe fn collect_argv(argc: c_int, argv: *const *const c_char) -> Vec<String> {
    if argv.is_null() || argc <= 0 { return Vec::new(); }
    let mut v = Vec::with_capacity(argc as usize);
    for i in 0..argc as isize {
        let p = *argv.offset(i);
        if p.is_null() { v.push(String::new()); } else { v.push(CStr::from_ptr(p).to_string_lossy().into_owned()); }
    }
    v
}

fn to_c(s: String) -> *mut c_char {
    // a NUL inside the payload would truncate the host's view; swap it for a space rather than fail
    let cleaned = if s.as_bytes().contains(&0) { s.replace('\0', " ") } else { s };
    match CString::new(cleaned) { Ok(c) => c.into_raw(), Err(_) => CString::new("").unwrap().into_raw() }
}

/// Run one command. Returns a NUL-terminated copy of what the CLI would write to **stdout**, which
/// the caller MUST release with `neuron_free`. Never returns NULL for a well-formed call; an
/// unknown command / usage error returns an empty string (see `neuron_call_ex` for the exit code).
///
/// `db` may be NULL/empty if argv itself carries `--db <path>`. `max` is the max-facts cap
/// (the CLI's `--max`, default 500); pass 0 to mean "use the default".
#[no_mangle]
pub unsafe extern "C" fn neuron_call(db: *const c_char, max: usize, argc: c_int, argv: *const *const c_char) -> *mut c_char {
    neuron_call_ex(db, max, argc, argv, std::ptr::null_mut())
}

/// `neuron_call` plus the process exit code the CLI would have used, written through `exit_code`
/// when non-NULL. 0 = ok, 2 = usage / unknown command, 3 = no match (get/recall/assoc/chain),
/// 1 = IO failure, 101 = a panic was caught inside the engine.
///
/// Hosts that today treat "subprocess exited non-zero" as a miss should branch on this exactly the
/// same way — that is what keeps `recall` misses and `coverage` 0.0 behaving as before.
#[no_mangle]
pub unsafe extern "C" fn neuron_call_ex(
    db: *const c_char,
    max: usize,
    argc: c_int,
    argv: *const *const c_char,
    exit_code: *mut c_int,
) -> *mut c_char {
    let dbs = if db.is_null() { String::new() } else { CStr::from_ptr(db).to_string_lossy().into_owned() };
    let args = collect_argv(argc, argv);
    let maxf = if max == 0 { 500 } else { max };

    // A panic must never cross the ABI boundary — it would be UB and would take the host down.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dispatch(&dbs, maxf, &args)));
    let (out, code) = match r { Ok(v) => v, Err(_) => (String::new(), 101) };
    if !exit_code.is_null() { *exit_code = code as c_int; }
    to_c(out)
}

/// Release a buffer returned by `neuron_call` / `neuron_call_ex`. NULL is a no-op.
#[no_mangle]
pub unsafe extern "C" fn neuron_free(p: *mut c_char) {
    if !p.is_null() { drop(CString::from_raw(p)); }
}

/// Enable (non-zero) or disable (0) the open-handle cache. OFF by default. Only turn it on once this
/// process is the sole writer of the db files it touches — see the CACHE note at the top. Disabling
/// also drops every cached handle.
#[no_mangle]
pub unsafe extern "C" fn neuron_cache(enable: c_int) {
    CACHE_ON.store(enable != 0, std::sync::atomic::Ordering::Relaxed);
    if enable == 0 {
        if let Some(m) = CACHE.get() {
            if let Ok(mut g) = m.lock() { g.clear(); }
        }
    }
}

/// Semantic version of this FFI shim (not of neuron-core). Static; do not free.
#[no_mangle]
pub extern "C" fn neuron_ffi_version() -> *const c_char {
    b"0.1.0\0".as_ptr() as *const c_char
}
