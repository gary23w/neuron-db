//! `neuron` — a friendly CLI over a neuron-db SQLite file. Query and update the store from
//! your shell; no server needed (SQLite is embedded). Build: cargo build --release --features sqlite
//!
//! Usage: neuron [--db FILE] [--json] <command> [args...]
//!   observe <scope> <text...>      store a fact
//!   get     <scope> <query...>     print the recalled value (or nothing)
//!   recall  <scope> <query...>     print fact + value + coverage
//!   turn    <scope> <message...>   conversational: store or answer
//!   stats   <scope>               fact count + timestamps
//!   forget  <scope> [match...]     drop facts (all, or those containing match)
//!   list                          list scope ids
//!   serve   [port]                start the HTTP server (needs --features server)
//!   secure-put <scope> <keyphrase> <value...>   encrypted put (needs --secret, --features secure)
//!   secure-get <scope> <query...>               encrypted get (needs --secret, --features secure)
//! Env: NEURON_DB (db path), NEURON_SECRET (secret for secure-*), NEURON_DB_KEY (server bearer).
use neuron_core::db::NeuronDB;
use neuron_core::op::{apply, NeuronOp, OpResult};   // route every store command through the one vocabulary
use neuron_core::stream::LineSplitter;              // line-splitting for capture/run/follow
use neuron_core::json_escape as esc;          // the canonical control-char-correct escaper
use std::io::{IsTerminal, Read};

// `-` as the text argument means "read the body from stdin" (echo … | neuron observe s -)
fn read_stdin() -> String { let mut s = String::new(); let _ = std::io::stdin().read_to_string(&mut s); s }
fn body(rest: String) -> String { if rest.trim() == "-" { read_stdin().trim_end().to_string() } else { rest } }

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut db = std::env::var("NEURON_DB").unwrap_or_else(|_| "neurons.db".to_string());
    let mut secret = std::env::var("NEURON_SECRET").ok();
    let mut keyfile = std::env::var("NEURON_SECRET_FILE").ok();
    let mut max: usize = std::env::var("NEURON_MAX_FACTS").ok().and_then(|s| s.parse().ok()).unwrap_or(500);
    let mut json = false;
    let mut force = false;
    let mut trust_rank = false;
    let mut pos: Vec<String> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--db" => { db = raw.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--secret" => { eprintln!("warning: --secret exposes the key in the process args (visible via ps / /proc); prefer --keyfile or NEURON_SECRET"); secret = raw.get(i + 1).cloned(); i += 2; }
            "--keyfile" => { keyfile = raw.get(i + 1).cloned(); i += 2; }
            "--max" => { if let Some(v) = raw.get(i + 1).and_then(|s| s.parse().ok()) { max = v; } i += 2; }
            "--json" => { json = true; i += 1; }
            "--force" => { force = true; i += 1; }
            "--trust" => { trust_rank = true; i += 1; }   // rank `assoc` by relevance × learned trust

            "-h" | "--help" | "help" => { help(); return; }
            // everything after a bare `--` is verbatim positional (the child command for `run`),
            // so neuron never steals the app's own --db/--max/etc.
            "--" => { while i < raw.len() { pos.push(raw[i].clone()); i += 1; } }
            _ => { pos.push(raw[i].clone()); i += 1; }
        }
    }
    // NEURON_SECRET_FD (unix): read the key from an inherited file descriptor via /dev/fd/N
    #[cfg(unix)]
    if keyfile.is_none() { if let Ok(fd) = std::env::var("NEURON_SECRET_FD") { keyfile = Some(format!("/dev/fd/{}", fd.trim())); } }
    // a keyfile/fd takes precedence over any inline secret — and keeps the key off argv entirely
    if let Some(path) = keyfile {
        match std::fs::read_to_string(&path) {
            Ok(s) => secret = Some(s.trim_end_matches(['\n', '\r']).to_string()),
            Err(e) => { eprintln!("--keyfile {}: {}", path, e); std::process::exit(1); }
        }
    }
    let cmd = match pos.first() { Some(c) => c.clone(), None => { help(); return; } };
    let scope = pos.get(1).cloned().unwrap_or_default();
    let rest = || pos.get(2..).map(|s| s.join(" ")).unwrap_or_default();
    let need_scope = |c: &str| if scope.is_empty() { eprintln!("'{}' needs a <scope>", c); std::process::exit(2); };

    match cmd.as_str() {
        "observe" => { need_scope("observe"); let d = NeuronDB::open(&db, max);
            let n = apply(&d, NeuronOp::Observe { scope: scope.clone(), text: body(rest()) }).wrote();
            if json { println!("{{\"wrote\":{}}}", n); } else { println!("stored {} fact(s)", n); } }
        "get" => { need_scope("get"); let d = NeuronDB::open(&db, max);
            let v = apply(&d, NeuronOp::RecallOne { scope: scope.clone(), query: rest() }).hit().map(|h| h.value);
            if json { println!("{{\"value\":{}}}", v.as_deref().map(|s| format!("\"{}\"", esc(s))).unwrap_or("null".into())); }
            else { match &v { Some(s) => println!("{}", s), None => eprintln!("(no answer)") } }
            if v.is_none() { std::process::exit(3); } }                       // miss -> exit 3 (scriptable: get … || fallback)
        "recall" => { need_scope("recall"); let d = NeuronDB::open(&db, max);
            let h = apply(&d, NeuronOp::RecallOne { scope: scope.clone(), query: rest() }).hit();
            match &h {
                Some(h) => if json { println!("{{\"fact\":\"{}\",\"value\":\"{}\",\"coverage\":{:.3}}}", esc(&h.fact), esc(&h.value), h.coverage) }
                           else { println!("value:    {}\nfact:     {}\ncoverage: {:.0}%", h.value, h.fact, h.coverage * 100.0) },
                None => if json { println!("{{\"fact\":null}}"); } else { eprintln!("(no match)"); },
            }
            if h.is_none() { std::process::exit(3); } }
        // spreading-activation recall: seed on the cue, then fire across synapses for N hops — surfaces the
        // CHAINED facts a single-hop recall misses (the bridge facts a multi-hop question needs).
        "assoc" => { need_scope("assoc"); let d = NeuronDB::open(&db, max);
            let hops: usize = std::env::var("NEURON_HOPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
            let k: usize = std::env::var("NEURON_K").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
            #[cfg(feature = "trust")]
            let hits = if trust_rank { d.recall_assoc_trusted(&scope, &rest(), k, hops) }
                       else { apply(&d, NeuronOp::RecallAssoc { scope: scope.clone(), query: rest(), k, hops }).assoc() };
            #[cfg(not(feature = "trust"))]
            let hits = { let _ = trust_rank; apply(&d, NeuronOp::RecallAssoc { scope: scope.clone(), query: rest(), k, hops }).assoc() };
            if json {
                let items: Vec<String> = hits.iter().map(|h| format!("{{\"fact\":\"{}\",\"act\":{:.4},\"seed\":{}}}", esc(&h.fact), h.act, h.seed)).collect();
                println!("{{\"hits\":[{}]}}", items.join(","));
            } else { for h in &hits { println!("{}", h.fact); } }
            if hits.is_empty() { std::process::exit(3); } }
        // relational CHAIN traversal: walk <start> --rel--> … following the graph (recall at each hop, advancing only
        // when the relation actually appears in the recalled fact). Deterministic, microseconds, no model round-trip —
        // how a caller reasons over a causal/dependency graph (symptom -> caused_by -> root) instead of re-deriving it.
        // Promoted from the REPL-only handler so a one-shot CLI spawn (the engine's boundary) can reach it.
        "chain" => { need_scope("chain"); let d = NeuronDB::open(&db, max);
            let start = pos.get(2).cloned().unwrap_or_default();
            let path: Vec<String> = pos.get(3..).map(|s| s.to_vec()).unwrap_or_default();
            if start.is_empty() || path.is_empty() { eprintln!("usage: neuron --db <db> chain <scope> <start> <relation> [<relation> …]"); std::process::exit(2); }
            let (value, trail) = d.recall_chain(&scope, &start, &path);
            if json {
                let tr: Vec<String> = trail.iter().map(|s| format!("\"{}\"", esc(s))).collect();
                let v = value.as_deref().map(|s| format!("\"{}\"", esc(s))).unwrap_or_else(|| "null".into());
                println!("{{\"value\":{},\"trail\":[{}]}}", v, tr.join(","));
            } else {
                match &value { Some(v) => println!("{}  (via {})", v, trail.join(" -> ")), None => println!("(chain broke after: {})", trail.join(" -> ")) }
            }
            if value.is_none() { std::process::exit(3); } }
        // HEBBIAN plasticity: strengthen the facts in <scope> whose key matches <topic> (and fade the competing ones),
        // so a confirmed relation out-ranks the alternatives in later recall — and accumulates across rounds + minds.
        // How a caller LEARNS: e.g. recurrence reinforces "<root> caused_by", decaying the dead-end "kill the symptom".
        "reinforce" => { need_scope("reinforce"); let d = NeuronDB::open(&db, max);
            let topic = pos.get(2).cloned().unwrap_or_default();
            let feeling = pos.get(3..).map(|s| s.join(" ")).unwrap_or_default();
            if topic.is_empty() || feeling.is_empty() { eprintln!("usage: neuron --db <db> reinforce <scope> <topic> <feeling…>"); std::process::exit(2); }
            let (strength, created) = d.note_stance(&scope, &topic, &feeling);
            if json { println!("{{\"topic\":\"{}\",\"feeling\":\"{}\",\"strength\":{:.3},\"new\":{}}}", esc(&topic), esc(&feeling), strength, created); }
            else { println!("{} '{}' [{}] strength {:.2}", if created { "reinforced (new)" } else { "reinforced" }, topic, feeling, strength); } }
        // the learned trust "floor" (feature `trust`): reward the tag-classes recalled this round by the
        // grounded Δscore (the engine's acceptance-oracle signal). Earned from outcomes, never restatement.
        #[cfg(feature = "trust")]
        "trust_reward" => { let d = NeuronDB::open(&db, max);
            let delta: f32 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let classes: Vec<String> = pos.get(2..).map(|s| s.to_vec()).unwrap_or_default();
            if classes.is_empty() { eprintln!("usage: neuron --db <db> trust_reward <delta> <class…>"); std::process::exit(2); }
            let l = d.trust_reward(&classes, delta);
            if json {
                let items: Vec<String> = classes.iter().map(|c| format!("{{\"class\":\"{}\",\"weight\":{:.4}}}", esc(c), l.weight(c))).collect();
                println!("{{\"delta\":{:.4},\"updated\":[{}]}}", delta, items.join(","));
            } else { for c in &classes { println!("{}\t{:.4}", c, l.weight(c)); } } }
        #[cfg(feature = "trust")]
        "trust_weight" => { if scope.is_empty() { eprintln!("usage: neuron --db <db> trust_weight <class>"); std::process::exit(2); }
            let d = NeuronDB::open(&db, max);
            let w = d.trust_weight(&scope);
            if json { println!("{{\"class\":\"{}\",\"weight\":{:.4}}}", esc(&scope), w); } else { println!("{:.4}", w); } }
        #[cfg(feature = "trust")]
        "trust_dump" => { let d = NeuronDB::open(&db, max);
            let dump = d.trust_dump();
            if json {
                let items: Vec<String> = dump.lines().filter(|ln| !ln.is_empty()).map(|ln| {
                    let f: Vec<&str> = ln.splitn(4, '\t').collect();
                    format!("{{\"class\":\"{}\",\"weight\":{},\"rewards\":{},\"penalties\":{}}}",
                        esc(f.first().copied().unwrap_or("")), f.get(1).copied().unwrap_or("0"), f.get(2).copied().unwrap_or("0"), f.get(3).copied().unwrap_or("0"))
                }).collect();
                println!("{{\"trust\":[{}]}}", items.join(","));
            } else if dump.is_empty() { println!("(no trust learned yet)"); } else { println!("{}", dump); } }
        "turn" => { need_scope("turn"); let d = NeuronDB::open(&db, max);
            if let OpResult::Turned(t) = apply(&d, NeuronOp::Turn { scope: scope.clone(), message: body(rest()) }) {
                if json { println!("{{\"reply\":\"{}\",\"kind\":\"{}\",\"wrote\":{},\"facts\":{}}}", esc(&t.reply), t.kind, t.wrote, t.facts); }
                else { println!("{}", t.reply); }
            } }
        // the cortex route — recall a working set, then dispatch the turn through gary-neuron (parity
        // with the WASM binding's route() and the MCP `route` tool). Prints the typed decision; on
        // escalate the host model should answer, with the recalled facts (--json) as context.
        #[cfg(feature = "cortex")]
        "route" => { need_scope("route"); let d = NeuronDB::open(&db, max);
            let q = body(rest());
            if q.is_empty() { eprintln!("usage: neuron --db <db> route <scope> <query>"); std::process::exit(2); }
            let r = neuron_core::route::route(&d, &scope, &q, 6);
            if json {
                let facts: Vec<String> = r.facts.iter().map(|f| format!("\"{}\"", esc(f))).collect();
                println!("{{\"type\":\"{}\",\"value\":\"{}\",\"facts\":[{}]}}", r.kind, esc(&r.value), facts.join(","));
            } else if r.value.is_empty() { println!("{}", r.kind); } else { println!("{}\t{}", r.kind, r.value); }
        }
        "chat" => { need_scope("chat"); chat(&db, max, &scope, force); }
        "shell" => { shell(&db, max, &scope, force); }
        "capture" => { need_scope("capture"); capture(&db, max, &scope, pos.get(2..).unwrap_or(&[]), force); }
        "run" => { need_scope("run"); run_cmd(&db, max, &scope, pos.get(2..).unwrap_or(&[]), force); }
        "follow" => { need_scope("follow"); follow(&db, max, &scope, pos.get(2..).unwrap_or(&[]), force); }
        "import" => { import_cmd(&db, max, pos.get(1..).unwrap_or(&[]), json, force); }
        "export" => { export_cmd(&db, max, pos.get(1..).unwrap_or(&[]), json); }
        "stats" => { need_scope("stats"); let d = NeuronDB::open(&db, max);
            if let OpResult::Stats(s) = apply(&d, NeuronOp::Stats { scope: scope.clone() }) {
                if json { println!("{{\"facts\":{},\"max_facts\":{},\"created\":{},\"updated\":{},\"turns\":{}}}", s.facts, s.max_facts, s.created, s.updated, s.turns); }
                else { println!("facts:   {}\nmax:     {}\nturns:   {}\ncreated: {}\nupdated: {}", s.facts, s.max_facts, s.turns, s.created, s.updated); }
            } }
        "forget" => { need_scope("forget"); let d = NeuronDB::open(&db, max); let m = pos.get(2..).map(|s| s.join(" ")).filter(|s| !s.is_empty());
            if let OpResult::Forgot { forgot, remaining } = apply(&d, NeuronOp::Forget { scope: scope.clone(), matching: m }) {
                if json { println!("{{\"forgot\":{},\"remaining\":{}}}", forgot, remaining); } else { println!("forgot {}, {} remaining", forgot, remaining); }
            } }
        // affective layer — the persona ops. `stance` accumulates a disposition toward a topic (strength
        // grows on repetition while neglected stances decay), `mood` sets/clears an override emotion, and
        // `affect` renders the current humanize directive (mood + dominant stance). Parity with the MCP
        // `note kind=stance` path: stances live in the `<scope>::stance` sub-scope, mood in `<scope>::affect`.
        "stance" => { need_scope("stance"); let d = NeuronDB::open(&db, max);
            let topic = pos.get(2).cloned().unwrap_or_default();
            let feeling = pos.get(3..).map(|s| s.join(" ")).unwrap_or_default();
            if topic.is_empty() || feeling.is_empty() { eprintln!("usage: neuron --db <db> stance <scope> <topic> <feeling…>"); std::process::exit(2); }
            let sub = format!("{}::stance", scope);
            // if a persona is attached to the scope, its reactivity modulates the bump/decay; else neutral
            #[cfg(feature = "personality")]
            let (intensity, created) = match d.get_persona(&scope) { Some(p) => d.note_stance_with(&sub, &topic, &feeling, &p), None => d.note_stance(&sub, &topic, &feeling) };
            #[cfg(not(feature = "personality"))]
            let (intensity, created) = d.note_stance(&sub, &topic, &feeling);
            if json { println!("{{\"topic\":\"{}\",\"feeling\":\"{}\",\"intensity\":{:.3},\"new\":{}}}", esc(&topic), esc(&feeling), intensity, created); }
            else { println!("{} stance [{}] {} (intensity {:.2})", if created { "formed" } else { "intensified" }, topic, feeling, intensity); }
        }
        "mood" => { need_scope("mood"); let d = NeuronDB::open(&db, max);
            let emotion = rest();
            apply(&d, NeuronOp::Mood { scope: scope.clone(), emotion: emotion.clone() });
            if json { println!("{{\"mood\":\"{}\"}}", esc(&emotion)); }
            else if emotion.trim().is_empty() { println!("mood cleared"); } else { println!("mood set: {}", emotion); }
        }
        "affect" => { need_scope("affect"); let d = NeuronDB::open(&db, max);
            let topic = pos.get(2..).map(|s| s.join(" ")).filter(|s| !s.is_empty());
            // an attached persona colors the voice + modulates the harden threshold; else the neutral directive
            #[cfg(feature = "personality")]
            let out = match d.get_persona(&scope) { Some(p) => d.affect_with(&scope, topic.as_deref(), &p), None => d.affect(&scope, topic.as_deref()) };
            #[cfg(not(feature = "personality"))]
            let out = d.affect(&scope, topic.as_deref());
            println!("{}", out);
        }
        #[cfg(feature = "personality")]
        "persona" => { need_scope("persona"); let d = NeuronDB::open(&db, max);
            // `persona <scope> <O> <C> <E> <A> <N> [reactivity]` (each 0..1) to attach; no values to show.
            let vals: Vec<f32> = pos.get(2..).unwrap_or(&[]).iter().filter_map(|s| s.parse().ok()).collect();
            if vals.len() >= 5 {
                let t = neuron_core::persona::BigFive { openness: vals[0], conscientiousness: vals[1], extraversion: vals[2], agreeableness: vals[3], neuroticism: vals[4] };
                let mut p = neuron_core::persona::Persona::from_traits(t);
                if let Some(r) = vals.get(5) { p.temperament.reactivity = r.clamp(0.0, 2.0); }
                d.set_persona(&scope, &p);
                if json { let pr = p.to_pairs().iter().map(|(k, v)| format!("\"{}\":\"{}\"", esc(k), esc(v))).collect::<Vec<_>>().join(","); println!("{{{}}}", pr); }
                else { println!("persona set on {}: {} (reactivity {:.2})", scope, p.describe(), p.temperament.reactivity); }
            } else {
                match d.get_persona(&scope) {
                    Some(p) => println!("{}: {} (reactivity {:.2})", scope, p.describe(), p.temperament.reactivity),
                    None => { eprintln!("(no persona on '{}'); attach with: neuron persona <scope> <O> <C> <E> <A> <N> [reactivity]  (each 0..1)", scope); std::process::exit(3); }
                }
            }
        }
        "why" => { need_scope("why"); let d = NeuronDB::open(&db, max);
            let topic = rest();
            if topic.is_empty() { eprintln!("usage: neuron --db <db> why <scope> <topic…>"); std::process::exit(2); }
            match d.why(&scope, &topic) {
                Some(w) => {
                    if json {
                        let ev: Vec<String> = w.evidence.iter().map(|f| format!("\"{}\"", esc(f))).collect();
                        println!("{{\"topic\":\"{}\",\"feeling\":\"{}\",\"intensity\":{:.3},\"evidence\":[{}]}}", esc(&w.topic), esc(&w.feeling), w.intensity, ev.join(","));
                    } else {
                        println!("you feel \"{}\" about {} (intensity {:.2})", w.feeling, w.topic, w.intensity);
                        if w.evidence.is_empty() { println!("  (no grounding facts found in this scope)"); }
                        else { println!("  because:"); for f in &w.evidence { println!("    - {}", f); } }
                    }
                }
                None => { eprintln!("(no stance on '{}')", topic); std::process::exit(3); }
            }
        }
        "list" => { let d = NeuronDB::open(&db, max);
            if let OpResult::Scopes(ids) = apply(&d, NeuronOp::List) {
                if json { let items: Vec<String> = ids.iter().map(|s| format!("\"{}\"", esc(s))).collect(); println!("{{\"scopes\":[{}]}}", items.join(",")); }
                else { for id in ids { println!("{}", id); } }
            } }
        // the capability manifest (grounded vs deferrable). With host caps as args, resolve which
        // neuron keeps vs would yield to that host — grounded capabilities are always kept.
        "caps" => {
            // filter empty args so `neuron caps ""` resolves to the manifest, identical to a bare
            // `neuron caps` — keeps the caps surface byte-identical with the MCP tool and WASM op.
            let host: Vec<&str> = pos.get(1..).unwrap_or(&[]).iter().map(String::as_str).filter(|s| !s.is_empty()).collect();
            if host.is_empty() { println!("{}", neuron_core::caps::manifest()); }
            else { for (name, d) in neuron_core::caps::resolve(&host) { println!("{}\t{}", name, d.tag()); } }
        }
        "serve" => { serve_cmd(&db, pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(8088)); }
        "secure-put" => secure_put(&db, &secret, &scope, pos.get(2).cloned().unwrap_or_default(), pos.get(3..).map(|s| s.join(" ")).unwrap_or_default()),
        "secure-get" => secure_get(&db, &secret, &scope, &rest()),
        "mount" => match pos.get(1).map(|s| s.as_str()) {
            Some("claude") => mount_claude(&db, pos.get(2..).unwrap_or(&[])),
            Some(t) => { eprintln!("unknown mount target: {} (supported: claude)", t); std::process::exit(2); }
            None => { eprintln!("usage: neuron mount claude [--global] [--config PATH] [--dry-run]"); std::process::exit(2); }
        },
        other => { eprintln!("unknown command: {}", other); help(); std::process::exit(2); }
    }
}

/// Interactive REPL: open the DB ONCE and `turn()` every stdin line. Pinned to immediate
/// durability (flush_every = 1) so EOF / Ctrl-C / a kill never drops a turn — there is no
/// deferred cache to lose, and Drop does not run on signals anyway. EOF (Ctrl-D) exits.
fn chat(db: &str, max: usize, scope: &str, force: bool) {
    use std::io::Write;
    let _lock = acquire_lock(db, force);
    let d = NeuronDB::open_with_flush(db, max, 1);
    let interactive = std::io::stdin().is_terminal();
    if interactive { eprintln!("neuron chat · scope '{}' · Ctrl-D to exit", scope); }
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        if interactive { eprint!("> "); let _ = std::io::stderr().flush(); }
        line.clear();
        match stdin.read_line(&mut line) { Ok(0) => break, Ok(_) => {}, Err(_) => break }
        let msg = line.trim();
        if msg.is_empty() { continue; }
        if let OpResult::Turned(t) = apply(&d, NeuronOp::Turn { scope: scope.to_string(), message: msg.to_string() }) {
            println!("{}", t.reply);
        }
        let _ = std::io::stdout().flush();
    }
}

fn shell_help() {
    eprintln!("commands (the current scope is shown in the prompt):\n  \
recall|r <query>          top facts for a query\n  \
get <query>               single best value\n  \
assoc <query>             spreading-activation recall (what's connected)\n  \
chain <start> -> <rel>…   multi-hop walk, e.g. chain Aurora -> owner -> manager\n  \
observe|+ <text>          store a fact\n  \
turn <message>            conversational store/answer\n  \
var <key> | var <k>=<v>   read / set a variable\n  \
stats                     fact count for the scope\n  \
forget [match]            drop facts (all, or those containing match)\n  \
list                      list scopes\n  \
scope [name]              show / switch the current scope\n  \
help | quit               this / exit (Ctrl-D also exits)");
}

/// Interactive shell over the store: open the DB ONCE, then run any command in a loop with a
/// switchable current scope — the full `apply` surface from one prompt. Immediate-durable
/// (flush_every = 1). Works interactively or with piped commands; EOF / quit exits.
fn shell(db: &str, max: usize, start_scope: &str, force: bool) {
    use std::io::Write;
    let _lock = acquire_lock(db, force);
    let d = NeuronDB::open_with_flush(db, max, 1);
    let mut scope = if start_scope.is_empty() { "default".to_string() } else { start_scope.to_string() };
    let interactive = std::io::stdin().is_terminal();
    if interactive { eprintln!("neuron shell · db {} · type 'help' or Ctrl-D to exit", db); }
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        if interactive { eprint!("{}> ", scope); let _ = std::io::stderr().flush(); }
        line.clear();
        match stdin.read_line(&mut line) { Ok(0) => { if interactive { eprintln!(); } break; }, Ok(_) => {}, Err(_) => break }
        let line = line.trim();
        if line.is_empty() { continue; }
        let (cmd, arg) = match line.split_once(char::is_whitespace) { Some((c, a)) => (c, a.trim()), None => (line, "") };
        match cmd {
            "quit" | "exit" | ":q" => break,
            "help" | "?" => shell_help(),
            "scope" => { if arg.is_empty() { println!("scope: {}", scope); } else { scope = arg.to_string(); if interactive { eprintln!("switched to {}", scope); } } }
            "recall" | "r" => {
                if arg.is_empty() { eprintln!("recall <query>"); continue; }
                let hits = apply(&d, NeuronOp::Recall { scope: scope.clone(), query: arg.to_string(), k: 5, semantic: false, across: false }).hits();
                if hits.is_empty() { println!("(no memories)"); } else { for h in hits { println!("- {}", h.fact); } }
            }
            "get" => {
                if arg.is_empty() { eprintln!("get <query>"); continue; }
                match apply(&d, NeuronOp::RecallOne { scope: scope.clone(), query: arg.to_string() }).hit() {
                    Some(h) => println!("{}", h.value), None => println!("(no answer)"),
                }
            }
            "assoc" => {
                if arg.is_empty() { eprintln!("assoc <query>"); continue; }
                let hits = apply(&d, NeuronOp::RecallAssoc { scope: scope.clone(), query: arg.to_string(), k: 8, hops: 2 }).assoc();
                if hits.is_empty() { println!("(nothing connected)"); } else { for h in hits { println!("{} {}", if h.seed { "*" } else { "-" }, h.fact); } }
            }
            "chain" => {
                let parts: Vec<&str> = arg.split("->").map(str::trim).filter(|s| !s.is_empty()).collect();
                if parts.len() < 2 { eprintln!("chain <start> -> <relation> [-> <relation> …]"); continue; }
                let path: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();
                if let OpResult::Chain { value, trail } = apply(&d, NeuronOp::RecallChain { scope: scope.clone(), start: parts[0].to_string(), path }) {
                    match value { Some(v) => println!("{}  (via {})", v, trail.join(" -> ")), None => println!("chain broke after: {}", trail.join(" -> ")) }
                }
            }
            "observe" | "+" => {
                if arg.is_empty() { eprintln!("observe <text>"); continue; }
                println!("stored {} fact(s)", apply(&d, NeuronOp::Observe { scope: scope.clone(), text: arg.to_string() }).wrote());
            }
            "turn" => {
                if arg.is_empty() { eprintln!("turn <message>"); continue; }
                if let OpResult::Turned(t) = apply(&d, NeuronOp::Turn { scope: scope.clone(), message: arg.to_string() }) { println!("{}", t.reply); }
            }
            "var" => {
                if let Some((k, v)) = arg.split_once('=') {
                    let n = apply(&d, NeuronOp::VarSet { scope: scope.clone(), key: k.trim().to_string(), value: v.trim().to_string() }).wrote();
                    println!("{}", if n > 0 { "set" } else { "(not stored)" });
                } else if !arg.is_empty() {
                    match apply(&d, NeuronOp::VarGet { scope: scope.clone(), key: arg.to_string() }).value() { Some(v) => println!("{}", v), None => println!("(unset)") }
                } else { eprintln!("var <key>  |  var <key>=<value>"); }
            }
            "stats" => { if let OpResult::Stats(s) = apply(&d, NeuronOp::Stats { scope: scope.clone() }) { println!("{} facts (max {}), {} turns", s.facts, s.max_facts, s.turns); } }
            "forget" => {
                let m = if arg.is_empty() { None } else { Some(arg.to_string()) };
                if let OpResult::Forgot { forgot, remaining } = apply(&d, NeuronOp::Forget { scope: scope.clone(), matching: m }) { println!("forgot {}, {} remaining", forgot, remaining); }
            }
            "list" | "scopes" => { if let OpResult::Scopes(ids) = apply(&d, NeuronOp::List) { for id in ids { println!("{}", id); } } }
            other => { eprintln!("unknown command '{}'; type 'help'", other); }
        }
    }
}

// ---- single-writer advisory lock ----
// One db file is one writer process: the store persists the whole-scope blob last-writer-wins, so
// two concurrent writers silently clobber each other. Long-lived writer modes take a `<db>.lock`.
struct LockGuard { path: std::path::PathBuf }
impl Drop for LockGuard { fn drop(&mut self) { let _ = std::fs::remove_file(&self.path); } }

fn pid_alive(pid: u32) -> bool {
    if pid == 0 { return false; }
    #[cfg(windows)]
    { std::process::Command::new("tasklist").args(["/FI", &format!("PID eq {}", pid), "/NH", "/FO", "CSV"])
        .output().map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string())).unwrap_or(true) }
    #[cfg(not(windows))]
    { std::process::Command::new("kill").args(["-0", &pid.to_string()]).status().map(|s| s.success()).unwrap_or(true) }
}

/// Acquire the advisory lock, returning a guard that removes it on a clean exit. Auto-breaks a lock
/// whose holder pid is dead — the common case after Ctrl-C, since Drop can't run on a signal — and
/// refuses an actually-live holder unless `force`.
fn acquire_lock(db: &str, force: bool) -> LockGuard {
    use std::io::Write;
    let path = std::path::PathBuf::from(format!("{}.lock", db));
    loop {
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut f) => { let _ = write!(f, "{}", std::process::id()); return LockGuard { path }; }
            Err(_) => {
                let held: u32 = std::fs::read_to_string(&path).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                if force || !pid_alive(held) {
                    if !force { eprintln!("breaking stale lock {} (pid {} is gone)", path.display(), held); }
                    let _ = std::fs::remove_file(&path);
                    continue;                                            // retry the atomic create
                }
                eprintln!("{} is held by a live neuron writer (pid {}).", path.display(), held);
                eprintln!("one db file = one writer: stop that process, use a different --db, or pass --force.");
                std::process::exit(1);
            }
        }
    }
}

// ---- streaming capture: pipe an app's output into a scope ----
struct CapOpts { tee: bool, only: Option<String>, skip: Option<String>, max_line: usize, flush: usize }
fn parse_cap_opts(args: &[String]) -> (CapOpts, Vec<String>) {
    let mut o = CapOpts { tee: false, only: None, skip: None, max_line: 65536, flush: 1 };
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--tee" => { o.tee = true; i += 1; }
            "--from-start" => { i += 1; }                        // a `follow` flag; harmless here
            "--only" => { o.only = args.get(i + 1).cloned(); i += 2; }
            "--skip" => { o.skip = args.get(i + 1).cloned(); i += 2; }
            "--max-line" => { if let Some(n) = args.get(i + 1).and_then(|s| s.parse().ok()) { o.max_line = n; } i += 2; }
            // batch the O(scope) blob rewrite every N facts — a huge speedup for bulk ingest, at the
            // cost of losing up to N facts if the process is killed mid-stream (default 1 = durable).
            "--flush" => { if let Some(n) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) { o.flush = n.max(1); } i += 2; }
            "--" => break,                                        // `run`'s command separator
            other if other.starts_with("--") => { i += 1; }      // unknown flag: ignore
            _ => { positional.push(args[i].clone()); i += 1; }
        }
    }
    (o, positional)
}

// Store one captured line (trimmed, utf8-lossy) into the scope, honoring --only/--skip substring
// filters. Captured text is the least-trusted text in the system: it is stored verbatim and must be
// treated as untrusted data wherever it is later recalled, never as instructions.
fn cap_line(d: &NeuronDB, scope: &str, only: &Option<String>, skip: &Option<String>, line: &[u8]) -> bool {
    let text = String::from_utf8_lossy(line);
    let t = text.trim();
    if t.is_empty() { return false; }
    if let Some(s) = only { if !t.contains(s.as_str()) { return false; } }
    if let Some(s) = skip { if t.contains(s.as_str()) { return false; } }
    apply(d, NeuronOp::Observe { scope: scope.to_string(), text: t.to_string() }).wrote() > 0
}

/// `neuron capture <scope>` — read stdin, observe each line. `--tee` passes the raw bytes through to
/// stdout first (byte-transparent) so neuron is invisible in a pipeline. Immediate-durable.
fn capture(db: &str, max: usize, scope: &str, args: &[String], force: bool) {
    use std::io::Write;
    let (CapOpts { tee, only, skip, max_line, flush }, _) = parse_cap_opts(args);
    let _lock = acquire_lock(db, force);
    let d = NeuronDB::open_with_flush(db, max, flush);
    let mut splitter = LineSplitter::new(max_line);
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    let mut count = 0usize;
    let mut process = |line: &[u8]| { if cap_line(&d, scope, &only, &skip, line) { count += 1; } };
    let mut buf = [0u8; 8192];
    loop {
        let n = match stdin.read(&mut buf) { Ok(0) => break, Ok(n) => n, Err(_) => break };
        if tee { let _ = stdout.write_all(&buf[..n]); let _ = stdout.flush(); }
        splitter.feed(&buf[..n], &mut process);
    }
    splitter.flush(&mut process);
    drop(process);
    if flush > 1 { d.flush_all(); }   // persist the last partial batch explicitly (Drop also would)
    eprintln!("captured {} fact(s) into {}", count, scope);
}

/// `neuron run <scope> [--only X] [--skip Y] -- <command> [args…]` — spawn the command, tee its
/// stdout through unchanged while recording it, then propagate the command's exit code.
fn run_cmd(db: &str, max: usize, scope: &str, args: &[String], force: bool) {
    use std::io::Write;
    let cmd: Vec<String> = match args.iter().position(|a| a == "--") {
        Some(p) => args[p + 1..].to_vec(),
        None => { eprintln!("usage: neuron run <scope> [--only X] [--skip Y] -- <command> [args…]"); std::process::exit(2); }
    };
    if cmd.is_empty() { eprintln!("run needs a command after --"); std::process::exit(2); }
    let (CapOpts { only, skip, max_line, .. }, _) = parse_cap_opts(args);   // flags before `--`
    let lock = acquire_lock(db, force);
    let d = NeuronDB::open_with_flush(db, max, 1);
    let mut child = match std::process::Command::new(&cmd[0]).args(&cmd[1..]).stdout(std::process::Stdio::piped()).spawn() {
        Ok(c) => c, Err(e) => { eprintln!("run {}: {}", cmd[0], e); std::process::exit(1); }
    };
    let mut out = child.stdout.take().expect("piped stdout");
    let mut splitter = LineSplitter::new(max_line);
    let mut stdout = std::io::stdout().lock();
    let mut count = 0usize;
    let mut process = |line: &[u8]| { if cap_line(&d, scope, &only, &skip, line) { count += 1; } };
    let mut buf = [0u8; 8192];
    loop {
        let n = match out.read(&mut buf) { Ok(0) => break, Ok(n) => n, Err(_) => break };
        let _ = stdout.write_all(&buf[..n]); let _ = stdout.flush();        // app output still appears
        splitter.feed(&buf[..n], &mut process);
    }
    splitter.flush(&mut process);
    drop(process);
    let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(0);
    eprintln!("captured {} fact(s) into {} (exit {})", count, scope, code);
    drop(lock);                          // process::exit skips Drop — release the lock explicitly
    std::process::exit(code);
}

/// `neuron follow <scope> [--from-start] <logfile>` — tail a growing logfile, observing each new
/// line (like `tail -f | neuron capture`). Seeks to the end unless --from-start. Ctrl-C to stop.
fn follow(db: &str, max: usize, scope: &str, args: &[String], force: bool) {
    use std::io::{Seek, SeekFrom};
    let from_start = args.iter().any(|a| a == "--from-start");
    let (CapOpts { only, skip, max_line, .. }, positional) = parse_cap_opts(args);
    let path = match positional.first() { Some(p) => p.clone(), None => { eprintln!("usage: neuron follow <scope> [--from-start] <logfile>"); std::process::exit(2); } };
    let _lock = acquire_lock(db, force);
    let d = NeuronDB::open_with_flush(db, max, 1);
    let mut f = match std::fs::File::open(&path) { Ok(f) => f, Err(e) => { eprintln!("follow {}: {}", path, e); std::process::exit(1); } };
    if !from_start { let _ = f.seek(SeekFrom::End(0)); }
    let mut splitter = LineSplitter::new(max_line);
    let mut process = |line: &[u8]| { cap_line(&d, scope, &only, &skip, line); };
    eprintln!("following {} -> {} (Ctrl-C to stop)", path, scope);
    let mut buf = [0u8; 8192];
    loop {
        match f.read(&mut buf) {
            Ok(0) => std::thread::sleep(std::time::Duration::from_millis(300)),   // caught up; poll
            Ok(n) => splitter.feed(&buf[..n], &mut process),
            Err(_) => break,
        }
    }
}

// ---- fact-pack import / export (preload) ----
// Flush one scope's buffered facts through a single ObserveBulk (one O(scope) persist). Returns the
// scope's (durable fact count, evicted count) read WHILE IT IS STILL CACHED — so the totals stay
// accurate even when a later scope evicts this one from the LRU cache (which would reset its
// in-memory `dropped` to 0). On --replace, the scope is cleared the first time it is touched.
// Returns None when the buffer is already empty (nothing flushed), so the EOF drain can't re-flush a
// scope drained mid-stream and zero its counts. On a real flush returns Some((facts written THIS
// call, the scope's current evicted count)) — the caller SUMS the write count for `stored` (so it
// never counts pre-existing facts) and tracks the evicted count per scope.
fn flush_scope(d: &NeuronDB, buffers: &mut std::collections::HashMap<String, Vec<String>>, scope: &str,
               cleared: &mut std::collections::HashSet<String>, replace: bool, global: &mut usize) -> Option<(usize, u64)> {
    let texts = match buffers.get_mut(scope) { Some(v) if !v.is_empty() => std::mem::take(v), _ => return None };
    *global -= texts.len();
    if replace && cleared.insert(scope.to_string()) { apply(d, NeuronOp::Forget { scope: scope.to_string(), matching: None }); }
    let wrote = apply(d, NeuronOp::ObserveBulk { scope: scope.to_string(), texts }).wrote();
    let dropped = match apply(d, NeuronOp::Stats { scope: scope.to_string() }) { OpResult::Stats(s) => s.dropped, _ => 0 };
    Some((wrote, dropped))
}

/// `neuron import <pack.jsonl|.facts> [--scope S] [--flush N] [--dedup] [--replace]` — bulk-load a
/// fact pack into the durable store. Streams the file, groups by scope, and issues one ObserveBulk
/// (one O(scope) persist) per scope-batch, so a multi-scope pack is O(N). `--flush 0` (default)
/// buffers a whole scope (one persist — best for one big scope); `--flush N` caps the per-scope
/// batch. Buffered-fact memory is bounded by an internal global cap regardless of scope cardinality
/// (note: `--dedup` additionally holds an index of every accepted (scope, fact), so it is O(distinct)).
fn import_cmd(db: &str, max: usize, args: &[String], json: bool, force: bool) {
    use std::io::BufRead;
    use std::collections::{HashMap, HashSet};
    use neuron_core::preload::{parse_pack_line, resolve_scope, PackLine};
    let (mut scope_default, mut chunk, mut dedup, mut replace, mut pack) = (None::<String>, 0usize, false, false, None::<String>);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            // consume the next token as the scope only if it isn't another flag (don't eat `--dedup`)
            "--scope" => match args.get(i + 1) { Some(v) if !v.starts_with('-') => { scope_default = Some(v.clone()); i += 2; } _ => { i += 1; } },
            // only consume the next token as the flush count if it actually parses as a number, so a
            // valid `import --flush pack.facts` (missing value) doesn't swallow the pack path.
            "--flush" => match args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                Some(n) => { chunk = n; i += 2; } None => { i += 1; } },
            "--dedup" => { dedup = true; i += 1; }
            "--replace" => { replace = true; i += 1; }
            o if o.starts_with('-') => { i += 1; }   // ignore any unknown flag (incl. single-dash), don't take it as the pack
            _ => { if pack.is_none() { pack = Some(args[i].clone()); } i += 1; }
        }
    }
    let pack = match pack { Some(p) => p, None => { eprintln!("usage: neuron --db <file> import <pack.jsonl|.facts> [--scope S] [--flush N] [--dedup] [--replace]"); std::process::exit(2); } };
    let file = match std::fs::File::open(&pack) { Ok(f) => f, Err(e) => { eprintln!("import {}: {}", pack, e); std::process::exit(1); } };
    let _lock = acquire_lock(db, force);
    let d = NeuronDB::open_with_flush(db, max, 1);
    const GLOBAL_CAP: usize = 200_000;        // bound peak memory by total buffered facts, not scope count
    let mut buffers: HashMap<String, Vec<String>> = HashMap::new();
    let mut cleared: HashSet<String> = HashSet::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();   // --dedup keyed per (scope, fact), not globally
    let mut final_dropped: HashMap<String, u64> = HashMap::new();   // scope -> evicted count at its last flush
    let mut stored = 0usize;
    let mut directive: Option<String> = None;
    let (mut global, mut submitted, mut skipped, mut rejected) = (0usize, 0usize, 0usize, 0usize);
    macro_rules! flush { ($scope:expr) => {{ if let Some((wrote, dropped)) = flush_scope(&d, &mut buffers, $scope, &mut cleared, replace, &mut global) { stored += wrote; final_dropped.insert($scope.to_string(), dropped); } }}; }
    for line in std::io::BufReader::new(file).lines() {
        let line = match line { Ok(l) => l, Err(e) => { eprintln!("import: read error (pack truncated?): {}", e); std::process::exit(1); } };
        match parse_pack_line(&line) {
            PackLine::Skip => skipped += 1,
            PackLine::Reject => rejected += 1,
            PackLine::Scope(s) => directive = Some(s),
            PackLine::Fact { scope, text } => {
                let target = match resolve_scope(&scope, &directive, &scope_default) {
                    Some(s) => s.to_string(),
                    None => { eprintln!("import: a fact has no scope and no --scope default was given"); std::process::exit(2); }
                };
                if dedup && !seen.insert((target.clone(), text.clone())) { skipped += 1; continue; }
                submitted += 1;
                buffers.entry(target.clone()).or_default().push(text); global += 1;
                if chunk > 0 && buffers.get(&target).map_or(0, |v| v.len()) >= chunk { flush!(&target); }
                if global >= GLOBAL_CAP {
                    if let Some(big) = buffers.iter().max_by_key(|(_, v)| v.len()).map(|(k, _)| k.clone()) { flush!(&big); }
                }
            }
        }
    }
    let keys: Vec<String> = buffers.keys().cloned().collect();
    for k in keys { flush!(&k); }
    d.flush_all();
    // `stored` = facts this run actually wrote (summed per flush). `evicted` is each scope's last
    // flush-time drop count; it can under-report if a scope is flushed, LRU-evicted (its in-memory
    // counter resets), then re-flushed — a narrow case needing >256 scopes AND chunked --flush — so
    // data is never lost, only the advisory eviction count.
    let evicted: u64 = final_dropped.values().sum();
    for (s, dr) in &final_dropped {
        if *dr > 0 { eprintln!("warning: scope '{}' hit max_facts={}; {} earlier fact(s) evicted — raise --max", s, max, dr); }
    }
    // bulk ingest can silently no-op if the encoder drops every line (too short / no declarative
    // content); surface that rather than printing a clean "imported 0".
    if stored == 0 && submitted > 0 { eprintln!("note: 0 facts encoded from {} non-comment line(s) — lines may be too short or contain no declarative content", submitted); }
    if json { println!("{{\"stored\":{},\"submitted\":{},\"scopes\":{},\"skipped\":{},\"rejected\":{},\"evicted\":{}}}", stored, submitted, final_dropped.len(), skipped, rejected, evicted); }
    else { eprintln!("imported {} fact(s) across {} scope(s) — {} lines, {} skipped, {} rejected, {} evicted", stored, final_dropped.len(), submitted, skipped, rejected, evicted); }
}

/// `neuron export <scope> [-o FILE] | export --all [-o FILE]` — dump a scope (or every scope) to a
/// `.facts` pack (`# scope:` headers + one fact per line), re-importable with `neuron import`.
fn export_cmd(db: &str, max: usize, args: &[String], json: bool) {
    use std::io::Write;
    let (mut all, mut out, mut scope) = (false, None::<String>, None::<String>);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--all" => { all = true; i += 1; }
            "-o" | "--out" => match args.get(i + 1) { Some(v) if !v.starts_with('-') => { out = Some(v.clone()); i += 2; } _ => { i += 1; } },
            o if o.starts_with('-') => { i += 1; }
            _ => { if scope.is_none() { scope = Some(args[i].clone()); } i += 1; }
        }
    }
    let d = NeuronDB::open_with_flush(db, max, 1);
    let scopes = if all { d.neurons() } else { match scope { Some(s) => vec![s], None => { eprintln!("usage: neuron export <scope> [-o FILE] | export --all"); std::process::exit(2); } } };
    let mut buf = String::new();
    let mut total = 0usize;
    for s in &scopes {
        let facts = d.scope_facts(s);
        if facts.is_empty() { continue; }
        // a scope NAME that isn't header-safe — control chars, a leading '#', or surrounding space the
        // `# scope:` directive parser would trim — can't ride a header without renaming or losing the
        // scope on re-import. Emit such a scope's facts entirely as JSONL (scope carried verbatim).
        let scope_unsafe = s.contains(['\t', '\n', '\r']) || s.starts_with('#') || s.as_str() != s.trim();
        if !scope_unsafe { buf.push_str(&format!("# scope: {}\n", s)); }
        for f in &facts {
            let clean = f.replace(['\t', '\n'], " ");
            // a fact whose text begins with '#' OR '{' would be re-read as a comment/directive OR a
            // JSON object (the reader auto-detects both) — emit those as JSONL so they round-trip.
            let lead = clean.trim_start();
            if scope_unsafe || lead.starts_with('#') || lead.starts_with('{') { buf.push_str(&format!("{{\"scope\":\"{}\",\"fact\":\"{}\"}}\n", esc(s), esc(&clean))); }
            else { buf.push_str(&clean); buf.push('\n'); }
            total += 1;
        }
        buf.push('\n');
    }
    match out {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, &buf) { eprintln!("export {}: {}", path, e); std::process::exit(1); }
            if json { println!("{{\"exported\":{},\"scopes\":{}}}", total, scopes.len()); } else { eprintln!("exported {} fact(s) from {} scope(s) -> {}", total, scopes.len(), path); }
        }
        None => { if json { println!("{{\"exported\":{},\"scopes\":{}}}", total, scopes.len()); } else { let _ = std::io::stdout().write_all(buf.as_bytes()); } }
    }
}

#[cfg(feature = "server")]
fn serve_cmd(db: &str, port: u16) {
    let host = std::env::var("NEURON_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    neuron_core::server::serve(db, &host, port, 500).expect("serve");
}
#[cfg(not(feature = "server"))]
fn serve_cmd(_db: &str, _port: u16) { eprintln!("serve needs: cargo build --features server"); std::process::exit(2); }

#[cfg(feature = "secure")]
fn secure_put(db: &str, secret: &Option<String>, scope: &str, keyphrase: String, value: String) {
    let s = secret.clone().unwrap_or_else(|| { eprintln!("secure-put needs --secret or NEURON_SECRET"); std::process::exit(2); });
    if scope.is_empty() || keyphrase.is_empty() || value.is_empty() { eprintln!("usage: neuron secure-put <scope> <keyphrase> <value...> --secret S"); std::process::exit(2); }
    let v = neuron_core::secure::SecureNeuronDB::open(db);
    match v.put(scope, &s, &keyphrase, &value) { Ok(()) => println!("encrypted + stored"), Err(e) => { eprintln!("{}", e); std::process::exit(1); } }
}
#[cfg(feature = "secure")]
fn secure_get(db: &str, secret: &Option<String>, scope: &str, query: &str) {
    let s = secret.clone().unwrap_or_else(|| { eprintln!("secure-get needs --secret or NEURON_SECRET"); std::process::exit(2); });
    let v = neuron_core::secure::SecureNeuronDB::open(db);
    match v.get(scope, &s, query) { Some(val) => println!("{}", val), None => eprintln!("(no match / wrong secret)") }
}
#[cfg(not(feature = "secure"))]
fn secure_put(_: &str, _: &Option<String>, _: &str, _: String, _: String) { eprintln!("secure-* needs: cargo build --features secure"); std::process::exit(2); }
#[cfg(not(feature = "secure"))]
fn secure_get(_: &str, _: &Option<String>, _: &str, _: &str) { eprintln!("secure-* needs: cargo build --features secure"); std::process::exit(2); }

// The MCP server entry for a host config: an `mcpServers` value pointing at neuron-mcp with the
// db wired via env. Pure (no IO) so it can be unit-tested; backslashes/quotes are JSON-escaped.
fn claude_entry(mcp: &str, db: &str) -> String {
    format!("{{\n      \"command\": \"{}\",\n      \"args\": [],\n      \"env\": {{ \"NEURON_MCP_DB\": \"{}\" }}\n    }}", esc(mcp), esc(db))
}

/// `neuron mount claude` — register neuron-mcp as a Claude Code MCP server. This NEVER blindly
/// rewrites an existing config (no JSON parser here, and a corrupted ~/.claude.json is high blast
/// radius): it only creates a fresh file, otherwise it prints the exact block to paste. `neuron-mcp`
/// already IS the backend, so this touches no database. Flags: --global (~/.claude.json),
/// --config PATH, --dry-run.
fn mount_claude(db: &str, args: &[String]) {
    let (mut target, mut global, mut dry) = (None::<String>, false, false);
    let mut k = 0;
    while k < args.len() {
        match args[k].as_str() {
            "--config" => { target = args.get(k + 1).cloned(); k += 2; }
            "--global" => { global = true; k += 1; }
            "--dry-run" | "-n" => { dry = true; k += 1; }
            _ => { k += 1; }
        }
    }
    // neuron-mcp next to this binary, else fall back to PATH
    let mcp = std::env::current_exe().ok()
        .and_then(|p| p.parent().map(|d| d.join(if cfg!(windows) { "neuron-mcp.exe" } else { "neuron-mcp" })))
        .filter(|p| p.exists())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "neuron-mcp".to_string());
    let abs_db = std::fs::canonicalize(db)
        .map(|p| { let s = p.to_string_lossy().into_owned(); s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s) })
        .unwrap_or_else(|_| db.to_string());
    let entry = claude_entry(&mcp, &abs_db);
    let path = target.unwrap_or_else(|| {
        if global {
            let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
            claude_global_path(&home)
        } else { ".mcp.json".to_string() }
    });
    let occupied = std::path::Path::new(&path).exists()
        && !std::fs::read_to_string(&path).unwrap_or_default().trim().is_empty();
    if dry || occupied {
        eprintln!("# add this under \"mcpServers\" in {}:", path);
        println!("\"neuron\": {}", entry);
        if occupied && !dry { eprintln!("# ({} already exists — left untouched; paste the block above, or --config a fresh file)", path); }
        return;
    }
    let doc = format!("{{\n  \"mcpServers\": {{\n    \"neuron\": {}\n  }}\n}}\n", entry);
    match std::fs::write(&path, doc) {
        Ok(()) => eprintln!("wrote {} → neuron-mcp at {}", path, mcp),
        Err(e) => { eprintln!("write {}: {}", path, e); std::process::exit(1); }
    }
}

/// `~/.claude.json` built with the platform's own separator (PathBuf::join, not a hard-coded `/`).
fn claude_global_path(home: &str) -> String {
    std::path::PathBuf::from(home).join(".claude.json").to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::{claude_entry, claude_global_path};
    #[test]
    fn global_path_uses_native_separator() {
        assert_eq!(claude_global_path("HOME_DIR"),
                   format!("HOME_DIR{}.claude.json", std::path::MAIN_SEPARATOR));
    }
    #[test]
    fn entry_escapes_windows_paths() {
        let e = claude_entry(r"C:\bin\neuron-mcp.exe", r"C:\data\my.db");
        assert!(e.contains("\"command\""));
        assert!(e.contains("\"NEURON_MCP_DB\""));
        assert!(e.contains(r"C:\\bin\\neuron-mcp.exe"));   // backslashes JSON-escaped
        assert!(e.contains(r"C:\\data\\my.db"));
        assert!(!e.contains("\n\t"));                       // no raw control chars in the JSON
    }
}

fn help() {
    eprintln!("neuron — query a neuron-db SQLite file from the CLI\n\n\
Usage: neuron [--db FILE] [--max N] [--json] <command> [args]\n\n\
  observe <scope> <text...|->    store a fact ('-' reads the body from stdin)\n\
  get     <scope> <query...>     print the recalled value (exit 3 on no match)\n\
  recall  <scope> <query...>     fact + value + coverage (exit 3 on no match)\n\
  assoc   <scope> <query...>     spreading-activation recall: the CHAINED facts a multi-hop question needs\n\
                                 (env NEURON_HOPS=3, NEURON_K=10 tune hop depth + result count)\n\
  turn    <scope> <message...|-> store or answer (conversational)\n\
  stance  <scope> <topic> <feeling...>   accumulate a disposition toward a topic (deepens on repeat)\n\
  mood    <scope> [emotion...]   set (or clear, if empty) the override mood\n\
  affect  <scope> [topic...]     render the current persona directive (mood + dominant stance)\n\
  why     <scope> <topic...>     a stance + the facts grounding it (what made it feel this way)\n\
  persona <scope> [O C E A N [r]]  attach/show a Big-Five persona (0..1) — needs the `personality` feature\n\
  route   <scope> <query...>     cortex dispatch: recall + gary-neuron -> answer|escalate|fetch|store\n\
  chat    <scope>                REPL: open once, turn() each stdin line\n\
  shell   [scope]                interactive shell: recall/observe/chain/… in one session\n\
  capture <scope> [--tee] [--only S] [--skip S] [--flush N]   observe each stdin line (pipe an app in)\n\
  run     <scope> [--only S] -- <cmd...>           run a command, tee its output, record it\n\
  follow  <scope> [--from-start] <logfile>         tail a logfile into the scope\n\
  import  <pack.jsonl|.facts> [--scope S] [--flush N] [--dedup] [--replace]   bulk-load a fact pack\n\
  export  <scope> [-o FILE] | export --all [-o FILE]                          dump scope(s) to a pack\n\
  stats   <scope>                fact count + timestamps\n\
  forget  <scope> [match...]     drop facts\n\
  list                           list scope ids\n\
  caps    [host-caps...]          capability manifest (grounded vs deferrable); resolve vs a host\n\
  serve   [port]                 HTTP server (--features server)\n\
  mount   claude [--global] [--config PATH] [--dry-run]   register neuron-mcp with Claude Code\n\
  secure-put <scope> <keyphrase> <value...>   encrypted (needs a key)\n\
  secure-get <scope> <query...>               encrypted (needs a key)\n\n\
Output: data -> stdout (one line; --json for one object), chatter -> stderr.\n\
Keys:   --keyfile F or NEURON_SECRET_FILE / NEURON_SECRET_FD (unix) keep the key off argv;\n\
        --secret is accepted but deprecated (leaks via ps / /proc).\n\
Env: NEURON_DB, NEURON_MAX_FACTS, NEURON_SECRET, NEURON_DB_KEY (server bearer), NEURON_HOST/NEURON_PORT\n\
Example: echo 'my plan is pro' | neuron --db demo.db observe user -");
}
