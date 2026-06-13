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

fn esc(s: &str) -> String { s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n") }

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut db = std::env::var("NEURON_DB").unwrap_or_else(|_| "neurons.db".to_string());
    let mut secret = std::env::var("NEURON_SECRET").ok();
    let mut json = false;
    let mut pos: Vec<String> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--db" => { db = raw.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--secret" => { secret = raw.get(i + 1).cloned(); i += 2; }
            "--json" => { json = true; i += 1; }
            "-h" | "--help" | "help" => { help(); return; }
            _ => { pos.push(raw[i].clone()); i += 1; }
        }
    }
    let cmd = match pos.first() { Some(c) => c.clone(), None => { help(); return; } };
    let scope = pos.get(1).cloned().unwrap_or_default();
    let rest = || pos.get(2..).map(|s| s.join(" ")).unwrap_or_default();
    let need_scope = |c: &str| if scope.is_empty() { eprintln!("'{}' needs a <scope>", c); std::process::exit(2); };

    match cmd.as_str() {
        "observe" => { need_scope("observe"); let d = NeuronDB::open(&db, 500); let n = d.observe(&scope, &rest());
            if json { println!("{{\"wrote\":{}}}", n); } else { println!("stored {} fact(s)", n); } }
        "get" => { need_scope("get"); let d = NeuronDB::open(&db, 500); let v = d.get(&scope, &rest());
            if json { println!("{{\"value\":{}}}", v.as_deref().map(|s| format!("\"{}\"", esc(s))).unwrap_or("null".into())); }
            else { match v { Some(s) => println!("{}", s), None => eprintln!("(no answer)") } } }
        "recall" => { need_scope("recall"); let d = NeuronDB::open(&db, 500); match d.recall(&scope, &rest()) {
            Some(h) => if json { println!("{{\"fact\":\"{}\",\"value\":\"{}\",\"coverage\":{:.3}}}", esc(&h.fact), esc(&h.value), h.coverage) }
                       else { println!("value:    {}\nfact:     {}\ncoverage: {:.0}%", h.value, h.fact, h.coverage * 100.0) },
            None => eprintln!("(no match)") } }
        "turn" => { need_scope("turn"); let d = NeuronDB::open(&db, 500); let t = d.turn(&scope, &rest());
            if json { println!("{{\"reply\":\"{}\",\"kind\":\"{}\",\"wrote\":{},\"facts\":{}}}", esc(&t.reply), t.kind, t.wrote, t.facts); }
            else { println!("{}", t.reply); } }
        "stats" => { need_scope("stats"); let d = NeuronDB::open(&db, 500); let s = d.stats(&scope);
            println!("{{\"facts\":{},\"max_facts\":{},\"created\":{},\"updated\":{},\"turns\":{}}}", s.facts, s.max_facts, s.created, s.updated, s.turns); }
        "forget" => { need_scope("forget"); let d = NeuronDB::open(&db, 500); let m = pos.get(2..).map(|s| s.join(" ")).filter(|s| !s.is_empty());
            let (f, r) = d.forget(&scope, m.as_deref()); println!("forgot {}, {} remaining", f, r); }
        "list" => { let d = NeuronDB::open(&db, 500); for id in d.neurons() { println!("{}", id); } }
        "serve" => { serve_cmd(&db, pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(8088)); }
        "secure-put" => secure_put(&db, &secret, &scope, pos.get(2).cloned().unwrap_or_default(), pos.get(3..).map(|s| s.join(" ")).unwrap_or_default()),
        "secure-get" => secure_get(&db, &secret, &scope, &rest()),
        other => { eprintln!("unknown command: {}", other); help(); std::process::exit(2); }
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

fn help() {
    eprintln!("neuron — query a neuron-db SQLite file from the CLI\n\n\
Usage: neuron [--db FILE] [--json] <command> [args]\n\n\
  observe <scope> <text...>     store a fact\n\
  get     <scope> <query...>    print the recalled value\n\
  recall  <scope> <query...>    fact + value + coverage\n\
  turn    <scope> <message...>  store or answer (conversational)\n\
  stats   <scope>               fact count + timestamps\n\
  forget  <scope> [match...]    drop facts\n\
  list                          list scope ids\n\
  serve   [port]                HTTP server (--features server)\n\
  secure-put <scope> <keyphrase> <value...>   encrypted (needs --secret)\n\
  secure-get <scope> <query...>               encrypted (needs --secret)\n\n\
Env: NEURON_DB, NEURON_SECRET, NEURON_DB_KEY (server bearer), NEURON_HOST/NEURON_PORT\n\
Example: neuron --db demo.db turn user 'my plan is pro'");
}
