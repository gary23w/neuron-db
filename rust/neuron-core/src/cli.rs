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
    let mut pos: Vec<String> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--db" => { db = raw.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--secret" => { eprintln!("warning: --secret exposes the key in the process args (visible via ps / /proc); prefer --keyfile or NEURON_SECRET"); secret = raw.get(i + 1).cloned(); i += 2; }
            "--keyfile" => { keyfile = raw.get(i + 1).cloned(); i += 2; }
            "--max" => { if let Some(v) = raw.get(i + 1).and_then(|s| s.parse().ok()) { max = v; } i += 2; }
            "--json" => { json = true; i += 1; }
            "-h" | "--help" | "help" => { help(); return; }
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
        "observe" => { need_scope("observe"); let d = NeuronDB::open(&db, max); let n = d.observe(&scope, &body(rest()));
            if json { println!("{{\"wrote\":{}}}", n); } else { println!("stored {} fact(s)", n); } }
        "get" => { need_scope("get"); let d = NeuronDB::open(&db, max); let v = d.get(&scope, &rest());
            if json { println!("{{\"value\":{}}}", v.as_deref().map(|s| format!("\"{}\"", esc(s))).unwrap_or("null".into())); }
            else { match &v { Some(s) => println!("{}", s), None => eprintln!("(no answer)") } }
            if v.is_none() { std::process::exit(3); } }                       // miss -> exit 3 (scriptable: get … || fallback)
        "recall" => { need_scope("recall"); let d = NeuronDB::open(&db, max); let h = d.recall(&scope, &rest());
            match &h {
                Some(h) => if json { println!("{{\"fact\":\"{}\",\"value\":\"{}\",\"coverage\":{:.3}}}", esc(&h.fact), esc(&h.value), h.coverage) }
                           else { println!("value:    {}\nfact:     {}\ncoverage: {:.0}%", h.value, h.fact, h.coverage * 100.0) },
                None => if json { println!("{{\"fact\":null}}"); } else { eprintln!("(no match)"); },
            }
            if h.is_none() { std::process::exit(3); } }
        "turn" => { need_scope("turn"); let d = NeuronDB::open(&db, max); let t = d.turn(&scope, &body(rest()));
            if json { println!("{{\"reply\":\"{}\",\"kind\":\"{}\",\"wrote\":{},\"facts\":{}}}", esc(&t.reply), t.kind, t.wrote, t.facts); }
            else { println!("{}", t.reply); } }
        "chat" => { need_scope("chat"); chat(&db, max, &scope); }
        "stats" => { need_scope("stats"); let d = NeuronDB::open(&db, max); let s = d.stats(&scope);
            if json { println!("{{\"facts\":{},\"max_facts\":{},\"created\":{},\"updated\":{},\"turns\":{}}}", s.facts, s.max_facts, s.created, s.updated, s.turns); }
            else { println!("facts:   {}\nmax:     {}\nturns:   {}\ncreated: {}\nupdated: {}", s.facts, s.max_facts, s.turns, s.created, s.updated); } }
        "forget" => { need_scope("forget"); let d = NeuronDB::open(&db, max); let m = pos.get(2..).map(|s| s.join(" ")).filter(|s| !s.is_empty());
            let (f, r) = d.forget(&scope, m.as_deref());
            if json { println!("{{\"forgot\":{},\"remaining\":{}}}", f, r); } else { println!("forgot {}, {} remaining", f, r); } }
        "list" => { let d = NeuronDB::open(&db, max); let ids = d.neurons();
            if json { let items: Vec<String> = ids.iter().map(|s| format!("\"{}\"", esc(s))).collect(); println!("{{\"scopes\":[{}]}}", items.join(",")); }
            else { for id in ids { println!("{}", id); } } }
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
fn chat(db: &str, max: usize, scope: &str) {
    use std::io::Write;
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
        println!("{}", d.turn(scope, msg).reply);
        let _ = std::io::stdout().flush();
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
            format!("{}/.claude.json", home)
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

#[cfg(test)]
mod tests {
    use super::claude_entry;
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
  turn    <scope> <message...|-> store or answer (conversational)\n\
  chat    <scope>                REPL: open once, turn() each stdin line\n\
  stats   <scope>                fact count + timestamps\n\
  forget  <scope> [match...]     drop facts\n\
  list                           list scope ids\n\
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
