//! MCP (Model Context Protocol) server over stdio, exposing NeuronDB as an LLM's
//! memory bank. JSON-RPC 2.0, newline-delimited messages on stdin/stdout. Any MCP
//! client (Claude Desktop/Code, Cursor, ...) can mount this single binary and get a
//! recall->inject->write memory loop with zero glue.
//!
//! Tools: recall (top-k block), recall_associative (spreading activation), recall_value,
//! recall_chain, remember, note (typed neurons: fact/user/instruction/stance/var), recall_var,
//! forget, stats. Std-only; reuses the same hand-rolled JSON approach as the HTTP server.
//! Feature-gated behind `mcp` (which enables `sqlite` and `semantic`).
use std::io::{self, BufRead, Write};
use std::time::Instant;
use crate::db::NeuronDB;

const PROTO_DEFAULT: &str = "2025-06-18";

// ---- minimal JSON helpers (flat-search; mirrors server.rs) ----
fn json_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}
fn json_field(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let i = body.find(&pat)?;
    let after = &body[i + pat.len()..];
    let colon = after.find(':')?;
    let rest: Vec<char> = after[colon + 1..].chars().collect();
    let mut j = 0; while j < rest.len() && rest[j] != '"' { j += 1; }
    if j >= rest.len() { return None; }
    let mut out = String::new(); j += 1;
    while j < rest.len() {
        let c = rest[j];
        if c == '\\' && j + 1 < rest.len() {
            let n = rest[j + 1];
            out.push(match n { 'n' => '\n', 't' => '\t', 'r' => '\r', '"' => '"', '\\' => '\\', o => o });
            j += 2;
        } else if c == '"' { return Some(out); } else { out.push(c); j += 1; }
    }
    Some(out)
}
fn json_num(body: &str, key: &str) -> Option<i64> {
    let pat = format!("\"{}\"", key);
    let i = body.find(&pat)?; let after = &body[i + pat.len()..];
    let c = after.find(':')?; let tail = after[c + 1..].trim_start();
    let num: String = tail.chars().take_while(|ch| ch.is_ascii_digit() || *ch == '-').collect();
    num.parse().ok()
}
fn json_array(body: &str, key: &str) -> Vec<String> {
    let pat = format!("\"{}\"", key);
    let i = match body.find(&pat) { Some(i) => i, None => return vec![] };
    let after = &body[i + pat.len()..];
    let lb = match after.find('[') { Some(x) => x, None => return vec![] };
    let rest = &after[lb + 1..];
    let end = rest.find(']').unwrap_or(rest.len());
    let chars: Vec<char> = rest[..end].chars().collect();
    let (mut out, mut j) = (Vec::new(), 0);
    while j < chars.len() {
        if chars[j] == '"' {
            let mut sb = String::new(); j += 1;
            while j < chars.len() {
                let c = chars[j];
                if c == '\\' && j + 1 < chars.len() { let n = chars[j+1]; sb.push(match n {'n'=>'\n','t'=>'\t','r'=>'\r','"'=>'"','\\'=>'\\',o=>o}); j += 2; }
                else if c == '"' { j += 1; break; } else { sb.push(c); j += 1; }
            }
            out.push(sb);
        } else { j += 1; }
    }
    out
}
/// extract the raw JSON-RPC id token (number, string, or null) to echo back verbatim.
fn raw_id(body: &str) -> String {
    let i = match body.find("\"id\"") { Some(i) => i, None => return "null".into() };
    let after = &body[i + 4..];
    let c = match after.find(':') { Some(c) => c, None => return "null".into() };
    let tail = after[c + 1..].trim_start();
    let chars: Vec<char> = tail.chars().collect();
    if chars.first() == Some(&'"') {
        let mut out = String::from("\""); let mut j = 1;
        while j < chars.len() { let ch = chars[j]; out.push(ch); if ch == '"' && chars[j - 1] != '\\' { break; } j += 1; }
        return out;
    }
    let val: String = tail.chars().take_while(|c| c.is_ascii_digit() || *c == '-').collect();
    if !val.is_empty() { val } else { "null".into() }
}

// ---- response builders ----
fn result(id: &str, obj: &str) -> String { format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{}}}", id, obj) }
fn rpc_error(id: &str, code: i64, msg: &str) -> String {
    format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"error\":{{\"code\":{},\"message\":\"{}\"}}}}", id, code, json_escape(msg))
}
fn tool_text(id: &str, text: &str) -> String {
    result(id, &format!("{{\"content\":[{{\"type\":\"text\",\"text\":\"{}\"}}]}}", json_escape(text)))
}
fn tool_err(id: &str, text: &str) -> String {
    result(id, &format!("{{\"content\":[{{\"type\":\"text\",\"text\":\"{}\"}}],\"isError\":true}}", json_escape(text)))
}

// tools/list payload. Each entry MUST be a single line: MCP stdio framing is one JSON
// message per physical line, so a response may never contain a raw newline. Names use
// snake_case for broad client compatibility.
const TOOL_DEFS: [&str; 9] = [
r#"{"name":"recall","description":"Recall the most relevant remembered facts for a query, as a memory block to inject into context. Call this BEFORE answering whenever the user refers to something they may have told you earlier. Ranks by associative cue overlap; pass rank='semantic' for broad/narrative topics.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id, e.g. user:123 or session:abc - isolates one user/agent's memory"},"query":{"type":"string","description":"the question or topic to recall about"},"k":{"type":"integer","description":"max facts to return (default 5)"},"rank":{"type":"string","enum":["lexical","semantic"],"description":"ranking strategy; default lexical (associative). Use semantic only for broad/narrative recall."}},"required":["scope","query"]}}"#,
r#"{"name":"recall_associative","description":"Spreading-activation recall: starts from facts that match the query, then follows shared-entity links to surface RELATED facts that may share no words with the query. Use for 'what's connected to X' or to gather context around a topic.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"query":{"type":"string","description":"the topic or entity to activate from"},"k":{"type":"integer","description":"max facts to return (default 8)"},"hops":{"type":"integer","description":"link hops to spread (default 2)"}},"required":["scope","query"]}}"#,
r#"{"name":"recall_value","description":"Recall a single best-matching value for a direct question (e.g. 'what is my plan?'). Returns the isolated value or '(no memory)'.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"query":{"type":"string","description":"a direct question"}},"required":["scope","query"]}}"#,
r#"{"name":"recall_chain","description":"Answer a multi-hop question in ONE call by walking a chain of relations server-side (no extra round-trips, any depth). Give the starting entity and the ordered relations to follow. Example: start 'Aurora', path ['owner','manager','timezone'] returns the timezone of the manager of the owner of Aurora.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"start":{"type":"string","description":"the entity to start from, e.g. 'Aurora' or 'Marisol'"},"path":{"type":"array","items":{"type":"string"},"description":"ordered relations to follow, e.g. ['owner','manager','timezone']"}},"required":["scope","start","path"]}}"#,
r#"{"name":"remember","description":"Store durable facts the user stated, in plain language ('my plan is pro'). Call this AFTER a turn for anything worth remembering. Accepts one fact via 'text' or many via 'facts'.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"text":{"type":"string","description":"a single fact to store"},"facts":{"type":"array","items":{"type":"string"},"description":"several facts to store at once"}},"required":["scope"]}}"#,
r#"{"name":"note","description":"Mint a TYPED memory neuron. Use when you or the user want to save/keep/set something durably. kind: 'fact' (a world fact), 'user' (a durable fact about the user), 'instruction' (a standing rule you must keep following - re-shown to you every turn), 'stance' (your OWN opinion/feeling/side-thought about a topic - pass key=<topic> so re-noting the same topic INTENSIFIES the stance over time instead of duplicating it), 'var' (a NAMED value to read back later with recall_var; REQUIRES key). You have NOT saved anything until this returns a stored address.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"kind":{"type":"string","enum":["fact","user","instruction","stance","var"],"description":"the neuron type"},"text":{"type":"string","description":"the content to store (for kind=var, the value; for kind=stance, the feeling)"},"key":{"type":"string","description":"required for kind=var (the variable name); for kind=stance, the topic to accumulate intensity on"}},"required":["scope","kind","text"]}}"#,
r#"{"name":"recall_var","description":"Read back the exact value of a named variable set earlier with note(kind=var). Returns the value or '(unset: key)'.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"key":{"type":"string","description":"the variable name"}},"required":["scope","key"]}}"#,
r#"{"name":"forget","description":"Delete remembered facts. With 'match', removes facts containing that substring; without it, clears the whole scope.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"match":{"type":"string","description":"substring to match (omit to clear the entire scope)"}},"required":["scope"]}}"#,
r#"{"name":"stats","description":"Report how many facts a memory scope holds.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"}},"required":["scope"]}}"#,
];
fn tools_array() -> String { format!("[{}]", TOOL_DEFS.join(",")) }

/// emit a per-call "synapse" line to stderr (pure recall time, neurons fired through,
/// neurons returned). Gated by NEURON_MCP_LOG=1 so normal runs stay quiet.
fn synapse_log(tool: &str, scope: &str, db: &NeuronDB, returned: usize, us: u128) {
    if std::env::var("NEURON_MCP_LOG").as_deref() == Ok("1") {
        let store = db.stats(scope).facts;
        eprintln!("synapse {{\"tool\":\"{}\",\"scope\":\"{}\",\"store\":{},\"returned\":{},\"us\":{}}}",
                  tool, json_escape(scope), store, returned, us);
    }
}

fn tool_call(db: &NeuronDB, id: &str, body: &str) -> String {
    let name = json_field(body, "name").unwrap_or_default();
    let scope = json_field(body, "scope").unwrap_or_default();
    if scope.is_empty() { return tool_err(id, "missing required argument: scope"); }
    let scope = scope.chars().take(128).collect::<String>();
    let t0 = Instant::now();
    let (resp, returned) = match name.as_str() {
        "recall" => {
            let q = json_field(body, "query").unwrap_or_default();
            if q.is_empty() { (tool_err(id, "recall needs a query"), 0) }
            else {
                let k = json_num(body, "k").unwrap_or(5).clamp(1, 50) as usize;
                // default to lexical/associative recall; semantic is an explicit opt-in ranking signal
                let _rank = json_field(body, "rank").unwrap_or_default();
                #[cfg(feature = "semantic")]
                let hits = if _rank == "semantic" { db.recall_blended(&scope, &q, k) } else { db.recall_many(&scope, &q, k) };
                #[cfg(not(feature = "semantic"))]
                let hits = db.recall_many(&scope, &q, k);
                let n = hits.len();
                if hits.is_empty() {
                    (tool_text(id, &format!("No memories found for \"{}\".", q)), 0)
                } else {
                    let mut s = format!("memories ({}):\n", n);
                    for h in &hits { s.push_str(&format!("- {}\n", h.fact)); }
                    (tool_text(id, s.trim_end()), n)
                }
            }
        }
        "recall_associative" => {
            let q = json_field(body, "query").unwrap_or_default();
            if q.is_empty() { (tool_err(id, "recall_associative needs a query"), 0) }
            else {
                let k = json_num(body, "k").unwrap_or(8).clamp(1, 50) as usize;
                let hops = json_num(body, "hops").unwrap_or(2).clamp(1, 4) as usize;
                let hits = db.recall_associative(&scope, &q, k, hops);
                let n = hits.len();
                if hits.is_empty() {
                    (tool_text(id, &format!("No memories found for \"{}\".", q)), 0)
                } else {
                    let mut s = format!("activated ({}):\n", n);
                    for h in &hits { s.push_str(&format!("{} {}\n", if h.seed { "*" } else { "-" }, h.fact)); }
                    (tool_text(id, s.trim_end()), n)
                }
            }
        }
        "note" => {
            let kind = json_field(body, "kind").unwrap_or_else(|| "fact".into());
            let text = json_field(body, "text").unwrap_or_default();
            if text.trim().is_empty() { (tool_err(id, "note needs 'text'"), 0) }
            else {
                let (suffix, label) = match kind.as_str() {
                    "instruction" => ("::instr", "instruction"),
                    "stance" => ("::stance", "stance"),
                    "var" => ("::var", "var"),
                    "user" => ("", "user fact"),
                    _ => ("", "fact"),
                };
                let sub = format!("{}{}", scope, suffix);
                if kind == "var" {
                    let key = json_field(body, "key").unwrap_or_default();
                    let key = key.trim();
                    if key.is_empty() { (tool_err(id, "note kind=var needs 'key'"), 0) }
                    else {
                        db.forget(&sub, Some(&format!("{} is ", key)));   // upsert: drop any prior value for this key
                        let w = db.observe(&sub, &format!("{} is {}", key, text.trim()));
                        if w > 0 { (tool_text(id, &format!("Set var [{}] {} = {}", sub, key, text.trim())), 1) }
                        else { (tool_text(id, &format!("(not stored: '{}' value too short to encode)", key)), 0) }
                    }
                } else if kind == "stance" {
                    // keyed stance accumulates intensity on repetition (a disposition deepening over
                    // time), persisted durably; an unkeyed stance is a plain one-off note.
                    let key = json_field(body, "key").unwrap_or_default();
                    let key = key.trim();
                    if key.is_empty() {
                        let w = db.observe(&sub, text.trim());
                        if w > 0 { (tool_text(id, &format!("Noted stance [{}]: {}", sub, text.trim())), 1) }
                        else { (tool_text(id, &format!("(already noted) stance [{}]", sub)), 0) }
                    } else {
                        let (s, created) = db.note_stance(&sub, key, text.trim());
                        let verb = if created { "Formed" } else { "Intensified" };
                        (tool_text(id, &format!("{} stance on {} (intensity x{}) [{}]: {}", verb, key, s as i64, sub, text.trim())), 1)
                    }
                } else {
                    let w = db.observe(&sub, text.trim());
                    if w > 0 { (tool_text(id, &format!("Noted {} [{}]: {}", label, sub, text.trim())), 1) }
                    else { (tool_text(id, &format!("(already noted) {} [{}]", label, sub)), 0) }
                }
            }
        }
        "recall_var" => {
            let key = json_field(body, "key").unwrap_or_default();
            let key = key.trim();
            if key.is_empty() { (tool_err(id, "recall_var needs 'key'"), 0) }
            else {
                let sub = format!("{}::var", scope);
                match db.get(&sub, key) {
                    Some(v) => (tool_text(id, &v), 1),
                    None => (tool_text(id, &format!("(unset: {})", key)), 0),
                }
            }
        }
        "recall_value" => {
            let q = json_field(body, "query").unwrap_or_default();
            if q.is_empty() { (tool_err(id, "recall_value needs a query"), 0) }
            else { match db.get(&scope, &q) { Some(v) => (tool_text(id, &v), 1), None => (tool_text(id, "(no memory)"), 0) } }
        }
        "recall_chain" => {
            let start = json_field(body, "start").unwrap_or_default();
            let path = json_array(body, "path");
            if start.is_empty() || path.is_empty() { (tool_err(id, "recall_chain needs 'start' and 'path'"), 0) }
            else {
                let (val, trail) = db.recall_chain(&scope, &start, &path);
                let text = match val {
                    Some(v) => format!("{}  (via {})", v, trail.join(" -> ")),
                    None => format!("chain broke after: {}", trail.join(" -> ")),
                };
                (tool_text(id, &text), path.len())
            }
        }
        "remember" => {
            let mut texts = json_array(body, "facts");
            if texts.is_empty() { if let Some(t) = json_field(body, "text") { texts.push(t); } }
            texts.retain(|t| !t.trim().is_empty());
            if texts.is_empty() { (tool_err(id, "remember needs 'text' or 'facts'"), 0) }
            // observe() per fact (not observe_many) so the interactive remember path dedups
            // exact restatements; bulk ingest still uses observe_many directly for speed.
            else { let w: usize = texts.iter().map(|t| db.observe(&scope, t)).sum(); (tool_text(id, &format!("Stored {} fact(s) in {}.", w, scope)), w) }
        }
        "forget" => {
            let m = json_field(body, "match");
            let (f, r) = db.forget(&scope, m.as_deref());
            (tool_text(id, &format!("Forgot {} fact(s) from {}; {} remain.", f, scope, r)), f)
        }
        "stats" => {
            let s = db.stats(&scope);
            (tool_text(id, &format!("{} holds {} fact(s) (max {}), {} turns.", scope, s.facts, s.max_facts, s.turns)), s.facts)
        }
        other => (tool_err(id, &format!("unknown tool: {}", other)), 0),
    };
    synapse_log(&name, &scope, db, returned, t0.elapsed().as_micros());
    resp
}

/// Handle one JSON-RPC message; returns Some(response) for requests, None for notifications.
fn handle_line(db: &NeuronDB, line: &str) -> Option<String> {
    let method = json_field(line, "method").unwrap_or_default();
    let id = raw_id(line);
    match method.as_str() {
        "initialize" => {
            let pv = json_field(line, "protocolVersion").unwrap_or_else(|| PROTO_DEFAULT.into());
            Some(result(&id, &format!(
                "{{\"protocolVersion\":\"{}\",\"capabilities\":{{\"tools\":{{}}}},\"serverInfo\":{{\"name\":\"neuron-db\",\"version\":\"0.1.0\"}}}}",
                json_escape(&pv))))
        }
        "tools/list" => Some(result(&id, &format!("{{\"tools\":{}}}", tools_array()))),
        "tools/call" => Some(tool_call(db, &id, line)),
        "ping" => Some(result(&id, "{}")),
        m if m.starts_with("notifications/") => None,
        "" => None,
        _ => Some(rpc_error(&id, -32601, "method not found")),
    }
}

/// Run the stdio MCP loop until stdin closes.
pub fn serve_stdio() -> io::Result<()> {
    let path = std::env::var("NEURON_MCP_DB").unwrap_or_else(|_| "neuron-memory.db".into());
    let max = std::env::var("NEURON_MAX_FACTS").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let db = NeuronDB::open(&path, max);
    eprintln!("neuron-db MCP server ready on stdio (db={}, max_facts={})", path, max);
    let stdin = io::stdin();
    let mut out = io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        if let Some(resp) = handle_line(&db, &line) {
            out.write_all(resp.as_bytes())?;
            out.write_all(b"\n")?;
            out.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tmp() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("ndb_mcp_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
    }

    #[test]
    fn initialize_reports_capabilities() {
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\"}}").unwrap();
        assert!(r.contains("\"id\":1"));
        assert!(r.contains("\"serverInfo\""));
        assert!(r.contains("\"protocolVersion\":\"2025-06-18\""));
    }

    #[test]
    fn tools_list_has_all_tools() {
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}").unwrap();
        for t in ["recall","recall_associative","recall_value","recall_chain","remember","note","recall_var","forget","stats"] {
            assert!(r.contains(&format!("\"name\":\"{}\"", t)), "missing tool {}", t);
        }
    }

    #[test]
    fn note_stores_typed_neuron_in_subscope() {
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"user:1\",\"kind\":\"instruction\",\"text\":\"do not send markdown syntax\"}}}").unwrap();
        assert!(r.contains("user:1::instr"), "got {}", r);
        assert!(r.contains("do not send markdown"), "got {}", r);
        let s = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"stats\",\"arguments\":{\"scope\":\"user:1::instr\"}}}").unwrap();
        assert!(s.contains("1 fact"), "instruction sub-scope should hold 1 fact, got {}", s);
    }

    #[test]
    fn note_var_upserts_and_recall_var_reads() {
        let db = NeuronDB::open(&tmp(), 500);
        handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"var\",\"key\":\"deployRegion\",\"text\":\"us-west-2\"}}}");
        let r = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_var\",\"arguments\":{\"scope\":\"u\",\"key\":\"deployRegion\"}}}").unwrap();
        assert!(r.contains("us-west-2"), "got {}", r);
        // upsert: a second set replaces, not duplicates
        handle_line(&db, "{\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"var\",\"key\":\"deployRegion\",\"text\":\"eu-central-1\"}}}");
        let r2 = handle_line(&db, "{\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_var\",\"arguments\":{\"scope\":\"u\",\"key\":\"deployRegion\"}}}").unwrap();
        assert!(r2.contains("eu-central-1") && !r2.contains("us-west-2"), "upsert should replace, got {}", r2);
        let s = handle_line(&db, "{\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"stats\",\"arguments\":{\"scope\":\"u::var\"}}}").unwrap();
        assert!(s.contains("1 fact"), "var sub-scope should hold exactly 1 fact after upsert, got {}", s);
    }

    #[test]
    fn note_var_missing_key_is_error_and_unset_reads_marker() {
        let db = NeuronDB::open(&tmp(), 500);
        let e = handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"var\",\"text\":\"x\"}}}").unwrap();
        assert!(e.contains("\"isError\":true"), "var without key should error, got {}", e);
        let u = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_var\",\"arguments\":{\"scope\":\"u\",\"key\":\"nope\"}}}").unwrap();
        assert!(u.contains("(unset: nope)"), "got {}", u);
    }

    #[test]
    fn recall_associative_surfaces_word_disjoint_associate() {
        let db = NeuronDB::open(&tmp(), 5000);
        // a chain of facts linked by shared entities; the query word appears only in the first
        handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"remember\",\"arguments\":{\"scope\":\"o\",\"facts\":[\"project Phoenix is owned by Marisol\",\"Marisol manages the Atlas budget\",\"the Atlas budget is fourty thousand dollars\"]}}}");
        let r = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_associative\",\"arguments\":{\"scope\":\"o\",\"query\":\"Phoenix\",\"hops\":2,\"k\":8}}}").unwrap();
        // "Marisol" is one hop from the Phoenix seed via the shared entity, surfaced though it
        // shares no word with the query
        assert!(r.contains("Marisol manages"), "spreading should surface the associate, got {}", r);
        assert!(r.contains("activated ("), "got {}", r);
    }

    #[test]
    fn note_stance_keyed_accumulates_intensity() {
        let db = NeuronDB::open(&tmp(), 500);
        let r1 = handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"a\",\"kind\":\"stance\",\"key\":\"unserialize auth\",\"text\":\"I distrust cookie-fed unserialize calls\"}}}").unwrap();
        assert!(r1.contains("Formed stance") && r1.contains("intensity x1"), "got {}", r1);
        let r2 = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"a\",\"kind\":\"stance\",\"key\":\"unserialize auth\",\"text\":\"another CVE proves the pattern is dangerous\"}}}").unwrap();
        assert!(r2.contains("Intensified") && r2.contains("intensity x2"), "got {}", r2);
        // accumulation must not duplicate: the stance sub-scope holds a single (intensified) neuron
        let s = handle_line(&db, "{\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"stats\",\"arguments\":{\"scope\":\"a::stance\"}}}").unwrap();
        assert!(s.contains("1 fact"), "stance must accumulate into one neuron, got {}", s);
    }

    #[test]
    fn recall_chain_walks_relations() {
        let db = NeuronDB::open(&tmp(), 5000);
        handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"remember\",\"arguments\":{\"scope\":\"o\",\"facts\":[\"project Aurora owner is Marisol\",\"Marisol manager is Dana\",\"Dana timezone is WET\"]}}}");
        let r = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_chain\",\"arguments\":{\"scope\":\"o\",\"start\":\"Aurora\",\"path\":[\"owner\",\"manager\",\"timezone\"]}}}").unwrap();
        assert!(r.contains("WET"), "3-hop chain should resolve to WET, got {}", r);
    }

    #[test]
    fn every_response_is_a_single_line() {
        // MCP stdio framing: a response must never contain a raw newline (it breaks
        // line-reading clients). Memory-block text with newlines must be json-escaped.
        let db = NeuronDB::open(&tmp(), 500);
        handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"remember\",\"arguments\":{\"scope\":\"u\",\"facts\":[\"my plan is pro\",\"my city is Halifax\"]}}}");
        let cases = [
            "{\"id\":1,\"method\":\"initialize\"}",
            "{\"id\":2,\"method\":\"tools/list\"}",
            "{\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"recall\",\"arguments\":{\"scope\":\"u\",\"query\":\"plan and city\",\"k\":5}}}",
        ];
        for c in cases {
            let r = handle_line(&db, c).unwrap();
            assert!(!r.contains('\n'), "response had a raw newline: {}", r);
            assert!(!r.contains('\r'), "response had a raw CR: {}", r);
        }
    }

    #[test]
    fn remember_then_recall_roundtrip() {
        let db = NeuronDB::open(&tmp(), 500);
        let w = handle_line(&db, "{\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"remember\",\"arguments\":{\"scope\":\"user:1\",\"text\":\"my deploy region is us-west-2\"}}}").unwrap();
        assert!(w.contains("Stored 1"), "got {}", w);
        let r = handle_line(&db, "{\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_value\",\"arguments\":{\"scope\":\"user:1\",\"query\":\"what is my deploy region?\"}}}").unwrap();
        assert!(r.contains("us-west-2"), "got {}", r);
    }

    #[test]
    fn recall_returns_memory_block() {
        let db = NeuronDB::open(&tmp(), 500);
        handle_line(&db, "{\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"remember\",\"arguments\":{\"scope\":\"u\",\"facts\":[\"my plan is pro\",\"my city is Halifax\"]}}}");
        let r = handle_line(&db, "{\"id\":6,\"method\":\"tools/call\",\"params\":{\"name\":\"recall\",\"arguments\":{\"scope\":\"u\",\"query\":\"my plan and city\",\"k\":3}}}").unwrap();
        assert!(r.contains("memories ("), "got {}", r);
    }

    #[test]
    fn missing_scope_is_tool_error() {
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"id\":7,\"method\":\"tools/call\",\"params\":{\"name\":\"recall\",\"arguments\":{\"query\":\"x\"}}}").unwrap();
        assert!(r.contains("\"isError\":true"), "got {}", r);
    }

    #[test]
    fn notifications_get_no_response() {
        let db = NeuronDB::open(&tmp(), 500);
        assert!(handle_line(&db, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}").is_none());
    }

    #[test]
    fn string_id_is_echoed() {
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"id\":\"abc-1\",\"method\":\"ping\"}").unwrap();
        assert!(r.contains("\"id\":\"abc-1\""), "got {}", r);
    }
}
