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
use crate::op::{apply, NeuronOp, OpResult};

const PROTO_DEFAULT: &str = "2025-06-18";
/// The MCP protocol versions the server actually speaks. `initialize` echoes the client's requested
/// version only when it is one of these; for anything else it answers with `PROTO_DEFAULT` (its own
/// latest), per the spec — so the client sees a real version it can accept or disconnect on, rather
/// than neuron rubber-stamping a version it doesn't support.
const SUPPORTED_PROTO: [&str; 1] = [PROTO_DEFAULT];

// ---- minimal JSON helpers (flat-search; mirrors server.rs) ----
use crate::json_escape;
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
    // scan from after '[' honoring string quoting — a ']' INSIDE an element must not end the array
    let chars: Vec<char> = after[lb + 1..].chars().collect();
    let (mut out, mut j) = (Vec::new(), 0);
    while j < chars.len() {
        match chars[j] {
            ']' => break,                  // real array terminator (we are not inside a string here)
            '"' => {
                let mut sb = String::new(); j += 1;
                while j < chars.len() {
                    let c = chars[j];
                    if c == '\\' && j + 1 < chars.len() { let n = chars[j+1]; sb.push(match n {'n'=>'\n','t'=>'\t','r'=>'\r','"'=>'"','\\'=>'\\',o=>o}); j += 2; }
                    else if c == '"' { j += 1; break; } else { sb.push(c); j += 1; }
                }
                out.push(sb);
            }
            _ => j += 1,
        }
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
        // find the real closing quote via escape PARITY (an odd backslash run escapes the quote; an
        // even one leaves it a real close — checking only the single preceding char mis-reads `\\"`).
        let mut out = String::from("\""); let mut j = 1; let (mut esc, mut closed) = (false, false);
        while j < chars.len() {
            let ch = chars[j]; out.push(ch);
            if esc { esc = false; } else if ch == '\\' { esc = true; } else if ch == '"' { closed = true; break; }
            j += 1;
        }
        // echo the token ONLY if it is a fully well-formed JSON string (no raw control char, every
        // escape legal, `\u` four-hex with correct surrogate pairing). The id is interpolated into the
        // response WITHOUT escaping, so anything less would emit invalid JSON — fail safe to null, as
        // the numeric branch does (this covers unterminated, raw-control, and malformed-`\u` ids).
        let oc: Vec<char> = out.chars().collect();
        return if closed && is_valid_json_string(&oc) { out } else { "null".into() };
    }
    // the numeric branch must yield a VALID JSON number or null — `take_while(digit|'-')` alone admits
    // malformed runs (`-`, `--5`, `5-3`, `-9-9`). Echo the token verbatim only if it is a well-formed
    // JSON integer (preserving an arbitrary-width id exactly, for correlation), else null.
    let val: String = tail.chars().take_while(|c| c.is_ascii_digit() || *c == '-').collect();
    if is_json_int(&val) { val } else { "null".into() }
}
/// True iff `s` is a syntactically valid JSON integer: an optional leading `-`, then a single `0` or a
/// non-zero-leading digit run (no leading zeros, no interior/repeated signs). Used to decide whether a
/// numeric id can be echoed verbatim into a response or must fall back to `null`.
fn is_json_int(s: &str) -> bool {
    let d = s.strip_prefix('-').unwrap_or(s);
    !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()) && (d == "0" || d.as_bytes()[0] != b'0')
}
/// Read exactly four hex digits of a `\uXXXX` escape starting at `at`, as a code unit. `None` if any
/// of the four is missing or non-hex (a truncated / malformed `\u`).
fn hex4(s: &[char], at: usize) -> Option<u32> {
    let mut v = 0u32;
    for k in 0..4 { v = v * 16 + s.get(at + k)?.to_digit(16)?; }
    Some(v)
}
/// True iff `s` (a char slice opening with `"`) is a complete, well-formed JSON string token per RFC
/// 8259: it closes with `"` as its LAST char, holds no raw control char (U+0000..U+001F), and every
/// `\` begins a legal escape — `\uXXXX` needs four hex digits, and a high surrogate (D800..DBFF) must
/// be followed by a `\u` low surrogate (DC00..DFFF); a lone/truncated surrogate is rejected. Gate for
/// echoing a string id verbatim into the unescaped response slot so the line is always valid JSON.
fn is_valid_json_string(s: &[char]) -> bool {
    if s.first() != Some(&'"') { return false; }
    let mut i = 1;
    loop {
        let c = match s.get(i) { Some(&c) => c, None => return false };  // ran off the end: never closed
        if c == '"' { return i == s.len() - 1; }                         // real close must be the final char
        if (c as u32) < 0x20 { return false; }                            // raw control char: illegal unescaped
        if c == '\\' {
            match s.get(i + 1) {
                Some('"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't') => i += 2,
                Some('u') => match hex4(s, i + 2) {
                    Some(hi) if (0xD800..=0xDBFF).contains(&hi) => {
                        // high surrogate -> must be paired with a following `\u` low surrogate
                        if s.get(i + 6) != Some(&'\\') || s.get(i + 7) != Some(&'u') { return false; }
                        match hex4(s, i + 8) { Some(lo) if (0xDC00..=0xDFFF).contains(&lo) => i += 12, _ => return false }
                    }
                    Some(cu) if (0xDC00..=0xDFFF).contains(&cu) => return false,   // lone low surrogate
                    Some(_) => i += 6,
                    None => return false,                                          // truncated / non-hex `\u`
                },
                _ => return false,                                                // illegal / dangling escape
            }
        } else {
            i += 1;
        }
    }
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
/// Render `["a","b",...]` from cap names (escaped defensively, though names are static identifiers).
fn json_str_list(items: &[&str]) -> String {
    format!("[{}]", items.iter().map(|s| format!("\"{}\"", json_escape(s))).collect::<Vec<_>>().join(","))
}
/// The TOP-LEVEL keys of the MCP `capabilities` object (e.g. `sampling`, `roots`, `experimental`).
/// A name counts only when it sits in KEY position directly inside the object — right after the
/// object's `{` or a `,`, at object-depth 1 and NOT inside any array, AND followed by `:`. That
/// follows the JSON object grammar (`{ key : value , … }`), so it can never be confused with: a
/// sibling field outside the object; a key nested deeper (e.g. `experimental.note`); a value inside
/// an array; or a string *value* that merely spells "sampling" — even one followed by a stray colon
/// in malformed input, because a value is not in key position. The scan tracks object-depth,
/// array-depth, and string/escape state and walks `chars`, so it is structure-correct and multi-byte
/// safe. Fails closed — an absent / non-object / unbalanced `capabilities` value yields no keys (cede
/// nothing). Used by `host_deferrable`; the only consumer.
fn capability_keys(line: &str) -> Vec<String> {
    const KEY: &str = "\"capabilities\"";
    // require the key's OWN colon (only whitespace between key and `:`), then an object `{` — so a
    // non-object value (`"capabilities":null`) or a key with no colon can't latch onto a sibling's.
    let after = match line.find(KEY) { Some(i) => &line[i + KEY.len()..], None => return vec![] };
    let body = match after.trim_start().strip_prefix(':') { Some(r) => r.trim_start(), None => return vec![] };
    if !body.starts_with('{') { return vec![]; }
    let (mut depth, mut adepth, mut in_str, mut esc) = (0i32, 0i32, false, false);
    let mut cur = String::new();             // the quoted string currently being read
    let mut expect_key = false;              // are we in KEY position directly inside the object?
    let mut pending: Option<String> = None;  // a top-level key string awaiting its `:` to be committed
    let mut keys = Vec::new();
    let top = |depth: i32, adepth: i32| depth == 1 && adepth == 0;  // directly inside the object
    for c in body.chars() {
        if in_str {
            match c {
                '\\' if !esc => { esc = true; cur.push(c); }
                '"' if !esc => {
                    in_str = false;
                    if top(depth, adepth) && expect_key { pending = Some(std::mem::take(&mut cur)); }
                    expect_key = false;      // a string closes the key-expecting slot
                }
                _ => { esc = false; cur.push(c); }
            }
        } else {
            match c {
                '"' => { in_str = true; cur.clear(); }
                '{' => { depth += 1; if top(depth, adepth) { expect_key = true; pending = None; } }
                '}' => { depth -= 1; if depth == 0 { return keys; } }
                '[' => adepth += 1,
                ']' => if adepth > 0 { adepth -= 1; },
                ':' if top(depth, adepth) => { if let Some(k) = pending.take() { keys.push(k); } }
                ',' if top(depth, adepth) => { expect_key = true; pending = None; }
                _ => {}
            }
        }
    }
    vec![]   // unbalanced — never saw the closing brace, so trust nothing (fail closed)
}
/// Map the host's advertised MCP client capabilities to the neuron *deferrable* capability names
/// they enable. `sampling` (the host can run an LLM on our behalf) unlocks the model-class text work
/// — summarize/embed/normalize; `roots` (the host exposes resource roots) unlocks `fetch`. Reads
/// only the TOP-LEVEL keys of the `capabilities` object (`capability_keys`), so a sibling field, a
/// nested key, or a string value spelling "sampling"/"roots" can't over-cede. These are only the
/// names neuron MIGHT cede — `caps::resolve` makes the final call, so a grounded cap never yields.
fn host_deferrable(line: &str) -> Vec<&'static str> {
    let keys = capability_keys(line);
    let mut out = Vec::new();
    if keys.iter().any(|k| k == "sampling") { out.extend(["summarize", "embed", "normalize"]); }
    if keys.iter().any(|k| k == "roots") { out.push("fetch"); }
    out
}

// tools/list payload. Each entry MUST be a single line: MCP stdio framing is one JSON
// message per physical line, so a response may never contain a raw newline. Names use
// snake_case for broad client compatibility.
const TOOL_DEFS: [&str; 10] = [
r#"{"name":"route","description":"Route a turn through the gary-neuron cortex — the cheap front gate. It recalls the working set for the query, then decides ONE route: answer (memory answers it; value is the answer), escalate (memory can't — hand the turn to YOUR model, using the returned facts as context), fetch (a live-world lookup; value is the topic to search), or store (a declarative to remember; value is the fact). Returns {type,value,facts}. Call this first each turn; only invoke your own model when type is escalate.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"query":{"type":"string","description":"the user's turn"},"k":{"type":"integer","description":"facts to recall as the working set (default 6)"}},"required":["scope","query"]}}"#,
r#"{"name":"recall","description":"Recall the most relevant remembered facts for a query, as a memory block to inject into context. Call this BEFORE answering whenever the user refers to something they may have told you earlier. Ranks by associative cue overlap; pass rank='semantic' for broad/narrative topics; pass documents=true to also search this user's stored documents.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id, e.g. user:123 or session:abc - isolates one user/agent's memory"},"query":{"type":"string","description":"the question or topic to recall about"},"k":{"type":"integer","description":"max facts to return (default 5)"},"rank":{"type":"string","enum":["lexical","semantic"],"description":"ranking strategy; default lexical (associative). Use semantic only for broad/narrative recall."},"documents":{"type":"boolean","description":"also search the user's stored document sub-scopes, not just the main scope (default false)"}},"required":["scope","query"]}}"#,
r#"{"name":"recall_associative","description":"Spreading-activation recall: starts from facts that match the query, then follows shared-entity links to surface RELATED facts that may share no words with the query. Use for 'what's connected to X' or to gather context around a topic.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"query":{"type":"string","description":"the topic or entity to activate from"},"k":{"type":"integer","description":"max facts to return (default 8)"},"hops":{"type":"integer","description":"link hops to spread (default 2)"}},"required":["scope","query"]}}"#,
r#"{"name":"recall_value","description":"Recall a single best-matching value for a direct question (e.g. 'what is my plan?'). Returns the isolated value or '(no memory)'.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"query":{"type":"string","description":"a direct question"}},"required":["scope","query"]}}"#,
r#"{"name":"recall_chain","description":"Answer a multi-hop question in ONE call by walking a chain of relations server-side (no extra round-trips, any depth). Give the starting entity and the ordered relations to follow. Example: start 'Aurora', path ['owner','manager','timezone'] returns the timezone of the manager of the owner of Aurora.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"start":{"type":"string","description":"the entity to start from, e.g. 'Aurora' or 'Marisol'"},"path":{"type":"array","items":{"type":"string"},"description":"ordered relations to follow, e.g. ['owner','manager','timezone']"}},"required":["scope","start","path"]}}"#,
r#"{"name":"remember","description":"Store durable facts the user stated, in plain language ('my plan is pro'). Call this AFTER a turn for anything worth remembering. Accepts one fact via 'text' or many via 'facts'.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"text":{"type":"string","description":"a single fact to store"},"facts":{"type":"array","items":{"type":"string"},"description":"several facts to store at once"}},"required":["scope"]}}"#,
r#"{"name":"note","description":"Mint a TYPED memory neuron. Use when you or the user want to save/keep/set something durably. kind: 'fact' (a world fact), 'user' (a durable fact about the user), 'instruction' (a standing ALWAYS/NEVER rule for how you behave, e.g. 'always answer in caps' or 'never use markdown' - re-shown to you every turn; use this, NOT var, for any durable behavior rule), 'stance' (your OWN opinion/feeling/side-thought about a topic - pass key=<topic> so re-noting the same topic INTENSIFIES the stance over time instead of duplicating it), 'var' (a NAMED value/datum to read back later with recall_var - a setting or fact, NOT a behavior rule; REQUIRES key). You have NOT saved anything until this returns a stored address.","inputSchema":{"type":"object","properties":{"scope":{"type":"string","description":"memory scope id"},"kind":{"type":"string","enum":["fact","user","instruction","stance","var"],"description":"the neuron type"},"text":{"type":"string","description":"the content to store (for kind=var, the value; for kind=stance, the feeling)"},"key":{"type":"string","description":"required for kind=var (the variable name); for kind=stance, the topic to accumulate intensity on"}},"required":["scope","kind","text"]}}"#,
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
    // §7: the capability manifest — scope-independent meta, so it answers before the scope check.
    // Hidden from tools/list (not a memory op) but callable by name; tells a host what neuron can do
    // and which capabilities it owns (grounded) vs would yield to a richer host (deferrable). With a
    // `host` array of neuron cap names the host claims, it resolves the live keep/defer disposition
    // (parity with `neuron caps <names...>`); with no `host`, the static manifest. Grounded names
    // resolve to `keep` even if the host lists them — grounded beats tier.
    if name == "caps" {
        let host = json_array(body, "host");
        // drop empty entries so an all-empty/degenerate host (`host:[""]`) resolves to the manifest,
        // identical to a host that was omitted — keeps caps byte-identical with the CLI and WASM.
        let hv: Vec<&str> = host.iter().map(String::as_str).filter(|s| !s.is_empty()).collect();
        if hv.is_empty() { return tool_text(id, &crate::caps::manifest()); }
        let lines = crate::caps::resolve(&hv).into_iter()
            .map(|(n, d)| format!("{}\t{}", n, d.tag())).collect::<Vec<_>>().join("\n");
        return tool_text(id, &lines);
    }
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
                let semantic = json_field(body, "rank").unwrap_or_default() == "semantic";
                let across = body.contains("\"documents\":true") || body.contains("\"documents\": true");
                let hits = apply(db, NeuronOp::Recall { scope: scope.clone(), query: q.clone(), k, semantic, across }).hits();
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
                let hits = apply(db, NeuronOp::RecallAssoc { scope: scope.clone(), query: q.clone(), k, hops }).assoc();
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
            if !matches!(kind.as_str(), "fact"|"user"|"instruction"|"stance"|"var") {
                // reject misspelled/hallucinated kinds loudly instead of silently filing a base-scope
                // fact (which would drop e.g. a standing instruction on the floor)
                return tool_err(id, &format!("unknown kind '{}'; valid: fact|user|instruction|stance|var", kind));
            }
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
                        // anchored upsert (no key collision)
                        let w = apply(db, NeuronOp::VarSet { scope: sub.clone(), key: key.to_string(), value: text.trim().to_string() }).wrote();
                        if w > 0 { (tool_text(id, &format!("Set var [{}] {} = {}", sub, key, text.trim())), 1) }
                        else { (tool_text(id, &format!("(not stored: '{}' value too short to encode)", key)), 0) }
                    }
                } else if kind == "stance" {
                    // keyed stance accumulates intensity on repetition (a disposition deepening over
                    // time), persisted durably; an unkeyed stance is a plain one-off note.
                    let key = json_field(body, "key").unwrap_or_default();
                    let key = key.trim();
                    if key.is_empty() {
                        let w = apply(db, NeuronOp::Observe { scope: sub.clone(), text: text.trim().to_string() }).wrote();
                        if w > 0 { (tool_text(id, &format!("Noted stance [{}]: {}", sub, text.trim())), 1) }
                        else { (tool_text(id, &format!("(already noted) stance [{}]", sub)), 0) }
                    } else {
                        let (s, created) = match apply(db, NeuronOp::Stance { scope: sub.clone(), topic: key.to_string(), feeling: text.trim().to_string() }) {
                            OpResult::Stance { intensity, created } => (intensity, created), _ => (0.0, false),
                        };
                        if s == 0.0 { (tool_text(id, "(not stored: stance text too short to encode)"), 0) }
                        else {
                            let verb = if created { "Formed" } else { "Intensified" };
                            (tool_text(id, &format!("{} stance on {} (intensity x{}) [{}]: {}", verb, key, s as i64, sub, text.trim())), 1)
                        }
                    }
                } else {
                    let w = apply(db, NeuronOp::Observe { scope: sub.clone(), text: text.trim().to_string() }).wrote();
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
                match apply(db, NeuronOp::VarGet { scope: sub, key: key.to_string() }).value() {
                    Some(v) => (tool_text(id, &v), 1),
                    None => (tool_text(id, &format!("(unset: {})", key)), 0),
                }
            }
        }
        "recall_value" => {
            let q = json_field(body, "query").unwrap_or_default();
            if q.is_empty() { (tool_err(id, "recall_value needs a query"), 0) }
            // main scope first, then a cross-document fallback — both live in apply(RecallValue)
            else {
                match apply(db, NeuronOp::RecallValue { scope: scope.clone(), query: q }).value() {
                    Some(v) => (tool_text(id, &v), 1), None => (tool_text(id, "(no memory)"), 0),
                }
            }
        }
        "recall_chain" => {
            let start = json_field(body, "start").unwrap_or_default();
            let path = json_array(body, "path");
            if start.is_empty() || path.is_empty() { (tool_err(id, "recall_chain needs 'start' and 'path'"), 0) }
            else {
                let n = path.len();
                let text = match apply(db, NeuronOp::RecallChain { scope: scope.clone(), start, path }) {
                    OpResult::Chain { value: Some(v), trail } => format!("{}  (via {})", v, trail.join(" -> ")),
                    OpResult::Chain { value: None, trail } => format!("chain broke after: {}", trail.join(" -> ")),
                    _ => String::new(),
                };
                (tool_text(id, &text), n)
            }
        }
        "remember" => {
            let mut texts = json_array(body, "facts");
            if texts.is_empty() { if let Some(t) = json_field(body, "text") { texts.push(t); } }
            texts.retain(|t| !t.trim().is_empty());
            if texts.is_empty() { (tool_err(id, "remember needs 'text' or 'facts'"), 0) }
            // observe() per fact (ObserveMany) so the interactive remember path dedups exact
            // restatements; bulk ingest still uses db.observe_many directly for speed.
            else { let w = apply(db, NeuronOp::ObserveMany { scope: scope.clone(), texts }).wrote(); (tool_text(id, &format!("Stored {} fact(s) in {}.", w, scope)), w) }
        }
        "forget" => {
            let m = json_field(body, "match");
            let (f, r) = match apply(db, NeuronOp::Forget { scope: scope.clone(), matching: m }) {
                OpResult::Forgot { forgot, remaining } => (forgot, remaining), _ => (0, 0),
            };
            (tool_text(id, &format!("Forgot {} fact(s) from {}; {} remain.", f, scope, r)), f)
        }
        "stats" => {
            match apply(db, NeuronOp::Stats { scope: scope.clone() }) {
                OpResult::Stats(s) => (tool_text(id, &format!("{} holds {} fact(s) (max {}), {} turns.", scope, s.facts, s.max_facts, s.turns)), s.facts),
                _ => (tool_err(id, "stats failed"), 0),
            }
        }
        // affective layer — handled but intentionally NOT advertised in tools/list (the harness
        // calls these by name; the mood override is the optional variable passed into the store).
        "feel" => {
            let emo = json_field(body, "emotion").unwrap_or_default();
            let emo = emo.trim();
            apply(db, NeuronOp::Mood { scope: scope.clone(), emotion: emo.to_string() });
            if emo.is_empty() { (tool_text(id, "(mood cleared; back to auto)"), 0) }
            else { (tool_text(id, &format!("now feeling {}", emo)), 1) }
        }
        "humanize" => {
            // optional topic biases the persona toward the asked-about stance (else the strongest)
            let topic = json_field(body, "topic").filter(|t| !t.trim().is_empty());
            (tool_text(id, &apply(db, NeuronOp::Affect { scope: scope.clone(), topic }).text()), 1)
        }
        // the cortex route — recall the working set, then dispatch the turn through gary-neuron. Returns
        // {type, value, facts}; the client acts on type (escalate -> its own model, with facts as
        // context). The cheap front gate so the host LLM runs only when memory genuinely can't answer.
        // (`mcp` requires the `cortex` feature, so this is always present — parity with CLI + WASM.)
        "route" => {
            let query = json_field(body, "query").unwrap_or_default();
            if query.trim().is_empty() { (tool_err(id, "missing required argument: query"), 0) }
            else {
                let k = json_num(body, "k").map(|n| n.max(1) as usize).unwrap_or(6);
                let r = crate::route::route(db, &scope, &query, k);
                let facts = r.facts.iter().map(|f| format!("\"{}\"", json_escape(f))).collect::<Vec<_>>().join(",");
                (tool_text(id, &format!("{{\"type\":\"{}\",\"value\":\"{}\",\"facts\":[{}]}}", r.kind, json_escape(&r.value), facts)), r.facts.len())
            }
        }
        other => (tool_err(id, &format!("unknown tool: {}", other)), 0),
    };
    synapse_log(&name, &scope, db, returned, t0.elapsed().as_micros());
    resp
}

/// Handle one JSON-RPC message; returns Some(response) for requests, None for notifications.
/// Public so the line handler can be embedded or parity-tested without spawning the stdio loop.
pub fn handle_line(db: &NeuronDB, line: &str) -> Option<String> {
    let method = json_field(line, "method").unwrap_or_default();
    let id = raw_id(line);
    match method.as_str() {
        "initialize" => {
            // answer with the client's requested version only if we actually speak it; otherwise our
            // own latest (PROTO_DEFAULT) — never rubber-stamp an unsupported/garbage version.
            let pv = match json_field(line, "protocolVersion") {
                Some(v) if SUPPORTED_PROTO.contains(&v.as_str()) => v,
                _ => PROTO_DEFAULT.to_string(),
            };
            // §7: read the CLIENT's advertised capabilities and ANSWER with neuron's negotiated
            // surface — instead of returning a static `{}`. Map the host's MCP caps to the neuron
            // *deferrable* names they unlock (sampling -> summarize/embed/normalize, roots -> fetch),
            // then let `caps::resolve` decide. Grounded caps (recall/chain/assess/var/stance/store)
            // are NEVER ceded, whatever the host claims. "Grounded beats tier." The result is carried
            // under `capabilities.experimental.neuron` (the MCP-blessed slot for non-standard caps):
            //   deferred = the store-free work neuron yields to THIS host (empty if it offered none),
            //   grounded = the store-touching work neuron always owns,
            //   manifestTool = the tool to call for the full name/owner/about manifest.
            let host_has = host_deferrable(line);
            let deferred = crate::caps::deferred_for(&host_has);
            let grounded = crate::caps::grounded_names();
            if std::env::var("NEURON_MCP_LOG").as_deref() == Ok("1") {
                eprintln!("mcp init: host unlocks {:?}; neuron defers {:?}, keeps grounded {:?}",
                          host_has, deferred, grounded);
            }
            let neuron = format!(
                "{{\"deferred\":{},\"grounded\":{},\"manifestTool\":\"caps\"}}",
                json_str_list(&deferred), json_str_list(&grounded));
            Some(result(&id, &format!(
                "{{\"protocolVersion\":\"{}\",\"capabilities\":{{\"tools\":{{}},\"experimental\":{{\"neuron\":{}}}}},\"serverInfo\":{{\"name\":\"neuron-db\",\"version\":\"0.1.0\"}}}}",
                json_escape(&pv), neuron)))
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
// Flush one scope's buffered facts through a single ObserveBulk (one persist). Returns count stored.
fn preload_flush(db: &NeuronDB, buffers: &mut std::collections::HashMap<String, Vec<String>>, scope: &str) -> usize {
    let texts = match buffers.get_mut(scope) { Some(v) if !v.is_empty() => std::mem::take(v), _ => return 0 };
    apply(db, NeuronOp::ObserveBulk { scope: scope.to_string(), texts }).wrote()
}

// A collision-proof default scope derived from the pack filename (so a pack with no per-fact scope
// lands somewhere unlikely to clash with a live runtime scope).
fn preload_default_scope(pack: &str) -> String {
    let base = pack.rsplit(['/', '\\']).next().unwrap_or(pack);
    format!("preload-{}", base.rsplit_once('.').map(|(s, _)| s).unwrap_or(base))
}

/// Load a fact pack into the durable store at boot. Streams the file, groups by scope, and writes
/// one ObserveBulk per scope-chunk. Per-scope idempotency: a scope that already holds facts is
/// skipped (so a restart with the env still set is a near no-op), unless NEURON_MCP_PRELOAD_FORCE=1
/// re-seeds it. All diagnostics go to stderr (stdout stays single-line-clean for the JSON-RPC framing).
fn preload_into(db: &NeuronDB, pack: &str) -> io::Result<()> {
    use std::io::BufRead;
    use std::collections::HashMap;
    use crate::preload::{parse_pack_line, resolve_scope, PackLine};
    let file = std::fs::File::open(pack)?;
    let force = std::env::var("NEURON_MCP_PRELOAD_FORCE").as_deref() == Ok("1");
    let default_scope = Some(std::env::var("NEURON_MCP_PRELOAD_SCOPE").ok().filter(|s| !s.is_empty())
        .unwrap_or_else(|| preload_default_scope(pack)));
    let mut buffers: HashMap<String, Vec<String>> = HashMap::new();
    let mut decision: HashMap<String, bool> = HashMap::new();   // scope -> load(true)/skip(false)
    let mut directive: Option<String> = None;
    let mut dropped = 0usize;
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        match parse_pack_line(&line) {
            PackLine::Skip | PackLine::Reject => { dropped += 1; }
            PackLine::Scope(s) => directive = Some(s),
            PackLine::Fact { scope, text } => {
                let target = match resolve_scope(&scope, &directive, &default_scope) { Some(s) => s.to_string(), None => { dropped += 1; continue; } };
                let load = *decision.entry(target.clone()).or_insert_with(|| {
                    let nonempty = matches!(apply(db, NeuronOp::Stats { scope: target.clone() }), OpResult::Stats(s) if s.facts > 0);
                    if nonempty && !force { false }
                    else { if nonempty && force { apply(db, NeuronOp::Forget { scope: target.clone(), matching: None }); } true }
                });
                if !load { continue; }
                // buffer the WHOLE scope; it is written in ONE ObserveBulk at EOF (all-or-nothing), so a
                // boot killed mid-preload leaves each scope either fully loaded or empty — never a
                // partial scope that the next boot's non-empty guard would then skip forever.
                buffers.entry(target).or_default().push(text);
            }
        }
    }
    let keys: Vec<String> = buffers.keys().cloned().collect();
    let loaded = keys.len();
    let (mut stored, mut evicted) = (0usize, 0u64);
    for k in &keys {
        preload_flush(db, &mut buffers, k);
        if let OpResult::Stats(s) = apply(db, NeuronOp::Stats { scope: k.clone() }) {
            stored += s.facts;
            if s.dropped > 0 { evicted += s.dropped; eprintln!("neuron-db preload: scope '{}' hit NEURON_MAX_FACTS; {} fact(s) evicted — raise it", k, s.dropped); }
        }
    }
    db.flush_all();
    let skipped_scopes = decision.values().filter(|&&v| !v).count();
    eprintln!("neuron-db preload: {} fact(s) into {} scope(s) from {} ({} already loaded, {} dropped, {} evicted)",
              stored, loaded, pack, skipped_scopes, dropped, evicted);
    Ok(())
}

pub fn serve_stdio() -> io::Result<()> {
    let path = std::env::var("NEURON_MCP_DB").unwrap_or_else(|_| "neuron-memory.db".into());
    let max = std::env::var("NEURON_MAX_FACTS").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    // NEURON_FLUSH_EVERY>1 opts into write-behind (persist every N observes; flushed on Drop/evict).
    // Default 1 = immediate per-write durability.
    let flush = std::env::var("NEURON_FLUSH_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let db = NeuronDB::open_with_flush(&path, max, flush);
    // Optional one-shot fact preload (NEURON_MCP_PRELOAD=<pack>): load a fact pack into the store at
    // boot, before the request loop, with zero lock contention. Pure opt-in (unset = skip). Wrapped
    // in catch_unwind so a preload failure (e.g. disk-full) logs to stderr and the server still
    // serves whatever persisted, rather than taking the request loop down with it.
    if let Ok(pack) = std::env::var("NEURON_MCP_PRELOAD") {
        if !pack.trim().is_empty() {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| preload_into(&db, &pack))) {
                Ok(Ok(())) => {}
                Ok(Err(e)) => eprintln!("neuron-db preload error: {}", e),
                Err(_) => eprintln!("neuron-db preload aborted (write error); serving what persisted"),
            }
        }
    }
    eprintln!("neuron-db MCP server ready on stdio (db={}, max_facts={})", path, max);
    let stdin = io::stdin();
    let mut out = io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        // Isolate each request: a panic mid-handler (e.g. SQLITE_FULL on a write) must not kill the
        // long-lived server. The store's locks de-poison on the next op, so it stays usable; we just
        // return a JSON-RPC internal error and keep serving.
        let resp = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle_line(&db, &line))) {
            Ok(r) => r,
            Err(_) => Some(r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"internal error"}}"#.to_string()),
        };
        if let Some(resp) = resp {
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
    fn initialize_with_no_host_caps_defers_nothing() {
        // §7: a bare host (advertises no capabilities) -> neuron yields nothing, keeps the grounded
        // floor. The handshake must carry the negotiated surface, not a static {}.
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{}}}").unwrap();
        assert!(r.contains("\"experimental\":{\"neuron\":"), "must advertise the neuron capability, got {}", r);
        assert!(r.contains("\"deferred\":[]"), "a bare host should be ceded nothing, got {}", r);
        assert!(r.contains("\"grounded\":[\"recall\""), "must list the grounded floor, got {}", r);
        assert!(r.contains("\"manifestTool\":\"caps\""), "got {}", r);
    }

    #[test]
    fn initialize_with_sampling_cedes_model_class_work_but_keeps_grounded() {
        // §7 grounded-beats-tier on the live wire: a sampling host (can run an LLM for us) is ceded
        // the store-free text work, never the store-touching ops.
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{\"sampling\":{},\"roots\":{\"listChanged\":true}}}}").unwrap();
        for ceded in ["summarize", "embed", "normalize", "fetch"] {
            assert!(r.contains(&format!("\"{}\"", ceded)), "sampling+roots host should be ceded {}, got {}", ceded, r);
        }
        // the deferred set must hold NONE of the grounded ops, even though the host can "sample"
        let deferred = r.split("\"deferred\":").nth(1).unwrap().split(']').next().unwrap();
        for grounded in ["recall", "chain", "assoc", "assess", "store", "var", "stance"] {
            assert!(!deferred.contains(&format!("\"{}\"", grounded)), "{} must never be ceded, got deferred={}", grounded, deferred);
        }
    }

    #[test]
    fn host_deferrable_is_scoped_to_the_capabilities_object() {
        // a real cap KEY at the TOP LEVEL of the capabilities object is detected, including when its
        // value is a sub-object; a `}` inside a quoted value must not close the object early.
        assert_eq!(host_deferrable("{\"params\":{\"capabilities\":{\"sampling\":{}}}}"), vec!["summarize", "embed", "normalize"]);
        assert_eq!(host_deferrable("{\"params\":{\"capabilities\":{\"roots\":{\"listChanged\":true}}}}"), vec!["fetch"]);
        assert_eq!(host_deferrable("{\"params\":{\"capabilities\":{\"sampling\":{},\"roots\":{}}}}"), vec!["summarize", "embed", "normalize", "fetch"]);
        assert_eq!(host_deferrable("{\"params\":{\"capabilities\":{\"experimental\":{\"note\":\"a } brace\"},\"roots\":{}}}}"), vec!["fetch"]);
        assert!(host_deferrable("{\"params\":{}}").is_empty());   // no capabilities key at all
        // a stray "sampling"/"roots" token in a SIBLING field — BEFORE or AFTER the capabilities
        // object (MCP members are unordered; clientInfo commonly serializes AFTER capabilities) —
        // must NOT trip detection.
        assert_eq!(host_deferrable("{\"params\":{\"clientInfo\":{\"name\":\"sampling-test\"},\"capabilities\":{}}}"), Vec::<&str>::new());
        assert_eq!(host_deferrable("{\"params\":{\"capabilities\":{},\"clientInfo\":{\"name\":\"roots\"}}}"), Vec::<&str>::new());
        assert_eq!(host_deferrable("{\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"experimental\":{\"roots\":1,\"sampling\":2}}}"), Vec::<&str>::new());
        // capabilities is NOT an object (null) — must not latch onto a sibling object's brace
        assert_eq!(host_deferrable("{\"params\":{\"capabilities\":null,\"clientInfo\":{\"name\":\"roots\"}}}"), Vec::<&str>::new());
        // the capabilities key has no colon of its own — must not latch onto the SIBLING's colon/object
        assert_eq!(host_deferrable("{\"foo\":1,\"capabilities\" ,\"bar\":{\"roots\":{}}}"), Vec::<&str>::new());
        // "sampling"/"roots" as a string VALUE (not a key), or as a key NESTED below depth 1, must
        // NOT be read as an advertised top-level capability.
        assert_eq!(host_deferrable("{\"params\":{\"capabilities\":{\"experimental\":{\"note\":\"sampling\"}}}}"), Vec::<&str>::new());
        assert_eq!(host_deferrable("{\"params\":{\"capabilities\":{\"x\":\"roots\"}}}"), Vec::<&str>::new());
        assert_eq!(host_deferrable("{\"params\":{\"capabilities\":{\"experimental\":{\"sampling\":{}}}}}"), Vec::<&str>::new());
        // a string VALUE spelling "sampling" followed by a stray colon (malformed) is in value
        // position, not key position — it must NOT be promoted to a top-level capability key.
        assert_eq!(host_deferrable("{\"capabilities\":{\"a\":\"sampling\":1}}"), Vec::<&str>::new());
        // a value that spells a cap but is followed by no colon at all is also never a key
        assert_eq!(host_deferrable("{\"capabilities\":{\"a\":\"roots\"}}"), Vec::<&str>::new());
        // inside an array (even a malformed `value:` shape) is not key position at object-depth 1
        assert_eq!(host_deferrable("{\"capabilities\":{\"a\":[\"x\",\"sampling\":{}]}}"), Vec::<&str>::new());
        assert_eq!(host_deferrable("{\"capabilities\":{\"a\":[{\"roots\":{}}]}}"), Vec::<&str>::new());
        // a real top-level key AFTER an array value is still found
        assert_eq!(host_deferrable("{\"capabilities\":{\"a\":[1,2],\"roots\":{}}}"), vec!["fetch"]);
    }

    #[test]
    fn caps_tool_manifest_and_resolve() {
        let db = NeuronDB::open(&tmp(), 500);
        // no host arg -> the static manifest (owner tags present)
        let m = handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"caps\",\"arguments\":{}}}").unwrap();
        assert!(m.contains("recall\\tgrounded") && m.contains("summarize\\tdeferrable"), "got {}", m);
        // with a host that claims recall (grounded) AND summarize (deferrable): recall stays keep,
        // summarize is ceded -> defer. This is grounded-beats-tier through the MCP tool surface.
        let r = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"caps\",\"arguments\":{\"host\":[\"recall\",\"summarize\"]}}}").unwrap();
        assert!(r.contains("recall\\tkeep"), "grounded recall must stay local even if host claims it, got {}", r);
        assert!(r.contains("summarize\\tdefer"), "offered deferrable must be ceded, got {}", r);
    }

    #[test]
    fn caps_tool_empty_host_is_the_manifest_like_no_host() {
        // parity: a degenerate all-empty host (`host:[""]`) must resolve to the manifest, identical to
        // an omitted host — so the caps surface stays byte-for-byte the same across CLI/MCP/WASM.
        let db = NeuronDB::open(&tmp(), 500);
        let none = handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"caps\",\"arguments\":{}}}").unwrap();
        let empty = handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"caps\",\"arguments\":{\"host\":[\"\"]}}}").unwrap();
        assert_eq!(none, empty, "empty host must match no-host (the manifest), got {} vs {}", none, empty);
        assert!(empty.contains("grounded") && !empty.contains("\\tkeep"), "must be the manifest, not the resolve table, got {}", empty);
    }

    #[test]
    fn initialize_answers_with_a_supported_protocol_version() {
        // MCP: the server answers with a version it actually speaks. A garbage/unsupported requested
        // version must NOT be echoed back as if neuron supported it — it falls back to PROTO_DEFAULT.
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"1999-01-01\",\"capabilities\":{}}}").unwrap();
        assert!(r.contains(&format!("\"protocolVersion\":\"{}\"", PROTO_DEFAULT)), "must advertise our real version, got {}", r);
        assert!(!r.contains("1999-01-01"), "must not echo an unsupported version, got {}", r);
    }

    #[test]
    fn raw_id_always_yields_a_valid_json_token() {
        // every response must be one line of valid JSON, so raw_id must return a well-formed token
        // (closed string / integer / null) for ANY id — including malformed/truncated request lines.
        // well-formed ids pass through unchanged, including an arbitrary-width integer beyond i64:
        assert_eq!(raw_id("{\"id\":42,\"method\":\"ping\"}"), "42");
        assert_eq!(raw_id("{\"id\":-5,\"method\":\"ping\"}"), "-5");
        assert_eq!(raw_id("{\"id\":0}"), "0");
        assert_eq!(raw_id("{\"id\":99999999999999999999,\"x\":1}"), "99999999999999999999");
        assert_eq!(raw_id("{\"id\":\"abc-1\"}"), "\"abc-1\"");
        // leading-zero integers are not valid JSON numbers -> null (don't guess/canonicalize):
        assert_eq!(raw_id("{\"id\":007}"), "null");
        // a string value ending in a backslash (escaped `\\`) must find the REAL close, not over-run:
        assert_eq!(raw_id("{\"id\":\"a\\\\\",\"method\":\"ping\"}"), "\"a\\\\\"");
        // an unterminated string falls back to null (not echoed verbatim):
        assert_eq!(raw_id("{\"method\":\"ping\",\"id\":\"abc"), "null");
        // a string id carrying a RAW control char (unescaped tab / CR) is not a well-formed JSON
        // string token -> null (a PROPERLY escaped \t is fine: it's the bytes backslash+t, not a tab):
        assert_eq!(raw_id("{\"id\":\"a\tb\",\"method\":\"ping\"}"), "null");
        assert_eq!(raw_id("{\"id\":\"a\rb\"}"), "null");
        assert_eq!(raw_id("{\"id\":\"a\\tb\"}"), "\"a\\tb\"");   // escaped tab round-trips verbatim
        // malformed `\u` escapes are not well-formed JSON strings -> null (build with a literal
        // backslash via '\\' so the source isn't a Rust unicode escape):
        let bs = '\\';
        assert_eq!(raw_id(&format!(r#"{{"id":"{bs}ud83d"}}"#)), "null");        // lone HIGH surrogate
        assert_eq!(raw_id(&format!(r#"{{"id":"{bs}ude00"}}"#)), "null");        // lone LOW surrogate
        assert_eq!(raw_id(&format!(r#"{{"id":"{bs}u"}}"#)), "null");            // truncated \u (no hex)
        assert_eq!(raw_id(&format!(r#"{{"id":"{bs}uZZZZ"}}"#)), "null");        // non-hex \u
        assert_eq!(raw_id(&format!(r#"{{"id":"{bs}x"}}"#)), "null");            // illegal escape
        // a valid surrogate PAIR and a valid \u escape ARE well-formed -> echoed verbatim:
        assert_eq!(raw_id(&format!(r#"{{"id":"{bs}ud83d{bs}ude00"}}"#)), format!(r#""{bs}ud83d{bs}ude00""#));
        assert_eq!(raw_id(&format!(r#"{{"id":"{bs}u00e9"}}"#)), format!(r#""{bs}u00e9""#));
        // malformed numbers (lone/leading/interior dashes) are NOT valid JSON numbers -> null:
        for bad in ["{\"id\":-}", "{\"id\":--5}", "{\"id\":5-3,\"x\":1}", "{\"method\":\"ping\",\"id\":-9-9}"] {
            assert_eq!(raw_id(bad), "null", "malformed number id must be null, input {}", bad);
        }
        // end to end: an unterminated id serializes as null and never leaks the open token
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"method\":\"ping\",\"id\":\"abc").unwrap();
        assert!(r.contains("\"id\":null") && !r.contains("\"id\":\"abc,"), "got {}", r);
    }

    #[test]
    fn tools_list_has_all_tools() {
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}").unwrap();
        for t in ["route","recall","recall_associative","recall_value","recall_chain","remember","note","recall_var","forget","stats"] {
            assert!(r.contains(&format!("\"name\":\"{}\"", t)), "missing tool {}", t);
        }
    }

    #[test]
    fn route_tool_dispatches_through_the_cortex() {
        // the cortex route on the MCP wire: recall the working set, then gary-neuron decides the route.
        // An extractive question over a stored fact answers from memory; an open-ended one escalates.
        let db = NeuronDB::open(&tmp(), 500);
        handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"remember\",\"arguments\":{\"scope\":\"u\",\"text\":\"the api key is zeta-9931\"}}}");
        let ans = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"route\",\"arguments\":{\"scope\":\"u\",\"query\":\"what is the api key?\"}}}").unwrap();
        assert!(ans.contains("\\\"type\\\":\\\"answer\\\"") && ans.contains("zeta-9931"), "extractive query should answer from memory, got {}", ans);
        let esc = handle_line(&db, "{\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"route\",\"arguments\":{\"scope\":\"u\",\"query\":\"write a short poem about the sea\"}}}").unwrap();
        assert!(esc.contains("\\\"type\\\":\\\"escalate\\\""), "an open-ended ask must escalate, got {}", esc);
        // missing query is a clean error, never a cortex call
        let err = handle_line(&db, "{\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"route\",\"arguments\":{\"scope\":\"u\"}}}").unwrap();
        assert!(err.contains("missing required argument: query"), "got {}", err);
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
    fn var_distinct_keys_do_not_clobber() {
        // regression: setting "region" must NOT delete "deployRegion" (anchored, not substring, upsert)
        let db = NeuronDB::open(&tmp(), 500);
        handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"var\",\"key\":\"deployRegion\",\"text\":\"eu-central-1\"}}}");
        handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"var\",\"key\":\"region\",\"text\":\"us-west-2\"}}}");
        let dr = handle_line(&db, "{\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_var\",\"arguments\":{\"scope\":\"u\",\"key\":\"deployRegion\"}}}").unwrap();
        assert!(dr.contains("eu-central-1"), "deployRegion must survive setting region, got {}", dr);
        let rg = handle_line(&db, "{\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_var\",\"arguments\":{\"scope\":\"u\",\"key\":\"region\"}}}").unwrap();
        assert!(rg.contains("us-west-2"), "got {}", rg);
    }

    #[test]
    fn var_value_containing_is_roundtrips_fully() {
        // regression: a value with " is " (or multiple words) must round-trip exactly, not get
        // cue-isolated down to a single token
        let db = NeuronDB::open(&tmp(), 500);
        handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"var\",\"key\":\"motto\",\"text\":\"trust is earned not given\"}}}");
        let m = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_var\",\"arguments\":{\"scope\":\"u\",\"key\":\"motto\"}}}").unwrap();
        assert!(m.contains("trust is earned not given"), "full value must round-trip, got {}", m);
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
    fn note_unknown_kind_is_rejected() {
        let db = NeuronDB::open(&tmp(), 500);
        let e = handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"instuction\",\"text\":\"never use markdown ever again please\"}}}").unwrap();
        assert!(e.contains("\"isError\":true") && e.contains("unknown kind"), "got {}", e);
        // and it must NOT have been silently filed into the base scope
        let s = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"stats\",\"arguments\":{\"scope\":\"u\"}}}").unwrap();
        assert!(s.contains("0 fact"), "misspelled kind must not write a base-scope fact, got {}", s);
    }

    #[test]
    fn stance_too_short_is_not_stored() {
        let db = NeuronDB::open(&tmp(), 500);
        let r = handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"stance\",\"key\":\"x\",\"text\":\"y\"}}}").unwrap();
        assert!(r.contains("not stored"), "phantom stance must be reported as not stored, got {}", r);
        let s = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"stats\",\"arguments\":{\"scope\":\"u::stance\"}}}").unwrap();
        assert!(s.contains("0 fact"), "no episode should exist, got {}", s);
    }

    #[test]
    fn forget_cascades_to_typed_subscopes() {
        let db = NeuronDB::open(&tmp(), 500);
        handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"remember\",\"arguments\":{\"scope\":\"u\",\"text\":\"the wifi password is hunter2\"}}}");
        handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"var\",\"key\":\"apikey\",\"text\":\"sk-secret-123\"}}}");
        handle_line(&db, "{\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"instruction\",\"text\":\"always reply in plain prose only\"}}}");
        handle_line(&db, "{\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"forget\",\"arguments\":{\"scope\":\"u\"}}}");
        // a full wipe must leave NO secret var / instruction behind
        let v = handle_line(&db, "{\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_var\",\"arguments\":{\"scope\":\"u\",\"key\":\"apikey\"}}}").unwrap();
        assert!(v.contains("(unset"), "var (secret) must be wiped by forget, got {}", v);
        let i = handle_line(&db, "{\"id\":6,\"method\":\"tools/call\",\"params\":{\"name\":\"stats\",\"arguments\":{\"scope\":\"u::instr\"}}}").unwrap();
        assert!(i.contains("0 fact"), "instructions must be wiped by forget, got {}", i);
    }

    #[test]
    fn var_set_unencodable_update_preserves_old_value() {
        // atomic upsert: an update whose "{key} is {value}" can't encode must NOT destroy the old
        // value. Use a short key so a stopword value makes the whole line unencodable.
        let db = NeuronDB::open(&tmp(), 500);
        handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"var\",\"key\":\"m\",\"text\":\"serious meaningful content here\"}}}");
        // "m is ok" -> no content word, no digit -> unencodable -> must be rejected, old kept
        handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"var\",\"key\":\"m\",\"text\":\"ok\"}}}");
        let v = handle_line(&db, "{\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_var\",\"arguments\":{\"scope\":\"u\",\"key\":\"m\"}}}").unwrap();
        assert!(v.contains("serious meaningful content here"), "old value must survive a failed update, got {}", v);
    }

    #[test]
    fn var_stopword_value_roundtrips() {
        // a stopword-class value (on/off/yes/no...) must read back as the value, not the key
        let db = NeuronDB::open(&tmp(), 500);
        handle_line(&db, "{\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"var\",\"key\":\"darkmode\",\"text\":\"on\"}}}");
        let v = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"recall_var\",\"arguments\":{\"scope\":\"u\",\"key\":\"darkmode\"}}}").unwrap();
        assert!(v.contains("on") && !v.contains("darkmode"), "got {}", v);
    }

    #[test]
    fn json_array_keeps_bracket_inside_element() {
        // a ']' inside a quoted element must not truncate a recall_chain path
        let got = json_array("{\"path\":[\"a]b\",\"manager\",\"timezone\"]}", "path");
        assert_eq!(got, vec!["a]b".to_string(), "manager".to_string(), "timezone".to_string()]);
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
    fn affect_layer_is_unlisted_but_humanize_reflects_mood_and_stance() {
        let db = NeuronDB::open(&tmp(), 500);
        // the affective tools must NOT appear in tools/list (secret)
        let list = handle_line(&db, "{\"id\":1,\"method\":\"tools/list\"}").unwrap();
        assert!(!list.contains("\"name\":\"feel\"") && !list.contains("\"name\":\"humanize\""), "affective tools must stay unlisted");
        // humanize is always a basis of context (baseline directive even with no mood/stance)
        let h0 = handle_line(&db, "{\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"humanize\",\"arguments\":{\"scope\":\"u\"}}}").unwrap();
        assert!(h0.contains("genuine feelings"), "got {}", h0);
        // the optional mood variable colors it
        handle_line(&db, "{\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"feel\",\"arguments\":{\"scope\":\"u\",\"emotion\":\"suspicious\"}}}");
        let h1 = handle_line(&db, "{\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"humanize\",\"arguments\":{\"scope\":\"u\"}}}").unwrap();
        assert!(h1.contains("you feel suspicious"), "got {}", h1);
        // an accumulated stance surfaces as the disposition
        handle_line(&db, "{\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"stance\",\"key\":\"this pattern\",\"text\":\"this insecure pattern keeps shipping\"}}}");
        handle_line(&db, "{\"id\":6,\"method\":\"tools/call\",\"params\":{\"name\":\"note\",\"arguments\":{\"scope\":\"u\",\"kind\":\"stance\",\"key\":\"this pattern\",\"text\":\"and it just failed again, worse\"}}}");
        let h2 = handle_line(&db, "{\"id\":7,\"method\":\"tools/call\",\"params\":{\"name\":\"humanize\",\"arguments\":{\"scope\":\"u\"}}}").unwrap();
        assert!(h2.contains("hardened view") && h2.contains("intensity x2"), "got {}", h2);
        // clearing the mood variable returns to auto (no override line)
        handle_line(&db, "{\"id\":8,\"method\":\"tools/call\",\"params\":{\"name\":\"feel\",\"arguments\":{\"scope\":\"u\",\"emotion\":\"\"}}}");
        let h3 = handle_line(&db, "{\"id\":9,\"method\":\"tools/call\",\"params\":{\"name\":\"humanize\",\"arguments\":{\"scope\":\"u\"}}}").unwrap();
        assert!(!h3.contains("Right now you feel"), "mood should clear, got {}", h3);
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

    #[test]
    fn preload_records_front_drain_so_boot_hook_can_warn() {
        // A pack with more facts than max_facts front-drains its oldest facts at load. The boot hook
        // must NOT lose them silently: it checks each loaded scope's Stats.dropped and warns on stderr
        // (PRELOAD.md's promise). Here we assert the exact trigger condition that warning is gated on —
        // the scope is capped at max_facts and the eviction is recorded in `dropped`. A `# scope:`
        // directive pins the target so the test never depends on the NEURON_MCP_PRELOAD_* env vars.
        let (max, n) = (4usize, 12usize);
        let mut body = String::from("# scope: evict_probe\n");
        for i in 0..n { body.push_str(&format!("subject{} states a distinct recorded fact.\n", i)); }
        let pack = std::env::temp_dir()
            .join(format!("ndb_preload_evict_{}_{}.facts", std::process::id(), n));
        std::fs::write(&pack, body).unwrap();

        let db = NeuronDB::open(&tmp(), max);
        preload_into(&db, pack.to_str().unwrap()).expect("preload should succeed");

        let st = match apply(&db, NeuronOp::Stats { scope: "evict_probe".into() }) {
            OpResult::Stats(s) => s, _ => panic!("stats"),
        };
        assert_eq!(st.facts, max, "scope must be capped at max_facts after the front-drain");
        assert!(st.dropped > 0, "eviction must be recorded so the boot hook warns (not silent loss)");

        let _ = std::fs::remove_file(&pack);
    }
}
