//! Minimal HTTP server over std TcpListener (no crate). Mirrors neuron_db/server.py:
//!   GET  /                 -> service info
//!   GET  /v1/{neuron}      -> stats            (auth)
//!   POST /v1/{neuron}      {message}  -> turn  (auth)
//!   POST /v1/{neuron}/get  {query}    -> {value}
//!   POST /v1/{neuron}/forget {match}  -> {forgot,remaining}
//! Bearer auth when NEURON_DB_KEY is set. Feature-gated behind `server`.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use crate::db::NeuronDB;
use crate::op::{apply, NeuronOp, OpResult};

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
fn json_num(body: &str, key: &str) -> Option<i64> {
    let pat = format!("\"{}\"", key);
    let i = body.find(&pat)?; let after = &body[i + pat.len()..];
    let c = after.find(':')?; let tail = after[c + 1..].trim_start();
    let num: String = tail.chars().take_while(|ch| ch.is_ascii_digit() || *ch == '-').collect();
    num.parse().ok()
}
fn respond(stream: &mut TcpStream, status: u16, body: &str) {
    let reason = match status { 200 => "OK", 204 => "No Content", 400 => "Bad Request", 401 => "Unauthorized", 404 => "Not Found", _ => "OK" };
    let resp = format!("HTTP/1.1 {} {}\r\ncontent-type: application/json\r\naccess-control-allow-origin: *\r\naccess-control-allow-headers: authorization, content-type\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", status, reason, body.len(), body);
    let _ = stream.write_all(resp.as_bytes());
}
fn clip(s: &str) -> String { s.chars().take(128).collect() }
fn cap(s: &str, n: usize) -> String { s.chars().take(n).collect() }
fn urldecode(s: &str) -> String {
    let b = s.as_bytes(); let mut o = Vec::new(); let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() { if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) { o.push(v); i += 3; continue; } }
        if b[i] == b'+' { o.push(b' '); } else { o.push(b[i]); }
        i += 1;
    }
    String::from_utf8_lossy(&o).to_string()
}

pub fn serve(db_path: &str, host: &str, port: u16, max_facts: usize) -> std::io::Result<()> {
    let db = Arc::new(NeuronDB::open(db_path, max_facts));
    let key = std::env::var("NEURON_DB_KEY").ok().filter(|s| !s.is_empty());
    let logmode: u8 = match std::env::var("NEURON_LOG").as_deref() { Ok("off") | Ok("0") => 0, Ok("json") => 2, _ => 1 };
    let reqs = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let listener = TcpListener::bind((host, port))?;
    eprintln!("neuron-db serving {} at http://{}:{}  (auth {}, log {})", db_path, host, port,
        if key.is_some() { "on" } else { "off" }, match logmode {0=>"off",2=>"json",_=>"text"});
    for s in listener.incoming().flatten() {
        let (db, key, reqs) = (db.clone(), key.clone(), reqs.clone());
        std::thread::spawn(move || handle(s, db, key, logmode, reqs, start));
    }
    Ok(())
}
fn hms() -> String { let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(); format!("{:02}:{:02}:{:02}", (n/3600)%24,(n/60)%60,n%60) }

/// Constant-time bearer-token comparison: every byte is compared regardless of where the first mismatch
/// is, so the auth check can't be timed to recover the key. Length is not the secret (the token bytes are),
/// so an early length check is acceptable and avoids subtle truncation bugs.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) { diff |= x ^ y; }
    diff == 0
}

fn handle(mut stream: TcpStream, db: Arc<NeuronDB>, key: Option<String>, logmode: u8, reqs: Arc<AtomicU64>, start: Instant) {
    let t0 = Instant::now();
    let n = reqs.fetch_add(1, Ordering::Relaxed) + 1;
    // bound how long a single peer can hold a worker thread (slow-loris) and how long a blocked write waits
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(15)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(15)));
    let peek = match stream.try_clone() { Ok(s) => s, Err(_) => return };
    let mut reader = BufReader::new(peek);
    let mut line = String::new(); if reader.read_line(&mut line).is_err() { return; }
    let mut it = line.split_whitespace();
    let method = it.next().unwrap_or("").to_string();
    let path = it.next().unwrap_or("/").to_string();
    let (mut clen, mut auth) = (0usize, String::new());
    loop {
        let mut h = String::new(); if reader.read_line(&mut h).is_err() { break; }
        let t = h.trim_end(); if t.is_empty() { break; }
        let low = t.to_lowercase();
        if let Some(v) = low.strip_prefix("content-length:") { clen = v.trim().parse().unwrap_or(0); }
        if low.starts_with("authorization:") { auth = t[t.find(':').unwrap() + 1..].trim().to_string(); }
    }
    // reject an oversized body BEFORE allocating it: a pre-auth Content-Length of 4 GiB must not OOM us
    const MAX_BODY: usize = 1 << 20; // 1 MiB
    if clen > MAX_BODY { respond(&mut stream, 413, "{\"error\":\"payload too large\"}"); return; }
    let mut buf = vec![0u8; clen]; if clen > 0 { let _ = reader.read_exact(&mut buf); }
    let body = String::from_utf8_lossy(&buf).to_string();
    let segs: Vec<String> = path.split('?').next().unwrap_or("").split('/').filter(|s| !s.is_empty()).map(urldecode).collect();
    let authed = key.is_none() || {
        // strip_prefix (not replace, which would strip every occurrence) + constant-time compare
        let token = auth.strip_prefix("Bearer ").unwrap_or(&auth).trim();
        ct_eq(token.as_bytes(), key.as_deref().unwrap_or("").as_bytes())
    };
    let scope = segs.get(1).map(|s| clip(s)).unwrap_or_default();

    let (status, resp) = route(&db, &method, &segs, authed, &body, n, start.elapsed().as_secs());
    respond(&mut stream, status, &resp);
    match logmode {
        1 => eprintln!("{} {:<7} {} scope={} -> {} {}ms", hms(), method, path, if scope.is_empty(){"-"}else{&scope}, status, t0.elapsed().as_millis()),
        2 => eprintln!("{{\"t\":\"{}\",\"method\":\"{}\",\"path\":\"{}\",\"scope\":\"{}\",\"status\":{},\"ms\":{}}}", hms(), method, json_escape(&path), json_escape(&scope), status, t0.elapsed().as_millis()),
        _ => {}
    }
}

fn route(db: &NeuronDB, method: &str, segs: &[String], authed: bool, body: &str, reqs: u64, uptime: u64) -> (u16, String) {
    if method == "OPTIONS" { return (204, String::new()); }
    if method == "GET" {
        if segs.is_empty() { return (200, "{\"service\":\"neuron-db\",\"endpoint\":\"POST /v1/{neuron}\"}".into()); }
        if segs.len() == 1 && segs[0] == "healthz" { return (200, "{\"ok\":true}".into()); }
        if segs.len() == 1 && segs[0] == "metrics" { return (200, format!("{{\"requests\":{},\"uptime_s\":{}}}", reqs, uptime)); }
        if !authed { return (401, "{\"error\":\"unauthorized\"}".into()); }
        if segs.len() >= 2 && segs[0] == "v1" {
            if let OpResult::Stats(s) = apply(db, NeuronOp::Stats { scope: clip(&segs[1]) }) {
                return (200, format!("{{\"facts\":{},\"max_facts\":{},\"created\":{},\"updated\":{},\"turns\":{}}}", s.facts, s.max_facts, s.created, s.updated, s.turns));
            }
        }
        return (404, "{\"error\":\"not found\"}".into());
    }
    if method == "POST" {
        if !authed { return (401, "{\"error\":\"unauthorized\"}".into()); }
        if segs.len() < 2 || segs[0] != "v1" { return (404, "{\"error\":\"POST /v1/{neuron}\"}".into()); }
        let nid = clip(&segs[1]);
        let sub = segs.get(2).map(|s| s.as_str()).unwrap_or("");
        if sub == "forget" {
            if let OpResult::Forgot { forgot, remaining } = apply(db, NeuronOp::Forget { scope: nid.clone(), matching: json_field(body, "match") }) {
                return (200, format!("{{\"forgot\":{},\"remaining\":{}}}", forgot, remaining));
            }
        }
        if sub == "observe" {
            let facts = json_array(body, "facts");
            if facts.is_empty() { return (400, "{\"error\":\"need {facts:[...]}\"}".into()); }
            let w = apply(db, NeuronOp::ObserveBulk { scope: nid.clone(), texts: facts }).wrote();
            return (200, format!("{{\"wrote\":{}}}", w));
        }
        if sub == "get" {
            let q = json_field(body, "query").or_else(|| json_field(body, "message")).unwrap_or_default();
            if q.is_empty() { return (400, "{\"error\":\"empty query\"}".into()); }
            let v = apply(db, NeuronOp::RecallOne { scope: nid.clone(), query: cap(&q, 4000) }).hit().map(|h| h.value);
            let vj = match v { Some(s) => format!("\"{}\"", json_escape(&s)), None => "null".to_string() };
            return (200, format!("{{\"value\":{}}}", vj));
        }
        if sub == "recall" {
            let q = json_field(body, "query").or_else(|| json_field(body, "message")).unwrap_or_default();
            if q.is_empty() { return (400, "{\"error\":\"empty query\"}".into()); }
            return match apply(db, NeuronOp::RecallOne { scope: nid.clone(), query: cap(&q, 4000) }).hit() {
                Some(h) => (200, format!("{{\"value\":\"{}\",\"fact\":\"{}\",\"coverage\":{:.4},\"overlap\":{},\"exact\":{}}}", json_escape(&h.value), json_escape(&h.fact), h.coverage, h.overlap, h.exact)),
                None => (200, "{\"value\":null,\"coverage\":0}".into()),
            };
        }
        if sub == "recall_many" {
            let q = json_field(body, "query").or_else(|| json_field(body, "message")).unwrap_or_default();
            if q.is_empty() { return (400, "{\"error\":\"empty query\"}".into()); }
            let k = json_num(body, "k").unwrap_or(5).clamp(1, 50) as usize;
            let items: Vec<String> = apply(db, NeuronOp::Recall { scope: nid.clone(), query: cap(&q, 4000), k, semantic: false, across: false }).hits().iter()
                .map(|h| format!("{{\"value\":\"{}\",\"fact\":\"{}\",\"coverage\":{:.4}}}", json_escape(&h.value), json_escape(&h.fact), h.coverage)).collect();
            return (200, format!("{{\"hits\":[{}]}}", items.join(",")));
        }
        let msg = json_field(body, "message").unwrap_or_default();
        if msg.is_empty() { return (400, "{\"error\":\"empty message\"}".into()); }
        return match apply(db, NeuronOp::Turn { scope: nid.clone(), message: cap(&msg, 4000) }) {
            OpResult::Turned(t) => (200, format!("{{\"reply\":\"{}\",\"kind\":\"{}\",\"wrote\":{},\"facts\":{},\"capacity_reached\":{}}}", json_escape(&t.reply), t.kind, t.wrote, t.facts, t.capacity_reached)),
            _ => (500, "{\"error\":\"turn failed\"}".into()),
        };
    }
    (404, "{\"error\":\"not found\"}".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn seg(p: &str) -> Vec<String> { p.split('/').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect() }

    // Pins the HTTP route surface end-to-end, now that every db call routes through op::apply.
    #[test]
    fn route_endpoints_through_apply() {
        let tmp = std::env::temp_dir().join(format!("neuron_route_test_{}.db", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let db = NeuronDB::open(tmp.to_str().unwrap(), 500);

        let (st, body) = route(&db, "POST", &seg("/v1/u/observe"), true, "{\"facts\":[\"my plan is pro\",\"the launch is Friday\"]}", 1, 0);
        assert_eq!(st, 200); assert!(body.contains("\"wrote\":2"), "observe: {}", body);

        let (st, body) = route(&db, "POST", &seg("/v1/u/get"), true, "{\"query\":\"what is my plan\"}", 1, 0);
        assert_eq!(st, 200); assert!(body.contains("pro"), "get: {}", body);

        let (st, body) = route(&db, "POST", &seg("/v1/u/recall"), true, "{\"query\":\"launch\"}", 1, 0);
        assert_eq!(st, 200); assert!(body.contains("\"fact\":") && body.contains("Friday"), "recall: {}", body);

        let (_st, body) = route(&db, "POST", &seg("/v1/u/recall_many"), true, "{\"query\":\"plan\",\"k\":3}", 1, 0);
        assert!(body.contains("\"hits\":["), "recall_many: {}", body);

        let (st, body) = route(&db, "GET", &seg("/v1/u"), true, "", 1, 0);
        assert_eq!(st, 200); assert!(body.contains("\"facts\":2"), "stats: {}", body);

        let (st, body) = route(&db, "POST", &seg("/v1/u"), true, "{\"message\":\"hi there\"}", 1, 0);
        assert_eq!(st, 200); assert!(body.contains("\"reply\":"), "turn: {}", body);

        let (st, body) = route(&db, "POST", &seg("/v1/u/forget"), true, "{\"match\":\"launch\"}", 1, 0);
        assert_eq!(st, 200); assert!(body.contains("\"forgot\":1"), "forget: {}", body);

        // gates that must still hold
        assert_eq!(route(&db, "GET", &seg("/v1/u"), false, "", 1, 0).0, 401);          // auth
        assert_eq!(route(&db, "POST", &seg("/v1/u/get"), true, "{}", 1, 0).0, 400);    // empty query
        assert_eq!(route(&db, "OPTIONS", &seg("/v1/u"), true, "", 1, 0).0, 204);       // CORS preflight

        let _ = std::fs::remove_file(&tmp);
    }
}
