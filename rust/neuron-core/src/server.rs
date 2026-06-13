//! Minimal HTTP server over std TcpListener (no crate). Mirrors neuron_db/server.py:
//!   GET  /                 -> service info
//!   GET  /v1/{neuron}      -> stats            (auth)
//!   POST /v1/{neuron}      {message}  -> turn  (auth)
//!   POST /v1/{neuron}/get  {query}    -> {value}
//!   POST /v1/{neuron}/forget {match}  -> {forgot,remaining}
//! Bearer auth when NEURON_DB_KEY is set. Feature-gated behind `server`.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use crate::db::NeuronDB;

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
    let listener = TcpListener::bind((host, port))?;
    eprintln!("neuron-db serving {} at http://{}:{}  (auth {})", db_path, host, port, if key.is_some() { "on" } else { "off" });
    for stream in listener.incoming() {
        if let Ok(s) = stream { let db = db.clone(); let key = key.clone(); std::thread::spawn(move || handle(s, db, key)); }
    }
    Ok(())
}
fn handle(mut stream: TcpStream, db: Arc<NeuronDB>, key: Option<String>) {
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
    let mut buf = vec![0u8; clen]; if clen > 0 { let _ = reader.read_exact(&mut buf); }
    let body = String::from_utf8_lossy(&buf).to_string();
    let segs: Vec<String> = path.split('?').next().unwrap_or("").split('/').filter(|s| !s.is_empty()).map(urldecode).collect();
    let authed = key.is_none() || auth.replace("Bearer ", "").trim() == key.as_deref().unwrap_or("");

    if method == "OPTIONS" { respond(&mut stream, 204, ""); return; }
    if method == "GET" {
        if segs.is_empty() { respond(&mut stream, 200, "{\"service\":\"neuron-db\",\"endpoint\":\"POST /v1/{neuron}\"}"); return; }
        if !authed { respond(&mut stream, 401, "{\"error\":\"unauthorized\"}"); return; }
        if segs.len() >= 2 && segs[0] == "v1" {
            let s = db.stats(&clip(&segs[1]));
            respond(&mut stream, 200, &format!("{{\"facts\":{},\"max_facts\":{},\"created\":{},\"updated\":{},\"turns\":{}}}", s.facts, s.max_facts, s.created, s.updated, s.turns));
            return;
        }
        respond(&mut stream, 404, "{\"error\":\"not found\"}"); return;
    }
    if method == "POST" {
        if !authed { respond(&mut stream, 401, "{\"error\":\"unauthorized\"}"); return; }
        if segs.len() < 2 || segs[0] != "v1" { respond(&mut stream, 404, "{\"error\":\"POST /v1/{neuron}\"}"); return; }
        let nid = clip(&segs[1]);
        if segs.len() >= 3 && segs[2] == "forget" {
            let (f, r) = db.forget(&nid, json_field(&body, "match").as_deref());
            respond(&mut stream, 200, &format!("{{\"forgot\":{},\"remaining\":{}}}", f, r)); return;
        }
        if segs.len() >= 3 && segs[2] == "get" {
            let q = json_field(&body, "query").or_else(|| json_field(&body, "message")).unwrap_or_default();
            if q.is_empty() { respond(&mut stream, 400, "{\"error\":\"empty query\"}"); return; }
            let v = db.get(&nid, &cap(&q, 4000));
            let vj = match v { Some(s) => format!("\"{}\"", json_escape(&s)), None => "null".to_string() };
            respond(&mut stream, 200, &format!("{{\"value\":{}}}", vj)); return;
        }
        let msg = json_field(&body, "message").unwrap_or_default();
        if msg.is_empty() { respond(&mut stream, 400, "{\"error\":\"empty message\"}"); return; }
        let t = db.turn(&nid, &cap(&msg, 4000));
        respond(&mut stream, 200, &format!("{{\"reply\":\"{}\",\"kind\":\"{}\",\"wrote\":{},\"facts\":{},\"capacity_reached\":{}}}", json_escape(&t.reply), t.kind, t.wrote, t.facts, t.capacity_reached));
        return;
    }
    respond(&mut stream, 404, "{\"error\":\"not found\"}");
}
