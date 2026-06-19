//! SecureNeuronDB: encrypted neurons. Values are AES-256-GCM ciphertext; the index is a
//! keyed hash of cue stems; the per-neuron secret is supplied per call and never stored, so
//! a stolen db file is opaque. Feature-gated behind `secure`. Faithful port of
//! neuron_db/secure.py (HKDF-SHA256 key schedule; Rust-native blob format).
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, KeyInit, Payload};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use std::collections::BTreeSet;
use crate::{content, stems_s};

type HmacSha256 = Hmac<Sha256>;
fn mac(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut m = <HmacSha256 as Mac>::new_from_slice(key).expect("hmac key");
    m.update(msg); m.finalize().into_bytes().to_vec()
}
/// HKDF-SHA256 (extract+expand), matching the Python _hkdf.
fn hkdf(key: &[u8], salt: &[u8], info: &[u8], n: usize) -> Vec<u8> {
    let prk = mac(salt, key);
    let mut out = Vec::new(); let mut t: Vec<u8> = Vec::new(); let mut i = 1u8;
    while out.len() < n {
        let mut msg = t.clone(); msg.extend_from_slice(info); msg.push(i);
        t = mac(&prk, &msg); out.extend_from_slice(&t); i = i.wrapping_add(1);
    }
    out.truncate(n); out
}
pub fn derive_key(secret: &str, nid: &str) -> Vec<u8> {
    let mut salt = b"neuron-db/".to_vec(); salt.extend_from_slice(nid.as_bytes());
    hkdf(secret.as_bytes(), &salt, b"key", 32)
}
// Additional authenticated data bound into every GCM tag, so a ciphertext can't be silently
// replayed under a different format version. The blob's leading byte is the format version: 1 =
// legacy (empty AAD, still readable), 2 = AAD-bound (what we write now).
const AAD_V2: &[u8] = b"neuron-secure-v2";
fn aead_encrypt(key: &[u8], pt: &[u8]) -> Vec<u8> {
    let mut nonce = [0u8; 12]; getrandom::getrandom(&mut nonce).expect("rng");
    let k = hkdf(key, b"neuron-aesgcm", b"v1", 32);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&k));
    let ct = cipher.encrypt(Nonce::from_slice(&nonce), Payload { msg: pt, aad: AAD_V2 }).expect("encrypt");
    let mut out = vec![2u8]; out.extend_from_slice(&nonce); out.extend_from_slice(&ct); out
}
fn aead_decrypt(key: &[u8], blob: &[u8]) -> Option<Vec<u8>> {
    if blob.len() < 13 { return None; }
    let aad: &[u8] = match blob[0] { 1 => b"", 2 => AAD_V2, _ => return None };  // read legacy + AAD-bound
    let k = hkdf(key, b"neuron-aesgcm", b"v1", 32);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&k));
    cipher.decrypt(Nonce::from_slice(&blob[1..13]), Payload { msg: &blob[13..], aad }).ok()
}

pub struct SecureNeuron { key: Vec<u8>, idx_key: Vec<u8>, entries: Vec<(Vec<String>, String)> }
impl SecureNeuron {
    pub fn new(key: Vec<u8>) -> Self {
        let idx_key = hkdf(&key, b"neuron-index", b"v1", 32);
        SecureNeuron { key, idx_key, entries: Vec::new() }
    }
    fn keyed(&self, stem: &str) -> String { B64.encode(&mac(&self.idx_key, stem.as_bytes())[..8]) }
    fn stems(&self, phrase: &str) -> BTreeSet<String> {
        stems_s(&content(phrase)).iter().map(|s| self.keyed(s)).collect()
    }
    pub fn put(&mut self, key_phrase: &str, value: &str) -> Result<(), &'static str> {
        let idx: Vec<String> = self.stems(key_phrase).into_iter().collect(); // BTreeSet -> sorted
        if idx.is_empty() { return Err("key phrase has no indexable content"); }
        let ct = B64.encode(aead_encrypt(&self.key, value.as_bytes()));
        self.entries.push((idx, ct)); Ok(())
    }
    pub fn get(&self, query: &str, min_cover: f64) -> Option<String> {
        let q = self.stems(query); if q.is_empty() { return None; }
        let mut best: Option<&(Vec<String>, String)> = None; let mut bk = (-1f64, -1i64);
        for (i, e) in self.entries.iter().enumerate() {
            let xs: BTreeSet<&String> = e.0.iter().collect();
            let ov = q.iter().filter(|s| xs.contains(s)).count();
            if ov == 0 { continue; }
            let cover = ov as f64 / e.0.len() as f64;
            if cover < min_cover { continue; }
            if cover > bk.0 || (cover == bk.0 && i as i64 > bk.1) { bk = (cover, i as i64); best = Some(e); }
        }
        let e = best?;
        let raw = B64.decode(&e.1).ok()?;
        let pt = aead_decrypt(&self.key, &raw)?;
        String::from_utf8(pt).ok()
    }
    pub fn dump(&self) -> String {
        self.entries.iter().map(|(idx, ct)| format!("{}|{}", idx.join(","), ct)).collect::<Vec<_>>().join("\n")
    }
    pub fn load(key: Vec<u8>, blob: &str) -> Self {
        let mut n = SecureNeuron::new(key);
        for line in blob.split('\n') {
            if line.is_empty() { continue; }
            if let Some((ix, ct)) = line.split_once('|') {
                let idx = ix.split(',').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
                n.entries.push((idx, ct.to_string()));
            }
        }
        n
    }
    pub fn count(&self) -> usize { self.entries.len() }
}

use rusqlite::{params, Connection};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS secure (id TEXT PRIMARY KEY, blob TEXT NOT NULL, updated INTEGER NOT NULL);";
pub struct SecureNeuronDB { conn: Mutex<Connection> }
impl SecureNeuronDB {
    pub fn open(path: &str) -> Self {
        let conn = Connection::open(path).expect("open");
        conn.execute(SCHEMA, []).expect("schema");
        SecureNeuronDB { conn: Mutex::new(conn) }
    }
    fn blob(&self, conn: &Connection, nid: &str) -> String {
        conn.query_row("SELECT blob FROM secure WHERE id=?1", params![nid], |r| r.get::<_, String>(0)).unwrap_or_else(|_| String::new())
    }
    fn write(&self, conn: &Connection, nid: &str, blob: &str) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;
        conn.execute("INSERT INTO secure(id,blob,updated) VALUES(?1,?2,?3) ON CONFLICT(id) DO UPDATE SET blob=excluded.blob,updated=excluded.updated", params![nid, blob, now]).expect("write");
    }
    pub fn put(&self, nid: &str, secret: &str, key_phrase: &str, value: &str) -> Result<(), &'static str> {
        let conn = self.conn.lock().unwrap();
        let k = derive_key(secret, nid);
        let mut n = SecureNeuron::load(k, &self.blob(&conn, nid));
        n.put(key_phrase, value)?; self.write(&conn, nid, &n.dump()); Ok(())
    }
    pub fn get(&self, nid: &str, secret: &str, query: &str) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        let k = derive_key(secret, nid);
        let n = SecureNeuron::load(k, &self.blob(&conn, nid));
        n.get(query, 0.5)
    }
}
