#![cfg(feature = "secure")]
use neuron_core::secure::SecureNeuronDB;
use std::time::{SystemTime, UNIX_EPOCH};
fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_sec_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}

#[test]
fn put_get_roundtrip_with_correct_secret() {
    let v = SecureNeuronDB::open(&tmp());
    v.put("alice", "alice-secret", "wifi password", "hunter2").unwrap();
    assert_eq!(v.get("alice", "alice-secret", "what is the wifi password?").as_deref(), Some("hunter2"));
}

#[test]
fn wrong_secret_reveals_nothing() {
    let v = SecureNeuronDB::open(&tmp());
    v.put("alice", "alice-secret", "wifi password", "hunter2").unwrap();
    assert_eq!(v.get("alice", "WRONG", "what is the wifi password?"), None);
    assert_eq!(v.get("bob", "alice-secret", "what is the wifi password?"), None); // wrong neuron id
}

#[test]
fn db_file_is_opaque() {
    let path = tmp();
    let v = SecureNeuronDB::open(&path);
    v.put("alice", "s3cr3t", "launch code", "ZEBRA-MAGENTA-7").unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let hay = String::from_utf8_lossy(&bytes);
    assert!(!hay.contains("ZEBRA-MAGENTA-7"), "plaintext leaked into the db file!");
    assert!(!hay.contains("launch"), "key phrase leaked into the db file!");
}
