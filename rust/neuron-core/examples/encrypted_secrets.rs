//! Store secrets encrypted at rest. The per-scope secret is supplied per call and never
//! written to disk; a stolen .db file is opaque.
//! Run: cargo run --release --example encrypted_secrets --features secure
use neuron_core::secure::SecureNeuronDB;

fn main() {
    let v = SecureNeuronDB::open("/tmp/neuron_vault.db");
    v.put("alice", "alice-secret", "wifi password", "hunter2").unwrap();
    v.put("alice", "alice-secret", "api token", "tok_abc123").unwrap();

    println!("correct secret -> {:?}", v.get("alice", "alice-secret", "what is the wifi password?"));
    println!("wrong secret   -> {:?}", v.get("alice", "WRONG",        "what is the wifi password?"));
    println!("wrong scope    -> {:?}", v.get("bob",   "alice-secret", "what is the wifi password?"));
}
