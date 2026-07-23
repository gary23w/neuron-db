//! The quantum-teleportation tier: entangled links, teleport recall, the no-cloning read budget,
//! superposed values, and entangled sharding. Run: cargo test --features quantum --test quantum_tier
//! (the durable half needs --features quantum-db).
#![cfg(feature = "quantum")]

use neuron_core::quantum::{EntangledStore, HasEntanglements, MemBack, QuantumBack, QuantumRouter, QuantumSide};

fn mem() -> EntangledStore<MemBack> { EntangledStore::new(MemBack::new()) }

#[test]
fn entangle_then_recall_triggers_side_effect() {
    let q = mem();
    q.entangle("user:42", "the gate code is 4491", "user:99", "the safe word is falcon", "copy", 3);
    let r = q.entangled_recall("user:42", "what is the gate code?").unwrap();
    assert_eq!(r.hit.value, "4491");
    assert_eq!(r.entangled.len(), 1, "the hit must surface its partner");
    assert_eq!(r.entangled[0].partner_scope, "user:99");
    assert_eq!(r.entangled[0].partner_value.as_deref(), Some("falcon"));
    // the correlated read is symmetric: recalling the OTHER side surfaces the first
    let r2 = q.entangled_recall("user:99", "what is the safe word?").unwrap();
    assert_eq!(r2.entangled[0].partner_scope, "user:42");
    // ... and non-consuming: the link still has its full budget
    assert_eq!(r2.entangled[0].link.ebits, 3);
}

#[test]
fn teleport_moves_association_not_fact() {
    let q = mem();
    q.entangle("alpha", "the gate code is 4491", "beta", "the gate code is ----", "copy", 2);
    let t = q.teleport("alpha", "what is the gate code?").unwrap();
    assert_eq!(t.value, "4491");
    assert_eq!(t.source_scope, "alpha");
    assert_eq!(t.dest_scope, "beta");
    assert_eq!(t.ebits_remaining, 1);
    // the source fact still exists (teleport moves the association, not the fact)
    assert!(q.inner.has_fact("alpha", "the gate code is 4491"));
    // the dest now answers with the source's value
    assert_eq!(q.recall("beta", "what is the gate code?").unwrap().value, "4491");
}

#[test]
fn teleport_ebit_exhaustion_disentangles() {
    let q = mem();
    let id = q.entangle("alpha", "the gate code is 4491", "beta", "the gate code is ----", "copy", 2);
    assert_eq!(q.teleport("alpha", "what is the gate code?").unwrap().ebits_remaining, 1);
    let last = q.teleport("alpha", "what is the gate code?").unwrap();
    assert_eq!(last.ebits_remaining, 0);
    // the budget is spent: the link is deleted and the next teleport finds nothing
    assert!(q.inner.read_entanglement(id).is_none(), "a fully-consumed link must be gone");
    assert!(q.teleport("alpha", "what is the gate code?").is_none());
}

#[test]
fn classical_channel_copy_swap_invert() {
    // copy: the dest takes the source's association
    let q = mem();
    q.entangle("c1", "the gate code is 4491", "c2", "the gate code is ----", "copy", 1);
    q.teleport("c1", "what is the gate code?").unwrap();
    assert_eq!(q.recall("c2", "what is the gate code?").unwrap().value, "4491");

    // swap: the two facts exchange
    q.entangle("s1", "the meeting room is atlas", "s2", "the meeting room is zephyr", "swap", 1);
    let t = q.teleport("s1", "what is the meeting room?").unwrap();
    assert_eq!(t.value, "atlas", "the teleported payload is the measured source value");
    assert_eq!(q.recall("s1", "what is the meeting room?").unwrap().value, "zephyr");
    assert_eq!(q.recall("s2", "what is the meeting room?").unwrap().value, "atlas");

    // invert: a numeric value negates on the dest
    q.entangle("i1", "the account balance is 250", "i2", "the account balance is 0", "invert", 1);
    let t = q.teleport("i1", "what is the account balance?").unwrap();
    assert_eq!(t.value, "-250");
    assert_eq!(q.recall("i2", "what is the account balance?").unwrap().value, "-250");

    // any other instruction: stored verbatim on the dest
    q.entangle("v1", "the gate code is 4491", "v2", "the gate code is ----", "the fallback code is 7777", 1);
    q.teleport("v1", "what is the gate code?").unwrap();
    assert!(q.inner.has_fact("v2", "the fallback code is 7777"));
}

#[test]
fn no_clone_fact_vanishes_after_max_reads() {
    let q = mem();
    q.write_once("user:42", "the launch code is gamma-7", 3);
    assert_eq!(q.reads_remaining("user:42", "the launch code is gamma-7"), Some(3));
    for i in 0..3 {
        let hit = q.recall_once("user:42", "what is the launch code?");
        assert_eq!(hit.unwrap().value, "gamma-7", "read {} of 3 must still return", i + 1);
    }
    // the third read spent the last budget and burned the fact
    assert!(q.recall_once("user:42", "what is the launch code?").is_none());
    assert!(!q.inner.has_fact("user:42", "the launch code is gamma-7"));
    assert_eq!(q.reads_remaining("user:42", "the launch code is gamma-7"), None);
}

#[test]
fn no_clone_fact_coexists_with_normal_facts() {
    let q = mem();
    q.observe("user:1", "my favorite color is teal");
    q.write_once("user:1", "the wifi password is 8842", 1);
    assert_eq!(q.recall_once("user:1", "what is the wifi password?").unwrap().value, "8842");
    assert!(q.recall_once("user:1", "what is the wifi password?").is_none(), "burned after its single read");
    // the ordinary fact is untouched and reads forever
    for _ in 0..5 {
        assert_eq!(q.recall_once("user:1", "what is my favorite color?").unwrap().value, "teal");
    }
}

#[test]
fn superposition_collapses_on_measurement() {
    let q = mem();
    q.store_super("user:42", "my favorite food is", &["pizza", "sushi", "tacos"]);
    // measurement is deterministic: highest amplitude wins, ties -> first stored
    assert_eq!(q.recall_super("user:42", "what is my favorite food?").unwrap(), "pizza");
    let alts = q.inner.super_get("user:42", "my favorite food is").unwrap();
    let winner = alts.iter().find(|(a, _)| a == "pizza").unwrap().1;
    assert!(winner > 1.0, "the measured candidate is Zeno-boosted: {}", winner);
    assert!(alts.iter().filter(|(a, _)| a != "pizza").all(|(_, w)| *w < 1.0), "the losers decay: {:?}", alts);
    // repeated measurement of the same value stabilizes it (quantum Zeno effect)
    assert_eq!(q.recall_super("user:42", "favorite food").unwrap(), "pizza");
    assert_eq!(q.recall_super("user:42", "favorite food").unwrap(), "pizza");
}

#[test]
fn superposition_removes_decayed_alternatives() {
    let q = mem();
    q.store_super("user:7", "my favorite food is", &["pizza", "sushi", "tacos"]);
    // amplitudes: winner x1.1 per measure, losers x0.5, pruned below 0.1 -> gone on the 4th
    for _ in 0..3 { q.recall_super("user:7", "favorite food").unwrap(); }
    let alts = q.inner.super_get("user:7", "my favorite food is").unwrap();
    assert_eq!(alts.len(), 3, "losers at 0.125 still hold on: {:?}", alts);
    assert_eq!(q.recall_super("user:7", "favorite food").unwrap(), "pizza");
    // the losers decayed below threshold and the lone survivor RESOLVED into a classical fact
    assert!(q.inner.super_get("user:7", "my favorite food is").is_none(), "the superposition entry is gone");
    assert!(q.inner.has_fact("user:7", "my favorite food is pizza"));
    assert_eq!(q.recall_once("user:7", "what is my favorite food?").unwrap().value, "pizza");
}

#[test]
fn quantum_router_fans_out_to_idle_shard() {
    let mut r = QuantumRouter::new(8);
    r.observe("the primary key is 111");
    r.observe("the cache ttl is 60");
    r.observe("the retry limit is 5");
    let id = r.entangle("the gate code is 4491", "the gate code is ----", "copy", 1);
    assert!(id > 0);
    // the dest landed on a fresh idle shard, never the busy source shard
    assert_eq!(r.shard_count(), 2);
    assert_eq!(r.shard_of("the gate code is 4491"), Some(0));
    assert_eq!(r.shard_of("the gate code is ----"), Some(1));
    assert_eq!(r.inner.shards[1].fact_count(), 1);
    let t = r.teleport("what is the gate code?").unwrap();
    assert_eq!(t.source_scope, "shard:0");
    assert_eq!(t.dest_scope, "shard:1", "the reconstruction lands on the idle shard");
    assert_eq!(t.value, "4491");
    // the idle shard now answers with the teleported association
    assert!(r.inner.shards[1].episodes.iter().any(|e| e.t == "the gate code is 4491"));
}

#[test]
fn entangled_scopes_are_independent_after_disentangle() {
    let q = mem();
    let id = q.entangle("a", "the gate code is 4491", "b", "the gate code is ----", "copy", 5);
    assert!(q.disentangle(id));
    // no cross-scope side effect is possible any more
    assert!(q.teleport("a", "what is the gate code?").is_none());
    let r = q.entangled_recall("a", "what is the gate code?").unwrap();
    assert!(r.entangled.is_empty());
    // and the dest scope is exactly as it was
    assert!(q.inner.has_fact("b", "the gate code is ----"));
    assert!(!q.inner.has_fact("b", "the gate code is 4491"));
}

// ---- the durable half (feature quantum-db): the same protocol over NeuronDB's side tables ----

#[cfg(feature = "quantum-db")]
#[test]
fn durable_quantum_state_survives_reopen() {
    use neuron_core::db::NeuronDB;
    use neuron_core::quantum as q;
    let tmp = std::env::temp_dir().join(format!("neuron_quantum_test_{}.db", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let path = tmp.to_str().unwrap().to_string();
    let id;
    {
        let d = NeuronDB::open(&path, 500);
        id = q::entangle(&d, "alpha", "the gate code is 4491", "beta", "the gate code is ----", "copy", 2);
        q::write_once(&d, "alpha", "the launch code is gamma-7", 1);
        q::store_super(&d, "alpha", "deploy region is", &["us-east", "eu-west"]);
        d.flush_all();
    }
    let d = NeuronDB::open(&path, 500);
    // the link survived the reopen and still teleports
    assert!(q::scope_entanglements(&d, "alpha").iter().any(|l| l.id == id));
    let t = q::teleport(&d, "alpha", "what is the gate code?").unwrap();
    assert_eq!(t.value, "4491");
    assert_eq!(t.ebits_remaining, 1);
    assert_eq!(d.recall("beta", "what is the gate code?").unwrap().value, "4491");
    // the read budget survived: one read returns the secret, then the fact is gone
    assert_eq!(q::recall_once(&d, "alpha", "what is the launch code?").unwrap().value, "gamma-7");
    assert!(!<NeuronDB as q::QuantumBack>::has_fact(&d, "alpha", "the launch code is gamma-7"));
    // the superposition survived and measures deterministically
    assert_eq!(q::recall_super(&d, "alpha", "what is the deploy region?").unwrap(), "us-east");
    let _ = std::fs::remove_file(&tmp);
}

#[cfg(feature = "quantum-db")]
#[test]
fn durable_noclone_burn_is_durable_across_reopen() {
    use neuron_core::db::NeuronDB;
    use neuron_core::quantum as q;
    let tmp = std::env::temp_dir().join(format!("neuron_quantum_burn_{}.db", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let path = tmp.to_str().unwrap().to_string();
    {
        let d = NeuronDB::open(&path, 500);
        q::write_once(&d, "vault", "the recovery phrase is zebra-9", 1);
        assert_eq!(q::recall_once(&d, "vault", "what is the recovery phrase?").unwrap().value, "zebra-9");
        d.flush_all();
    }
    // the burn happened before the reopen: a fresh handle must NOT resurrect the fact
    let d = NeuronDB::open(&path, 500);
    assert!(q::recall_once(&d, "vault", "what is the recovery phrase?").is_none());
    assert!(!<NeuronDB as q::QuantumBack>::has_fact(&d, "vault", "the recovery phrase is zebra-9"));
    let _ = std::fs::remove_file(&tmp);
}
