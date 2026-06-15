//! PlasticNeuron + neurotransmitter recall, and the two caveat fixes (exact-key
//! disambiguation, multi-word values). Run: cargo test --release --test plastic

use neuron_core::{Neuron, plastic::PlasticNeuron};

fn pn() -> PlasticNeuron { PlasticNeuron::new(10_000_000, Some(1e9), 3) }

#[test]
fn adaptation_strength_overtakes_recency() {
    let mut n = pn();
    n.observe("the meeting is on monday");
    n.observe("the meeting is on friday");
    assert_eq!(n.recall("when is the meeting?").unwrap().value.to_lowercase(), "friday");
    // find the monday episode id and reinforce it a few times
    let id = n.base.episodes.iter().find(|e| e.t.contains("monday")).unwrap().id;
    for _ in 0..4 { n.reinforce(id, 1.0); }
    assert_eq!(n.recall("when is the meeting?").unwrap().value.to_lowercase(), "monday");
}

#[test]
fn multi_hop_surfaces_unshared_fact() {
    let mut n = pn();
    n.observe("the phoenix project is owned by Dana");
    n.observe("the rollout ships in the third quarter");
    n.observe("the budget was approved by finance");
    for _ in 0..4 { n.recall("who owns phoenix?"); n.recall("when does the rollout ship?"); }
    for _ in 0..4 { n.recall("when does the rollout ship?"); n.recall("who approved the budget?"); }
    let res = n.recall_spreading("who owns phoenix?", 2, 5, 0.6, 6);
    let facts: Vec<String> = res.iter().map(|r| r.fact.to_lowercase()).collect();
    assert!(facts.iter().any(|f| f.contains("phoenix")), "seed must fire");
    let budget = res.iter().find(|r| r.fact.to_lowercase().contains("budget"));
    assert!(budget.is_some(), "2-hop associate not surfaced: {:?}", facts);
    assert!(!budget.unwrap().seed, "budget should be reached by spreading, not a seed");
}

#[test]
fn reuptake_attenuates_with_distance() {
    let mut n = pn();
    n.observe("the alpha signal is strong");
    n.observe("the relay forwards traffic");
    for _ in 0..4 { n.recall("what about alpha?"); n.recall("what does the relay do?"); }
    let res = n.recall_spreading("what about alpha?", 1, 10, 0.6, 6);
    let seed = res.iter().find(|r| r.fact.contains("alpha")).map(|r| r.activation).unwrap();
    let relay = res.iter().find(|r| r.fact.contains("relay")).map(|r| r.activation).unwrap_or(0.0);
    assert!(relay > 0.0, "neighbour should receive activation");
    assert!(relay < seed, "a 1-hop neighbour must be weaker than the seed (reuptake)");
}

#[test]
fn inhibitory_relation_gate() {
    let mut n = pn();
    n.observe("my dog's name is Rex");
    n.observe("my brother's name is Sam");
    let res = n.recall_spreading("what is my dog's name?", 1, 5, 0.6, 6);
    let joined: String = res.iter().map(|r| r.fact.to_lowercase()).collect::<Vec<_>>().join(" ");
    assert!(joined.contains("rex"), "correct relation should fire");
    assert!(!joined.contains("sam"), "conflicting relation must be inhibited: {}", joined);
}

#[test]
fn caveat_exact_key_disambiguation() {
    // project17 vs project170 collide on the 6-char stem; the exact-token tier resolves it.
    let mut n = Neuron::new(1000);
    n.observe("the project17 token is aaa111");
    n.observe("the project170 token is bbb222");
    assert_eq!(n.recall("what is the project170 token?").unwrap().value, "bbb222");
    assert_eq!(n.recall("what is the project17 token?").unwrap().value, "aaa111");
}

#[test]
fn caveat_multi_word_value() {
    let mut n = Neuron::new(1000);
    n.observe("my favorite tool is Search Console");
    assert_eq!(n.recall("what is my favorite tool?").unwrap().value, "Search Console");
}
