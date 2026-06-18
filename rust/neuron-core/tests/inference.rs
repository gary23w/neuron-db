#![cfg(feature = "cortex")] // the dispatcher tests need the baked cortex; skip them in a store-only build
use neuron_core::model::{GaryModel, Dispatch};

// gary-neuron is the middle DISPATCHER: each turn routes to ANSWER / ESCALATE / FETCH / STORE.
// These pin known-good routes (greedy => deterministic). For the ANSWER route the host grounds
// the literal value from the store's deterministic recall; the cortex decides the route.
#[test] fn dispatch_answers_from_memory() {
    let m = GaryModel::embedded();
    let ws = vec!["the api key is zeta-9931".to_string(), "the launch is on Friday".to_string()];
    match m.dispatch(&ws, "what is the api key?") {
        Dispatch::Answer(v) => assert!(v.contains("zeta-9931"), "got {v:?}"),
        d => panic!("expected Answer, got {d:?}"),
    }
}
#[test] fn dispatch_chains_two_facts() {
    let m = GaryModel::embedded();
    let ws = vec!["Aurora's owner is Marisol".to_string(), "Marisol's manager is Dana".to_string()];
    match m.dispatch(&ws, "who is the manager of Aurora's owner?") {
        Dispatch::Answer(v) => assert!(v.contains("Dana"), "got {v:?}"),
        d => panic!("expected Answer, got {d:?}"),
    }
}
#[test] fn dispatch_escalates_when_not_in_memory() {
    let m = GaryModel::embedded();
    let ws = vec!["the api key is zeta-9931".to_string()];
    assert_eq!(m.dispatch(&ws, "explain how DNS works"), Dispatch::Escalate);
}
#[test] fn cortex_knows_it_is_gary_neuron() {
    let m = GaryModel::embedded();
    match m.dispatch(&[], "who made you?") {
        Dispatch::Answer(v) => assert!(v.to_lowercase().contains("gary-neuron"), "got {v:?}"),
        d => panic!("expected Answer, got {d:?}"),
    }
}
#[test] fn bpe_roundtrip() {
    let m = GaryModel::embedded();
    let ids = m.encode("U: the wifi password is vekam73\nG:");
    assert_eq!(&ids[..2], &[53u32, 26]);   // known tokenization prefix
}
