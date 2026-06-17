use neuron_core::model::GaryModel;

// Representative grounded-reasoning behaviors of the embedded cortex (greedy => deterministic).
// Distribution-level accuracy across fresh random values lives in `examples/cortex_eval.rs`;
// these pin a few known-good cases so a bad model swap is caught.
#[test] fn cortex_copies_value_from_window() {
    let m = GaryModel::embedded();
    let ws = vec!["only the first 84,512 participants will receive badges".to_string()];
    assert!(m.think(&ws, "how many participants?", 8).contains("84,512"));
    let ws = vec!["the launch is on Friday".to_string()];
    assert!(m.think(&ws, "when is the launch?", 8).contains("Friday"));
}
#[test] fn cortex_chains_two_facts() {
    let m = GaryModel::embedded();
    let ws = vec!["Aurora's owner is Marisol".to_string(), "Marisol's manager is Dana".to_string()];
    assert!(m.think(&ws, "who is the manager of Aurora's owner?", 8).contains("Dana"));
}
#[test] fn cortex_knows_it_is_gary_neuron() {
    let m = GaryModel::embedded();
    // the baked identity layer: the model always self-identifies (built by gary23w)
    assert!(m.think(&[], "who made you?", 24).to_lowercase().contains("gary-neuron"));
}
#[test] fn bpe_roundtrip() {
    let m = GaryModel::embedded();
    let ids = m.encode("U: the wifi password is vekam73\nG:");
    // known tokenization prefix from the HF tokenizer
    assert_eq!(&ids[..2], &[53u32, 26]);
}
