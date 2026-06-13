use neuron_core::model::GaryModel;
#[test] fn cortex_copies_value_from_window() {
    let m = GaryModel::embedded();
    let ws = vec!["only the first 84,512 participants will receive badges".to_string()];
    assert!(m.think(&ws, "how many participants?", 8).contains("84,512"));
    let ws = vec!["the wifi password is vekam73".to_string()];
    assert!(m.think(&ws, "what is the wifi password?", 8).contains("vekam73"));
}
#[test] fn bpe_roundtrip() {
    let m = GaryModel::embedded();
    let ids = m.encode("U: the wifi password is vekam73\nG:");
    // known tokenization prefix from the HF tokenizer
    assert_eq!(&ids[..2], &[53u32, 26]);
}
