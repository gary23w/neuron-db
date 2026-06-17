//! Characterize the embedded gary-neuron cortex: extractive copy, multi-fact, paraphrase,
//! abstention, multi-hop, arithmetic, and open generation (coherence). Run: cargo run --example cortex_probe
use neuron_core::model::GaryModel;

fn main() {
    let m = GaryModel::embedded();
    let probe = |label: &str, facts: &[&str], q: &str| {
        let fv: Vec<String> = facts.iter().map(|s| s.to_string()).collect();
        let a = m.think(&fv, q, 16);
        println!("[{label}]\n  facts: {facts:?}\n  Q: {q}\n  → {a:?}\n");
    };

    println!("==== gary-neuron probe (1.13M params, 8L/E96/384ctx/2048 vocab, greedy) ====\n");

    // 1) extractive copy — the trained sweet spot
    probe("copy-value", &["the wifi password is vekam73"], "what is the wifi password?");
    probe("copy-number", &["only the first 84,512 participants will receive badges"], "how many participants?");
    probe("copy-date", &["the launch is on Friday"], "when is the launch?");

    // 2) multiple facts in the window — must pick the right one
    probe("select-among-3", &["the wifi password is vekam73","the launch is on Friday","the api key is zeta-9931"], "what is the api key?");

    // 3) paraphrase — query words don't match the fact
    probe("paraphrase", &["the capital of France is Paris"], "which city is the French capital?");

    // 4) abstention — answer not present
    probe("abstain", &["the launch is on Friday"], "what is the wifi password?");

    // 5) multi-hop over two facts (real reasoning)
    probe("multi-hop", &["Aurora's owner is Marisol","Marisol's manager is Dana"], "who is the manager of Aurora's owner?");

    // 6) tiny arithmetic / comparison over facts
    probe("compare", &["Alice has 3 apples","Bob has 5 apples"], "who has more apples?");

    // 7) open generation, no facts — pure coherence of the base LM
    probe("open-1", &[], "the weather today is");
    probe("open-2", &[], "i think that");
    probe("open-long", &[], "tell me about dogs");

    // 8) degradation: let it run longer to see repetition/incoherence
    let long = m.think(&[], "once upon a time", 40);
    println!("[open-40tok]\n  Q: once upon a time\n  → {long:?}\n");

    // TOKENIZER GROUND TRUTH — to validate the Python BPE replica matches exactly
    println!("==== TOK ====");
    for s in ["the wifi password is vekam73", "U: when is the launch?\nG:", "who is gary23w?"] {
        println!("TOK\t{:?}\t{:?}", s, m.encode(s));
    }
}
