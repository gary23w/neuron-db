use neuron_core::Neuron;
use neuron_core::model::GaryModel;
fn main() {
    let mut store = Neuron::new(500);
    for f in ["the wifi password is vekam73", "only the first 84,512 participants will receive badges",
              "the launch is on Friday"] { store.observe(f); }
    let m = GaryModel::embedded();
    for q in ["what is the wifi password?", "how many participants?", "when is the launch?"] {
        let facts: Vec<String> = store.recall(q).map(|r| vec![r.fact]).unwrap_or_default();
        println!("Q: {}\n  store -> {:?}\n  cortex -> {:?}", q, facts, m.think(&facts, q, 10));
    }
}
