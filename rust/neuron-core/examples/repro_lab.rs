//! Repro of the lab's degeneration: the lab's `gary()` wasm op calls cortex().think(facts, query, 24)
//! and DISPLAYS the raw string. This drives the SAME think() natively to show that the garbage on
//! elaboration questions over a partial working set is the cortex itself — not the §7 change (which
//! never touches the forward pass). Extractive queries are included as a control (they answer clean).
use neuron_core::model::GaryModel;

fn main() {
    let m = GaryModel::embedded();
    // approximate the lab's partial world-news working set (it "saved 8" from a web search incl. vit-D)
    let news: Vec<String> = [
        "Vitamin D deficiency is a current worldwide health status update.",
        "Vitamin D deficiency affects many people globally.",
        "Taiwan plays a crucial role in the semiconductor industry.",
        "Philipp Lahm announced his retirement from football.",
    ].iter().map(|s| s.to_string()).collect();

    println!("== ELABORATION over a partial working set (the lab's failing case) ==");
    for q in [
        "tell me more about this vitamin D deficiency pandemic",
        "why is vitamin D important?",
        "what is going on with vitamin D?",
    ] {
        println!("q={q:?}\n  think(24) -> {:?}", m.think(&news, q, 24));
    }

    println!("\n== EXTRACTIVE control (the cortex's actual competence) ==");
    let kv: Vec<String> = ["the api key is zeta-9931", "the wifi password is Maple-Cedar-71"]
        .iter().map(|s| s.to_string()).collect();
    for q in ["what is the api key?", "what is the wifi password?"] {
        println!("q={q:?}\n  think(24) -> {:?}", m.think(&kv, q, 24));
    }

    println!("\n== the SAME elaboration queries, but as dispatch() (clean enum mapping) ==");
    for q in ["tell me more about this vitamin D deficiency pandemic", "why is vitamin D important?"] {
        println!("q={q:?}\n  dispatch() -> {:?}", m.dispatch(&news, q));
    }
}
