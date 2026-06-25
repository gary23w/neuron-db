//! Deterministic dispatch battery: prints the cortex's raw generation for a fixed set of
//! (working-set, query) cases through the real Rust runtime. Diff the output before vs after a
//! forward-pass change (e.g. SIMD) to prove token-for-token equivalence (greedy => deterministic).
use neuron_core::model::GaryModel;

fn main() {
    let m = GaryModel::embedded();
    let keys = ["api key", "wifi password", "gate code", "database port", "deploy region",
                "launch date", "building manager", "firmware version", "seat assignment", "monthly rent"];
    let vals = ["zeta-9931", "Maple-Cedar-71", "4471", "5432", "us-west-2",
                "Friday", "Priya", "v3.1.0", "22A", "2100 dollars"];
    let names = ["Marisol", "Dana", "Tony", "Lena", "Omar", "Aisha"];
    let mut cases: Vec<(Vec<String>, String)> = Vec::new();

    // ANSWER over a range of working-set sizes (the v2 failure axis), rotating which key is asked
    for ws in [1usize, 2, 3, 5, 8, 11, 14, 17] {
        let facts: Vec<String> = (0..ws).map(|i| format!("the {} is {}", keys[i % keys.len()], vals[i % vals.len()])).collect();
        for tk in [0usize, ws / 2, ws.saturating_sub(1)] {
            cases.push((facts.clone(), format!("what is the {}?", keys[tk % keys.len()])));
        }
    }
    // routes: escalate, fetch, store
    cases.push((vec!["the api key is zeta-9931".into()], "explain how DNS works".into()));
    cases.push((vec!["the gate code is 4471".into(), "the seat assignment is 22A".into()], "write a short poem about the sea".into()));
    cases.push((vec![], "what is the capital of France?".into()));
    cases.push((vec![], "what's the weather in Tokyo?".into()));
    cases.push((vec!["the api key is zeta-9931".into()], "my wifi password is hunter2".into()));
    // multihop + compare
    for (a, b, c) in [("Aurora", "Marisol", "Dana"), ("Nimbus", "Tony", "Lena"), ("Vega", "Omar", "Aisha")] {
        cases.push((vec![format!("{a}'s owner is {b}"), format!("{b}'s manager is {c}")], format!("who is the manager of {a}'s owner?")));
    }
    for (x, y, nx, ny) in [("Ravi", "Caleb", 97, 86), ("Hana", "Greta", 75, 52), ("Idris", "Yara", 20, 41)] {
        cases.push((vec![format!("{x} has {nx} coins"), format!("{y} has {ny} coins")], "who has more coins?".into()));
        cases.push((vec![format!("{x} has {nx} coins"), format!("{y} has {ny} coins")], "who has fewer coins?".into()));
    }
    let _ = names;
    cases.push((vec![], "who made you?".into()));

    for (i, (facts, q)) in cases.iter().enumerate() {
        let out = m.think(facts, q, 24);
        println!("{:>3} ws={:<2} q={:<42} -> {:?}", i, facts.len(), q, out);
    }
    println!("== {} cases ==", cases.len());
}
