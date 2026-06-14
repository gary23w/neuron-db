//! Basic CRUD against a durable NeuronDB. Run:
//!   cargo run --release --example quickstart --features sqlite
use neuron_core::db::NeuronDB;

fn main() {
    let db = NeuronDB::open(&std::env::temp_dir().join("neuron_quickstart.db").to_string_lossy(), 500);

    // INSERT: state facts in plain language
    db.observe("user:42", "the plan is pro");
    db.observe("user:42", "the region is us-west-2");
    db.observe("user:42", "the seat count is 12");

    // READ: ask questions, get values back
    println!("plan   = {:?}", db.get("user:42", "what plan?"));        // Some("pro")
    println!("region = {:?}", db.get("user:42", "where is the region?"));
    println!("seats  = {:?}", db.get("user:42", "how many seats?"));   // number-aware

    // "UPDATE": just re-state it — the newest matching fact wins
    db.observe("user:42", "the plan is enterprise");
    println!("plan'  = {:?}", db.get("user:42", "what plan?"));        // Some("enterprise")

    // RECALL with metadata (fact + coverage), useful for ranking/debugging
    if let Some(hit) = db.recall("user:42", "what is the region?") {
        println!("recall -> value={} coverage={:.0}% fact={:?}", hit.value, hit.coverage * 100.0, hit.fact);
    }

    // DELETE: by substring, or wipe the scope with None
    let (forgot, remaining) = db.forget("user:42", Some("seat"));
    println!("forgot {} fact(s), {} remaining", forgot, remaining);

    // INTROSPECT
    let s = db.stats("user:42");
    println!("stats: {} facts, {} turns", s.facts, s.turns);
    println!("scopes: {:?}", db.neurons());
}
