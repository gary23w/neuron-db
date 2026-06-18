//! Cold-case probe: verifies the detective demo's case data against the real recall engine.
//!
//! The browser demo (docs/demos/cold-case.html) drives neuron-db through the same three ops this
//! probe exercises natively, so whatever resolves here resolves identically in the .wasm:
//!   - recall_many  -> "pull the file" on a person or object
//!   - recall_spreading -> the cork-board "red string": who/what is wired to a seed
//!   - chain        -> multi-hop deduction (victim -> partner -> creditor), walked one relation at a time
//!
//! Run: cargo run --example cold_case_probe

use neuron_core::Neuron;

// The case file. Phrased so each relation the player can follow ("partner", "creditor", "alibi",
// "buyer") appears literally in its fact and resolves to a capitalized name — that is what lets the
// chain walk advance instead of abstaining.
const CASE: &[&str] = &[
    "Lena Marsh owned the Vermillion Room, a jazz club on Halsted Street.",
    "Lena Marsh was found dead in the back room on the night of October 3rd.",
    "Lena Marsh's business partner was Marcus Vane.",
    "Lena Marsh's lover was Vivian Sloane, the club's headline singer.",
    "Marcus Vane co-owned the Vermillion Room with Lena Marsh.",
    "Marcus Vane was deep in gambling debt at the Athena card room.",
    "Marcus Vane's creditor was Eliza Crowe, a private loan shark.",
    "Marcus Vane would inherit the whole club if Lena Marsh died.",
    "Marcus Vane's buyer was Cornelius Pike, a rival club owner.",
    "Eliza Crowe was calling in Marcus Vane's debt that week.",
    "Eliza Crowe's enforcer was a man named Sal Roon.",
    "Cornelius Pike wanted to buy the Vermillion Room for years.",
    "Cornelius Pike's offer was rejected by Lena Marsh twice.",
    "Roman Dett was the bartender at the Vermillion Room.",
    "Roman Dett kept the only key to the back room.",
    "Roman Dett's alibi was Vivian Sloane.",
    "Vivian Sloane's alibi was Roman Dett.",
    "Vivian Sloane sang at the Vermillion Room every Friday night.",
    "Eddie the busboy saw Marcus Vane near the back room at eleven.",
    "The murder weapon was a brass bar rail missing from the club.",
    "A ledger vanished from Lena Marsh's office the same night.",
    "The ledger recorded Marcus Vane's debt to Eliza Crowe.",
];

fn chain(n: &mut Neuron, start: &str, path: &[&str]) -> (String, Vec<String>) {
    let mut current = start.to_string();
    let mut trail = vec![current.clone()];
    let mut broke = false;
    for rel in path {
        let rel_words: Vec<&str> = rel.split_whitespace().filter(|w| w.len() >= 3).collect();
        match n.recall(&format!("{} {}", current, rel)) {
            Some(h)
                if rel_words.is_empty()
                    || rel_words
                        .iter()
                        .any(|rw| h.fact.split_whitespace().any(|w| neuron_core::rel_matches(w, rw))) =>
            {
                current = h.value.clone();
                trail.push(current.clone());
            }
            _ => {
                broke = true;
                break;
            }
        }
    }
    (if broke { String::new() } else { current }, trail)
}

fn main() {
    let mut n = Neuron::new(100_000);
    for f in CASE {
        n.observe(f);
    }
    println!("case loaded: {} facts\n", n.episodes.len());

    println!("== recall (pull the file) ==");
    for q in ["Marcus Vane", "ledger", "murder weapon", "back room key", "Vivian Sloane"] {
        println!("  ? {q}");
        for r in n.recall_many(q, 3) {
            println!("      - {}", r.fact);
        }
    }

    println!("\n== assoc (red string) ==");
    for seed in ["Marcus Vane", "ledger", "back room"] {
        println!("  ~ {seed}");
        for s in n.recall_spreading(seed, 6, 2) {
            println!("      - {}", s.fact);
        }
    }

    println!("\n== chain (deduce) ==");
    let chains: &[(&str, &[&str])] = &[
        ("Lena Marsh", &["partner", "creditor"]),
        ("Lena Marsh", &["partner", "buyer"]),
        ("Lena Marsh", &["lover", "alibi"]),
        ("Roman Dett", &["alibi"]),
        ("Lena Marsh", &["partner", "enforcer"]),
    ];
    for (start, path) in chains {
        let (val, trail) = chain(&mut n, start, path);
        let verdict = if val.is_empty() { "(chain broke)".into() } else { format!("=> {val}") };
        println!("  {start} . {}  {verdict}", path.join(" . "));
        println!("      {}", trail.join("  ->  "));
    }
}
