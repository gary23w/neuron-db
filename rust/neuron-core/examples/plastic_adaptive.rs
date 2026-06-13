//! Memory that adapts: strength on use, decay on disuse, and neurotransmitter-style
//! spreading recall. No database, no features needed.
//! Run: cargo run --release --example plastic_adaptive
use neuron_core::plastic::PlasticNeuron;

fn main() {
    let mut n = PlasticNeuron::new(10_000_000, Some(1e9), 3);
    n.observe("the meeting is on monday");
    n.observe("the meeting is on friday");
    println!("fresh recall (recency): {}", n.recall("when is the meeting?").unwrap().value);

    // reinforce the monday fact so it wins despite being older
    let id = n.base.episodes.iter().find(|e| e.t.contains("monday")).unwrap().id;
    for _ in 0..4 { n.reinforce(id, 1.0); }
    println!("after use (adapted):    {}", n.recall("when is the meeting?").unwrap().value);

    // spreading activation surfaces a wired-but-unshared fact
    n.observe("the phoenix project is owned by Dana");
    n.observe("the rollout ships in q3");
    for _ in 0..4 { n.recall("who owns phoenix?"); n.recall("when does the rollout ship?"); }
    println!("\nspreading from 'phoenix':");
    for s in n.recall_spreading("who owns phoenix?", 2, 5, 0.6, 6) {
        println!("  {:<40} activation={:.3} seed={}", s.fact, s.activation, s.seed);
    }
}
