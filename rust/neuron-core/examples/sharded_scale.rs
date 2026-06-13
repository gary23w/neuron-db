//! Hold a lot of facts with sharp recall by sharding across many small neurons.
//! Run: cargo run --release --example sharded_scale
use neuron_core::router::NeuronRouter;

fn main() {
    let mut r = NeuronRouter::new(128); // facts per shard
    for i in 0..5000 {
        r.observe(&format!("the gate{} code is val{}", i, i));
    }
    println!("{} facts across {} shards", r.fact_count(), r.shard_count());
    println!("gate4831 -> {:?}", r.get("what is the gate4831 code?")); // fan-out, exact hit
    println!("gate77   -> {:?}", r.get("what is the gate77 code?"));
}
