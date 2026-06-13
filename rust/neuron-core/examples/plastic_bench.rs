use neuron_core::plastic::PlasticNeuron;
use std::time::Instant;

fn main() {
    let adj = ["north","south","east","west","gold","iron","jade","onyx","ruby","teal","main","spare"];
    let noun = ["server","log","report","ticket","build","deploy","alert","metric","trace","span"];
    let n_facts = 50_000;
    let mut pn = PlasticNeuron::new(100_000_000, Some(1e9), 3);
    let t = Instant::now();
    for i in 0..n_facts {
        pn.observe(&format!("the {} {} {} reads code{}", adj[i%adj.len()], noun[(i/12)%noun.len()], i, i));
    }
    let obs = t.elapsed().as_secs_f64();
    println!("observe: {} facts in {:.3}s = {:.0} facts/sec", n_facts, obs, n_facts as f64/obs);

    // selective recall
    pn.recall("jade trace 1450");
    let reps = 2000;
    let t = Instant::now();
    for _ in 0..reps { pn.recall("jade trace 1450"); }
    println!("recall (selective): {:.1} us/op", t.elapsed().as_secs_f64()/reps as f64*1e6);

    // spreading activation
    pn.recall_spreading("jade trace 1450", 2, 10, 0.6, 6);
    let t = Instant::now();
    for _ in 0..reps { pn.recall_spreading("jade trace 1450", 2, 10, 0.6, 6); }
    println!("recall_spreading:   {:.1} us/op", t.elapsed().as_secs_f64()/reps as f64*1e6);
}
