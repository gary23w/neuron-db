use neuron_core::Neuron;
use std::time::Instant;

fn main() {
    // creation rate
    let t = Instant::now(); let mut made = 0usize;
    while t.elapsed().as_millis() < 1000 {
        let mut n = Neuron::new(500);
        n.observe("my name is User. my plan is pro. my city is Halifax");
        std::hint::black_box(&n); made += 1;
    }
    println!("creation: {} neurons (3 facts) in ~1000ms = {}/sec", made, made);

    // recall throughput at N=500, distinct keys
    let nouns = ["wifi","door","bank","email","garage","locker","safe","router","vault","gate"];
    let adjs = ["north","south","east","west","main","spare","old","new","blue","red","work","home","beach","city","lake","hill","park","river","farm","studio"];
    let things = ["password","code","pin","key","number","combo","token","secret"];
    let mut n = Neuron::new(2000);
    let mut probes = Vec::new();
    let mut i = 0;
    for a in adjs { for no in nouns { for th in things {
        if i >= 500 { break; }
        let val = format!("{}{}", 1000 + i, (b'A' + (i % 26) as u8) as char);
        n.observe(&format!("the {} {} {} is {}", a, no, th, val));
        probes.push((format!("what is the {} {} {}?", a, no, th), val));
        i += 1;
    }}}
    let mut hits = 0;
    for (q, a) in &probes { if let Some(r) = n.recall(q) { if &r.value == a { hits += 1; } } }
    println!("recall accuracy (N={}, distinct keys): {}/{}", probes.len(), hits, probes.len());
    let t = Instant::now(); let iters = 50_000;
    for k in 0..iters { let (q,_) = &probes[k % probes.len()]; std::hint::black_box(n.recall(q)); }
    let per = t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
    println!("recall latency (N={}): {:.2} us/call = {:.0} recalls/sec", probes.len(), per, 1e6/per);
}
