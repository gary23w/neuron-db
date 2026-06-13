use neuron_core::router::NeuronRouter;

#[test]
fn spills_into_shards_and_recalls() {
    let mut r = NeuronRouter::new(8);
    for i in 0..40 { r.observe(&format!("the gate{} code is val{}", i, i)); }
    assert!(r.shard_count() >= 5, "should spill: {} shards", r.shard_count());
    assert_eq!(r.fact_count(), 40);
    assert_eq!(r.get("what is the gate17 code?").as_deref(), Some("val17"));
    assert_eq!(r.get("what is the gate3 code?").as_deref(), Some("val3"));
}

#[test]
fn fanout_scale_distinct_keys() {
    let adj=["north","south","east","west","gold","iron","jade","onyx"];
    let noun=["gate","vault","door","panel","relay","valve","hatch","port"];
    let mut r = NeuronRouter::new(128);
    let mut keys=vec![];
    for i in 0..1000 {
        let k=format!("{} {} {}", adj[i%8], noun[(i/8)%8], i);
        keys.push((k.clone(), format!("v{}",i)));
        r.observe(&format!("the {} reads v{}", k, i));
    }
    let mut ok=0;
    for (k,v) in keys.iter().step_by(7) {
        if r.get(&format!("what does the {} read?", k)).as_deref()==Some(v.as_str()) { ok+=1; }
    }
    let n=keys.iter().step_by(7).count();
    assert!(ok*100 >= n*98, "fan-out recall {}/{}", ok, n);
}

#[test]
fn dump_load_roundtrip() {
    let mut r = NeuronRouter::new(4);
    for i in 0..12 { r.observe(&format!("the node{} owner is owner{}", i, i)); }
    let blobs = r.dump();
    let mut r2 = NeuronRouter::load(&blobs, 4);
    assert_eq!(r2.fact_count(), 12);
    assert_eq!(r2.get("who is the node5 owner?").as_deref(), Some("owner5"));
}
