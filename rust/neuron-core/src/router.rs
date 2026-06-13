//! NeuronRouter: chain many small neuron shards into one memory and fan a query out across
//! them. A single neuron's recall degrades as it grows (common cues collide); the router
//! keeps each shard small and picks the single best-matching value across all shards.
//! Faithful std-only port of neuron_db/router.py.

use crate::{Neuron, Recall};

const BIG: usize = usize::MAX / 2; // shards never self-truncate; the router controls size

pub struct NeuronRouter {
    pub per_shard: usize,
    pub shards: Vec<Neuron>,
}

impl NeuronRouter {
    pub fn new(per_shard: usize) -> Self {
        NeuronRouter { per_shard, shards: vec![Neuron::new(BIG)] }
    }

    /// Fill the current shard; spill into a fresh one when it reaches `per_shard`.
    pub fn observe(&mut self, text: &str) -> usize {
        if self.shards.last().unwrap().fact_count() >= self.per_shard {
            self.shards.push(Neuron::new(BIG));
        }
        self.shards.last_mut().unwrap().observe(text)
    }

    /// Fan out across shards, keep the strongest hit by (overlap, coverage).
    pub fn recall(&mut self, query: &str) -> Option<Recall> {
        let mut best: Option<Recall> = None;
        let mut bk = (-1i64, -1i64, -1f64);
        for sh in self.shards.iter_mut() {
            if let Some(h) = sh.recall(query) {
                let sc = (h.exact as i64, h.overlap as i64, h.coverage);
                let better = sc.0 > bk.0 || (sc.0 == bk.0 && sc.1 > bk.1)
                    || (sc.0 == bk.0 && sc.1 == bk.1 && sc.2 > bk.2);
                if better { bk = sc; best = Some(h); }
            }
        }
        best
    }
    pub fn get(&mut self, query: &str) -> Option<String> { self.recall(query).map(|h| h.value) }
    pub fn fact_count(&self) -> usize { self.shards.iter().map(|s| s.fact_count()).sum() }
    pub fn shard_count(&self) -> usize { self.shards.len() }
    pub fn dump(&self) -> Vec<String> { self.shards.iter().map(|s| s.dump()).collect() }
    pub fn load(blobs: &[String], per_shard: usize) -> Self {
        let shards: Vec<Neuron> = blobs.iter().map(|b| Neuron::load(b, BIG)).collect();
        NeuronRouter { per_shard, shards: if shards.is_empty() { vec![Neuron::new(BIG)] } else { shards } }
    }
}
