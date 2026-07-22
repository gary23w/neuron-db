//! PlasticNeuron: the adaptive store tier in pure Rust. A Neuron whose recall changes with
//! use -- strength on recall, lazy exponential decay, Hebbian links, and a neurotransmitter
//! style spreading-activation recall. All O(1) / O(neighbors) scalar updates, no neural net.
//! Faithful port of neuron_db/plastic.py.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use super::{Neuron, Recall, content, stems_s, stem1, rel_s, pets, stopval_s, pick_value, has_stem};

/// One readout of the spreading-activation pass.
#[derive(Clone, Debug)]
pub struct Spread { pub fact: String, pub value: String, pub activation: f64, pub seed: bool }

pub struct PlasticNeuron {
    pub base: Neuron,
    pub half_life: Option<f64>,   // None / <=0 / inf => no decay
    pub link_window: usize,
    pub tick: i64,
    w: HashMap<i64, f64>,
    t: HashMap<i64, i64>,
    links: HashMap<i64, HashMap<i64, f64>>,
    recent: Vec<i64>,
    pinned: HashSet<i64>,
    next_id: i64,
}

#[inline]
fn gt(a: &(i64, f64, i64, i64, i64, i64), b: &(i64, f64, i64, i64, i64, i64)) -> bool {
    if a.0 != b.0 { return a.0 > b.0; }
    if a.1 != b.1 { return a.1 > b.1; }
    if a.2 != b.2 { return a.2 > b.2; }
    if a.3 != b.3 { return a.3 > b.3; }
    if a.4 != b.4 { return a.4 > b.4; }
    a.5 > b.5
}

impl PlasticNeuron {
    pub fn new(max_facts: usize, half_life: Option<f64>, link_window: usize) -> Self {
        PlasticNeuron { base: Neuron::new(max_facts), half_life, link_window, tick: 0,
            w: HashMap::new(), t: HashMap::new(), links: HashMap::new(),
            recent: Vec::new(), pinned: HashSet::new(), next_id: 0 }
    }
    pub fn fact_count(&self) -> usize { self.base.fact_count() }
    pub fn strength(&self, id: i64) -> f64 { self.eff(id) }

    fn eff(&self, id: i64) -> f64 {
        let w = *self.w.get(&id).unwrap_or(&1.0);
        match self.half_life {
            None => w,
            Some(h) if h <= 0.0 || h.is_infinite() => w,
            Some(h) => { let age = (self.tick - *self.t.get(&id).unwrap_or(&self.tick)) as f64; w * 0.5f64.powf(age / h) }
        }
    }
    pub fn pin(&mut self, id: i64) { self.pinned.insert(id); }
    fn link(&mut self, a: i64, b: i64, d: f64) {
        if a == b { return; }
        *self.links.entry(a).or_default().entry(b).or_insert(0.0) += d;
        *self.links.entry(b).or_default().entry(a).or_insert(0.0) += d;
    }
    pub fn reinforce(&mut self, id: i64, amount: f64) {
        let e = self.eff(id); self.w.insert(id, e + amount); self.t.insert(id, self.tick);
    }

    pub fn observe(&mut self, text: &str) -> usize {
        let before = self.base.episodes.len();
        let n = self.base.observe(text);
        // PlasticNeuron is meant to run with a large max_facts (no drain); the new episodes
        // are the last `n` appended.
        let mut ids = Vec::new();
        let len = self.base.episodes.len();
        if before + n <= len {
            for k in 0..n {
                let idx = before + k;
                let id = self.next_id; self.next_id += 1;
                self.base.episodes[idx].id = id;
                self.w.insert(id, 1.0); self.t.insert(id, self.tick);
                ids.push(id);
            }
        }
        for i in 0..ids.len() { for j in i + 1..ids.len() { self.link(ids[i], ids[j], 1.0); } }
        self.tick += 1;
        n
    }

    /// Same matching as Neuron::recall, but ties break on learned strength, and the hit is
    /// reinforced and wired to recently-recalled facts (the adaptive side effects).
    pub fn recall(&mut self, query: &str) -> Option<Recall> {
        let cue: HashSet<String> = stems_s(&content(query));
        if cue.is_empty() { return None; }
        let qraw: HashSet<String> = content(query);
        let pet_query = cue.contains(&stem1("pet")) || cue.contains(&stem1("animal"));
        let name_query = cue.contains("name") && cue.intersection(rel_s()).count() == 0;
        let order = self.base.candidates(&cue, pet_query);
        let mut best: Option<usize> = None;
        let mut bk = (-1i64, -1f64, -1i64, -1i64, -1i64, -1i64);
        for i in order {
            let e = &self.base.episodes[i];
            let mut ov = cue.iter().filter(|c| has_stem(&e.s, c)).count();
            let es_pet = e.s.iter().any(|s| pets().contains(s.as_ref()));
            if ov < 1 && pet_query && es_pet { ov = 1; }
            if ov < 1 { continue; }
            if e.s.iter().any(|s| rel_s().contains(s.as_ref()) && !cue.contains(s.as_ref())) && !(pet_query && es_pet) { continue; }
            if cue.iter().any(|s| rel_s().contains(s) && !has_stem(&e.s, s)) && !(pet_query && es_pet) { continue; }
            let exact_ov = qraw.iter().filter(|wd| has_stem(&e.raw, wd)).count() as i64;
            let selfp = if name_query && e.self_flag { 1 } else { 0 };
            let subj = if cue.contains(&e.head) { 1 } else { 0 };
            let spec = -(e.s.iter().filter(|s| !cue.contains(s.as_ref()) && !stopval_s().contains(s.as_ref())).count() as i64);
            let sc = (exact_ov, self.eff(e.id), selfp, subj, spec, i as i64);
            if gt(&sc, &bk) { bk = sc; best = Some(i); }
        }
        let bi = best?;
        let (fact, sset, id) = { let e = &self.base.episodes[bi]; (e.t.clone(), e.s.clone(), e.id) };
        let mut cov = cue.iter().filter(|c| has_stem(&sset, c)).count() as f64 / (cue.len().max(1) as f64);
        if pet_query && sset.iter().any(|s| pets().contains(s.as_ref())) { cov = 1.0; }
        let overlap = cue.iter().filter(|c| has_stem(&sset, c)).count();
        let exact_n = bk.0 as usize;
        let want_num = cue.contains("many") || cue.contains("much") || cue.contains(&stem1("number"));
        let (val, echo) = pick_value(&self.base.episodes[bi], &cue, want_num);
        if id >= 0 {
            self.reinforce(id, 1.0);
            let recents: Vec<i64> = self.recent.iter().rev().take(self.link_window).cloned().collect();
            for prev in recents { self.link(id, prev, 0.5); }
            self.recent.push(id);
            let l = self.recent.len(); if l > 32 { self.recent.drain(0..l - 32); }
            self.tick += 1;
        }
        Some(Recall { fact, value: val, coverage: cov, overlap, exact: exact_n, echo, idx: bi })
    }

    /// The hit plus its strongest associates (one-hop spreading).
    pub fn recall_related(&mut self, query: &str, k: usize) -> Vec<(String, String, f64)> {
        let hit = match self.recall(query) { Some(h) => h, None => return Vec::new() };
        let eid = self.base.episodes.iter().find(|e| e.t == hit.fact).map(|e| e.id).unwrap_or(-1);
        let mut out = vec![(hit.fact.clone(), hit.value.clone(), 0.0f64)];
        let mut nbrs: Vec<(i64, f64)> = self.links.get(&eid).map(|m| m.iter().map(|(a, b)| (*a, *b)).collect()).unwrap_or_default();
        nbrs.sort_by(|a, b| (b.1 * self.eff(b.0)).partial_cmp(&(a.1 * self.eff(a.0))).unwrap());
        let by_id: HashMap<i64, usize> = self.base.episodes.iter().enumerate().map(|(i, e)| (e.id, i)).collect();
        for (nid, w) in nbrs.into_iter().take(k) {
            if let Some(&ix) = by_id.get(&nid) { let e = &self.base.episodes[ix]; out.push((e.t.clone(), e.v.clone(), w)); }
        }
        out
    }

    /// Neurotransmitter-style recall: excitatory release at cue matches, inhibitory relation
    /// gating, multi-hop spread across synapses with reuptake decay. Pure read (no mutation).
    pub fn recall_spreading(&mut self, query: &str, hops: usize, k: usize, reuptake: f64, fan: usize) -> Vec<Spread> {
        let cue: HashSet<String> = stems_s(&content(query));
        if cue.is_empty() { return Vec::new(); }
        let pet_query = cue.contains(&stem1("pet")) || cue.contains(&stem1("animal"));
        let order = self.base.candidates(&cue, pet_query);
        let by_id: HashMap<i64, usize> = self.base.episodes.iter().enumerate().map(|(i, e)| (e.id, i)).collect();
        let mut act: HashMap<i64, f64> = HashMap::new();
        for i in order {
            let e = &self.base.episodes[i];
            let mut ov = cue.iter().filter(|c| has_stem(&e.s, c)).count();
            let es_pet = e.s.iter().any(|s| pets().contains(s.as_ref()));
            if ov < 1 && pet_query && es_pet { ov = 1; }
            if ov < 1 { continue; }
            if e.s.iter().any(|s| rel_s().contains(s.as_ref()) && !cue.contains(s.as_ref())) && !(pet_query && es_pet) { continue; }
            if cue.iter().any(|s| rel_s().contains(s) && !has_stem(&e.s, s)) && !(pet_query && es_pet) { continue; }
            if e.id < 0 { continue; }
            *act.entry(e.id).or_insert(0.0) += ov as f64 * self.eff(e.id).max(1e-6);
        }
        if act.is_empty() { return Vec::new(); }
        let seeds: HashSet<i64> = act.keys().cloned().collect();
        let mut frontier = act.clone();
        for _ in 0..hops {
            let mut nxt: HashMap<i64, f64> = HashMap::new();
            for (&src, &a) in frontier.iter() {
                if a <= 1e-9 { continue; }
                let nbrs = match self.links.get(&src) { Some(m) => m, None => continue };
                let mut top: Vec<(i64, f64)> = nbrs.iter().map(|(a, b)| (*a, *b)).collect();
                top.sort_by(|x, y| (y.1 * self.eff(y.0)).partial_cmp(&(x.1 * self.eff(x.0))).unwrap());
                top.truncate(fan);
                let norm: f64 = top.iter().map(|(_, w)| *w).sum::<f64>().max(1.0);
                for (nid, w) in top {
                    let sig = a * (1.0 - reuptake) * (w / norm) * self.eff(nid).min(1.0);
                    if sig <= 1e-9 { continue; }
                    *nxt.entry(nid).or_insert(0.0) += sig;
                }
            }
            for (nid, s) in nxt.iter() { *act.entry(*nid).or_insert(0.0) += *s; }
            if nxt.is_empty() { break; }
            frontier = nxt;
        }
        let mut ranked: Vec<(i64, f64)> = act.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(k);
        ranked.into_iter().filter_map(|(id, a)| by_id.get(&id).map(|&ix| {
            let e = &self.base.episodes[ix];
            Spread { fact: e.t.clone(), value: e.v.clone(), activation: (a * 1e4).round() / 1e4, seed: seeds.contains(&id) }
        })).collect()
    }

    /// "Sleep": merge facts with identical stem-sets (sum strength), prune decayed-away facts.
    pub fn consolidate(&mut self, prune_below: f64) -> (usize, usize, usize) {
        let mut by_key: HashMap<Vec<Arc<str>>, i64> = HashMap::new();
        let mut keep: Vec<usize> = Vec::new();
        let (mut merged, mut pruned) = (0usize, 0usize);
        for i in 0..self.base.episodes.len() {
            let key = self.base.episodes[i].s.clone();
            let id = self.base.episodes[i].id;
            if let Some(&ai) = by_key.get(&key) {
                let e = self.eff(ai) + self.eff(id); self.w.insert(ai, e); self.t.insert(ai, self.tick);
                if let Some(m) = self.links.remove(&id) { for (nid, w) in m { self.link(ai, nid, w); } }
                merged += 1;
            } else { by_key.insert(key, id); keep.push(i); }
        }
        let mut survivors: Vec<usize> = Vec::new();
        for &i in &keep {
            let (id, is_self) = { let e = &self.base.episodes[i]; (e.id, e.self_flag) };
            if self.eff(id) >= prune_below || is_self || self.pinned.contains(&id) { survivors.push(i); }
            else { pruned += 1; self.w.remove(&id); self.t.remove(&id); self.links.remove(&id); }
        }
        let survivor_set: HashSet<usize> = survivors.iter().cloned().collect();
        let mut new_eps = Vec::new();
        for i in 0..self.base.episodes.len() { if survivor_set.contains(&i) { new_eps.push(self.base.episodes[i].clone()); } }
        self.base.episodes = new_eps;
        // drop any link edge that points at a merged-away or pruned id — both endpoints must still be
        // alive — so spreading recall never chases a dangling neighbor (incoming refs, not just outgoing).
        let alive: HashSet<i64> = self.base.episodes.iter().map(|e| e.id).collect();
        self.links.retain(|id, _| alive.contains(id));
        for nbrs in self.links.values_mut() { nbrs.retain(|nid, _| alive.contains(nid)); }
        self.base.invalidate_index();
        (merged, pruned, self.base.episodes.len())
    }
}
