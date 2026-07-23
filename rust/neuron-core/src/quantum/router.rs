//! QuantumRouter: entangled sharding over the plain `NeuronRouter`. Entangling places the dest
//! fact on an IDLE shard (least-loaded, never the source's own, spawning a fresh one when
//! everything else is full), so a teleport's reconstruction — the write half of the recall —
//! always lands off the busy shard. Load-balanced recall by construction, not by a scheduler.
//! Shard scopes are named "shard:<index>" in the records it keeps.

use super::entangle::EntanglementRecord;
use super::teleport::{reconstruct, TeleportResult};
use super::{now_ms, rewrite_in};
use crate::router::NeuronRouter;
use crate::{Neuron, Recall};

const BIG: usize = usize::MAX / 2;   // shards never self-truncate; the router controls size

pub struct QuantumRouter {
    pub inner: NeuronRouter,
    links: Vec<EntanglementRecord>,
    next_id: u64,
}

fn scope_name(i: usize) -> String { format!("shard:{}", i) }

impl QuantumRouter {
    pub fn new(per_shard: usize) -> Self {
        QuantumRouter { inner: NeuronRouter::new(per_shard), links: Vec::new(), next_id: 1 }
    }

    pub fn observe(&mut self, text: &str) -> usize { self.inner.observe(text) }
    pub fn recall(&mut self, query: &str) -> Option<Recall> { self.inner.recall(query) }
    pub fn get(&mut self, query: &str) -> Option<String> { self.inner.get(query) }
    pub fn fact_count(&self) -> usize { self.inner.fact_count() }
    pub fn shard_count(&self) -> usize { self.inner.shard_count() }
    /// The links currently held (for inspection; consumed links are gone).
    pub fn entanglements(&self) -> &[EntanglementRecord] { &self.links }

    /// Which shard holds a fact with this exact text (highest shard wins — the freshest copy).
    pub fn shard_of(&self, text: &str) -> Option<usize> {
        self.inner.shards.iter().enumerate().rev()
            .find(|(_, sh)| sh.episodes.iter().any(|e| e.t == text))
            .map(|(i, _)| i)
    }

    /// The least-loaded shard OTHER than `avoid` that still has room; when no such shard exists
    /// (single-shard router, or everything else full), a fresh one is spawned. This is where the
    /// teleport's reconstruction will land — never the busy source shard.
    fn idle_shard(&mut self, avoid: usize) -> usize {
        let per = self.inner.per_shard;
        let pick = self.inner.shards.iter().enumerate()
            .filter(|(i, sh)| *i != avoid && sh.fact_count() < per)
            .min_by_key(|(_, sh)| sh.fact_count())
            .map(|(i, _)| i);
        match pick {
            Some(i) => i,
            None => { self.inner.shards.push(Neuron::new(BIG)); self.inner.shards.len() - 1 }
        }
    }

    /// Entangle: store the source through the router's normal fill/spill path, place the dest on
    /// the idle shard, link them. Returns the link id (0 when the source text did not encode —
    /// nothing was linked).
    pub fn entangle(&mut self, src_text: &str, dst_text: &str, classical: &str, ebits: u32) -> u64 {
        if self.shard_of(src_text).is_none() && self.inner.observe(src_text) == 0 { return 0; }
        let src_shard = match self.shard_of(src_text) { Some(i) => i, None => return 0 };
        let dst_shard = match self.shard_of(dst_text) {
            Some(i) if i != src_shard => i,   // dest already stored elsewhere: link in place
            found => {
                let i = self.idle_shard(src_shard);
                if let Some(cur) = found {
                    // stored on the source's own shard: relocate so the two scopes stay distinct
                    // (the rebind bookkeeping in teleport() relies on it)
                    let n = &mut self.inner.shards[cur];
                    n.episodes.retain(|e| e.t != dst_text);
                    n.invalidate_index();
                }
                if self.inner.shards[i].observe(dst_text) == 0 { return 0; }
                i
            }
        };
        let id = self.next_id; self.next_id += 1;
        self.links.push(EntanglementRecord {
            id,
            source_scope: scope_name(src_shard), source_text: src_text.to_string(),
            dest_scope: scope_name(dst_shard), dest_text: dst_text.to_string(),
            classical: classical.to_string(), ebits: ebits.max(1), created_at: now_ms(),
        });
        id
    }

    pub fn disentangle(&mut self, id: u64) -> bool {
        let before = self.links.len();
        self.links.retain(|l| l.id != id);
        self.links.len() < before
    }

    /// Teleport across shards: recall the source (fan-out, as ever), spend an e-bit, apply the
    /// classical instruction to the dest fact ON ITS OWN (idle) shard. The result's scopes name
    /// the shards involved.
    pub fn teleport(&mut self, cue: &str) -> Option<TeleportResult> {
        let hit = self.inner.recall(cue)?;
        let src_shard = self.shard_of(&hit.fact)?;
        let src_scope = scope_name(src_shard);
        let li = self.links.iter().position(|l| l.source_scope == src_scope && l.source_text == hit.fact && l.ebits > 0)?;
        self.links[li].ebits -= 1;
        let link = self.links[li].clone();
        let remaining = link.ebits;
        if remaining == 0 { self.links.remove(li); }
        let dst_shard: usize = link.dest_scope.strip_prefix("shard:").and_then(|s| s.parse().ok())?;
        if dst_shard >= self.inner.shards.len() { return None; }
        let (new_dest, value) = if link.classical == "swap" {
            rewrite_in(&mut self.inner.shards[src_shard], &hit.fact, &link.dest_text);
            (hit.fact.clone(), hit.value.clone())
        } else {
            reconstruct(&link.classical, &hit.fact, &hit.value)
        };
        if rewrite_in(&mut self.inner.shards[dst_shard], &link.dest_text, &new_dest) {
            // shards are distinct scopes here, so a field-wise rebind never collides
            for l in self.links.iter_mut() {
                if l.dest_scope == link.dest_scope && l.dest_text == link.dest_text { l.dest_text = new_dest.clone(); }
                if l.source_scope == link.dest_scope && l.source_text == link.dest_text { l.source_text = new_dest.clone(); }
                if link.classical == "swap" {
                    if l.dest_scope == src_scope && l.dest_text == hit.fact { l.dest_text = link.dest_text.clone(); }
                    if l.source_scope == src_scope && l.source_text == hit.fact { l.source_text = link.dest_text.clone(); }
                }
            }
        }
        Some(TeleportResult {
            value,
            source_scope: src_scope,
            source_fact: hit.fact,
            dest_scope: link.dest_scope,
            dest_fact: new_dest,
            classical_used: link.classical,
            ebits_remaining: remaining,
        })
    }
}
