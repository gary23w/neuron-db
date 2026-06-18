// Hand-written numeric kernels below index parallel weight/activation arrays by position; the
// index-based loops mirror the reference math and read clearer than enumerate/zip chains.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! GaryModel — the emergence cortex + tokenizer, BUNDLED in the binary (include_bytes!).
//! Give it a working set (facts the store retrieved) and a question; it generates the answer.
use crate::cortex::{Cortex, Kv};
use crate::bpe::Bpe;

pub struct GaryModel { cortex: Cortex, bpe: Bpe }

/// The routing decision the middle-layer cortex makes for one turn. v3 emits this as a short
/// action-prefixed string; `dispatch` parses it so the host (lab / WASM / MCP) can act:
///   Answer  -> memory answered; serve `.0` directly (no host-model round trip)
///   Escalate-> memory can't; hand the turn up to the larger host model
///   Fetch   -> live-world question; go to the web for `.0`
///   Store   -> a declarative; remember `.0`
#[derive(Debug, Clone, PartialEq)]
pub enum Dispatch { Answer(String), Escalate, Fetch(String), Store(String) }

impl GaryModel {
    /// Load the emergence model baked into the binary at compile time. No files, no network.
    pub fn embedded() -> GaryModel {
        let bin = include_bytes!("../model/cortex.bin");
        let man = include_str!("../model/manifest.tsv");
        let vocab = include_str!("../model/vocab.tsv");
        let merges = include_str!("../model/petite_merges.txt");
        GaryModel { cortex: Cortex::load(bin, man), bpe: Bpe::load(vocab, merges) }
    }

    pub fn think(&self, facts: &[String], query: &str, max_new: usize) -> String {
        let mut prompt = String::new();
        for f in facts { prompt.push_str(&format!("U: {}\nG: noted.\n", f)); }
        prompt.push_str(&format!("U: {}\nG:", query));
        let ids: Vec<usize> = self.bpe.encode(&prompt).iter().map(|&x| x as usize).collect();
        let blk = self.cortex.cfg.blk;
        // the positional table has `blk` slots; if the prompt is longer keep its last blk tokens
        // (matches the old sliding window). Prefill once, then decode one token at a time.
        let start: &[usize] = if ids.len() > blk { &ids[ids.len()-blk..] } else { &ids[..] };
        let mut cache = Kv::new(self.cortex.cfg.l);
        let mut lg = self.cortex.forward(start, &mut cache);
        let mut out: Vec<u32> = Vec::new();
        for _ in 0..max_new {
            let mut best = 0usize; let mut bv = f32::NEG_INFINITY;
            for v in 1..lg.len() { if lg[v] > bv { bv = lg[v]; best = v; } }
            if best == 0 { break; }
            let piece = self.bpe.decode(&[best as u32]);
            if piece.contains('\n') { break; }
            out.push(best as u32);
            if cache.len() >= blk { break; }   // positional table exhausted
            lg = self.cortex.forward(&[best], &mut cache);
        }
        self.bpe.decode(&out).trim().to_string()
    }

    /// Run the cortex as the middle dispatcher: generate the action and parse it into a route.
    /// `facts` is the working set neuron-db recalled; `query` is the turn. Unknown/!-prefixed
    /// output defaults to Escalate (fail safe: hand up to the host model rather than guess).
    pub fn dispatch(&self, facts: &[String], query: &str) -> Dispatch {
        let raw = self.think(facts, query, 24);
        let raw = raw.trim();
        if let Some(v) = raw.strip_prefix("ANSWER ") { Dispatch::Answer(v.trim().to_string()) }
        else if let Some(v) = raw.strip_prefix("FETCH ") { Dispatch::Fetch(v.trim().to_string()) }
        else if let Some(v) = raw.strip_prefix("STORE ") { Dispatch::Store(v.trim().to_string()) }
        else { Dispatch::Escalate }
    }

    pub fn encode(&self, t: &str) -> Vec<u32> { self.bpe.encode(t) }
}
