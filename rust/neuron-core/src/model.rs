//! GaryModel — the emergence cortex + tokenizer, BUNDLED in the binary (include_bytes!).
//! Give it a working set (facts the store retrieved) and a question; it generates the answer.
use crate::cortex::Cortex;
use crate::bpe::Bpe;

pub struct GaryModel { cortex: Cortex, bpe: Bpe }

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
        let mut ids: Vec<usize> = self.bpe.encode(&prompt).iter().map(|&x| x as usize).collect();
        let blk = self.cortex.cfg.blk;
        let mut out: Vec<u32> = Vec::new();
        for _ in 0..max_new {
            let win: &[usize] = if ids.len() > blk { &ids[ids.len()-blk..] } else { &ids[..] };
            let lg = self.cortex.forward_last(win);
            let mut best = 0usize; let mut bv = f32::NEG_INFINITY;
            for v in 1..lg.len() { if lg[v] > bv { bv = lg[v]; best = v; } }
            if best == 0 { break; }
            let piece = self.bpe.decode(&[best as u32]);
            if piece.contains('\n') { break; }
            ids.push(best); out.push(best as u32);
        }
        self.bpe.decode(&out).trim().to_string()
    }

    pub fn encode(&self, t: &str) -> Vec<u32> { self.bpe.encode(t) }
}
