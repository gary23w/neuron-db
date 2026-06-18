// Hand-written numeric kernels below index parallel weight/activation arrays by position; the
// index-based loops mirror the reference math and read clearer than enumerate/zip chains.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! Pure-Rust forward pass for the gary-neuron cortex. std-only, runs natively and in wasm.
//! Mirrors the reference math (numpy/TS port matched to 0.0000 MSE).
//!
//! Two perf decisions matter for the (browser-)wasm hot path:
//!   * weights are resolved into directly-indexed `Block` slices ONCE at load — the forward
//!     loop does no `format!()`/HashMap lookups per layer per token.
//!   * generation is INCREMENTAL via a per-layer K/V cache (`Kv`): the prompt is encoded once,
//!     then each new token is a single-token step instead of re-forwarding the whole window.
//! The arithmetic is byte-identical to the old full-window forward, so outputs are unchanged.
use std::collections::HashMap;

const C: f32 = 0.797_884_6; const A: f32 = 0.044_715;
fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + (C * (x + A * x * x * x)).tanh()) }

pub struct Cfg { pub e: usize, pub h: usize, pub l: usize, pub blk: usize, pub vocab: usize }

/// One transformer block's weights, resolved to flat slices at load (no per-call lookups).
struct Block {
    ln1_w: Vec<f32>, ln1_b: Vec<f32>,
    in_w: Vec<f32>, in_b: Vec<f32>,        // attn.in_proj  [3e, e]
    out_w: Vec<f32>, out_b: Vec<f32>,      // attn.out_proj [e, e]
    ln2_w: Vec<f32>, ln2_b: Vec<f32>,
    mlp0_w: Vec<f32>, mlp0_b: Vec<f32>,    // mlp.0 [4e, e]
    mlp2_w: Vec<f32>, mlp2_b: Vec<f32>,    // mlp.2 [e, 4e]
}

pub struct Cortex {
    pub cfg: Cfg,
    tok: Vec<f32>, pos: Vec<f32>,
    lnf_w: Vec<f32>, lnf_b: Vec<f32>,
    blocks: Vec<Block>,
}

/// Per-layer key/value cache for incremental decoding. `k[li]`/`v[li]` hold `len` positions,
/// each a full e-dim vector laid out as `[pos*e + dim]`.
pub struct Kv { k: Vec<Vec<f32>>, v: Vec<Vec<f32>>, len: usize }
impl Kv {
    pub fn new(layers: usize) -> Kv { Kv { k: vec![Vec::new(); layers], v: vec![Vec::new(); layers], len: 0 } }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl Cortex {
    /// bin = cortex.bin ; manifest = manifest.tsv (first line "#cfg\tE\tH\tL\tBLK\tVOCAB")
    pub fn load(bin: &[u8], manifest: &str) -> Cortex {
        let mut cfg = Cfg { e: 96, h: 4, l: 8, blk: 384, vocab: 2048 };
        let mut t: HashMap<String, Vec<f32>> = HashMap::new();
        for line in manifest.lines() {
            if line.starts_with("#cfg") {
                let p: Vec<&str> = line.split('\t').collect();
                cfg = Cfg { e: p[1].parse().unwrap(), h: p[2].parse().unwrap(), l: p[3].parse().unwrap(),
                            blk: p[4].parse().unwrap(), vocab: p[5].parse().unwrap() };
                continue;
            }
            if line.is_empty() { continue; }
            let p: Vec<&str> = line.split('\t').collect();
            let (name, qtype, off, rows, cols) =
                (p[0], p[1], p[2].parse::<usize>().unwrap(), p[4].parse::<usize>().unwrap(), p[5].parse::<usize>().unwrap());
            let mut data = vec![0f32; rows * cols];
            if qtype == "i8" {
                let mut o = off;
                let mut scale = vec![0f32; rows];
                for r in 0..rows { scale[r] = f32::from_le_bytes([bin[o], bin[o+1], bin[o+2], bin[o+3]]); o += 4; }
                for r in 0..rows { let s = scale[r]; for c in 0..cols { data[r*cols+c] = (bin[o] as i8) as f32 * s; o += 1; } }
            } else {
                let mut o = off;
                for i in 0..rows*cols { data[i] = f32::from_le_bytes([bin[o], bin[o+1], bin[o+2], bin[o+3]]); o += 4; }
            }
            t.insert(name.to_string(), data);
        }
        // resolve every tensor to a directly-held slice ONCE (kills format!()/HashMap in forward)
        let mut take = |name: String| t.remove(&name).unwrap_or_else(|| panic!("manifest missing tensor {name}"));
        let blocks = (0..cfg.l).map(|li| {
            let p = format!("blocks.{li}.");
            Block {
                ln1_w: take(format!("{p}ln1.weight")), ln1_b: take(format!("{p}ln1.bias")),
                in_w: take(format!("{p}attn.in_proj_weight")), in_b: take(format!("{p}attn.in_proj_bias")),
                out_w: take(format!("{p}attn.out_proj.weight")), out_b: take(format!("{p}attn.out_proj.bias")),
                ln2_w: take(format!("{p}ln2.weight")), ln2_b: take(format!("{p}ln2.bias")),
                mlp0_w: take(format!("{p}mlp.0.weight")), mlp0_b: take(format!("{p}mlp.0.bias")),
                mlp2_w: take(format!("{p}mlp.2.weight")), mlp2_b: take(format!("{p}mlp.2.bias")),
            }
        }).collect();
        let (tok, pos) = (take("tok.weight".into()), take("pos.weight".into()));
        let (lnf_w, lnf_b) = (take("lnf.weight".into()), take("lnf.bias".into()));
        Cortex { cfg, tok, pos, lnf_w, lnf_b, blocks }
    }

    /// LayerNorm of a single e-vector into `out`.
    fn ln(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
        let e = x.len();
        let mu: f32 = x.iter().sum::<f32>() / e as f32;
        let var: f32 = x.iter().map(|v| (v-mu)*(v-mu)).sum::<f32>() / e as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for j in 0..e { out[j] = (x[j]-mu)*inv*w[j] + b[j]; }
    }
    /// Dot product of two equal-length slices, eight independent lane accumulators so LLVM emits
    /// SIMD (f32x4/f32x8 native, +simd128 wasm — see .cargo/config.toml). This REORDERS the float
    /// sum vs a strict left-to-right accumulate: numerically equivalent (pairwise) but not
    /// bit-identical, so greedy decode is re-verified (cortex_battery) whenever this changes.
    #[inline]
    fn dot(x: &[f32], w: &[f32]) -> f32 {
        let mut acc = [0f32; 8];
        let mut xi = x.chunks_exact(8);
        let mut wi = w.chunks_exact(8);
        for (xc, wc) in xi.by_ref().zip(wi.by_ref()) {
            let xc: &[f32; 8] = xc.try_into().unwrap();
            let wc: &[f32; 8] = wc.try_into().unwrap();
            for l in 0..8 { acc[l] += xc[l] * wc[l]; }
        }
        let mut s = ((acc[0]+acc[1]) + (acc[2]+acc[3])) + ((acc[4]+acc[5]) + (acc[6]+acc[7]));
        for (a, b) in xi.remainder().iter().zip(wi.remainder()) { s += a * b; }
        s
    }
    /// out[o] = b[o] + sum_i x[i]*w[o*i_dim + i]  (single token).
    fn linear(x: &[f32], i_dim: usize, o_dim: usize, w: &[f32], b: &[f32], out: &mut [f32]) {
        let x = &x[..i_dim];
        for o in 0..o_dim { out[o] = b[o] + Self::dot(x, &w[o*i_dim..o*i_dim + i_dim]); }
    }

    /// Encode `ids` incrementally on top of `cache` (positions continue from `cache.len`),
    /// appending each token's per-layer K/V. Returns logits for the LAST token in `ids`.
    /// Same arithmetic as a full-window forward over [prior cache ++ ids], last position.
    pub fn forward(&self, ids: &[usize], cache: &mut Kv) -> Vec<f32> {
        let (e, h, l, vocab) = (self.cfg.e, self.cfg.h, self.cfg.l, self.cfg.vocab);
        let hd = e / h; let sca = 1.0 / (hd as f32).sqrt();
        let mut h1 = vec![0f32; e]; let mut qkv = vec![0f32; 3*e];
        let mut attn = vec![0f32; e]; let mut o = vec![0f32; e];
        let mut h2 = vec![0f32; e]; let mut m0 = vec![0f32; 4*e]; let mut m2 = vec![0f32; e];
        let mut x = vec![0f32; e];
        let mut logits = vec![0f32; vocab];
        let mut sc = vec![0f32; self.cfg.blk.max(8)];   // attention scores, hoisted out of the per-head/token loop
        let last = ids.len() - 1;
        for (n, &id) in ids.iter().enumerate() {
            let posi = cache.len;                       // absolute position of this token
            for j in 0..e { x[j] = self.tok[id*e + j] + self.pos[posi*e + j]; }
            for li in 0..l {
                let bl = &self.blocks[li];
                Self::ln(&x, &bl.ln1_w, &bl.ln1_b, &mut h1);
                Self::linear(&h1, e, 3*e, &bl.in_w, &bl.in_b, &mut qkv);
                cache.k[li].extend_from_slice(&qkv[e..2*e]);     // append this token's K (full e)
                cache.v[li].extend_from_slice(&qkv[2*e..3*e]);   // and V
                let np = posi + 1;                                // positions to attend over (incl. self)
                let (kc, vc) = (&cache.k[li], &cache.v[li]);
                for head in 0..h {
                    let qo = head*hd; let q = &qkv[qo..qo+hd];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..np {
                        let kb = j*e + head*hd;
                        let d = Self::dot(q, &kc[kb..kb+hd]) * sca;
                        sc[j] = d; if d > mx { mx = d; }
                    }
                    let mut sum = 0f32;
                    for j in 0..np { sc[j] = (sc[j]-mx).exp(); sum += sc[j]; }
                    let ob = head*hd;
                    for c in 0..hd { attn[ob+c] = 0.0; }
                    for j in 0..np { let w = sc[j]/sum; let vb = j*e + head*hd; for c in 0..hd { attn[ob+c] += w*vc[vb+c]; } }
                }
                Self::linear(&attn, e, e, &bl.out_w, &bl.out_b, &mut o);
                for j in 0..e { x[j] += o[j]; }
                Self::ln(&x, &bl.ln2_w, &bl.ln2_b, &mut h2);
                Self::linear(&h2, e, 4*e, &bl.mlp0_w, &bl.mlp0_b, &mut m0);
                for v in m0.iter_mut() { *v = gelu(*v); }
                Self::linear(&m0, 4*e, e, &bl.mlp2_w, &bl.mlp2_b, &mut m2);
                for j in 0..e { x[j] += m2[j]; }
            }
            cache.len += 1;
            if n == last {
                let mut xf = vec![0f32; e];
                Self::ln(&x, &self.lnf_w, &self.lnf_b, &mut xf);
                for v in 0..vocab { logits[v] = Self::dot(&xf, &self.tok[v*e..v*e + e]); }
            }
        }
        logits
    }
}
