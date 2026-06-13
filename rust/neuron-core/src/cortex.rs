//! Pure-Rust forward pass for the gary-neuron cortex (the emergence model, 8L/E96/384ctx).
//! Mirrors the TypeScript port (which matched numpy to 0.0000 MSE). std-only.
//! Weights load from the int8 blob + flat manifest bundled in model/.
use std::collections::HashMap;

const C: f32 = 0.797_884_56; const A: f32 = 0.044_715;
fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + (C * (x + A * x * x * x)).tanh()) }

pub struct Cfg { pub e: usize, pub h: usize, pub l: usize, pub blk: usize, pub vocab: usize }
pub struct Cortex { pub cfg: Cfg, t: HashMap<String, Vec<f32>> }

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
        Cortex { cfg, t }
    }

    fn ln(&self, x: &[f32], n: usize, e: usize, w: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..n {
            let base = i * e;
            let mu: f32 = x[base..base+e].iter().sum::<f32>() / e as f32;
            let var: f32 = x[base..base+e].iter().map(|v| (v-mu)*(v-mu)).sum::<f32>() / e as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for j in 0..e { out[base+j] = (x[base+j]-mu)*inv*w[j] + b[j]; }
        }
    }
    fn linear(&self, x: &[f32], n: usize, i_dim: usize, o_dim: usize, w: &[f32], b: &[f32], out: &mut [f32]) {
        for ti in 0..n {
            let xb = ti*i_dim; let ob = ti*o_dim;
            for o in 0..o_dim {
                let mut acc = b[o]; let wb = o*i_dim;
                for i in 0..i_dim { acc += x[xb+i]*w[wb+i]; }
                out[ob+o] = acc;
            }
        }
    }

    /// logits for the LAST position only
    pub fn forward_last(&self, ids: &[usize]) -> Vec<f32> {
        let (e, h, l, vocab) = (self.cfg.e, self.cfg.h, self.cfg.l, self.cfg.vocab);
        let hd = e / h; let tt = ids.len();
        let tok = &self.t["tok.weight"]; let pos = &self.t["pos.weight"];
        let mut x = vec![0f32; tt*e];
        for i in 0..tt { for j in 0..e { x[i*e+j] = tok[ids[i]*e+j] + pos[i*e+j]; } }
        let mut h1 = vec![0f32; tt*e]; let mut qkv = vec![0f32; tt*3*e];
        let mut o = vec![0f32; tt*e]; let mut ao = vec![0f32; tt*e];
        let mut h2 = vec![0f32; tt*e]; let mut m0 = vec![0f32; tt*4*e]; let mut m2 = vec![0f32; tt*e];
        for li in 0..l {
            let p = format!("blocks.{}.", li);
            self.ln(&x, tt, e, &self.t[&format!("{}ln1.weight",p)], &self.t[&format!("{}ln1.bias",p)], &mut h1);
            self.linear(&h1, tt, e, 3*e, &self.t[&format!("{}attn.in_proj_weight",p)], &self.t[&format!("{}attn.in_proj_bias",p)], &mut qkv);
            for v in o.iter_mut() { *v = 0.0; }
            let sca = 1.0 / (hd as f32).sqrt();
            for head in 0..h {
                let (qo, ko, vo) = (head*hd, e + head*hd, 2*e + head*hd);
                for ti in 0..tt {
                    let mut sc = vec![0f32; ti+1]; let mut mx = f32::NEG_INFINITY;
                    for j in 0..=ti {
                        let mut d = 0f32; let qb = ti*3*e+qo; let kb = j*3*e+ko;
                        for c in 0..hd { d += qkv[qb+c]*qkv[kb+c]; }
                        d *= sca; sc[j] = d; if d > mx { mx = d; }
                    }
                    let mut sum = 0f32;
                    for j in 0..=ti { sc[j] = (sc[j]-mx).exp(); sum += sc[j]; }
                    let ob = ti*e + head*hd;
                    for j in 0..=ti { let w = sc[j]/sum; let vb = j*3*e+vo; for c in 0..hd { o[ob+c] += w*qkv[vb+c]; } }
                }
            }
            self.linear(&o, tt, e, e, &self.t[&format!("{}attn.out_proj.weight",p)], &self.t[&format!("{}attn.out_proj.bias",p)], &mut ao);
            for i in 0..tt*e { x[i] += ao[i]; }
            self.ln(&x, tt, e, &self.t[&format!("{}ln2.weight",p)], &self.t[&format!("{}ln2.bias",p)], &mut h2);
            self.linear(&h2, tt, e, 4*e, &self.t[&format!("{}mlp.0.weight",p)], &self.t[&format!("{}mlp.0.bias",p)], &mut m0);
            for v in m0.iter_mut() { *v = gelu(*v); }
            self.linear(&m0, tt, 4*e, e, &self.t[&format!("{}mlp.2.weight",p)], &self.t[&format!("{}mlp.2.bias",p)], &mut m2);
            for i in 0..tt*e { x[i] += m2[i]; }
        }
        let mut last = vec![0f32; e]; let mut xf = vec![0f32; e];
        for j in 0..e { last[j] = x[(tt-1)*e+j]; }
        self.ln(&last, 1, e, &self.t["lnf.weight"], &self.t["lnf.bias"], &mut xf);
        let mut logits = vec![0f32; vocab];
        for v in 0..vocab { let mut acc = 0f32; let wb = v*e; for c in 0..e { acc += xf[c]*tok[wb+c]; } logits[v] = acc; }
        logits
    }
}
