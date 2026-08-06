//! fisher.rs — two-class Fisher linear discriminant heads over plain feature vectors.
//!
//! The classical closed form, kept deliberately two-class: one linear solve, no eigenproblem,
//! no linalg crate. `w ∝ S_λ⁻¹(μ₊ − μ₋)` with shrinkage toward the scaled identity, threshold
//! `c = w·(μ₊+μ₋)/2 + ln(n₋/n₊)`, and scores reported in within-class sigma units (the axis is
//! normalized so `wᵀ S_λ w = 1`). Shrinkage is what makes the head safe in a 256-d space where a
//! class may hold eight samples: `S_λ` is positive-definite by construction, so the solve always
//! succeeds and a data-starved head degrades toward a nearest-mean classifier instead of exploding.
//!
//! State is class-agnostic and O(d²) ONCE, not per class: the head keeps one global second-moment
//! matrix (packed upper triangle, f64) plus a first moment per class (open taxonomy, exactly like
//! the trust ledger — nothing is privileged a priori). The within-class scatter for ANY class pair
//! materializes by subtraction, so one head answers every pairing (helped-vs-hurt, scope-vs-rest,
//! tag-vs-tag) without a matrix per pair. `S_w` is pooled across ALL observed classes — the
//! common-covariance assumption of classical LDA over the whole population — which both shares
//! statistical strength and keeps the state one matrix.
//!
//! The learning-rule guardrails mirror trust.rs: the head is INERT (`axis` returns None,
//! contributes nothing) until both classes clear a sample floor; scores are clamped at the read
//! (the STRENGTH_CAP idiom); every accumulator decays by an exponential forgetting factor so the
//! axis tracks drift and cannot lock in; and the head only ever RANKS — it never gates truth,
//! never mints a fact, and a build that never feeds it is byte-identical in behavior.
//!
//! Deterministic by construction: closed-form, no RNG anywhere. Std-only; feature `fisher`.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

/// A class is silent until it has this much effective (decayed) weight behind it.
const N_MIN: f64 = 8.0;
/// Scores are clamped here at the read, so one extreme fact modulates rank, never dominates it.
pub const Z_CAP: f32 = 4.0;
/// Exponential forgetting timescale, in labeled observations: each new sample multiplies the
/// weight of everything before it by (1 - 1/FORGET). Old evidence fades; the axis stays movable.
const FORGET: f64 = 4096.0;
/// Renormalize the raw accumulators when the running weight boost exceeds this (the boost trick
/// makes each update O(1) extra instead of decaying every accumulator per observation).
const REBASE_AT: f64 = 1e30;

/// A computed discriminant axis: direction `w` (unit within-class variance), threshold `c`
/// (midpoint + log prior odds), and the effective per-class weights behind it.
#[derive(Clone, Debug)]
pub struct Axis {
    pub w: Vec<f32>,
    pub c: f32,
    pub n_pos: f64,
    pub n_neg: f64,
}

impl Axis {
    /// Score a vector along the axis: z > 0 leans `pos`, z < 0 leans `neg`, in within-class sigma
    /// units, clamped to ±Z_CAP so the signal is bounded rank evidence, never a runaway weight.
    pub fn score(&self, x: &[f32]) -> f32 {
        let mut z = 0f64;
        let n = self.w.len().min(x.len());
        for i in 0..n { z += self.w[i] as f64 * x[i] as f64; }
        ((z as f32) - self.c).clamp(-Z_CAP, Z_CAP)
    }
}

#[derive(Clone, Debug, Default)]
struct ClassMoment { sum: Vec<f64>, n: f64 }

/// The head: one packed second moment + per-class first moments, with exponential forgetting.
/// All accumulators share one raw/boost representation (newest sample carries weight 1 after
/// dividing by `boost`), so means and scatters stay mutually consistent under decay.
pub struct FisherHead {
    dim: usize,
    m2: Vec<f64>,                          // packed upper triangle of Σ w_i x xᵀ
    n: f64,                                // Σ w_i (raw)
    boost: f64,                            // current sample weight (γ^-t); effective n = n/boost
    updates: u64,                          // labeled observations ever seen (the cache epoch)
    classes: HashMap<String, ClassMoment>, // open taxonomy; a class exists once it is observed
    cache: HashMap<(String, String), (Axis, u64)>, // (pos,neg) -> axis + updates-at-compute
}

#[inline]
fn tri(d: usize, i: usize, j: usize) -> usize {
    // packed upper triangle (i <= j): row i starts after i rows of shrinking length. The
    // subtraction-free form of i*d - i(i-1)/2 (which underflows usize at i = 0 in debug).
    debug_assert!(i <= j && j < d);
    i * (2 * d + 1 - i) / 2 + (j - i)
}

impl FisherHead {
    pub fn new(dim: usize) -> Self {
        FisherHead {
            dim,
            m2: vec![0.0; dim * (dim + 1) / 2],
            n: 0.0,
            boost: 1.0,
            updates: 0,
            classes: HashMap::new(),
            cache: HashMap::new(),
        }
    }
    pub fn dim(&self) -> usize { self.dim }
    pub fn updates(&self) -> u64 { self.updates }
    /// Effective (decayed) sample weight behind `class`; 0.0 for a class never observed.
    pub fn class_n(&self, class: &str) -> f64 {
        self.classes.get(class).map(|c| c.n / self.boost).unwrap_or(0.0)
    }
    /// The observed classes with their effective weights (for inspection; order unspecified).
    pub fn classes(&self) -> Vec<(String, f64)> {
        self.classes.iter().map(|(k, v)| (k.clone(), v.n / self.boost)).collect()
    }

    /// Fold one labeled vector in. O(d²/2) for the rank-1 update; the exponential forgetting is
    /// O(1) via the boost trick (new samples enter with growing raw weight instead of every old
    /// accumulator being decayed per observation — identical math, rebase keeps it finite).
    pub fn observe_labeled(&mut self, class: &str, x: &[f32]) {
        if x.len() != self.dim || class.is_empty() { return; }
        self.boost /= 1.0 - 1.0 / FORGET;
        let w = self.boost;
        let d = self.dim;
        for i in 0..d {
            let wi = w * x[i] as f64;
            let base = tri(d, i, i);
            for j in i..d { self.m2[base + (j - i)] += wi * x[j] as f64; }
        }
        let e = self.classes.entry(class.to_string()).or_insert_with(|| ClassMoment { sum: vec![0.0; d], n: 0.0 });
        for i in 0..d { e.sum[i] += w * x[i] as f64; }
        e.n += w;
        self.n += w;
        self.updates += 1;
        if self.boost > REBASE_AT { self.rebase(); }
    }

    /// Divide every raw accumulator by the current boost so the newest sample weighs 1 again.
    /// Pure re-representation: every effective quantity (means, scatters, counts) is unchanged.
    fn rebase(&mut self) {
        let b = self.boost;
        for v in self.m2.iter_mut() { *v /= b; }
        for c in self.classes.values_mut() { c.n /= b; for v in c.sum.iter_mut() { *v /= b; } }
        self.n /= b;
        self.boost = 1.0;
    }

    /// The two-class axis separating `pos` from `neg`, or None while either class is below the
    /// sample floor (an inert head contributes nothing — the trust-ledger posture). Cached per
    /// pair and lazily recomputed once the head has seen twice the updates it was computed at
    /// (the emb_cache drift-bound idiom), so recall never pays the solve on its hot path.
    pub fn axis(&mut self, pos: &str, neg: &str) -> Option<Axis> {
        let key = (pos.to_string(), neg.to_string());
        if let Some((ax, at)) = self.cache.get(&key) {
            if self.updates < at.saturating_mul(2).max(at + 16) { return Some(ax.clone()); }
        }
        let ax = self.solve(pos, neg)?;
        self.cache.insert(key, (ax.clone(), self.updates));
        Some(ax)
    }

    fn solve(&self, pos: &str, neg: &str) -> Option<Axis> {
        let d = self.dim;
        let (p, q) = (self.classes.get(pos)?, self.classes.get(neg)?);
        let (np, nq) = (p.n / self.boost, q.n / self.boost);
        if np < N_MIN || nq < N_MIN { return None; }
        let mu_p: Vec<f64> = p.sum.iter().map(|s| s / p.n).collect();
        let mu_q: Vec<f64> = q.sum.iter().map(|s| s / q.n).collect();

        // pooled within-class scatter over ALL classes: S_w = M2 - Σ_c n_c μ_c μ_cᵀ. A nonneg-
        // weighted sum of centered outer products, so PSD in exact arithmetic; shrinkage below
        // makes it strictly PD regardless of sample count or f64 roundoff.
        let mut s = self.m2.clone();
        for c in self.classes.values() {
            if c.n <= 0.0 { continue; }
            for i in 0..d {
                let mi = c.sum[i] / c.n;
                let base = tri(d, i, i);
                for j in i..d { s[base + (j - i)] -= c.n * mi * (c.sum[j] / c.n); }
            }
        }
        // raw accumulators share one scale, so S_w/raw_n is the proper weighted covariance (the
        // boost cancels); only the shrinkage strength needs the EFFECTIVE sample count.
        let nw = self.n.max(1.0);
        let n_eff = (self.n / self.boost).max(1.0);
        let mut trace = 0.0;
        for i in 0..d { trace += s[tri(d, i, i)] / nw; }
        let mtr = trace / d as f64;
        if !(mtr > 0.0) || !mtr.is_finite() { return None; }   // degenerate: no spread observed yet

        // shrinkage toward the scaled identity: heavier when data is thin relative to d, floored
        // so the solve stays comfortably PD even under adversarial roundoff.
        let lam = (d as f64 / (d as f64 + n_eff)).clamp(0.05, 0.95);
        // S_λ as a full row-major matrix for the Cholesky (transient, ~d² f64)
        let mut a = vec![0.0f64; d * d];
        for i in 0..d {
            for j in i..d {
                let v = (1.0 - lam) * (s[tri(d, i, j)] / nw) + if i == j { lam * mtr } else { 0.0 };
                a[i * d + j] = v;
                a[j * d + i] = v;
            }
        }
        // Cholesky a = L Lᵀ (lower), then solve L y = Δμ, Lᵀ w = y
        let mut l = vec![0.0f64; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut sum = a[i * d + j];
                for k2 in 0..j { sum -= l[i * d + k2] * l[j * d + k2]; }
                if i == j {
                    if sum <= 0.0 { return None; }   // shrinkage should prevent this; abstain if not
                    l[i * d + j] = sum.sqrt();
                } else {
                    l[i * d + j] = sum / l[j * d + j];
                }
            }
        }
        let dmu: Vec<f64> = (0..d).map(|i| mu_p[i] - mu_q[i]).collect();
        let mut y = vec![0.0f64; d];
        for i in 0..d {
            let mut sum = dmu[i];
            for k2 in 0..i { sum -= l[i * d + k2] * y[k2]; }
            y[i] = sum / l[i * d + i];
        }
        let mut w = vec![0.0f64; d];
        for i in (0..d).rev() {
            let mut sum = y[i];
            for k2 in (i + 1)..d { sum -= l[k2 * d + i] * w[k2]; }
            w[i] = sum / l[i * d + i];
        }
        // wᵀ S_λ w = wᵀ Δμ (because S_λ w = Δμ): normalize so scores land in sigma units.
        let quad: f64 = (0..d).map(|i| w[i] * dmu[i]).sum();
        if !(quad > 1e-12) || !quad.is_finite() { return None; }   // means indistinguishable: abstain
        let inv = 1.0 / quad.sqrt();
        let wf: Vec<f32> = w.iter().map(|v| (v * inv) as f32).collect();
        let mut c = 0.0f64;
        for i in 0..d { c += w[i] * inv * (mu_p[i] + mu_q[i]) * 0.5; }
        c += (nq / np).ln();
        Some(Axis { w: wf, c: c as f32, n_pos: np, n_neg: nq })
    }

    /// Persistence in the store's tab-line convention. Rebases first so the dump is canonical
    /// (boost 1.0); f64 Display is shortest-round-trippable, so reload is exact.
    pub fn dump(&mut self) -> String {
        use std::fmt::Write as _;
        self.rebase();
        let mut out = String::new();
        let _ = write!(out, "dim\t{}\nn\t{}\nupdates\t{}\nm2\t", self.dim, self.n, self.updates);
        for (i, v) in self.m2.iter().enumerate() {
            if i > 0 { out.push(' '); }
            let _ = write!(out, "{}", v);
        }
        let mut names: Vec<&String> = self.classes.keys().collect();
        names.sort();   // deterministic dump order
        for name in names {
            let c = &self.classes[name];
            let key = name.replace(['\t', '\n'], " ");
            let _ = write!(out, "\nclass\t{}\t{}\t", key, c.n);
            for (i, v) in c.sum.iter().enumerate() {
                if i > 0 { out.push(' '); }
                let _ = write!(out, "{}", v);
            }
        }
        out
    }

    pub fn load(blob: &str) -> Option<Self> {
        let mut dim = 0usize;
        let mut n = 0f64;
        let mut updates = 0u64;
        let mut m2: Vec<f64> = Vec::new();
        let mut classes: HashMap<String, ClassMoment> = HashMap::new();
        for line in blob.split('\n') {
            let mut it = line.splitn(2, '\t');
            match (it.next(), it.next()) {
                (Some("dim"), Some(v)) => dim = v.parse().ok()?,
                (Some("n"), Some(v)) => n = v.parse().ok()?,
                (Some("updates"), Some(v)) => updates = v.parse().ok()?,
                (Some("m2"), Some(v)) => m2 = v.split(' ').filter_map(|x| x.parse().ok()).collect(),
                (Some("class"), Some(rest)) => {
                    let mut f = rest.splitn(3, '\t');
                    let name = f.next()?.to_string();
                    let cn: f64 = f.next()?.parse().ok()?;
                    let sum: Vec<f64> = f.next()?.split(' ').filter_map(|x| x.parse().ok()).collect();
                    classes.insert(name, ClassMoment { sum, n: cn });
                }
                _ => {}
            }
        }
        if dim == 0 || m2.len() != dim * (dim + 1) / 2 { return None; }
        if classes.values().any(|c| c.sum.len() != dim) { return None; }
        Some(FisherHead { dim, m2, n, boost: 1.0, updates, classes, cache: HashMap::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // deterministic gaussian sampler: splitmix64 uniforms through Box-Muller. No rand crate.
    fn splitmix(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 { self.0 = splitmix(self.0); ((self.0 >> 11) as f64 / (1u64 << 53) as f64).max(1e-16) }
        fn gauss(&mut self) -> f64 {
            let (u1, u2) = (self.uniform(), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }
    /// class sample: x = μ + σ ⊙ g with per-dim σ (a known diagonal covariance)
    fn sample(rng: &mut Rng, mu: &[f32], sig: &[f32]) -> Vec<f32> {
        mu.iter().zip(sig).map(|(m, s)| m + s * rng.gauss() as f32).collect()
    }
    fn cosine(a: &[f32], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * y).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();
        dot / (na * nb).max(1e-12)
    }

    // Recovers the analytic two-class direction Σ⁻¹Δμ on anisotropic gaussians — the axis must
    // find the LOW-VARIANCE separating direction, which raw Δμ or cosine ranking would miss.
    #[test]
    fn recovers_two_gaussian_axis() {
        let d = 16;
        // μ differs on dims 0 and 3; dim 0 is high-noise, dim 3 low-noise -> the discriminant
        // must weight dim 3 far above dim 0 (Σ⁻¹ does; Δμ alone does not).
        let mut mu_p = vec![0.0f32; d]; mu_p[0] = 1.0; mu_p[3] = 0.6;
        let mut mu_n = vec![0.0f32; d]; mu_n[0] = -1.0; mu_n[3] = -0.6;
        let mut sig = vec![0.7f32; d]; sig[0] = 3.0; sig[3] = 0.25;
        let mut h = FisherHead::new(d);
        let mut rng = Rng(7);
        for _ in 0..400 {
            h.observe_labeled("+", &sample(&mut rng, &mu_p, &sig));
            h.observe_labeled("-", &sample(&mut rng, &mu_n, &sig));
        }
        let ax = h.axis("+", "-").expect("well-fed head must produce an axis");
        // analytic direction: Σ⁻¹Δμ, diagonal Σ -> Δμ_i / σ_i²
        let truth: Vec<f64> = (0..d).map(|i| (mu_p[i] - mu_n[i]) as f64 / (sig[i] as f64).powi(2)).collect();
        let cos = cosine(&ax.w, &truth);
        assert!(cos > 0.95, "axis should align with Σ⁻¹Δμ, cosine {cos}");
        // and it must classify held-out samples
        let mut ok = 0;
        for _ in 0..200 {
            if ax.score(&sample(&mut rng, &mu_p, &sig)) > 0.0 { ok += 1; }
            if ax.score(&sample(&mut rng, &mu_n, &sig)) < 0.0 { ok += 1; }
        }
        assert!(ok >= 380, "held-out accuracy too low: {ok}/400");
        // sigma-unit sanity: class means should sit a few sigma apart, not thousands
        let zp = ax.score(&mu_p);
        assert!(zp > 0.5 && zp <= Z_CAP, "mean score out of sigma range: {zp}");
    }

    // Ten samples per class in 256-d: shrinkage must keep the solve PD and the scores finite.
    // (Ten, not eight: exponential forgetting leaves 8 nominal samples a hair under the 8.0 floor.)
    #[test]
    fn shrinkage_survives_n_below_d() {
        let d = 256;
        let mut h = FisherHead::new(d);
        let mut rng = Rng(42);
        let mut mu_p = vec![0.0f32; d]; mu_p[5] = 1.0;
        let mut mu_n = vec![0.0f32; d]; mu_n[5] = -1.0;
        let sig = vec![1.0f32; d];
        for _ in 0..10 {
            h.observe_labeled("+", &sample(&mut rng, &mu_p, &sig));
            h.observe_labeled("-", &sample(&mut rng, &mu_n, &sig));
        }
        let ax = h.axis("+", "-").expect("shrinkage must make 8 samples solvable at d=256");
        let z = ax.score(&mu_p);
        assert!(z.is_finite() && z.abs() <= Z_CAP);
        assert!(ax.w.iter().all(|v| v.is_finite()));
    }

    // Below the sample floor the head is inert — it must contribute nothing, not guess.
    #[test]
    fn inert_below_sample_floor() {
        let mut h = FisherHead::new(8);
        let mut rng = Rng(3);
        for _ in 0..7 {
            h.observe_labeled("+", &sample(&mut rng, &[1.0; 8], &[0.5; 8]));
            h.observe_labeled("-", &sample(&mut rng, &[-1.0; 8], &[0.5; 8]));
        }
        assert!(h.axis("+", "-").is_none(), "7 < N_MIN samples must stay inert");
        assert!(h.axis("+", "missing").is_none(), "an unobserved class must stay inert");
    }

    // The trust_is_relearnable mirror: when the world flips, forgetting lets the axis flip too.
    #[test]
    fn relearns_a_flipped_world() {
        let d = 8;
        let mut h = FisherHead::new(d);
        let mut rng = Rng(11);
        let (mu_a, mu_b, sig) = ([1.0f32; 8], [-1.0f32; 8], [0.6f32; 8]);
        for _ in 0..200 {
            h.observe_labeled("+", &sample(&mut rng, &mu_a, &sig));
            h.observe_labeled("-", &sample(&mut rng, &mu_b, &sig));
        }
        let before = h.axis("+", "-").unwrap();
        assert!(before.score(&mu_a) > 0.0);
        // the world flips: what was "+" now looks like mu_b and vice versa. Feed enough that the
        // exponentially-forgotten old evidence loses to the new.
        for _ in 0..4000 {
            h.observe_labeled("+", &sample(&mut rng, &mu_b, &sig));
            h.observe_labeled("-", &sample(&mut rng, &mu_a, &sig));
        }
        let after = h.axis("+", "-").unwrap();
        assert!(after.score(&mu_b) > 0.0, "axis must follow the flip: {}", after.score(&mu_b));
        assert!(after.score(&mu_a) < 0.0, "old direction must have decayed: {}", after.score(&mu_a));
    }

    // A third class in the pool must not break a pair's axis (pooled covariance is by design).
    #[test]
    fn extra_classes_pool_without_breaking_pairs() {
        let d = 12;
        let mut h = FisherHead::new(d);
        let mut rng = Rng(19);
        let mut mu_p = vec![0.0f32; d]; mu_p[1] = 1.2;
        let mut mu_n = vec![0.0f32; d]; mu_n[1] = -1.2;
        let mut mu_x = vec![0.0f32; d]; mu_x[7] = 2.0;   // an unrelated scope's moment
        let sig = vec![0.8f32; d];
        for _ in 0..120 {
            h.observe_labeled("+", &sample(&mut rng, &mu_p, &sig));
            h.observe_labeled("-", &sample(&mut rng, &mu_n, &sig));
            h.observe_labeled("scope:other", &sample(&mut rng, &mu_x, &sig));
        }
        let ax = h.axis("+", "-").expect("pair axis with a third class pooled");
        assert!(ax.score(&mu_p) > 0.0 && ax.score(&mu_n) < 0.0);
        // and the scope-vs-rest style pairing works off the same single head
        let ax2 = h.axis("scope:other", "+").expect("any observed pair is answerable");
        assert!(ax2.score(&mu_x) > 0.0);
    }

    // Scores are bounded at the read no matter how extreme the input.
    #[test]
    fn scores_stay_bounded() {
        let mut h = FisherHead::new(4);
        let mut rng = Rng(23);
        for _ in 0..50 {
            h.observe_labeled("+", &sample(&mut rng, &[2.0; 4], &[0.3; 4]));
            h.observe_labeled("-", &sample(&mut rng, &[-2.0; 4], &[0.3; 4]));
        }
        let ax = h.axis("+", "-").unwrap();
        assert_eq!(ax.score(&[1e6; 4]), Z_CAP);
        assert_eq!(ax.score(&[-1e6; 4]), -Z_CAP);
    }

    // The head survives a restart exactly: same axis before and after dump/load.
    #[test]
    fn dump_load_round_trips() {
        let d = 8;
        let mut h = FisherHead::new(d);
        let mut rng = Rng(31);
        for _ in 0..60 {
            h.observe_labeled("+", &sample(&mut rng, &[1.0; 8], &[0.5; 8]));
            h.observe_labeled("-", &sample(&mut rng, &[-1.0; 8], &[0.5; 8]));
        }
        let a1 = h.axis("+", "-").unwrap();
        let blob = h.dump();
        let mut r = FisherHead::load(&blob).expect("dump must reload");
        let a2 = r.axis("+", "-").unwrap();
        assert_eq!(a1.w.len(), a2.w.len());
        for (x, y) in a1.w.iter().zip(&a2.w) { assert!((x - y).abs() < 1e-9, "{x} vs {y}"); }
        assert!((a1.c - a2.c).abs() < 1e-9);
        assert_eq!(h.updates(), r.updates());
        // corrupted blobs abstain rather than misload
        assert!(FisherHead::load("dim\t0").is_none());
        assert!(FisherHead::load("").is_none());
    }
}
