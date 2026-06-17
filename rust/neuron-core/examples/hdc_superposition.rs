//! (b) HDC / VSA superposition capacity micro-test — the honest version of "pack many facts into
//! one vector's dimensions to store more on a smaller device".
//!
//! Hyperdimensional computing stores a key->value map in ONE fixed D-dimensional vector:
//!   bind(key, value) = key ⊙ value        (elementwise product of bipolar {-1,+1} vectors)
//!   memory           = sign( Σ_i bind(key_i, value_i) )   <- one D-bit record holding ALL pairs
//!   recall(key_j)    = cleanup( memory ⊙ key_j )          <- unbind, then nearest value in codebook
//! Unbinding returns value_j plus crosstalk from the other K-1 pairs; past a capacity the crosstalk
//! buries the signal and cleanup picks the wrong value. Theory: capacity ~ D / (2 ln V) ish — it
//! grows LINEARLY with D. This bench measures the accuracy-vs-load curve and the capacity K* (the
//! largest K with >=90% recall), then reports the resulting bytes/pair to compare against the
//! exact store's ~42 bytes/fact.
//!
//! Run: cargo run --release --example hdc_superposition   (pure std, no features)

const V: usize = 64;          // value codebook size (cleanup picks among these)
const TRIALS: usize = 8;      // random memories per (D,K) point, averaged
const TEST_KEYS: usize = 150; // recalls sampled per memory (cap so big-K stays tractable)

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
/// deterministic bipolar atom for (kind, id): D entries of +-1, derived from a hash so the same
/// symbol always yields the same hypervector without storing it.
fn atom(kind: u64, id: u64, d: usize, out: &mut [i8]) {
    let base = splitmix(kind.wrapping_mul(0x100000001B3) ^ id.wrapping_mul(0x9E3779B1));
    for (p, slot) in out.iter_mut().enumerate().take(d) {
        let h = splitmix(base ^ (p as u64).wrapping_mul(0xD1B54A32D192ED03));
        *slot = if h & 1 == 0 { 1 } else { -1 };
    }
}

/// One random memory of K pairs at dimension D: accuracy = fraction of sampled keys recalled right.
fn trial(d: usize, k: usize, seed: u64) -> f64 {
    // value codebook
    let mut vcode = vec![vec![0i8; d]; V];
    for (v, vc) in vcode.iter_mut().enumerate() { atom(2, v as u64, d, vc); }
    // build the superposed memory: accumulate sign-less integer sum, then binarize
    let mut acc = vec![0i32; d];
    let mut key_buf = vec![0i8; d];
    let mut stored_val = Vec::with_capacity(k);
    for i in 0..k {
        let key_id = splitmix(seed ^ i as u64) % 1_000_000_007;
        let val_id = (splitmix(seed.wrapping_add(0x5555) ^ i as u64) % V as u64) as usize;
        atom(1, key_id, d, &mut key_buf);
        for p in 0..d { acc[p] += (key_buf[p] as i32) * (vcode[val_id][p] as i32); }
        stored_val.push((key_id, val_id));
    }
    let mem: Vec<i8> = acc.iter().map(|&x| if x >= 0 { 1 } else { -1 }).collect(); // D-bit record

    // recall a sample of the stored keys: unbind, cleanup against the value codebook
    let tested = k.min(TEST_KEYS);
    let mut correct = 0usize;
    let mut ubuf = vec![0i8; d];
    for t in 0..tested {
        let (key_id, want) = stored_val[t * k / tested.max(1)];
        atom(1, key_id, d, &mut key_buf);
        for p in 0..d { ubuf[p] = (mem[p] as i32 * key_buf[p] as i32) as i8; } // unbind
        // nearest value atom by dot product
        let (mut best, mut bestdot) = (0usize, i64::MIN);
        for (v, vc) in vcode.iter().enumerate() {
            let dot: i64 = (0..d).map(|p| ubuf[p] as i64 * vc[p] as i64).sum();
            if dot > bestdot { bestdot = dot; best = v; }
        }
        if best == want { correct += 1; }
    }
    correct as f64 / tested.max(1) as f64
}

fn accuracy(d: usize, k: usize) -> f64 {
    let mut s = 0.0;
    for t in 0..TRIALS { s += trial(d, k, 0xC0FFEE ^ ((d as u64) << 20) ^ ((k as u64) << 4) ^ t as u64); }
    s / TRIALS as f64
}

fn main() {
    println!("== (b) HDC superposition capacity: K key->value pairs in ONE D-bit vector ==");
    println!("   bind=elementwise product, bundle=sign(sum), recall=unbind+cleanup over {} values.", V);
    println!("   Accuracy is exact-value recall; capacity K* = largest K with >=90% recall.\n");

    for d in [256usize, 1024, 4096] {
        // sweep K as fractions of D, find where accuracy crosses 90% and 50%
        let ks: Vec<usize> = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]
            .iter().map(|f| ((d as f64) * f).round() as usize).filter(|&k| k >= 1).collect();
        println!("   D = {:>5}  ({} bytes per fixed record)", d, d / 8);
        print!("      K:");
        for &k in &ks { print!(" {:>5}", k); }
        println!();
        print!("    acc%:");
        let mut accs = Vec::new();
        for &k in &ks { let a = accuracy(d, k); accs.push((k, a)); print!(" {:>5.0}", a * 100.0); }
        println!();
        // capacity estimate: last K with accuracy >= 0.90
        let kstar = accs.iter().rev().find(|(_, a)| *a >= 0.90).map(|(k, _)| *k).unwrap_or(0);
        if kstar > 0 {
            println!("      => capacity K*(>=90%) ~ {} pairs  ({:.3} x D)  -> {:.1} bytes/pair at capacity",
                     kstar, kstar as f64 / d as f64, (d as f64 / 8.0) / kstar as f64);
        } else {
            println!("      => even the smallest K tested fell below 90%");
        }
        println!();
    }

    println!("   Takeaways:");
    println!("   - Capacity grows ~linearly with D (more dimensions => more pairs), exactly the");
    println!("     'extend dimensions to store more' intuition — but it is LOSSY: past K* recall");
    println!("     collapses, and even below K* it is probabilistic, not exact.");
    println!("   - bytes/pair at capacity is the fair comparison: if it isn't well under ~42 B/fact");
    println!("     AND you can tolerate approximate recall AND you store values from a fixed codebook");
    println!("     (not arbitrary text), HDC packing wins; otherwise the exact store is better.");
    println!("   - The text of a fact still has to live somewhere — HDC packs SYMBOL bindings, not");
    println!("     sentences, so it is a different capability (associative, robust, parallel), not a");
    println!("     drop-in density multiplier for a language memory.");
}
