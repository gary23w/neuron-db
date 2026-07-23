//! Quantum tier vs classical equivalents, head to head on the durable store. Measures:
//!   1. the quantum-aware read's overhead over plain recall (dormant + active quantum state)
//!   2. teleport vs its classical 3-op equivalent (recall + forget + observe = move an association)
//!   3. write_once + burning read vs the classical get + forget round trip
//!   4. superposition store/measure cost
//!   5. entangle throughput + link lookup at scale (the teleport hot path)
//!   6. unbounded relay cascade (A->B->C->...) — the "how far can we push it" probe. The chain
//!      LENGTH below is test-data size, not a cap: the cascade itself runs until the e-bits
//!      drain (hops are unbounded everywhere; conservation terminates, not a budget).
//!
//! Run: cargo run --release --features quantum-db --example quantum_bench
//! The report prints to stdout AND lands in %TEMP%\quantum_bench_report.txt (WDAC-safe capture).

use neuron_core::db::NeuronDB;
use neuron_core::quantum as q;
use std::fmt::Write as _;
use std::time::Instant;

const SEED_FACTS: usize = 5_000;
const READS: usize = 2_000;
const PAIRS: usize = 300;
const LINK_SCALE: usize = 10_000;
const RELAY_CHAIN: usize = 64;   // depth of the BUILT chain (data), not a hop cap

fn db(tag: &str) -> (NeuronDB, std::path::PathBuf) {
    let p = std::env::temp_dir().join(format!("qbench_{}_{}.db", tag, std::process::id()));
    let _ = std::fs::remove_file(&p);
    (NeuronDB::open(p.to_str().unwrap(), 1_000_000), p)
}

fn us(t: Instant, n: usize) -> f64 { t.elapsed().as_secs_f64() * 1e6 / n as f64 }

fn main() {
    let mut rpt = String::new();
    macro_rules! say { ($($a:tt)*) => {{ let line = format!($($a)*); println!("{}", line); let _ = writeln!(rpt, "{}", line); }} }

    say!("quantum_bench — {} seed facts, {} reads, {} pairs, {} links, relay chain {}",
        SEED_FACTS, READS, PAIRS, LINK_SCALE, RELAY_CHAIN);
    say!("");

    // ---- 1. read overhead: plain recall vs the quantum-aware read ----
    // entity names stay <= 5 chars so every fact keeps a UNIQUE stem (the store truncates longer
    // words, which would collapse them into one hub stem and turn every read into a full scan —
    // that regime is measured separately below as the honest worst case).
    let (d, p1) = db("reads");
    let texts: Vec<String> = (0..SEED_FACTS).map(|i| format!("the w{} unit serial is c{}x", i, i)).collect();
    let t = Instant::now();
    d.observe_many("bench", &texts);
    say!("[seed]      observe_many {} facts          {:>9.1} us/fact", SEED_FACTS, us(t, SEED_FACTS));

    let queries: Vec<String> = (0..READS).map(|i| format!("what is the w{} unit serial?", i * 2 % SEED_FACTS)).collect();
    let t = Instant::now();
    let mut hits = 0;
    for qy in &queries { if d.recall("bench", qy).is_some() { hits += 1; } }
    let base_read = us(t, READS);
    say!("[read]      plain recall (indexed path)     {:>9.1} us/op   ({} hits)", base_read, hits);

    let t = Instant::now();
    let mut hits = 0;
    for qy in &queries { if q::recall_once(&d, "bench", qy).is_some() { hits += 1; } }
    let dormant_read = us(t, READS);
    say!("[read]      recall_once, quantum DORMANT    {:>9.1} us/op   ({} hits, {:+.1}% vs plain)", dormant_read, hits, (dormant_read / base_read - 1.0) * 100.0);

    // arm one superposition + one no-clone fact elsewhere in the db: the worst realistic case for
    // ordinary reads (quantum state EXISTS, just not for this scope/cue)
    q::store_super(&d, "elsewhere", "the standby mode is", &["armed", "idle"]);
    q::write_once(&d, "elsewhere", "the drop point is pier 9", 999);
    let t = Instant::now();
    let mut hits = 0;
    for qy in &queries { if q::recall_once(&d, "bench", qy).is_some() { hits += 1; } }
    let active_read = us(t, READS);
    say!("[read]      recall_once, quantum ACTIVE     {:>9.1} us/op   ({} hits, {:+.1}% vs plain)", active_read, hits, (active_read / base_read - 1.0) * 100.0);

    // worst case: every cue stem is a hub (df = whole scope), so the engine scores the full
    // candidate pool — the upper-bound read regime for BOTH paths
    let t = Instant::now();
    for _ in 0..200 { let _ = d.recall("bench", "what is the unit serial?"); }
    let hub_plain = us(t, 200);
    let t = Instant::now();
    for _ in 0..200 { let _ = q::recall_once(&d, "bench", "what is the unit serial?"); }
    let hub_once = us(t, 200);
    say!("[read]      hub-cue full-scan regime        {:>9.1} us/op plain vs {:.1} us/op recall_once ({:+.1}%)", hub_plain, hub_once, (hub_once / hub_plain - 1.0) * 100.0);
    say!("");

    // ---- 2. move an association: teleport vs the classical 3-op dance ----
    let (d2, p2) = db("move");
    for i in 0..PAIRS {
        q::entangle(&d2, "src", &format!("the v{} combo is {}z", i, 1000 + i),
                    "dst", &format!("the v{} combo is tbd", i), "copy", 1);
    }
    let t = Instant::now();
    let mut ok = 0;
    for i in 0..PAIRS { if q::teleport(&d2, "src", &format!("what is the v{} combo?", i)).is_some() { ok += 1; } }
    let tele = us(t, PAIRS);
    say!("[move]      teleport (1 atomic op)          {:>9.1} us/move ({}/{} ok)", tele, ok, PAIRS);

    let (d3, p3) = db("classic");
    for i in 0..PAIRS {
        d3.observe("src", &format!("the v{} combo is {}z", i, 1000 + i));
        d3.observe("dst", &format!("the v{} combo is tbd", i));
    }
    let t = Instant::now();
    let mut ok = 0;
    for i in 0..PAIRS {
        // the classical equivalent of one teleport: measure the source, clear the placeholder,
        // write the moved association — three separate ops, with a copies-both-exist window
        // (and forget's secure-delete WAL checkpoint, which the atomic rewrite never needs)
        if let Some(h) = d3.recall("src", &format!("what is the v{} combo?", i)) {
            d3.forget("dst", Some(&format!("the v{} combo is tbd", i)));
            d3.observe("dst", &h.fact);
            ok += 1;
        }
    }
    let classic = us(t, PAIRS);
    say!("[move]      classical recall+forget+observe {:>9.1} us/move ({}/{} ok, {:.2}x teleport)", classic, ok, PAIRS, classic / tele);
    say!("");

    // ---- 3. one-shot secret: write_once + burning read vs get + forget ----
    let (d4, p4) = db("burn");
    let t = Instant::now();
    for i in 0..PAIRS {
        q::write_once(&d4, "vault", &format!("the d{} code is g{}x", i, i), 1);
        let _ = q::recall_once(&d4, "vault", &format!("what is the d{} code?", i));   // the read burns it
    }
    let burn = us(t, PAIRS);
    say!("[one-shot]  write_once + burning read       {:>9.1} us/cycle (facts left: {})", burn, d4.stats("vault").facts);

    let (d5, p5) = db("burnclassic");
    let t = Instant::now();
    for i in 0..PAIRS {
        d5.observe("vault", &format!("the d{} code is g{}x", i, i));
        let _ = d5.recall("vault", &format!("what is the d{} code?", i));
        d5.forget("vault", Some(&format!("d{} code", i)));   // the manual cleanup teleport makes unnecessary
    }
    let burn_c = us(t, PAIRS);
    say!("[one-shot]  observe + get + manual forget   {:>9.1} us/cycle (facts left: {}, {:.2}x write_once)", burn_c, d5.stats("vault").facts, burn_c / burn);
    say!("");

    // ---- 4. superposition ----
    let (d6, p6) = db("super");
    let t = Instant::now();
    for i in 0..PAIRS { q::store_super(&d6, "amb", &format!("s{} state is", i), &["nominal", "degraded", "offline"]); }
    say!("[super]     store_super (3 alternatives)    {:>9.1} us/op", us(t, PAIRS));
    let t = Instant::now();
    let mut got = 0;
    for i in 0..PAIRS { if q::recall_super(&d6, "amb", &format!("what is s{} state?", i)).is_some() { got += 1; } }
    say!("[super]     recall_super (measure+persist)  {:>9.1} us/op   ({}/{} measured)", us(t, PAIRS), got, PAIRS);
    say!("");

    // ---- 5. entangle throughput + the link lookup at scale ----
    let (d7, p7) = db("links");
    let t = Instant::now();
    for i in 0..LINK_SCALE {
        q::entangle(&d7, "a", &format!("the r{} channel is {}", i, i), "b", &format!("the r{} mirror is blank", i), "copy", 3);
    }
    say!("[links]     entangle throughput             {:>9.1} us/link ({} links)", us(t, LINK_SCALE), LINK_SCALE);
    let probe: Vec<String> = (0..READS).map(|i| format!("the r{} channel is {}", i * 3 % LINK_SCALE, i * 3 % LINK_SCALE)).collect();
    let t = Instant::now();
    let mut found = 0;
    for (i, txt) in probe.iter().enumerate() { let _ = i; if !d7_find(&d7, txt).is_empty() { found += 1; } }
    say!("[links]     find_entanglements @ {}k links  {:>9.1} us/op   ({} found)", LINK_SCALE / 1000, us(t, READS), found);
    say!("");

    // ---- 6. unbounded relay: ONE cascade call runs until the entanglement graph settles ----
    let (d8, p8) = db("relay");
    let secret = "the payload token is zx9000";
    q::entangle(&d8, "hop0", secret, "hop1", "the payload token is blank1", "copy", 1);
    for i in 1..RELAY_CHAIN {
        // each hop's dest is pre-entangled to the NEXT hop on the text the teleport will create
        q::entangle(&d8, &format!("hop{}", i), secret, &format!("hop{}", i + 1), &format!("the payload token is blank{}", i + 1), "copy", 1);
        // (the entangle observed `secret` into hop{i} — remove it so only the cascade can place it)
        <NeuronDB as q::QuantumBack>::forget_exact(&d8, &format!("hop{}", i), secret);
    }
    let t = Instant::now();
    let trail = q::teleport_cascade(&d8, "hop0", "what is the payload token?");
    let depth = trail.len();
    say!("[relay]     teleport_cascade (no hop cap)   {:>9.1} us/hop  (settled at depth {} over a {}-hop chain)", us(t, depth.max(1)), depth, RELAY_CHAIN);
    let last = trail.last().map(|h| h.value.clone()).unwrap_or_default();
    say!("[relay]     payload at the final hop: '{}'", last);

    let report = std::env::temp_dir().join("quantum_bench_report.txt");
    let _ = std::fs::write(&report, &rpt);
    say!("");
    say!("(report written to {})", report.display());

    for p in [p1, p2, p3, p4, p5, p6, p7, p8] { let _ = std::fs::remove_file(p); }
}

fn d7_find(d: &NeuronDB, text: &str) -> Vec<q::EntanglementRecord> {
    <NeuronDB as q::HasEntanglements>::find_entanglements(d, "a", text)
}
