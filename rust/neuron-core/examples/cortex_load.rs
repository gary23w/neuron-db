//! gary-neuron v2 under NORMAL LOAD — the middle dispatcher serving a sustained, realistic stream.
//! Sweeps the WORKING-SET SIZE (how many facts neuron-db recalled into context) to map where the
//! model holds up and where it degrades: per size, the answerable accuracy, the escalate/abstain
//! accuracy, and the p50/p99 latency. This is the number that decides how to tune the cortex.
//!
//! Run: cargo run --release --example cortex_load        (REPS env per item, default 5)
use neuron_core::model::GaryModel;
use std::time::Instant;

fn abstained(a: &str) -> bool {
    let a = a.to_lowercase();
    a.len() < 2 || a.contains("don't have") || a.contains("do not have")
        || a.contains("not in") || a.contains("isn't in") || a.contains("don't know")
}

fn main() {
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let verbose = std::env::var("VERBOSE").is_ok();
    let m = GaryModel::embedded();

    // answerable: (fact that holds the answer, query, expected substring)
    let items: Vec<(&str, &str, &str)> = vec![
        ("the wifi password is Maple-Cedar-71", "what is the wifi password?", "Maple-Cedar-71"),
        ("the api key is zeta-9931",            "what is the api key?",       "zeta-9931"),
        ("the launch is on Friday",             "when is the launch?",        "Friday"),
        ("the gate code is 4471",               "what is the gate code?",     "4471"),
        ("the database port is 5432",           "which port does the database run on?", "5432"),
        ("the deploy region is us-west-2",      "what is the deploy region?", "us-west-2"),
        ("the building manager is Priya",       "who is the building manager?", "Priya"),
        ("the invoice total is 4820 dollars",   "what is the invoice total?", "4820"),
    ];
    // filler facts to pad the working set to a target size (none answers any item's query)
    let distractors: Vec<&str> = vec![
        "the meeting room is 14C", "the seat assignment is 22A", "the firmware version is v3.1.0",
        "the account number is ACME-50833", "the checkout time is 11 AM", "the tracking number is 1ZQ77",
        "the emergency contact is Omar", "the monthly rent is 2100 dollars", "the printer address is 10.0.0.4",
        "the safe combination is 12-34-56", "the project deadline is March 9th", "the backup wifi is Birch-Otter-22",
        "the conference line is 7782", "the parking spot is B12", "the badge id is 99031",
    ];
    let oom: Vec<&str> = vec![
        "what is the boiling point of mercury?", "who won the 2019 world cup?", "explain quantum entanglement",
    ];
    let sizes = [1usize, 3, 6, 9, 12, 15];

    // build a working set of `ws` facts that INCLUDES `target` (deterministic, target rotated in)
    let ws_with = |target: &str, ws: usize, slot: usize| -> Vec<String> {
        let mut set: Vec<String> = distractors.iter().take(ws.saturating_sub(1)).map(|s| s.to_string()).collect();
        let pos = if ws == 0 { 0 } else { slot % ws };
        set.insert(pos.min(set.len()), target.to_string());
        set
    };

    println!("==== gary-neuron v2 — load profile vs working-set size ====\n");
    println!("{:>3} | {:>14} | {:>14} | {:>18}", "WS", "answerable", "escalate(OOM)", "latency p50/p99 ms");
    println!("{}", "-".repeat(60));
    for &ws in &sizes {
        let mut lat: Vec<u128> = Vec::new();
        let (mut ans_ok, mut ans_tot, mut esc_ok, mut esc_tot) = (0usize, 0usize, 0usize, 0usize);
        for rep in 0..reps {
            for (i, (fact, q, exp)) in items.iter().enumerate() {
                let set = ws_with(fact, ws, i + rep);
                let s = Instant::now(); let a = m.think(&set, q, 16); lat.push(s.elapsed().as_micros());
                ans_tot += 1; if a.contains(exp) { ans_ok += 1; }
                if verbose && rep == 0 && ws == 12 { println!("   [{}] {:<36} -> {:?}", if a.contains(exp){"ok "}else{"MISS"}, q, a); }
            }
            for (i, q) in oom.iter().enumerate() {              // out-of-memory: pad with ws distractors, none answer it
                let set: Vec<String> = distractors.iter().cycle().skip(i).take(ws).map(|s| s.to_string()).collect();
                let s = Instant::now(); let a = m.think(&set, q, 16); lat.push(s.elapsed().as_micros());
                esc_tot += 1; if abstained(&a) { esc_ok += 1; }
            }
        }
        lat.sort_unstable();
        let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len()-1)] as f64 / 1000.0;
        println!("{:>3} | {:>6}/{:<3} {:>3.0}% | {:>6}/{:<3} {:>3.0}% | {:>8.0} / {:<.0}",
                 ws, ans_ok, ans_tot, 100.0*ans_ok as f64/ans_tot as f64,
                 esc_ok, esc_tot, 100.0*esc_ok as f64/esc_tot as f64, pct(0.50), pct(0.99));
    }
}
