// gary-neuron driving the system as the cortex/router: a mixed query stream is dispatched by the
// cortex over a recalled working set. Reports per-intent routing accuracy, the share of turns
// memory resolves without a host-model call, and dispatch latency.
use neuron_core::model::{Dispatch, GaryModel};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let m = GaryModel::embedded();
    let ws: Vec<String> = [
        "the api key is zeta-9931",
        "the wifi password is Maple-Cedar-71",
        "the gate code is 4471",
        "the deploy region is us-west-2",
        "the building manager is Priya",
        "the launch date is Friday",
    ].iter().map(|s| s.to_string()).collect();

    let cases: &[(&str, &str)] = &[
        ("what is the api key?", "ANSWER"),
        ("what is the wifi password?", "ANSWER"),
        ("when is the launch?", "ANSWER"),
        ("who is the building manager?", "ANSWER"),
        ("explain how TLS works", "ESCALATE"),
        ("write a haiku about latency", "ESCALATE"),
        ("what is the capital of Brazil?", "FETCH"),
        ("what's the weather in Oslo?", "FETCH"),
        ("my favorite editor is neovim", "STORE"),
        ("remember the office closes at 6pm", "STORE"),
    ];

    let reps = 50;
    let n = reps * cases.len();
    let mut lat = Vec::with_capacity(n);
    let mut correct = 0usize;
    let mut handled = 0usize;
    let mut by: HashMap<&str, (usize, usize)> = HashMap::new();

    for _ in 0..reps {
        for (q, want) in cases {
            let t = Instant::now();
            let d = m.dispatch(&ws, q);
            lat.push(t.elapsed().as_micros());
            let got = match &d {
                Dispatch::Answer(_) => "ANSWER",
                Dispatch::Escalate => "ESCALATE",
                Dispatch::Fetch(_) => "FETCH",
                Dispatch::Store(_) => "STORE",
            };
            let slot = by.entry(want).or_default();
            slot.1 += 1;
            if got == *want {
                correct += 1;
                slot.0 += 1;
            }
            if matches!(d, Dispatch::Answer(_)) {
                handled += 1;
            }
        }
    }

    lat.sort_unstable();
    let pct = |q: f64| lat[((n as f64 * q) as usize).min(n - 1)] as f64 / 1000.0;
    println!("gary-neuron as cortex/router: {n} dispatches over a {}-fact working set", ws.len());
    println!("routing accuracy: {correct}/{n} ({:.0}%)", 100.0 * correct as f64 / n as f64);
    for route in ["ANSWER", "ESCALATE", "FETCH", "STORE"] {
        let (c, t) = by[route];
        println!("  {route:<9} {c}/{t} ({:.0}%)", 100.0 * c as f64 / t as f64);
    }
    println!("resolved by memory (no host-model call): {handled}/{n} ({:.0}%)", 100.0 * handled as f64 / n as f64);
    let mean = lat.iter().sum::<u128>() as f64 / n as f64 / 1000.0;
    println!("dispatch latency: p50 {:.2} ms | p99 {:.2} ms | mean {mean:.2} ms", pct(0.5), pct(0.99));
}
