//! User-testing benchmark: drive NeuronDB the way a real app would. Simulate a fleet
//! of users, each with a per-user scope; write a realistic profile, then ask a battery
//! of natural questions (direct lookups, alias paraphrases, numeric, and abstention)
//! and score recall accuracy + latency. This is an end-to-end "does it work for users"
//! benchmark, distinct from the micro-benchmark in db_bench.rs.
//!
//! Run: cargo run --release --features sqlite --example scenario_bench
use neuron_core::db::NeuronDB;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn tmp() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ndb_scn_{}_{}.db", std::process::id(), n)).to_string_lossy().into_owned()
}

const NAMES: [&str; 10] = ["Marisol","Dana","Kenji","Amara","Bjorn","Priya","Tariq","Lena","Mateo","Nadia"];
const PLANS: [&str; 3] = ["pro","free","enterprise"];
const CITIES: [&str; 6] = ["Halifax","Lisbon","Nairobi","Osaka","Bogota","Tallinn"];
const EDITORS: [&str; 4] = ["neovim","vscode","emacs","helix"];
const TZS: [&str; 4] = ["UTC","PST","EST","CET"];

struct Cat { name: &'static str, hit: u64, total: u64, }
impl Cat { fn new(n: &'static str) -> Self { Cat { name: n, hit: 0, total: 0 } }
    fn record(&mut self, ok: bool) { self.total += 1; if ok { self.hit += 1; } }
    fn pct(&self) -> f64 { if self.total == 0 { 0.0 } else { 100.0 * self.hit as f64 / self.total as f64 } } }

fn main() {
    let users = 1_000usize;
    let db = NeuronDB::open(&tmp(), 1_000);
    println!("== NeuronDB user-testing benchmark: {} simulated users ==\n", users);

    // ---- ingest: write each user's profile ----
    let mut writes = 0u64;
    let t = Instant::now();
    for i in 0..users {
        let s = format!("user:{}", i);
        let facts = [
            format!("my name is {}", NAMES[i % NAMES.len()]),
            format!("my plan is {}", PLANS[i % PLANS.len()]),
            format!("my city is {}", CITIES[i % CITIES.len()]),
            format!("my manager is {}", NAMES[(i + 3) % NAMES.len()]),
            format!("my editor is {}", EDITORS[i % EDITORS.len()]),
            format!("my timezone is {}", TZS[i % TZS.len()]),
            format!("my seat count is {}", 1 + (i % 50)),
        ];
        for f in &facts { writes += db.observe(&s, f) as u64; }
    }
    let ingest = t.elapsed().as_secs_f64();
    println!("ingest: {} facts across {} scopes in {:.2}s ({:.0} writes/sec)\n",
             writes, users, ingest, writes as f64 / ingest);

    // ---- query battery ----
    let mut direct = Cat::new("direct lookup     ");
    let mut alias  = Cat::new("alias paraphrase  ");
    let mut numeric = Cat::new("numeric           ");
    let mut abstain = Cat::new("abstention        ");
    // separate, NOT folded into the overall score: a query whose word stems into a
    // stored fact's stem ("planet" -> "plan"), exposing aggressive-stemming false positives.
    let mut collide = Cat::new("no-collision check ");
    let mut lat: Vec<u128> = Vec::with_capacity(users * 10);

    let mut probe = |db: &NeuronDB, scope: &str, q: &str, lat: &mut Vec<u128>| -> Option<String> {
        let t = Instant::now();
        let r = db.get(scope, q);
        lat.push(t.elapsed().as_nanos());
        r
    };

    for i in 0..users {
        let s = format!("user:{}", i);
        let name = NAMES[i % NAMES.len()];
        let plan = PLANS[i % PLANS.len()];
        let city = CITIES[i % CITIES.len()];
        let mgr = NAMES[(i + 3) % NAMES.len()];
        let editor = EDITORS[i % EDITORS.len()];
        let seats = (1 + (i % 50)).to_string();

        // direct lookups
        direct.record(probe(&db, &s, "what is my name?", &mut lat).as_deref() == Some(name));
        direct.record(probe(&db, &s, "what plan am i on?", &mut lat).as_deref() == Some(plan));
        direct.record(probe(&db, &s, "what is my city?", &mut lat).as_deref() == Some(city));

        // alias paraphrases (cue word never appears literally in the stored fact)
        alias.record(probe(&db, &s, "what is my subscription?", &mut lat).as_deref() == Some(plan));
        alias.record(probe(&db, &s, "who is my boss?", &mut lat).as_deref() == Some(mgr));
        alias.record(probe(&db, &s, "what is my ide?", &mut lat).as_deref() == Some(editor));

        // numeric
        numeric.record(probe(&db, &s, "how many seats do i have?", &mut lat).as_deref() == Some(seats.as_str()));

        // abstention (never stored, no stem collision -> must be None)
        abstain.record(probe(&db, &s, "what is my blood type?", &mut lat).is_none());
        abstain.record(probe(&db, &s, "what is the weather today?", &mut lat).is_none());

        // collision-prone: "planet" stems to "plan" and matches the stored plan fact.
        // Expected None; failures here quantify aggressive-stemming false positives.
        collide.record(probe(&db, &s, "what is my favorite planet?", &mut lat).is_none());
    }

    // ---- scorecard ----
    println!("recall accuracy by category:");
    for c in [&direct, &alias, &numeric, &abstain] {
        println!("  {} {:>5.1}%  ({}/{})", c.name, c.pct(), c.hit, c.total);
    }
    let total_hit: u64 = [&direct, &alias, &numeric, &abstain].iter().map(|c| c.hit).sum();
    let total_all: u64 = [&direct, &alias, &numeric, &abstain].iter().map(|c| c.total).sum();
    println!("  {:<18} {:>5.1}%  ({}/{})", "OVERALL", 100.0 * total_hit as f64 / total_all as f64, total_hit, total_all);
    println!("\nstemming precision probe (separate):");
    println!("  {} {:>5.1}%  ({}/{})  <- 'planet' wrongly matches stored 'plan' when low",
             collide.name, collide.pct(), collide.hit, collide.total);

    lat.sort_unstable();
    let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)] as f64 / 1000.0;
    println!("recall latency over {} queries: p50 {:.2} us | p95 {:.2} us | p99 {:.2} us | max {:.2} us",
             lat.len(), pct(0.50), pct(0.95), pct(0.99), *lat.last().unwrap() as f64 / 1000.0);

    println!("\n== done ==");
}
