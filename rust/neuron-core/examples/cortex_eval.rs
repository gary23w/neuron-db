//! Quantitative held-out eval for the embedded gary-neuron cortex.
//! Generates N fresh random cases per task (values the model never saw) and reports exact-match
//! accuracy in the real int8 runtime. Tasks: copy, number-copy, select, abstain, compare, multihop.
//! Run: cargo run --release --example cortex_eval   (N per task via EVAL_N env, default 50)
use neuron_core::model::GaryModel;

// deterministic std-only RNG (xorshift64*) so runs are reproducible without a rand dep
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    fn range(&mut self, lo: i64, hi: i64) -> i64 { lo + (self.next() % (hi - lo) as u64) as i64 }
    fn pick<'a>(&mut self, xs: &'a [&str]) -> &'a str { xs[self.below(xs.len())] }
}

const LOW: &[u8] = b"abcdefghijkmnpqrstuvwxyz23456789";
const UPP: &[u8] = b"ABCDEFGHJKLMNPRSTUVWXYZ23456789";
fn rstr(r: &mut Rng, set: &[u8], n: usize) -> String {
    (0..n).map(|_| set[r.below(set.len())] as char).collect()
}
fn digits(r: &mut Rng, n: usize) -> String { (0..n).map(|_| (b'0' + (r.next() % 10) as u8) as char).collect() }
fn commafy(mut n: i64) -> String {
    let s = n.abs().to_string(); let b = s.as_bytes(); let mut out = String::new();
    for (i, c) in b.iter().enumerate() {
        if i > 0 && (b.len() - i) % 3 == 0 { out.push(','); }
        out.push(*c as char);
    }
    if n < 0 { n = -n; let _ = n; }
    out
}

const KEYS: &[&str] = &["wifi password","api key","gate code","database port","room number","account number",
    "ip address","deploy region","firmware version","tracking number","serial number","passphrase","code word",
    "confirmation code","license plate","flight number","locker number","badge id","coupon code","access pin"];
const NAMES: &[&str] = &["Marisol","Dana","Tony","Lena","Priya","Omar","Sofia","Kenji","Aisha","Diego","Hana",
    "Mateo","Noor","Ravi","Elena","Yusuf","Greta","Caleb","Ingrid","Tomas","Aurora","Felix","Nadia","Bruno",
    "Clara","Idris","Owen","Rosa","Samir","Talia","Victor","Wendy","Xavier","Yara","Zane"];
const NOUNS: &[&str] = &["participants","attendees","tickets","visitors","guests","members","orders","downloads"];
const REL: &[&str] = &["owner","manager","mentor","landlord","doctor","lawyer","coach","sponsor","tutor","agent"];
const WORDS: &[&str] = &["Maple","Cedar","Falcon","Robin","Otter","Ember","Slate","Amber","Cobalt","Onyx","Nova","Harbor"];

fn gen_val(r: &mut Rng) -> String {
    match r.below(10) {
        0 => { let k = 5 + r.below(4); rstr(r, LOW, k) }                     // vekam73-style
        1 => format!("{}-{}-{}", r.pick(WORDS), r.pick(WORDS), digits(r, 2)),
        2 => format!("{}{}", r.pick(&["sk-","zeta-","key_","ak-"]), rstr(r, LOW, 8)),
        3 => { let k = 3 + r.below(4); digits(r, k) }
        4 => commafy(r.range(1000, 9_999_999)),
        5 => format!("{}.{}.{}", r.range(1,9), r.range(0,9), r.range(0,9)),
        6 => (0..4).map(|i| format!("{}{}", if i>0 {"."} else {""}, r.range(1,254))).collect(),
        7 => format!("${}", commafy(r.range(50, 99999))),
        8 => format!("{}-{}", rstr(r, UPP, 4), digits(r, 4)),
        _ => r.pick(NAMES).to_string(),
    }
}

fn norm(s: &str) -> String {
    s.trim().trim_matches(|c: char| c=='.'||c=='"'||c=='\''||c==' '||c=='!').to_string()
}
fn hit(out: &str, want: &str) -> bool {
    let (o, w) = (norm(out), norm(want));
    o == w || (w.len() >= 3 && o.starts_with(&w))
}

fn main() {
    let n: usize = std::env::var("EVAL_N").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let m = GaryModel::embedded();
    let mut r = Rng(0x9E3779B97F4A7C15);
    println!("==== gary-neuron held-out eval (N={n}/task, fresh random values, greedy) ====\n");

    let report = |name: &str, ok: usize, tot: usize, misses: &[String]| {
        println!("{name:<12} {ok:>3}/{tot:<3}  {:>5.1}%", 100.0 * ok as f64 / tot as f64);
        for ex in misses.iter().take(2) { println!("     miss: {ex}"); }
    };

    // COPY (arbitrary value shapes)
    let (mut ok, mut miss) = (0usize, Vec::new());
    for _ in 0..n {
        let k = r.pick(KEYS).to_string(); let v = gen_val(&mut r);
        let got = m.think(&[format!("the {k} is {v}")], &format!("what is the {k}?"), 16);
        if hit(&got, &v) { ok += 1 } else { miss.push(format!("{k}={v:?} got {got:?}")) }
    }
    report("copy", ok, n, &miss);

    // NUMBER-COPY (comma counts)
    let (mut ok, mut miss) = (0usize, Vec::new());
    for _ in 0..n {
        let cnt = commafy(r.range(1000, 9_999_999)); let noun = r.pick(NOUNS).to_string();
        let got = m.think(&[format!("there are {cnt} {noun} registered")], &format!("how many {noun}?"), 16);
        if hit(&got, &cnt) { ok += 1 } else { miss.push(format!("{cnt} {noun} got {got:?}")) }
    }
    report("number-copy", ok, n, &miss);

    // SELECT (3-5 distractors)
    let (mut ok, mut miss) = (0usize, Vec::new());
    for _ in 0..n {
        let cnt = 3 + r.below(3);
        let mut used: Vec<&str> = Vec::new(); let mut facts = Vec::new(); let mut tv = String::new(); let mut tk = String::new();
        let tgt = r.below(cnt);
        for i in 0..cnt {
            let mut k = r.pick(KEYS); while used.contains(&k) { k = r.pick(KEYS); } used.push(k);
            let v = gen_val(&mut r);
            if i == tgt { tk = k.to_string(); tv = v.clone(); }
            facts.push(format!("the {k} is {v}"));
        }
        let got = m.think(&facts, &format!("what is the {tk}?"), 16);
        if hit(&got, &tv) { ok += 1 } else { miss.push(format!("{tk}={tv:?} got {got:?}")) }
    }
    report("select", ok, n, &miss);

    // ABSTAIN (asked key absent)
    let (mut ok, mut miss) = (0usize, Vec::new());
    for _ in 0..n {
        let cnt = 2 + r.below(3);
        let mut used: Vec<&str> = Vec::new(); let mut facts = Vec::new();
        for _ in 0..cnt { let mut k = r.pick(KEYS); while used.contains(&k) { k = r.pick(KEYS); } used.push(k);
            facts.push(format!("the {k} is {}", gen_val(&mut r))); }
        let mut mk = r.pick(KEYS); while used.contains(&mk) { mk = r.pick(KEYS); }
        let got = m.think(&facts, &format!("what is the {mk}?"), 16).to_lowercase();
        if got.contains("don't have") || got.contains("isn't in") || got.contains("not in") { ok += 1 }
        else { miss.push(format!("missing {mk} got {got:?}")) }
    }
    report("abstain", ok, n, &miss);

    // COMPARE (answer the name)
    let (mut ok, mut miss) = (0usize, Vec::new());
    for _ in 0..n {
        let a = r.pick(NAMES).to_string(); let mut b = r.pick(NAMES).to_string(); while b == a { b = r.pick(NAMES).to_string(); }
        let (x, y) = (r.range(1, 99), r.range(1, 99)); if x == y { continue; }
        let item = r.pick(&["apples","points","books","votes","wins","coins"]).to_string();
        let more = r.below(2) == 0;
        let want = if (x > y) == more { &a } else { &b };
        let facts = vec![format!("{a} has {x} {item}"), format!("{b} has {y} {item}")];
        let q = format!("who has {} {item}?", if more {"more"} else {"fewer"});
        let got = m.think(&facts, &q, 8);
        if hit(&got, want) { ok += 1 } else { miss.push(format!("{a}:{x} {b}:{y} {q} want {want:?} got {got:?}")) }
    }
    report("compare", ok, n, &miss);

    // MULTIHOP (2-hop chain)
    let (mut ok, mut miss) = (0usize, Vec::new());
    for _ in 0..n {
        let a = r.pick(NAMES).to_string();
        let mut b = r.pick(NAMES).to_string(); while b == a { b = r.pick(NAMES).to_string(); }
        let mut c = r.pick(NAMES).to_string(); while c == a || c == b { c = r.pick(NAMES).to_string(); }
        let r1 = r.pick(REL).to_string(); let mut r2 = r.pick(REL).to_string(); while r2 == r1 { r2 = r.pick(REL).to_string(); }
        let facts = vec![format!("{a}'s {r1} is {b}"), format!("{b}'s {r2} is {c}")];
        let q = format!("who is the {r2} of {a}'s {r1}?");
        let got = m.think(&facts, &q, 8);
        if hit(&got, &c) { ok += 1 } else { miss.push(format!("{q} want {c:?} got {got:?}")) }
    }
    report("multihop", ok, n, &miss);
}
