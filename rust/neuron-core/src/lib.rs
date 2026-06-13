//! neuron-core: the associative memory ("neuron") in pure Rust, standard library only.
//! Faithful port of the Python prototype's store: write facts in plain language, recall
//! by cue, isolate the value nearest the asked-about word, abstain when nothing matches.
//! A stem->fact inverted index keeps recall sub-linear.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

fn set(words: &str) -> HashSet<&'static str> {
    // leak once into 'static; sets are tiny and built one time
    let leaked: &'static str = Box::leak(words.to_string().into_boxed_str());
    leaked.split_whitespace().collect()
}
fn stop() -> &'static HashSet<&'static str> { static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| set("what is my the a an you your i me do did does how was were it to of and or in on at that this s u g who whats tell about im ive id ill am are be will wont cant dont yes no really have has had its these those there here remember recall know knew think guess again still any some mine get got getting go going want wanted would could should can please us we they them he she him her his hers their our ours")) }
fn stopval() -> &'static HashSet<&'static str> { static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| set("had has have having like likes liked want wants wanted went going goes got get gets day days week thing things something anything nothing everything lot bit time times really very name named names favorite favourite color colour food dog cat hello hi hey thanks thank okay yes lol bye good great nice long suppose supposed gonna wanna kinda sorta maybe probably definitely oh ya yeah yep nah hmm right sure fine cool wow oops used use using still now for with from into onto over under after before out off down up around through there here then than while because but not no so just too also even back well if as by be been being actually never always sometimes today tomorrow yesterday tonight sister brother mom dad mother father wife husband son daughter grandma grandpa aunt uncle cousin friend boss live lives lived drive drives new anyway hows heres theres lets gotta mostly honestly basically literally")) }
fn rel() -> &'static HashSet<&'static str> { static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| set("dog cat pet bird fish sister brother mom dad mother father wife husband son daughter grandma grandpa aunt uncle cousin car truck bike cats dogs puppy kitten puppies kittens hamster")) }
fn numwords() -> &'static HashSet<&'static str> { static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| set("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty hundred thousand million dozen")) }
fn adv() -> &'static HashSet<&'static str> { static S: OnceLock<HashSet<&'static str>> = OnceLock::new();
    S.get_or_init(|| set("actually anyway honestly basically literally oh ok okay well yeah yep nah hmm wow oops so but and also still just then wait sorry hey um uh no yes listen look")) }
fn irr(w: &str) -> &str {
    static M: OnceLock<HashMap<&'static str,&'static str>> = OnceLock::new();
    let m = M.get_or_init(|| {
        let pairs = "drank drink ate eat went go goes go saw see met meet took take bought buy made make ran run drove drive wrote write slept sleep gave give told tell said say felt feel day today days today old age aged age work job working job employed job occupation job profession job career job kids kid children kid hometown city town city favourite favorite colour color";
        let v: Vec<&str> = pairs.split_whitespace().collect();
        let mut h = HashMap::new(); let mut i=0; while i+1<v.len() { h.insert(v[i], v[i+1]); i+=2; } h
    });
    *m.get(w).unwrap_or(&w)
}

fn w1(w: &str) -> String {
    let t: &str = w.trim_matches(|c: char| "?.!,;:'\"’><)([]}{".contains(c));
    let t = t.to_lowercase();
    if t.ends_with("'s") || t.ends_with("\u{2019}s") { t[..t.len()-2].to_string() } else { t }
}
fn words(s: &str) -> HashSet<String> { s.split_whitespace().map(w1).filter(|x| !x.is_empty()).collect() }
fn content(s: &str) -> HashSet<String> { words(s).into_iter().filter(|w| !stop().contains(w.as_str())).collect() }
fn stem1(w: &str) -> String {
    let mut w = irr(w).to_string();
    if w.len() >= 5 && w.ends_with("ies") { w = format!("{}y", &w[..w.len()-3]); }
    else if w.len() >= 4 && w.ends_with('s') && !w.ends_with("ss") { w.pop(); }
    if w.len() >= 8 { w[..6].to_string() } else if w.len() >= 4 { w[..4].to_string() } else { w }
}
fn stems<'a, I: IntoIterator<Item = &'a String>>(it: I) -> HashSet<String> { it.into_iter().map(|w| stem1(w)).collect() }
fn stems_s(it: &HashSet<String>) -> HashSet<String> { it.iter().map(|w| stem1(w)).collect() }
fn rel_s() -> &'static HashSet<String> { static S: OnceLock<HashSet<String>> = OnceLock::new();
    S.get_or_init(|| rel().iter().map(|w| stem1(w)).collect()) }
fn pets() -> &'static HashSet<String> { static S: OnceLock<HashSet<String>> = OnceLock::new();
    S.get_or_init(|| ["dog","cat","pet","bird","fish","puppy","kitten","hamster"].iter().map(|w| stem1(w)).collect()) }
fn stopval_s() -> &'static HashSet<String> { static S: OnceLock<HashSet<String>> = OnceLock::new();
    S.get_or_init(|| stopval().iter().map(|w| stem1(w)).collect()) }

fn is_num(w: &str) -> bool {
    w.chars().any(|c| c.is_ascii_digit()) || numwords().contains(w.trim_matches(|c:char| "?.!,'\"()".contains(c)).to_lowercase().as_str())
}
fn clip(s: &str) -> String { s.trim_matches(|c:char| "?.!,;:'\"()[]{}".contains(c)).to_string() }
fn surprise(w: &str, i: usize) -> f64 {
    let mut s = 0.0; let core = w.to_lowercase();
    if core.chars().any(|c| c.is_ascii_digit()) { s += 3.0; }
    else if w.chars().next().map_or(false,|c| c.is_uppercase()) && i>0 { s += 2.0; }
    if core.len() >= 7 { s += 0.6; }
    s
}

#[derive(Clone, Debug)]
pub struct Episode { pub t: String, pub v: String, pub c: Vec<String>, pub s: Vec<String>, pub head: String, pub self_flag: bool }

#[derive(Clone, Debug)]
pub struct Recall { pub fact: String, pub value: String, pub coverage: f64, pub overlap: usize, pub echo: bool }

fn sentences(u: &str, cap: usize) -> Vec<String> {
    let mut parts = Vec::new(); let mut cur = String::new();
    let chars: Vec<char> = u.trim().chars().collect();
    for (i,&c) in chars.iter().enumerate() {
        cur.push(c);
        let brk = matches!(c, '.'|'!'|'?'|';') && chars.get(i+1).map_or(true,|n| n.is_whitespace());
        if brk || c=='\n' { let t=cur.trim().to_string(); if !t.is_empty(){parts.push(t);} cur.clear(); }
    }
    let t = cur.trim().to_string(); if !t.is_empty() { parts.push(t); }
    if parts.is_empty() { parts.push(u.trim().to_string()); }
    parts.truncate(cap); parts
}

fn encode(text: &str, entity: Option<&str>) -> Option<Episode> {
    let u = text.trim();
    if u.is_empty() { return None; }
    let cont = content(u);
    let has_digit = cont.iter().any(|w| w.chars().any(|c| c.is_ascii_digit()));
    if cont.len() < 2 && !has_digit { return None; }
    if u.split_whitespace().count() < 3 && !has_digit { return None; }
    let uw = words(u);
    let selfish = (uw.contains("my")||uw.contains("i")||uw.contains("im")||uw.contains("mine"))
        && !(uw.contains("her")||uw.contains("his")||uw.contains("its")||uw.contains("their")||uw.contains("your"));
    let mut cands: Vec<(String,f64)> = Vec::new();
    for (i,raw) in u.split_whitespace().enumerate() {
        let w = clip(raw); let wl = w.to_lowercase();
        if wl.is_empty() || stop().contains(wl.as_str()) || stopval().contains(wl.as_str()) { continue; }
        if !wl.chars().any(|c| c.is_alphanumeric()) { continue; }
        if wl.len() < 3 && !wl.chars().all(|c| c.is_ascii_digit()) { continue; }
        cands.push((w, surprise(&w_clone(raw), i) + 0.15*(i as f64)));
    }
    if cands.is_empty() { return None; }
    cands.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    let mut keep: Vec<String> = cands.iter().take(5).map(|(w,_)| w.clone()).collect();
    for (w,_) in cands.iter().skip(5) {
        if keep.len() >= 10 { break; }
        if is_num(w) || w.chars().next().map_or(false,|c| c.is_uppercase()) { keep.push(w.clone()); }
    }
    let self_name = selfish && stems_s(&cont).contains("name");
    let mut head = String::new();
    for w in u.split_whitespace() { let x = w1(w); if !x.is_empty() && !stop().contains(x.as_str()) && !adv().contains(x.as_str()) { head = stem1(&x); break; } }
    let _ = entity;
    let s_set: HashSet<String> = stems_s(&cont);
    let mut s: Vec<String> = s_set.into_iter().collect(); s.sort();
    Some(Episode { t: text.to_string(), v: keep[0].clone(), c: keep, s, head, self_flag: self_name })
}
fn w_clone(raw: &str) -> String { clip(raw) }

fn pick_value(ep: &Episode, cue: &HashSet<String>, want_num: bool) -> (String, bool) {
    let words: Vec<&str> = ep.t.split_whitespace().collect();
    let cue_pos: Vec<usize> = words.iter().enumerate()
        .filter(|(_,w)| { let st = stem1(&w1(w)); cue.contains(&st) }).map(|(i,_)| i).collect();
    let pos_of = |c: &str| -> usize {
        let cl = clip(c).to_lowercase();
        for (i,w) in words.iter().enumerate() { if clip(w).to_lowercase()==cl || w1(w)==w1(c) { return i; } }
        1_000_000
    };
    let mut pool: Vec<String> = ep.c.iter().filter(|c| { let st = stem1(&c.to_lowercase()); !cue.contains(&st) }).cloned().collect();
    if want_num { let nums: Vec<String> = pool.iter().filter(|c| is_num(c)).cloned().collect(); if !nums.is_empty() { pool = nums; } }
    if pool.is_empty() { return (ep.v.clone(), true); }
    if want_num && !cue_pos.is_empty() && pool.len() > 1 {
        pool.sort_by_key(|c| { let p = pos_of(c) as i64; cue_pos.iter().map(|&q| { let q=q as i64; ((p-q).abs(), if p<=q {0} else {1}) }).min().unwrap() });
    }
    (pool[0].clone(), false)
}

pub struct Neuron {
    pub episodes: Vec<Episode>,
    pub max_facts: usize,
    index: Option<HashMap<String, Vec<usize>>>,
    index_len: usize,
}
impl Neuron {
    pub fn new(max_facts: usize) -> Self { Neuron { episodes: Vec::new(), max_facts, index: None, index_len: usize::MAX } }

    fn build_index(&mut self) {
        let mut idx: HashMap<String, Vec<usize>> = HashMap::new();
        for (i,e) in self.episodes.iter().enumerate() { for s in &e.s { idx.entry(s.clone()).or_default().push(i); } }
        self.index = Some(idx); self.index_len = self.episodes.len();
    }

    pub fn observe(&mut self, text: &str) -> usize {
        if text.trim().is_empty() || text.contains('?') { return 0; }
        let mut n = 0;
        for s in sentences(text, 24) {
            if let Some(e) = encode(&s, None) { self.episodes.push(e); n += 1; }
        }
        if self.episodes.len() > self.max_facts {
            let start = self.episodes.len() - self.max_facts;
            self.episodes.drain(0..start);
        }
        n
    }

    pub fn recall(&mut self, query: &str) -> Option<Recall> {
        let cue: HashSet<String> = stems_s(&content(query));
        if cue.is_empty() { return None; }
        let pet_query = cue.contains(&stem1("pet")) || cue.contains(&stem1("animal"));
        let name_query = cue.contains("name") && cue.intersection(rel_s()).count()==0;
        if self.index.is_none() || self.index_len != self.episodes.len() { self.build_index(); }
        let idx = self.index.as_ref().unwrap();
        let mut cand: HashSet<usize> = HashSet::new();
        for s in &cue { if let Some(v) = idx.get(s) { cand.extend(v); } }
        if pet_query { for s in pets() { if let Some(v) = idx.get(s) { cand.extend(v); } } }
        let mut order: Vec<usize> = cand.into_iter().collect(); order.sort();
        let mut best: Option<usize> = None;
        let mut bk: (i64,i64,i64,i64,i64) = (-1,-1,-1,0,-1);
        for i in order {
            let e = &self.episodes[i];
            let es: HashSet<&String> = e.s.iter().collect();
            let mut ov = cue.iter().filter(|c| es.contains(c)).count();
            let es_pet = e.s.iter().any(|s| pets().contains(s));
            if ov < 1 && pet_query && es_pet { ov = 1; }
            if ov < 1 { continue; }
            let unbound_es = e.s.iter().any(|s| rel_s().contains(s) && !cue.contains(s));
            if unbound_es && !(pet_query && es_pet) { continue; }
            let unbound_cue = cue.iter().any(|s| rel_s().contains(s) && !es.contains(s));
            if unbound_cue && !(pet_query && es_pet) { continue; }
            let selfp = if name_query && e.self_flag { 1 } else { 0 };
            let subj = if cue.contains(&e.head) { 1 } else { 0 };
            let spec = -(e.s.iter().filter(|s| !cue.contains(*s) && !stopval_s().contains(*s)).count() as i64);
            let sc = (ov as i64, selfp, subj, spec, i as i64);
            if sc > bk { bk = sc; best = Some(i); }
        }
        let bi = best?;
        let e = &self.episodes[bi];
        let bes: HashSet<&String> = e.s.iter().collect();
        let mut cov = cue.iter().filter(|c| bes.contains(c)).count() as f64 / (cue.len().max(1) as f64);
        if pet_query && e.s.iter().any(|s| pets().contains(s)) { cov = 1.0; }
        let want_num = cue.contains("many") || cue.contains("much") || cue.contains(&stem1("number"));
        let (val, echo) = pick_value(e, &cue, want_num);
        Some(Recall { fact: e.t.clone(), value: val, coverage: cov, overlap: bk.0 as usize, echo })
    }

    /// minimal persistence: "<flag>\t<text>" per line; index rebuilt on load
    pub fn dump(&self) -> String {
        self.episodes.iter().map(|e| format!("{}\t{}", if e.self_flag {1} else {0}, e.t)).collect::<Vec<_>>().join("\n")
    }
    pub fn load(blob: &str, max_facts: usize) -> Self {
        let mut n = Neuron::new(max_facts);
        for line in blob.split('\n') {
            if line.is_empty() { continue; }
            let text = line.splitn(2,'\t').nth(1).unwrap_or("");
            if let Some(e) = encode(text, None) { n.episodes.push(e); }
        }
        n
    }
    pub fn fact_count(&self) -> usize { self.episodes.len() }
}

// ---- inference tier: the emergence cortex, bundled (optional to use) ----
pub mod cortex;
pub mod bpe;
pub mod model;
