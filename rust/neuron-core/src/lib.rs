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
    m.get(w).unwrap_or(&w)
}

fn w1(w: &str) -> String {
    let t: &str = w.trim_matches(|c: char| "?.!,;:'\"’><)([]}{".contains(c));
    let t = t.to_lowercase();
    // strip a possessive suffix char-safely; t[..t.len()-2] panics when the apostrophe
    // is a multibyte curly quote (\u{2019}, 3 bytes) rather than ASCII '.
    if let Some(s) = t.strip_suffix("'s").or_else(|| t.strip_suffix("\u{2019}s")) { s.to_string() } else { t }
}
fn words(s: &str) -> HashSet<String> { s.split_whitespace().map(w1).filter(|x| !x.is_empty()).collect() }
fn content(s: &str) -> HashSet<String> { words(s).into_iter().filter(|w| !stop().contains(w.as_str())).collect() }
fn stem1(w: &str) -> String {
    let mut w = irr(w).to_string();
    if w.len() >= 5 && w.ends_with("ies") { w = format!("{}y", &w[..w.len()-3]); }
    else if w.len() >= 4 && w.ends_with('s') && !w.ends_with("ss") { w.pop(); }
    // truncate by CHARS, not bytes: a byte slice (w[..6]) panics on multibyte UTF-8.
    // Keep mid-length words at 5 chars (not 4) so 5-6 char words don't collapse onto a
    // shorter word's stem ("planet"/"plant"/"plane" no longer all become "plan").
    let n = w.chars().count();
    if n >= 8 { w.chars().take(6).collect() }
    else if n >= 5 { w.chars().take(5).collect() }
    else { w }
}
/// Public morphological root (owner/owned/owns -> own). Used by recall_chain to verify a
/// hop's relation actually appears in the recalled fact before advancing.
pub fn root_token(w: &str) -> String { root(w) }
/// Map a word to its canonical synonym if known (plural-tolerant): "reports"/"report" ->
/// "manager", "lives" -> "city". Returns the word's own normalized form otherwise.
fn canon(w: &str) -> String {
    let wl = w1(w);
    if let Some(c) = aliases().get(wl.as_str()) { return (*c).to_string(); }
    let s = wl.trim_end_matches('s');
    if s.len() >= 3 { if let Some(c) = aliases().get(s) { return (*c).to_string(); } }
    wl
}
/// Escape a string for embedding in a JSON string literal (control chars -> \uXXXX). The one
/// canonical escaper shared by every text wire surface (CLI --json, the MCP server, the HTTP
/// server) so they can't drift; std-only, no serde.
pub fn json_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

/// Whether two words name the same relation: same morphological root (owner/owned), same
/// stem (dependency/depends), or the same canonical synonym (reports/manager, lives/city).
pub fn rel_matches(a: &str, b: &str) -> bool {
    root(a) == root(b) || stem1(&w1(a)) == stem1(&w1(b)) || {
        let (ca, cb) = (canon(a), canon(b));
        root(&ca) == root(&cb) || stem1(&ca) == stem1(&cb)
    }
}
/// Morphological root for the fuzzy fallback: strip a common suffix so owner/owned/owns
/// normalize together. ONLY used on a primary recall miss, so it never affects the fast path.
fn root(w: &str) -> String {
    let base = w.trim_matches(|c: char| "?.!,;:'\"\u{2019}><)([]}{".contains(c)).to_lowercase();
    let base = irr(&base).to_string();
    const SUF: [&str; 12] = ["ization","ation","ments","ment","ness","ing","ion","ies","es","ed","er","s"];
    for suf in SUF {
        if base.len() >= suf.len() + 3 && base.ends_with(suf) {
            return base[..base.len() - suf.len()].to_string();
        }
    }
    base
}
fn stems_s(it: &HashSet<String>) -> HashSet<String> { it.iter().map(|w| stem1(w)).collect() }
fn aliases() -> &'static HashMap<&'static str, &'static str> {
    static M: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    M.get_or_init(|| {
        // curated synonym -> canonical; applied to BOTH the query cue and (in the fallback)
        // the stored facts, so synonyms recall regardless of which side used which word.
        [("subscription","plan"),("tier","plan"),("membership","plan"),
         ("boss","manager"),("supervisor","manager"),("report","manager"),("reports","manager"),
         ("manages","manager"),("manage","manager"),("lead","manager"),("reporting","manager"),
         ("role","job"),("occupation","job"),("profession","job"),("position","job"),("title","job"),
         ("ide","editor"),("tz","timezone"),("zone","timezone"),("mail","email"),
         ("username","handle"),("user","handle"),("due","deadline"),("cell","phone"),("mobile","phone"),
         // residence -> city
         ("lives","city"),("live","city"),("living","city"),("resides","city"),("reside","city"),
         ("based","city"),("located","city"),("location","city"),("home","city"),("hometown","city"),
         // ownership / dependency relation words
         ("owned","owner"),("owns","owner"),("own","owner"),("dependency","depends"),
         ("blocker","depends"),("requires","depends"),("needs","depends")]
        .iter().cloned().collect()
    })
}
fn expand_cue(query: &str, cue: &mut HashSet<String>) {
    for w in content(query) { if let Some(c) = aliases().get(w.as_str()) { cue.insert(stem1(c)); } }
}
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
/// Escape the record separators (tab/newline) and the escape char itself so a fact's text can
/// never be mistaken for a field/record boundary in the dump format.
fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() { match c { '\\' => o.push_str("\\\\"), '\t' => o.push_str("\\t"), '\n' => o.push_str("\\n"), c => o.push(c) } }
    o
}
fn unesc(s: &str) -> String {
    if !s.contains('\\') { return s.to_string(); }   // fast path: escape-free (incl. all legacy blobs)
    let mut o = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() { Some('t') => o.push('\t'), Some('n') => o.push('\n'), Some('\\') => o.push('\\'), Some(x) => { o.push('\\'); o.push(x); }, None => o.push('\\') }
        } else { o.push(c); }
    }
    o
}
fn surprise(w: &str, i: usize) -> f64 {
    let mut s = 0.0; let core = w.to_lowercase();
    if core.chars().any(|c| c.is_ascii_digit()) { s += 3.0; }
    else if w.chars().next().is_some_and(|c| c.is_uppercase()) && i>0 { s += 2.0; }
    if core.len() >= 7 { s += 0.6; }
    s
}

#[derive(Clone, Debug)]
pub struct Episode { pub t: String, pub v: String, pub c: Vec<String>, pub s: Vec<String>, pub raw: Vec<String>, pub pos: Vec<u32>, pub head: String, pub self_flag: bool, pub id: i64, pub strength: f32 }

#[derive(Clone, Debug)]
pub struct Recall { pub fact: String, pub value: String, pub coverage: f64, pub overlap: usize, pub exact: usize, pub echo: bool }
/// A spreading-activation hit: a fact reached by following shared-entity links from the cue.
/// `seed` marks a fact that directly matched the query; the rest are associates surfaced by spread.
#[derive(Debug, Clone)]
pub struct Spread { pub fact: String, pub value: String, pub seed: bool, pub act: f64 }

// Result shapes for a conversational turn and a scope's stats. Defined at the crate root (not in the
// sqlite-gated db.rs) so the op vocabulary + the Store trait can name them in a no-sqlite wasm build.
#[derive(Debug, Clone, Default)]
pub struct TurnOut { pub reply: String, pub kind: String, pub wrote: usize, pub facts: usize, pub capacity_reached: bool }
#[derive(Debug, Clone, Default)]
pub struct Stats { pub facts: usize, pub max_facts: usize, pub created: i64, pub updated: i64, pub turns: i64, pub dropped: u64 }

fn sentences(u: &str, cap: usize) -> Vec<String> {
    let mut parts = Vec::new(); let mut cur = String::new();
    let trimmed = u.trim();
    // stream the chars with one-char lookahead (peek) instead of materializing a Vec<char> — so a
    // multi-MB paste isn't fully resident as a char vector during ingest.
    let mut it = trimmed.chars().peekable();
    while let Some(c) = it.next() {
        cur.push(c);
        let brk = matches!(c, '.'|'!'|'?'|';') && it.peek().is_none_or(|n| n.is_whitespace());
        if brk || c=='\n' { let t=cur.trim().to_string(); if !t.is_empty(){parts.push(t);} cur.clear(); }
    }
    let t = cur.trim().to_string(); if !t.is_empty() { parts.push(t); }
    if parts.is_empty() { parts.push(trimmed.to_string()); }
    parts.truncate(cap); parts
}

fn encode(text: &str, entity: Option<&str>) -> Option<Episode> {
    let u = text.trim();
    if u.is_empty() { return None; }
    let cont = content(u);
    let has_digit = cont.iter().any(|w| w.chars().any(|c| c.is_ascii_digit()));
    // need at least one content word (or a number), and at least 3 words total — so an explicit
    // short fact like "i am tired" or "call me ace" is kept, but bare "ok" / "hello" is not.
    // A ':' marks a deliberate structured entry (a typed neuron, e.g. a terse stance
    // "bureaucracy: draining"), so it's exempt from the min-word heuristic — bare chatter has none.
    if cont.is_empty() && !has_digit { return None; }
    if u.split_whitespace().count() < 3 && !has_digit && !u.contains(':') { return None; }
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
        if is_num(w) || w.chars().next().is_some_and(|c| c.is_uppercase()) { keep.push(w.clone()); }
    }
    let self_name = selfish && stems_s(&cont).contains("name");
    let mut head = String::new();
    for w in u.split_whitespace() { let x = w1(w); if !x.is_empty() && !stop().contains(x.as_str()) && !adv().contains(x.as_str()) { head = stem1(&x); break; } }
    let _ = entity;
    let s_set: HashSet<String> = stems_s(&cont);
    let mut s: Vec<String> = s_set.into_iter().collect(); s.sort();
    // earliest raw-token position of each content stem, aligned with the sorted `s`. This lets
    // recall() compute the subject-position tiebreak via a binary_search per cue word instead
    // of re-tokenizing + re-stemming the whole fact for every candidate (the hot-loop cost).
    let mut pos = vec![u32::MAX; s.len()];
    for (i, tok) in u.split_whitespace().enumerate() {
        if let Ok(k) = s.binary_search(&stem1(&w1(tok))) {
            if pos[k] == u32::MAX { pos[k] = i as u32; }
        }
    }
    let mut raw: Vec<String> = cont.into_iter().collect(); raw.sort(); raw.dedup();
    Some(Episode { t: text.to_string(), v: keep[0].clone(), c: keep, s, raw, pos, head, self_flag: self_name, id: -1, strength: 1.0 })
}
fn w_clone(raw: &str) -> String { clip(raw) }

fn expand_value(text: &str, val: &str) -> String {
    // upgrade a single-token value to a full Capitalized phrase ("Search Console")
    if val.chars().any(|c| c.is_ascii_digit()) { return val.to_string(); }
    let toks: Vec<&str> = text.split_whitespace().collect();
    let vl = clip(val).to_lowercase();
    let mut idx = None;
    for (i,w) in toks.iter().enumerate() { if clip(w).to_lowercase()==vl { idx=Some(i); break; } }
    let i = match idx { Some(i)=>i, None=>return val.to_string() };
    let is_cap = |w:&str| clip(w).chars().next().is_some_and(|c| c.is_uppercase());
    if !is_cap(toks[i]) { return val.to_string(); }
    let blocked = |w:&str| { let wl = clip(w).to_lowercase(); stop().contains(wl.as_str()) || stopval().contains(wl.as_str()) };
    let (mut lo, mut hi) = (i, i);
    while lo>0 && is_cap(toks[lo-1]) && !blocked(toks[lo-1]) { lo-=1; }
    while hi+1<toks.len() && is_cap(toks[hi+1]) && !blocked(toks[hi+1]) { hi+=1; }
    if hi>lo { toks[lo..=hi].iter().map(|w| clip(w)).collect::<Vec<_>>().join(" ") } else { val.to_string() }
}

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
    if pool.is_empty() { return (expand_value(&ep.t, &ep.v), true); }
    // The value normally follows the subject+relation. If any candidate sits AFTER the cue,
    // prefer those — this stops a leading structural word ("project Graus status is paused"
    // -> "project") from beating a short value. Numeric recall keeps proximity instead, since
    // a number can precede its cue ("12 engineers" -> "how many engineers").
    if !want_num && !cue_pos.is_empty() && pool.len() > 1 {
        let last_cue = *cue_pos.iter().max().unwrap();
        let after: Vec<String> = pool.iter().filter(|c| pos_of(c) > last_cue).cloned().collect();
        if !after.is_empty() { pool = after; }
    }
    if want_num && !cue_pos.is_empty() && pool.len() > 1 {
        pool.sort_by_key(|c| { let p = pos_of(c) as i64; cue_pos.iter().map(|&q| { let q=q as i64; ((p-q).abs(), if p<=q {0} else {1}) }).min().unwrap() });
    }
    (expand_value(&ep.t, &pool[0]), false)
}

pub struct Neuron {
    pub episodes: Vec<Episode>,
    pub max_facts: usize,
    index: Option<HashMap<String, Vec<usize>>>,
    index_len: usize,
    pub dropped: u64,   // oldest facts evicted by the max_facts front-drain. Per-session: NOT persisted
                        // by dump()/load(), so it resets on reload/eviction — read it right after a write.
}
impl Neuron {
    pub fn new(max_facts: usize) -> Self { Neuron { episodes: Vec::new(), max_facts, index: None, index_len: usize::MAX, dropped: 0 } }

    fn build_index(&mut self) {
        let mut idx: HashMap<String, Vec<usize>> = HashMap::new();
        // clone a stem String only the first time it's seen (once per unique stem), not once per
        // posting — at millions of facts this is ~unique-stems clones instead of ~total-postings.
        for (i,e) in self.episodes.iter().enumerate() {
            for s in &e.s { match idx.get_mut(s) { Some(v) => v.push(i), None => { idx.insert(s.clone(), vec![i]); } } }
        }
        self.index = Some(idx); self.index_len = self.episodes.len();
    }

    /// Ensure the inverted index covers all current episodes. Incrementally indexes appended
    /// facts in O(new) instead of rebuilding the whole index in O(N); any fact removal nulls
    /// the index (see observe() drain / db forget) so this path only ever sees pure appends.
    fn ensure_index(&mut self) {
        match &mut self.index {
            Some(idx) if self.index_len <= self.episodes.len() => {
                for i in self.index_len..self.episodes.len() {
                    for s in &self.episodes[i].s { match idx.get_mut(s) { Some(v) => v.push(i), None => { idx.insert(s.clone(), vec![i]); } } }
                }
                self.index_len = self.episodes.len();
            }
            _ => self.build_index(),
        }
    }

    pub fn observe(&mut self, text: &str) -> usize {
        if text.trim().is_empty() { return 0; }
        let mut n = 0;
        // Split into sentences first, then keep DECLARATIVES and skip questions per-sentence —
        // so a paragraph of prose is captured comprehensively, not dropped wholesale just because
        // it contains a question. Cap is high so long pasted text is ingested, not truncated.
        for s in sentences(text, 400) {
            if s.contains('?') { continue; } // questions aren't facts
            if let Some(e) = encode(&s, None) { self.episodes.push(e); n += 1; }
        }
        if self.episodes.len() > self.max_facts {
            let start = self.episodes.len() - self.max_facts;
            self.dropped = self.dropped.saturating_add(start as u64);   // record the eviction (not silent)
            self.episodes.drain(0..start);
            self.index = None; self.index_len = usize::MAX; // front-drain shifts indices -> rebuild
        }
        n
    }

    pub fn recall(&mut self, query: &str) -> Option<Recall> {
        let mut cue: HashSet<String> = stems_s(&content(query));
        if cue.is_empty() { return None; }
        expand_cue(query, &mut cue);
        let qraw: HashSet<String> = content(query);
        let pet_query = cue.contains(&stem1("pet")) || cue.contains(&stem1("animal"));
        let name_query = cue.contains("name") && cue.intersection(rel_s()).count()==0;
        self.ensure_index();
        let idx = self.index.as_ref().unwrap();
        let mut cand: HashSet<usize> = HashSet::new();
        for s in &cue { if let Some(v) = idx.get(s) { cand.extend(v); } }
        if pet_query { for s in pets() { if let Some(v) = idx.get(s) { cand.extend(v); } } }
        let mut order: Vec<usize> = cand.into_iter().collect(); order.sort();
        let mut best: Option<usize> = None;
        let mut bk: (i64,i64,i64,i64,i64,i64,i64) = (-1,-1,-1,-1,0,-100000,-1);
        for i in order {
            let e = &self.episodes[i];
            let mut ov = cue.iter().filter(|c| e.s.binary_search(c).is_ok()).count();
            let es_pet = e.s.iter().any(|s| pets().contains(s));
            if ov < 1 && pet_query && es_pet { ov = 1; }
            if ov < 1 { continue; }
            let unbound_es = e.s.iter().any(|s| rel_s().contains(s) && !cue.contains(s));
            if unbound_es && !(pet_query && es_pet) { continue; }
            let unbound_cue = cue.iter().any(|s| rel_s().contains(s) && e.s.binary_search(s).is_err());
            if unbound_cue && !(pet_query && es_pet) { continue; }
            let exact_ov = qraw.iter().filter(|wd| e.raw.binary_search(wd).is_ok()).count() as i64;
            let selfp = if name_query && e.self_flag { 1 } else { 0 };
            let subj = if cue.contains(&e.head) { 1 } else { 0 };
            let spec = -(e.s.iter().filter(|s| !cue.contains(*s) && !stopval_s().contains(*s)).count() as i64);
            // prefer the fact where the query's words appear EARLIEST (the subject), so
            // "Aurora depends on" beats "X depends on Aurora". Tiebreak before recency.
            let first_cue = cue.iter()
                .filter_map(|c| e.s.binary_search(c).ok().map(|k| e.pos[k] as i64))
                .min().unwrap_or(9999);
            let sc = (exact_ov, ov as i64, selfp, subj, spec, -first_cue, i as i64);
            if sc > bk { bk = sc; best = Some(i); }
        }
        let bi = match best { Some(b) => b, None => return self.root_scan(query, 1).into_iter().next() };
        // If the relation didn't fully bind (some cue words matched nothing -- e.g. the query
        // says "owner" but the fact says "owned"), a morphological scan may match more of the
        // query. Prefer it when it does. This rescues the entity-only-overlap case where the
        // primary would otherwise pick an arbitrary fact about the right entity.
        let prim_ov = { let bs = &self.episodes[bi].s;
                        cue.iter().filter(|c| bs.binary_search(c).is_ok()).count() };
        // The O(N) root_scan rescue only helps the relation-MORPHOLOGY case (query "owner" vs fact "owned"):
        // a missing NON-relation word (e.g. "where", "what") can't be matched by a morph scan, so don't pay
        // O(N) for it. Gating on an unbound relation cue keeps recall O(candidates) for ordinary questions —
        // value/assess/chain (which all call recall) drop from ~ms to ~µs on a large scope.
        if prim_ov < cue.len() {
            let bs = &self.episodes[bi].s;
            let unbound_rel = cue.iter().any(|c| rel_s().contains(c) && bs.binary_search(c).is_err());
            if unbound_rel {
                if let Some(r) = self.root_scan(query, 1).into_iter().next() {
                    if r.overlap > prim_ov { return Some(r); }
                }
            }
        }
        let e = &self.episodes[bi];
        let mut cov = cue.iter().filter(|c| e.s.binary_search(c).is_ok()).count() as f64 / (cue.len().max(1) as f64);
        if pet_query && e.s.iter().any(|s| pets().contains(s)) { cov = 1.0; }
        let want_num = cue.contains("many") || cue.contains("much") || cue.contains(&stem1("number"));
        let (val, echo) = pick_value(e, &cue, want_num);
        Some(Recall { fact: e.t.clone(), value: val, coverage: cov, overlap: bk.1 as usize, exact: bk.0 as usize, echo })
    }

    /// Top-k relevant facts (richest first), for building a memory block. Same scoring as
    /// recall, but returns several hits instead of one.
    pub fn recall_many(&mut self, query: &str, k: usize) -> Vec<Recall> {
        let mut cue: HashSet<String> = stems_s(&content(query));
        if cue.is_empty() { return Vec::new(); }
        expand_cue(query, &mut cue);
        let qraw: HashSet<String> = content(query);
        let pet_query = cue.contains(&stem1("pet")) || cue.contains(&stem1("animal"));
        let order = self.candidates(&cue, pet_query);
        // top-k via a bounded min-heap: O(candidates · log k) with k memory, instead of scoring then
        // sorting the whole candidate set. The key carries Reverse(index) so equal scores break to the
        // lower index — identical ordering to the previous stable sort + truncate.
        use std::cmp::Reverse;
        type Key = ((i64, i64, i64, i64), Reverse<usize>);
        let mut heap: std::collections::BinaryHeap<Reverse<Key>> = std::collections::BinaryHeap::with_capacity(k + 1);
        for i in order {
            let e = &self.episodes[i];
            let ov = cue.iter().filter(|c| e.s.binary_search(c).is_ok()).count();
            if ov < 1 { continue; }
            let exact = qraw.iter().filter(|wd| e.raw.binary_search(wd).is_ok()).count() as i64;
            let spec = -(e.s.iter().filter(|s| !cue.contains(*s) && !stopval_s().contains(*s)).count() as i64);
            let first_cue = cue.iter()
                .filter_map(|c| e.s.binary_search(c).ok().map(|kk| e.pos[kk] as i64))
                .min().unwrap_or(9999);
            let key: Key = ((exact, ov as i64, spec, -first_cue), Reverse(i));
            if heap.len() < k { heap.push(Reverse(key)); }
            else if heap.peek().is_some_and(|Reverse(min)| key > *min) { heap.pop(); heap.push(Reverse(key)); }
        }
        let mut scored: Vec<((i64,i64,i64,i64), usize)> =
            heap.into_iter().map(|Reverse((sc, Reverse(i)))| (sc, i)).collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));   // best first; ties -> ascending index
        let want_num = cue.contains("many") || cue.contains("much") || cue.contains(&stem1("number"));
        let out: Vec<Recall> = scored.into_iter().map(|(sc,i)| {
            let e = &self.episodes[i];
            let cov = cue.iter().filter(|c| e.s.binary_search(c).is_ok()).count() as f64 / (cue.len().max(1) as f64);
            let (val, echo) = pick_value(e, &cue, want_num);
            Recall { fact: e.t.clone(), value: val, coverage: cov, overlap: sc.1 as usize, exact: sc.0 as usize, echo }
        }).collect();
        if out.is_empty() { self.root_scan(query, k) } else { out }
    }

    /// Spreading-activation recall over the shared-stem co-occurrence graph. Facts that match the
    /// cue are seeded, then activation flows along DISCRIMINATIVE shared stems (rare entities link
    /// strongly; common/hub stems are ignored) for `hops`, so facts that share no words with the
    /// query but are wired to a match still surface. This is association-based recall — it traverses
    /// structure the raw text never stated — not keyword or cosine ranking. Pure read of the index.
    pub fn recall_spreading(&mut self, query: &str, k: usize, hops: usize) -> Vec<Spread> {
        let cue: HashSet<String> = stems_s(&content(query));
        if cue.is_empty() || self.episodes.is_empty() { return Vec::new(); }
        self.ensure_index();
        let idx = self.index.as_ref().unwrap();
        let n = self.episodes.len();
        // sparse activation: cost tracks the ACTIVE set, not total facts (a large base scope with a
        // narrow query lights only a handful of episodes — no O(N) allocation or final O(N) scan).
        let mut act: HashMap<usize, f64> = HashMap::new();
        let mut seed: HashSet<usize> = HashSet::new();
        let mut frontier: Vec<usize> = Vec::new();
        for s in &cue {
            if let Some(v) = idx.get(s) {
                for &j in v { if seed.insert(j) { frontier.push(j); } *act.entry(j).or_insert(0.0) += 1.0; }
            }
        }
        if frontier.is_empty() { return Vec::new(); }
        let dfcap = ((n as f64) * 0.25).max(4.0) as usize;   // hub stems link too much to be discriminative
        for _ in 0..hops {
            let mut next: HashMap<usize, f64> = HashMap::new();
            for &i in &frontier {
                let ai = act[&i];
                for s in &self.episodes[i].s {
                    if stopval_s().contains(s) { continue; }
                    let posting = match idx.get(s) { Some(p) => p, None => continue };
                    let df = posting.len();
                    if df < 2 || df > dfcap { continue; }    // unique stem = no link; hub = skipped
                    let w = 0.5 / df as f64;                 // rarer shared entity = stronger link
                    for &j in posting { if j != i { *next.entry(j).or_insert(0.0) += ai * w; } }
                }
            }
            frontier.clear();
            for (j, add) in next {
                let e = act.entry(j).or_insert(0.0);
                if *e == 0.0 { frontier.push(j); }           // newly lit -> spread on the next hop
                *e += add;
            }
            if frontier.is_empty() { break; }
        }
        let mut order: Vec<(usize, f64)> = act.into_iter().filter(|&(_, a)| a > 0.0).collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        order.truncate(k);
        order.into_iter().map(|(i, a)| {
            let e = &self.episodes[i];
            Spread { fact: e.t.clone(), value: e.v.clone(), seed: seed.contains(&i), act: a }
        }).collect()
    }

    /// Fuzzy fallback: a root-normalized linear scan used only when exact-stem recall finds
    /// nothing. Unifies morphological variants (owner/owned/owns) and alias synonyms, so a
    /// query whose words don't stem-match the stored facts can still recall. O(facts), but
    /// only on a miss, so the warm fast path keeps its flat microsecond cost.
    fn root_scan(&self, query: &str, k: usize) -> Vec<Recall> {
        let qc = content(query);
        if qc.is_empty() { return Vec::new(); }
        // canonicalize both sides through the synonym map so reports<->manager, lives<->city, etc. match
        let mut qroots: HashSet<String> = qc.iter().map(|w| root(w)).collect();
        for w in &qc { qroots.insert(root(&canon(w))); }
        let want_num = qc.contains("many") || qc.contains("much") || qc.contains("number");
        let in_q = |w: &str| qroots.contains(&root(w)) || qroots.contains(&root(&canon(w)));
        let mut scored: Vec<((i64,i64,i64,i64), usize)> = Vec::new();
        // bound this fuzzy fallback to the most-recent window so a lexical MISS stays cheap as a
        // scope grows across many chats (it only runs on a miss, but was O(N) over the whole scope).
        const ROOT_SCAN_CAP: usize = 4000;
        let start = self.episodes.len().saturating_sub(ROOT_SCAN_CAP);
        for (i, e) in self.episodes.iter().enumerate().skip(start) {
            let mut er: HashSet<String> = HashSet::new();
            for w in &e.raw { er.insert(root(w)); er.insert(root(&canon(w))); }
            let ov = qroots.iter().filter(|r| er.contains(*r)).count() as i64;
            if ov < 1 { continue; }
            let spec = -(e.raw.len() as i64);
            // subject-position tiebreak: prefer where the query's words appear earliest, so
            // "Dana manager" picks "Dana reports to X" over "Y reports to Dana".
            let first_pos = e.t.split_whitespace().enumerate()
                .filter(|(_, w)| in_q(w)).map(|(p, _)| p as i64).min().unwrap_or(9999);
            scored.push(((ov, spec, -first_pos, i as i64), i));
        }
        scored.sort_by_key(|e| std::cmp::Reverse(e.0));
        scored.truncate(k);
        scored.into_iter().map(|(sc, i)| {
            let e = &self.episodes[i];
            let cue: HashSet<String> = e.raw.iter()
                .filter(|w| qroots.contains(&root(w)) || qroots.contains(&root(&canon(w))))
                .map(|w| stem1(w)).collect();
            let (val, echo) = pick_value(e, &cue, want_num);
            let cov = sc.0 as f64 / qroots.len().max(1) as f64;
            Recall { fact: e.t.clone(), value: val, coverage: cov, overlap: sc.0 as usize, exact: 0, echo }
        }).collect()
    }

    /// Persistence: "<flag>\t<text>\t<strength>" per line; index rebuilt on load. Strength carries
    /// accumulated salience (e.g. a stance that intensified with repetition) durably across restarts.
    pub fn dump(&self) -> String { self.dump_from(0) }
    /// Serialize `episodes[from..]` in the same line format as dump(). The durable append-log uses this
    /// to write ONLY the newly-appended facts, instead of rewriting the whole scope blob on each observe.
    pub fn dump_from(&self, from: usize) -> String {
        use std::fmt::Write as _;
        // one pass into a single pre-sized buffer (no intermediate Vec<String>, no join copy).
        let eps = self.episodes.get(from..).unwrap_or(&[]);
        let mut out = String::with_capacity(eps.len().saturating_mul(48));
        for (i, e) in eps.iter().enumerate() {
            if i > 0 { out.push('\n'); }
            let _ = write!(out, "{}\t{}\t{}", if e.self_flag {1} else {0}, esc(&e.t), e.strength);
        }
        out
    }
    pub fn load(blob: &str, max_facts: usize) -> Self {
        let mut n = Neuron::new(max_facts);
        for line in blob.split('\n') {
            if line.is_empty() { continue; }
            // line = "flag\ttext[\tstrength]". Slice past the flag, then peel an optional trailing
            // strength only if the last tab-field parses as f32 — zero allocation per line, and any
            // tabs that were inside legacy text are preserved (the strength peel just won't fire).
            let after_flag = match line.find('\t') { Some(i) => &line[i + 1..], None => continue };
            let (text, strength) = match after_flag.rfind('\t') {
                Some(j) => match after_flag[j + 1..].parse::<f32>() {
                    Ok(s) => (&after_flag[..j], s),
                    Err(_) => (after_flag, 1.0),
                },
                None => (after_flag, 1.0),
            };
            if let Some(mut e) = encode(&unesc(text), None) { e.strength = strength; n.episodes.push(e); }
        }
        n.build_index();   // load is pure append -> build once here, so the first recall pays no O(N) rebuild
        n
    }

    /// Reinforce the stance whose text begins "<topic>:" (case-insensitive), accumulating its
    /// strength by `bump`; if none exists, create it at strength `bump`. This is how a disposition
    /// intensifies with repeated exposure — and because strength is persisted (see dump), the
    /// accumulation survives restarts. Returns (new_strength, created_new).
    pub fn reinforce_prefix(&mut self, topic: &str, feeling: &str, bump: f32) -> (f32, bool) {
        let feeling = feeling.split_whitespace().collect::<Vec<_>>().join(" "); // collapse to one tidy line
        let pat = format!("{}:", topic.trim().to_lowercase());
        let stored = format!("{}: {}", topic.trim(), feeling);
        match self.episodes.iter().position(|e| e.t.to_lowercase().starts_with(&pat)) {
            Some(i) => {
                let s = self.episodes[i].strength + bump;
                if let Some(mut e) = encode(&stored, None) {   // refine wording, carry strength
                    self.episodes.remove(i); self.index = None; // removal shifts indices -> rebuild
                    e.strength = s; self.episodes.push(e);
                } else {
                    self.episodes[i].strength = s;             // unencodable refinement: just intensify
                }
                (s, false)
            }
            None => match encode(&stored, None) {
                Some(mut e) => { e.strength = bump; self.episodes.push(e); (bump, true) }
                None => (0.0, false),   // nothing encoded/stored; strength 0 signals not-stored to callers
            },
        }
    }
    /// Multiplicatively decay the strength of every prefix-group EXCEPT `keep`, floored at `floor`.
    /// Called when one group is reinforced so the ones not being revisited slowly fade — a
    /// disposition that can shift over time instead of only ever hardening.
    pub fn decay_prefix_others(&mut self, keep: &str, factor: f32, floor: f32) {
        let pat = format!("{}:", keep.trim().to_lowercase());
        for e in self.episodes.iter_mut() {
            if !e.t.to_lowercase().starts_with(&pat) {
                e.strength = (e.strength * factor).max(floor);
            }
        }
    }
    /// Remove episodes whose text begins with `prefix` (case-insensitive). Anchored at the start,
    /// so removing "region is " never touches "deployRegion is …". Returns the removed count.
    pub fn forget_prefix(&mut self, prefix: &str) -> usize {
        let pl = prefix.to_lowercase();
        let before = self.episodes.len();
        self.episodes.retain(|e| !e.t.to_lowercase().starts_with(&pl));
        let removed = before - self.episodes.len();
        if removed > 0 { self.index = None; self.index_len = usize::MAX; } // removals shift indices
        removed
    }

    pub fn fact_count(&self) -> usize { self.episodes.len() }
    pub(crate) fn invalidate_index(&mut self) { self.index = None; }

    /// Candidate episode indices for a cue (ensures the index is current). Used by PlasticNeuron.
    pub(crate) fn candidates(&mut self, cue: &HashSet<String>, pet_query: bool) -> Vec<usize> {
        self.ensure_index();
        let idx = self.index.as_ref().unwrap();
        // df-aware gating: a stem in >25% of a large scope is a hub — gathering on it pulls O(scope)
        // candidates. When the cue ALSO has a discriminative (rare) stem, skip the hub postings and
        // gather only from the rare ones, so block recall is O(rare-df), not O(scope). A genuinely
        // broad query (every cue stem is a hub) falls back to gathering all. The dfcap floor (64) keeps
        // behavior byte-identical at small scope — only large scopes are gated.
        let dfcap = ((self.episodes.len() as f64) * 0.25).max(64.0) as usize;
        let has_rare = cue.iter().any(|s| idx.get(s).is_some_and(|v| v.len() <= dfcap));
        let mut cand: HashSet<usize> = HashSet::new();
        for s in cue {
            if let Some(v) = idx.get(s) {
                if has_rare && v.len() > dfcap { continue; }   // skip the hub posting; a rarer cue covers it
                cand.extend(v);
            }
        }
        if pet_query { for s in pets() { if let Some(v) = idx.get(s) { cand.extend(v); } } }
        let mut order: Vec<usize> = cand.into_iter().collect(); order.sort(); order
    }
}

// ---- inference tier: the emergence cortex, bundled (optional to use) ----

pub mod plastic;
pub mod router;
pub mod turn;
pub mod stream;   // line-splitting for piping app output into a scope (capture/run/follow)
pub mod preload;  // shared fact-pack reader for `neuron import` + the MCP preload boot hook
pub mod affect;   // the one shared mood + stance + humanize-directive layer (db.rs and wasm both use it)
pub mod caps;     // the capability manifest (grounded vs deferrable) — the polymorphism spine (§7)
#[cfg(feature = "sqlite")] pub mod db;
pub mod op;     // the one op vocabulary + apply() every transport routes through (std-only, generic over Store)
#[cfg(feature = "secure")] pub mod secure;
#[cfg(feature = "server")] pub mod server;
#[cfg(feature = "mcp")] pub mod mcp;
#[cfg(feature = "semantic")] pub mod semantic;
#[cfg(feature = "compress")] pub mod codec;
pub mod cortex;
pub mod bpe;
pub mod model;
#[cfg(feature = "cortex")] pub mod route;   // the shared recall→cortex-dispatch path (CLI/MCP/WASM parity)

pub mod wasm;
