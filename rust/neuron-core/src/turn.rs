//! turn(): deterministic conversation routing over a Neuron. A statement is stored and
//! acknowledged; a question is answered from memory or with an honest "i don't know";
//! arithmetic (+ - * /, word forms too) is evaluated. std-only, no regex crate, wasm-safe.
//! Faithful port of neuron_db/turn.py.

use crate::{Neuron, content, words, is_num};

pub struct Turn { pub reply: String, pub kind: String, pub wrote: usize }
fn rr(reply: &str, kind: &str) -> Turn { Turn { reply: reply.into(), kind: kind.into(), wrote: 0 } }

const QWORDS: [&str; 7] = ["what","whats","where","who","when","how","which"];
const YNWORDS: [&str; 14] = ["am","is","are","do","does","did","can","could","will","would","was","were","have","has"];
const COMMANDS: [&str; 16] = ["count","explain","list","show","give","make","stop","repeat","calculate","compute","draw","sing","translate","define","describe","summarize"];
const GREET_ROOTS: [&str; 6] = ["hi","hey","hello","yo","sup","howdy"];
const GREETS: [&str; 4] = ["hey. tell me things; i'll remember them.","hi. what should i remember?",
    "ready. give me facts, ask them back later.","memory online. go ahead."];
const CHATR: [&str; 4] = ["noted.","okay.","got it.","if it matters, say it like a fact -- i'll keep it."];
const IDK: &str = "i don't know right now.";

fn lc(s: &str) -> String { s.to_lowercase() }
fn first_word(u: &str) -> String {
    u.split_whitespace().next().unwrap_or("").trim_matches(|c: char| "?,!".contains(c)).to_lowercase()
}
// squeeze a trailing run of the last char to one ("hiii"->"hi", "heyyy"->"hey")
fn squeeze_tail(s: &str) -> String {
    let mut v: Vec<char> = s.chars().collect();
    while v.len() > 2 && v[v.len()-1] == v[v.len()-2] { v.pop(); }
    v.into_iter().collect()
}
fn is_greet(u: &str) -> bool {
    let t = lc(u.trim());
    if t.starts_with("good morning") || t.starts_with("good afternoon") || t.starts_with("good evening") { return true; }
    let head = squeeze_tail(&first_word(&t));
    GREET_ROOTS.contains(&head.as_str())
}
fn starts_any(u: &str, pres: &[&str]) -> bool { let t = lc(u); pres.iter().any(|p| t.starts_with(p)) }
fn contains_any(u: &str, subs: &[&str]) -> bool { let t = lc(u); subs.iter().any(|s| t.contains(s)) }

// arithmetic: a space-delimited "<int> <op> <int>" (so dates like 2026-06-13 are NOT math)
fn find_math(s: &str) -> Option<(i64, char, i64)> {
    let mut t = format!(" {} ", lc(s));
    for (w, sym) in [("divided by"," / "),("plus"," + "),("minus"," - "),("times"," * ")] {
        t = t.replace(&format!(" {} ", w), sym);
    }
    let toks: Vec<&str> = t.split_whitespace().collect();
    for w in toks.windows(3) {
        if let (Ok(a), Ok(b)) = (w[0].parse::<i64>(), w[2].parse::<i64>()) {
            if w[1].len() == 1 {
                let op = w[1].chars().next().unwrap();
                if "+-*/".contains(op) { return Some((a, op, b)); }
            }
        }
    }
    None
}

pub fn turn(n: &mut Neuron, u: &str) -> Turn {
    let u = u.trim();
    let uw = words(u);
    let first = first_word(u);
    let mut questionish = u.contains('?') || QWORDS.contains(&first.as_str());
    let about = (uw.contains("your") || uw.contains("you") || uw.contains("yours"))
        && !(uw.contains("my") || uw.contains("i") || uw.contains("im") || uw.contains("mine"));
    let yn = YNWORDS.contains(&first.as_str()) || lc(u).contains("yes or no");
    if yn { questionish = true; }

    if about && contains_any(u, &["what's your name","what is your name","whats your name","who are you"]) {
        return rr("i'm a neuron -- an associative memory. you're the one i remember things about.", "self");
    }
    if about && contains_any(u, &["real person","human","alive","robot"," ai ","sentient","a machine"]) {
        return rr("i'm a tiny memory, not a person. but i'll remember what you tell me.", "self");
    }
    if is_greet(u) { return rr(GREETS[n.fact_count() % GREETS.len()], "smalltalk"); }
    if starts_any(u, &["lol","lmao","haha","hehe"]) { return rr("glad that landed.", "smalltalk"); }
    if starts_any(u, &["thanks","thank you","thx","ty "]) || lc(u) == "ty" { return rr("anytime.", "smalltalk"); }
    if starts_any(u, &["bye","goodbye","later","gtg","good night","goodnight"]) { return rr("later. i'll remember.", "smalltalk"); }
    if questionish && contains_any(u, &["how's it going","how is it going","how are you","how you doing","how are things"]) {
        return rr("running fine -- all of it yours.", "smalltalk");
    }
    if COMMANDS.contains(&first.as_str()) && !(uw.contains("my")||uw.contains("i")||uw.contains("im")||uw.contains("mine")) {
        return rr("i can't do that -- i remember facts and do arithmetic.", "smalltalk");
    }

    if !lc(u).contains('=') {
        if let Some((a, op, b)) = find_math(u) {
            let reply = match op {
                '+' => Some(format!("{} + {} = {}", a, b, a + b)),
                '-' => Some(format!("{} - {} = {}", a, b, a - b)),
                '*' => Some(format!("{} * {} = {}", a, b, a * b)),
                '/' => if b != 0 { Some(format!("{} / {} = {}", a, b, (((a as f64 / b as f64) * 1e6).round() / 1e6))) } else { None },
                _ => None,
            };
            if let Some(rep) = reply { return rr(&rep, "math"); }
        }
    }

    if questionish {
        let cw = content(u);
        let hit = if !cw.is_empty() { n.recall(u) }
                  else if lc(u).contains("who am i") { n.recall("name") }
                  else { None };
        if yn && !cw.is_empty() {
            return match &hit {
                Some(h) if h.coverage >= 0.99 => rr("yes.", "recall"),
                Some(h) => rr(&format!("hmm -- what i remember is: {}", h.fact), "recall"),
                None => rr("not that i know.", "recall"),
            };
        }
        if let Some(h) = &hit {
            let v = &h.value;
            let cap = v.chars().next().map_or(false, |c| c.is_uppercase());
            let conf = is_num(v) || (cap && h.fact.split_whitespace().count() <= 6);
            if conf && !h.echo { return rr(&format!("{}.", v), "recall"); }
            if h.echo || h.fact.split_whitespace().count() > 6 {
                let q = h.fact.trim();
                let shown = if q.chars().count() <= 140 { q.to_string() }
                            else { format!("{}\u{2026}", q.chars().take(137).collect::<String>()) };
                return rr(&format!("what i remember: \"{}\"", shown), "recall");
            }
            return rr(&format!("{}.", v), "recall");
        }
        if !cw.is_empty() { return rr(IDK, "idk"); }
    }

    let wrote = n.observe(u);
    if wrote > 0 {
        let val = n.episodes.last().map(|e| e.v.clone()).unwrap_or_default();
        let reply = if wrote > 1 { format!("noted -- {} facts.", wrote) }
            else { match n.fact_count() % 3 { 0 => format!("got it -- {}.", val), 1 => format!("noted -- {}.", val), _ => format!("okay -- {}.", val) } };
        return Turn { reply, kind: "ack".into(), wrote };
    }
    rr(CHATR[u.len() % CHATR.len()], "smalltalk")
}
