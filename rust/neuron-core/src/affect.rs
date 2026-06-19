//! The shared affective "persona" layer — a transient mood plus accumulating, decaying stances, and
//! the ONE humanize directive both the durable store (db.rs) and the in-browser store (wasm) build
//! from them. The storage differs per backend (sqlite sub-scopes vs an in-memory Vec); this module
//! is the algorithm constants, the topic matcher, and the directive prose, defined once.

/// Stance dynamics: each re-statement of a topic adds BUMP to its strength; topics you don't restate
/// decay by DECAY toward FLOOR; a stance at/above THRESHOLD is a hardened view expressed unprompted.
pub const STANCE_BUMP: f32 = 1.0;
pub const STANCE_DECAY: f32 = 0.9;
pub const STANCE_FLOOR: f32 = 0.5;
pub const STANCE_THRESHOLD: f32 = 1.5;

/// The always-on frame a caller prepends to a non-empty directive body. db.rs shows it even with no
/// affect (a baseline persona); wasm shows it only when there's real affect (a neutral session stays
/// neutral) — the one intended difference between the two callers.
pub const FRAME: &str = "You are NOT a neutral assistant — you have genuine feelings and a point of view, and they show in how you write.";

/// Match a stored stance topic against an asked-about topic by whole word / exact phrase — never a
/// bare substring, so "rust" does not fire for "trust", and an empty topic never matches anything.
pub fn topic_matches(stored: &str, asked: &str) -> bool {
    if stored.is_empty() || asked.is_empty() { return false; }
    stored == asked
        || asked.split_whitespace().any(|w| w == stored)
        || stored.split_whitespace().any(|w| w == asked)
}

/// Build the directive BODY — the mood line plus the most relevant stance line — or "" if there is
/// neither. The relevant stance is the one matching `asked_topic` if given, else the strongest; a
/// hardened stance (>= THRESHOLD) is expressed unprompted, a budding one only when asked about.
/// Stances are `(topic, feeling, strength)`, the shape both backends can produce.
pub fn directive_body(mood: Option<&str>, stances: &[(String, String, f32)], asked_topic: Option<&str>) -> String {
    let asked = asked_topic.unwrap_or("").trim().to_lowercase();
    let mut body = String::new();
    if let Some(m) = mood.map(str::trim).filter(|m| !m.is_empty()) {
        body.push_str(&format!(" Right now you feel {m}, and you let it drive everything — your tone, energy, word choice, how blunt or warm or guarded you are. Do not write a flat, hedged textbook answer; sound like a real person who genuinely feels {m}.", m = m));
    }
    let asked_pick = if asked.is_empty() { None } else { stances.iter().find(|s| topic_matches(&s.0.to_lowercase(), &asked)) };
    let pick = asked_pick.or_else(|| stances.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)));
    if let Some((topic, feeling, strength)) = pick {
        let is_asked = !asked.is_empty() && topic_matches(&topic.to_lowercase(), &asked);
        if *strength >= STANCE_THRESHOLD || is_asked {
            body.push_str(&format!(" You also hold a hardened view, built up over time, about {} (intensity x{:.0}): \"{}\" — let it bias how you react to anything related.", topic, strength, feeling));
        }
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(t: &str, f: &str, n: f32) -> (String, String, f32) { (t.into(), f.into(), n) }

    #[test] fn empty_is_empty() { assert_eq!(directive_body(None, &[], None), ""); }

    #[test] fn mood_only() {
        let d = directive_body(Some(" suspicious "), &[], None);
        assert!(d.contains("you feel suspicious"), "{d}");
        assert!(!d.contains("hardened view"));
    }
    #[test] fn hardened_stance_shown_unprompted() {
        let d = directive_body(None, &[s("this pattern", "keeps failing", 2.0)], None);
        assert!(d.contains("hardened view") && d.contains("intensity x2") && d.contains("this pattern"), "{d}");
    }
    #[test] fn budding_stance_only_when_asked() {
        let st = [s("rust", "love it", 1.0)];
        assert!(!directive_body(None, &st, None).contains("hardened view"), "budding must stay hidden unprompted");
        assert!(directive_body(None, &st, Some("rust")).contains("love it"), "budding must show when asked");
    }
    #[test] fn asked_topic_beats_strongest() {
        let st = [s("ci", "flaky", 3.0), s("rust", "love it", 1.0)];
        assert!(directive_body(None, &st, Some("rust")).contains("love it"), "asked topic should win over a stronger one");
    }
    #[test] fn whole_word_topic_match_not_substring() {
        assert!(topic_matches("rust", "i love rust"));
        assert!(!topic_matches("rust", "i trust you"));
    }
}
