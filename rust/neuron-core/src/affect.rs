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

/// Presentation knobs the OPTIONAL personality layer (persona.rs) hands in to color the directive prose,
/// each in `0.0..=1.0` with `0.5` neutral. Defined here so this module stays personality-agnostic — it
/// consumes a `Style`, it never imports a `Persona`. A neutral/`None` style adds nothing, so the base
/// directive is unchanged when no persona is attached.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    pub energy: f32,      // extraversion: spare+measured .. animated+expansive
    pub warmth: f32,      // agreeableness: blunt+guarded .. warm+generous
    pub volatility: f32,  // neuroticism: even-keeled .. reactions run hot
    pub curiosity: f32,   // openness: concrete+literal .. chases tangents
    pub structure: f32,   // conscientiousness: loose+spontaneous .. organized+deliberate
}

fn join_and(parts: &[&str]) -> String {
    match parts.len() {
        0 => String::new(),
        1 => parts[0].to_string(),
        2 => format!("{} and {}", parts[0], parts[1]),
        _ => format!("{}, and {}", parts[..parts.len() - 1].join(", "), parts[parts.len() - 1]),
    }
}

/// The "voice" sentence OCEAN adds — only the axes that are notably high/low (so an average persona is
/// silent and reproduces the base directive exactly).
fn style_line(s: &Style) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if s.energy >= 0.66 { parts.push("animated and expansive"); } else if s.energy <= 0.34 { parts.push("spare and measured"); }
    if s.warmth >= 0.66 { parts.push("warm and generous"); } else if s.warmth <= 0.34 { parts.push("blunt and guarded"); }
    if s.curiosity >= 0.66 { parts.push("curious, chasing tangents and connecting ideas"); } else if s.curiosity <= 0.34 { parts.push("concrete and literal"); }
    if s.structure >= 0.66 { parts.push("organized and deliberate"); } else if s.structure <= 0.34 { parts.push("loose and spontaneous"); }
    let mut line = if parts.is_empty() { String::new() } else { format!(" Your voice is {}.", join_and(&parts)) };
    if s.volatility >= 0.66 { line.push_str(" Your reactions run hot and shift quickly."); }
    else if s.volatility <= 0.34 { line.push_str(" You stay even-keeled, slow to rattle."); }
    line
}

/// Build the directive BODY — the mood line plus the most relevant stance line — or "" if there is
/// neither. The relevant stance is the one matching `asked_topic` if given, else the strongest; a
/// hardened stance (>= THRESHOLD) is expressed unprompted, a budding one only when asked about.
/// Stances are `(topic, feeling, strength)`, the shape both backends can produce.
pub fn directive_body(mood: Option<&str>, stances: &[(String, String, f32)], asked_topic: Option<&str>) -> String {
    directive_body_styled(mood, stances, asked_topic, None, STANCE_THRESHOLD)
}

/// As `directive_body`, but the optional personality layer passes a `Style` (to color the voice) and a
/// persona-modulated `threshold` (Openness raises the bar at which a budding stance hardens). With
/// `style = None` and `threshold = STANCE_THRESHOLD` this is byte-identical to `directive_body`.
pub fn directive_body_styled(mood: Option<&str>, stances: &[(String, String, f32)], asked_topic: Option<&str>, style: Option<&Style>, threshold: f32) -> String {
    let asked = asked_topic.unwrap_or("").trim().to_lowercase();
    let mut body = String::new();
    if let Some(m) = mood.map(str::trim).filter(|m| !m.is_empty()) {
        body.push_str(&format!(" Right now you feel {m}, and you let it drive everything — your tone, energy, word choice, how blunt or warm or guarded you are. Do not write a flat, hedged textbook answer; sound like a real person who genuinely feels {m}.", m = m));
    }
    let asked_pick = if asked.is_empty() { None } else { stances.iter().find(|s| topic_matches(&s.0.to_lowercase(), &asked)) };
    let pick = asked_pick.or_else(|| stances.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)));
    if let Some((topic, feeling, strength)) = pick {
        let is_asked = !asked.is_empty() && topic_matches(&topic.to_lowercase(), &asked);
        if *strength >= threshold || is_asked {
            body.push_str(&format!(" You also hold a hardened view, built up over time, about {} (intensity x{:.0}): \"{}\" — let it bias how you react to anything related.", topic, strength, feeling));
        }
    }
    if let Some(s) = style { body.push_str(&style_line(s)); }
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

    fn neutral_style() -> Style { Style { energy: 0.5, warmth: 0.5, volatility: 0.5, curiosity: 0.5, structure: 0.5 } }

    #[test] fn neutral_style_reproduces_base_directive() {
        let st = [s("ci", "flaky", 2.0)];
        let base = directive_body(Some("tired"), &st, None);
        let styled = directive_body_styled(Some("tired"), &st, None, Some(&neutral_style()), STANCE_THRESHOLD);
        assert_eq!(base, styled, "an average persona must add nothing");
    }

    #[test] fn style_colors_the_voice() {
        let warm_open = Style { energy: 0.9, warmth: 0.9, volatility: 0.9, curiosity: 0.9, structure: 0.5 };
        let d = directive_body_styled(None, &[], None, Some(&warm_open), STANCE_THRESHOLD);
        assert!(d.contains("animated") && d.contains("warm") && d.contains("curious") && d.contains("run hot"), "{d}");
        let cold_blunt = Style { energy: 0.1, warmth: 0.1, volatility: 0.1, curiosity: 0.1, structure: 0.1 };
        let d2 = directive_body_styled(None, &[], None, Some(&cold_blunt), STANCE_THRESHOLD);
        assert!(d2.contains("spare") && d2.contains("blunt") && d2.contains("even-keeled"), "{d2}");
    }

    #[test] fn openness_threshold_gates_unprompted_hardening() {
        let st = [s("climate", "furious", 1.2)];
        // a low (rigid) threshold lets the 1.2-strength view harden unprompted; a high (open) one hides it
        assert!(directive_body_styled(None, &st, None, None, 1.0).contains("hardened view"), "low threshold should surface it");
        assert!(!directive_body_styled(None, &st, None, None, 2.0).contains("hardened view"), "high threshold should keep it budding");
    }
}
