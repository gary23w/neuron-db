//! Shared pack reader for the fact-preload feature. A *pack* is a list of declarative facts loaded
//! into neuron-db scopes — consumed identically by the CLI (`neuron import`) and the MCP boot hook,
//! and matching the line shapes the WASM `loadmany` op expects.
//!
//! Two text forms, one reader (auto-detected per line by the leading character):
//!   - `.jsonl` — one JSON object per line: `{"fact":"…","scope":"…"}` (scope optional)
//!   - `.facts` — one bare fact per line; a `# scope: <id>` line sets the scope for the lines after it
//!
//! A blank line or a plain `#` comment is skipped. A fact carrying a **tab or newline** is REJECTED:
//! those are field/record separators on the WASM wire protocol, so rejecting them keeps a pack
//! portable across every surface (the durable path would survive them, but we reject uniformly).
//! Std-only, no serde — a pack line is a single flat object, so a tiny extractor suffices.

/// The classification of one pack line.
#[derive(Debug, PartialEq)]
pub enum PackLine {
    /// a fact plus its per-line scope override, if the JSON object carried one
    Fact { scope: Option<String>, text: String },
    /// a `# scope: <id>` directive — sets the active scope for subsequent bare-fact lines
    Scope(String),
    /// blank line, plain `#` comment, or an empty/absent fact — skipped (counted, not stored)
    Skip,
    /// a fact carrying a tab or newline (a wire separator) — rejected (counted, not stored)
    Reject,
}

/// Extract a JSON string field's value, decoding the standard escapes. Mirrors the MCP server's
/// `json_field`; a pack line is one flat object so a flat scan is sufficient and std-only.
fn json_str_field(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let i = body.find(&pat)?;
    let rest: Vec<char> = body[i + pat.len()..].chars().collect();
    let colon = rest.iter().position(|&c| c == ':')?;
    let mut j = colon + 1;
    while j < rest.len() && rest[j] != '"' { j += 1; }      // advance to the opening quote
    if j >= rest.len() { return None; }
    j += 1;
    let mut out = String::new();
    while j < rest.len() {
        let c = rest[j];
        if c == '\\' && j + 1 < rest.len() {
            match rest[j + 1] {
                'n' => { out.push('\n'); j += 2; }
                't' => { out.push('\t'); j += 2; }
                'r' => { out.push('\r'); j += 2; }
                'b' => { out.push('\u{0008}'); j += 2; }
                'f' => { out.push('\u{000C}'); j += 2; }
                '"' => { out.push('"'); j += 2; }
                '\\' => { out.push('\\'); j += 2; }
                '/' => { out.push('/'); j += 2; }
                // \uXXXX — decode the 4 hex digits to a char (json_escape emits this for control
                // bytes, so failing to decode it would round-trip "" as the literal text "u0001")
                'u' if j + 6 <= rest.len() => {
                    let hex: String = rest[j + 2..j + 6].iter().collect();
                    match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                        Some(ch) => out.push(ch),
                        None => { out.push('\\'); out.push('u'); }
                    }
                    j += 6;
                }
                o => { out.push('\\'); out.push(o); j += 2; }   // unknown escape: keep both chars, don't drop the '\'
            }
        } else if c == '"' { return Some(out); } else { out.push(c); j += 1; }
    }
    Some(out)
}

fn classify_fact(scope: Option<String>, fact: &str) -> PackLine {
    let fact = fact.trim();
    if fact.is_empty() { return PackLine::Skip; }
    // reject ANY control char (tab/newline are wire separators; others like  don't survive the
    // json_escape→json_str_field round-trip cleanly) so a corruptible byte never enters the store
    if fact.chars().any(char::is_control) { return PackLine::Reject; }
    if scope.as_deref().is_some_and(|s| s.chars().any(char::is_control)) { return PackLine::Reject; }
    PackLine::Fact { scope, text: fact.to_string() }
}

/// Classify one raw pack line. JSONL (`{…}`) vs plain (`.facts`) is auto-detected by the leading
/// non-space character; `#` is a comment or a `# scope:` directive.
pub fn parse_pack_line(raw: &str) -> PackLine {
    // strip a UTF-8 BOM (Windows editors / PowerShell `Out-File -Encoding utf8` prepend one to the
    // first line) so it doesn't break the leading `{` / `#` detection
    let raw = raw.strip_prefix('\u{feff}').unwrap_or(raw);
    let line = raw.trim_end_matches(['\r', '\n']);
    let t = line.trim_start();
    if t.is_empty() { return PackLine::Skip; }
    if let Some(rest) = t.strip_prefix('#') {
        let r = rest.trim_start();
        if let Some(s) = r.strip_prefix("scope:").map(str::trim) {
            if !s.is_empty() { return PackLine::Scope(s.to_string()); }
        }
        return PackLine::Skip;                              // plain comment
    }
    if t.starts_with('{') {
        match json_str_field(line, "fact") {
            Some(fact) => classify_fact(json_str_field(line, "scope"), &fact),
            None => PackLine::Reject,                        // a `{`-line with no "fact" key — surfaced, not silently dropped
        }
    } else {
        classify_fact(None, line)                          // bare .facts line
    }
}

/// Resolve the target scope for a fact: per-fact JSON scope → active `# scope:` directive → the
/// caller's default. `None` only when no scope exists at any level (the caller errors on that).
pub fn resolve_scope<'a>(per_fact: &'a Option<String>, directive: &'a Option<String>, default: &'a Option<String>) -> Option<&'a str> {
    // a blank/whitespace scope at ANY level is treated as absent (don't route into a degenerate ""
    // scope) — it falls through to the next level instead.
    let nonblank = |o: &'a Option<String>| o.as_deref().filter(|s| !s.trim().is_empty());
    nonblank(per_fact).or_else(|| nonblank(directive)).or_else(|| nonblank(default))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn jsonl_basic() {
        assert_eq!(parse_pack_line(r#"{"scope":"s","fact":"the sky is blue"}"#),
                   PackLine::Fact { scope: Some("s".into()), text: "the sky is blue".into() });
        assert_eq!(parse_pack_line(r#"{"fact":"no scope here"}"#),
                   PackLine::Fact { scope: None, text: "no scope here".into() });
    }
    #[test] fn facts_and_directives() {
        assert_eq!(parse_pack_line("# scope: chem"), PackLine::Scope("chem".into()));
        assert_eq!(parse_pack_line("water is wet"), PackLine::Fact { scope: None, text: "water is wet".into() });
        assert_eq!(parse_pack_line("# just a comment"), PackLine::Skip);
        assert_eq!(parse_pack_line("   "), PackLine::Skip);
    }
    #[test] fn rejects_wire_separators() {
        assert_eq!(parse_pack_line("a\tfact with tab"), PackLine::Reject);
        assert_eq!(parse_pack_line(r#"{"fact":"line one\nline two"}"#), PackLine::Reject);  // \n decoded then rejected
    }
    #[test] fn empty_fact_skipped() {
        assert_eq!(parse_pack_line(r#"{"fact":"   "}"#), PackLine::Skip);
        assert_eq!(parse_pack_line(r#"{"scope":"s"}"#), PackLine::Reject);                   // no fact key -> surfaced
    }
    #[test] fn unicode_escapes_and_control_bytes() {
        // a valid \uXXXX escape decodes (not stored as the literal text "u00e9")
        let uni = format!(r#"{{"fact":"caf{}u00e9 au lait"}}"#, '\\');   // builds the literal é escape
        assert_eq!(parse_pack_line(&uni), PackLine::Fact { scope: None, text: "café au lait".into() });
        // a control byte (json_escape emits  for it) is rejected, never stored corruptibly
        assert_eq!(parse_pack_line(r#"{"fact":"badbyte here"}"#), PackLine::Reject);
    }
    #[test] fn scope_precedence() {
        let pf = Some("perfact".to_string()); let dir = Some("dir".to_string()); let def = Some("def".to_string());
        assert_eq!(resolve_scope(&pf, &dir, &def), Some("perfact"));
        assert_eq!(resolve_scope(&None, &dir, &def), Some("dir"));
        assert_eq!(resolve_scope(&None, &None, &def), Some("def"));
        assert_eq!(resolve_scope(&None, &None, &None), None);
        // a blank per-fact scope falls through instead of routing into ""
        assert_eq!(resolve_scope(&Some("  ".to_string()), &dir, &def), Some("dir"));
        assert_eq!(resolve_scope(&Some("".to_string()), &None, &def), Some("def"));
    }
}
