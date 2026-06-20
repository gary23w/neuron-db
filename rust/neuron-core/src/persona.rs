//! The opt-in "polymorphic mind" — a `Persona` (Big Five traits + an innate `Temperament` + values)
//! that MODULATES the affective and learning layers. Feature-gated behind `personality` AND inert until
//! a Persona is explicitly attached to a scope: the default store stays a neutral memory with no point of
//! view. Some users would rather the AI have no personality at all, so this is never on implicitly.
//!
//! Nothing here is a neural net — the traits are five scalars that turn the dials the affective layer
//! (`affect.rs`) already exposes (`STANCE_BUMP` / `STANCE_DECAY` / `STANCE_FLOOR` / `STANCE_THRESHOLD`)
//! and shapes the prose of the humanize directive. A neutral persona (all 0.5, reactivity 1.0) reproduces
//! the un-modulated behavior exactly, so "personality on but average" == "no personality" by construction.

use crate::affect;

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

/// The Big Five (OCEAN) dispositional axes, each in `0.0..=1.0` where `0.5` is average/neutral.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BigFive {
    pub openness: f32,           // imagination, curiosity, willingness to revise a view
    pub conscientiousness: f32,  // self-discipline, organization, goal persistence
    pub extraversion: f32,       // energy / expressiveness drawn from stimulation
    pub agreeableness: f32,      // warmth, empathy, cooperation, trust
    pub neuroticism: f32,        // emotional reactivity — stress, anxiety, mood swings (low = stable)
}

impl Default for BigFive {
    /// The average person: every trait at the midpoint. A persona built from this changes nothing.
    fn default() -> Self {
        BigFive { openness: 0.5, conscientiousness: 0.5, extraversion: 0.5, agreeableness: 0.5, neuroticism: 0.5 }
    }
}

impl BigFive {
    pub fn clamped(self) -> Self {
        BigFive {
            openness: self.openness.clamp(0.0, 1.0),
            conscientiousness: self.conscientiousness.clamp(0.0, 1.0),
            extraversion: self.extraversion.clamp(0.0, 1.0),
            agreeableness: self.agreeableness.clamp(0.0, 1.0),
            neuroticism: self.neuroticism.clamp(0.0, 1.0),
        }
    }
    /// Pull each live trait a fraction `rate` toward another vector (used to drift back toward temperament).
    pub fn toward(self, anchor: BigFive, rate: f32) -> BigFive {
        let r = rate.clamp(0.0, 1.0);
        BigFive {
            openness: lerp(self.openness, anchor.openness, r),
            conscientiousness: lerp(self.conscientiousness, anchor.conscientiousness, r),
            extraversion: lerp(self.extraversion, anchor.extraversion, r),
            agreeableness: lerp(self.agreeableness, anchor.agreeableness, r),
            neuroticism: lerp(self.neuroticism, anchor.neuroticism, r),
        }
    }
}

/// Temperament: the innate, biologically-given baseline a mind is "born with". It anchors the live traits
/// (experience nudges them, but they drift back toward this floor) and sets the raw emotional-reactivity
/// gain applied on top of Neuroticism. This is the part that does not change — the constitution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Temperament {
    pub baseline: BigFive,
    pub reactivity: f32,   // 0.0..=2.0 global gain on how hard events move mood/stance (1.0 = neutral)
}

impl Default for Temperament {
    fn default() -> Self { Temperament { baseline: BigFive::default(), reactivity: 1.0 } }
}

/// The presentation knobs OCEAN hands to the affective layer to color the humanize directive. Plain
/// scalars so `affect.rs` stays personality-agnostic — it consumes a `Style`, it never imports a Persona.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    pub energy: f32,      // extraversion — terse+calm .. animated+expansive
    pub warmth: f32,      // agreeableness — blunt+guarded .. warm+cooperative
    pub volatility: f32,  // neuroticism — even-keeled .. reactive+intense
    pub curiosity: f32,   // openness — concrete+literal .. exploratory+associative
    pub structure: f32,   // conscientiousness — loose .. orderly+deliberate
}

/// The attached mind. `traits` is the live (drifting) state; `temperament` is the innate anchor; `values`
/// are canonical morals/meaning (name -> weight) that bias judgment.
#[derive(Clone, Debug, PartialEq)]
pub struct Persona {
    pub traits: BigFive,
    pub temperament: Temperament,
    pub values: Vec<(String, f32)>,
}

impl Default for Persona {
    fn default() -> Self { Persona { traits: BigFive::default(), temperament: Temperament::default(), values: Vec::new() } }
}

impl Persona {
    /// An explicitly neutral mind — behaves identically to having no persona at all.
    pub fn neutral() -> Self { Persona::default() }

    /// Build from Big Five traits, taking those as both the live state and the innate baseline (reactivity 1.0).
    pub fn from_traits(traits: BigFive) -> Self {
        let traits = traits.clamped();
        Persona { traits, temperament: Temperament { baseline: traits, reactivity: 1.0 }, values: Vec::new() }
    }

    // --- the dials OCEAN turns on the stance dynamics (base = the affect.rs constant) ---

    /// Reinforcement bump per re-statement of a topic. Reactive minds (high Neuroticism) and a hot innate
    /// temperament spike harder; even-keeled minds barely move. Neutral persona => `base` unchanged.
    pub fn stance_bump(&self, base: f32) -> f32 {
        (base * self.temperament.reactivity * (0.5 + self.traits.neuroticism)).max(0.0)
    }

    /// Multiplicative decay factor for stances NOT being revisited (higher = stickier). Rumination
    /// (Conscientiousness + Neuroticism) makes old feelings linger; low values let them fade fast.
    pub fn stance_decay(&self, base: f32) -> f32 {
        let rumination = (self.traits.conscientiousness + self.traits.neuroticism) * 0.5; // 0..1, 0.5 neutral
        (base + (rumination - 0.5) * 0.2).clamp(0.5, 0.99)
    }

    /// Strength at which a budding stance becomes a hardened, unprompted view. Open minds demand more
    /// evidence before committing (and revise easily); rigid (low-Openness) minds harden quickly.
    pub fn stance_threshold(&self, base: f32) -> f32 {
        base * (0.7 + 0.6 * self.traits.openness)
    }

    /// Floor a neglected stance decays toward — Neuroticism keeps a residue of old feeling around.
    pub fn stance_floor(&self, base: f32) -> f32 {
        (base + (self.traits.neuroticism - 0.5) * 0.4).clamp(0.0, base.max(0.9))
    }

    /// Habit reinforcement gain on the plastic layer — Conscientiousness builds stickier habits.
    pub fn habit_gain(&self, base: f64) -> f64 {
        base * (0.5 + self.traits.conscientiousness as f64)
    }

    /// The presentation style the affect layer uses to color the directive prose.
    pub fn style(&self) -> Style {
        Style {
            energy: self.traits.extraversion,
            warmth: self.traits.agreeableness,
            volatility: self.traits.neuroticism,
            curiosity: self.traits.openness,
            structure: self.traits.conscientiousness,
        }
    }

    /// One step of experience: the live traits drift a little toward the innate temperament baseline (the
    /// constitution reasserting itself), so transient pushes fade unless reinforced. Cheap, O(1).
    pub fn relax_toward_temperament(&mut self, rate: f32) {
        self.traits = self.traits.toward(self.temperament.baseline, rate).clamped();
    }

    // --- durable round-trip as flat (key, value) pairs (the caller persists via var_set or any backend) ---

    pub fn to_pairs(&self) -> Vec<(String, String)> {
        let t = &self.traits; let b = &self.temperament.baseline;
        let mut v = vec![
            ("openness".into(), format!("{:.4}", t.openness)),
            ("conscientiousness".into(), format!("{:.4}", t.conscientiousness)),
            ("extraversion".into(), format!("{:.4}", t.extraversion)),
            ("agreeableness".into(), format!("{:.4}", t.agreeableness)),
            ("neuroticism".into(), format!("{:.4}", t.neuroticism)),
            ("base_openness".into(), format!("{:.4}", b.openness)),
            ("base_conscientiousness".into(), format!("{:.4}", b.conscientiousness)),
            ("base_extraversion".into(), format!("{:.4}", b.extraversion)),
            ("base_agreeableness".into(), format!("{:.4}", b.agreeableness)),
            ("base_neuroticism".into(), format!("{:.4}", b.neuroticism)),
            ("reactivity".into(), format!("{:.4}", self.temperament.reactivity)),
        ];
        for (name, w) in &self.values { v.push((format!("value::{name}"), format!("{w:.4}"))); }
        v
    }

    pub fn from_pairs(pairs: &[(String, String)]) -> Persona {
        let get = |k: &str, d: f32| pairs.iter().find(|(n, _)| n == k).and_then(|(_, v)| v.parse().ok()).unwrap_or(d);
        let traits = BigFive {
            openness: get("openness", 0.5), conscientiousness: get("conscientiousness", 0.5),
            extraversion: get("extraversion", 0.5), agreeableness: get("agreeableness", 0.5),
            neuroticism: get("neuroticism", 0.5),
        }.clamped();
        let baseline = BigFive {
            openness: get("base_openness", traits.openness), conscientiousness: get("base_conscientiousness", traits.conscientiousness),
            extraversion: get("base_extraversion", traits.extraversion), agreeableness: get("base_agreeableness", traits.agreeableness),
            neuroticism: get("base_neuroticism", traits.neuroticism),
        }.clamped();
        let values = pairs.iter().filter_map(|(n, v)| n.strip_prefix("value::").map(|name| (name.to_string(), v.parse().unwrap_or(0.0)))).collect();
        Persona { traits, temperament: Temperament { baseline, reactivity: get("reactivity", 1.0).clamp(0.0, 2.0) }, values }
    }

    /// A one-line human summary of the dominant traits, for `persona show` / the directive header.
    pub fn describe(&self) -> String {
        let t = &self.traits;
        let axis = |hi: &str, lo: &str, v: f32| -> Option<String> {
            if v >= 0.66 { Some(hi.to_string()) } else if v <= 0.34 { Some(lo.to_string()) } else { None }
        };
        let parts: Vec<String> = [
            axis("curious", "conventional", t.openness),
            axis("disciplined", "easygoing", t.conscientiousness),
            axis("outgoing", "reserved", t.extraversion),
            axis("warm", "blunt", t.agreeableness),
            axis("volatile", "even-keeled", t.neuroticism),
        ].into_iter().flatten().collect();
        if parts.is_empty() { "balanced, even-tempered".into() } else { parts.join(", ") }
    }
}

/// The base stance constants a neutral persona reproduces — re-exported so callers modulate from one source.
pub fn base_bump() -> f32 { affect::STANCE_BUMP }
pub fn base_decay() -> f32 { affect::STANCE_DECAY }
pub fn base_floor() -> f32 { affect::STANCE_FLOOR }
pub fn base_threshold() -> f32 { affect::STANCE_THRESHOLD }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_persona_changes_nothing() {
        let p = Persona::neutral();
        assert!((p.stance_bump(affect::STANCE_BUMP) - affect::STANCE_BUMP).abs() < 1e-6, "neutral bump must equal base");
        assert!((p.stance_decay(affect::STANCE_DECAY) - affect::STANCE_DECAY).abs() < 1e-6, "neutral decay must equal base");
        assert!((p.stance_threshold(affect::STANCE_THRESHOLD) - affect::STANCE_THRESHOLD).abs() < 1e-6, "neutral threshold must equal base");
    }

    #[test]
    fn neuroticism_drives_reactivity() {
        let calm = Persona::from_traits(BigFive { neuroticism: 0.0, ..Default::default() });
        let hot  = Persona::from_traits(BigFive { neuroticism: 1.0, ..Default::default() });
        assert!(hot.stance_bump(1.0) > calm.stance_bump(1.0), "a neurotic mind must spike harder");
        // a hot temperament amplifies further
        let mut hotter = hot.clone(); hotter.temperament.reactivity = 2.0;
        assert!(hotter.stance_bump(1.0) > hot.stance_bump(1.0));
    }

    #[test]
    fn openness_raises_the_harden_threshold() {
        let rigid = Persona::from_traits(BigFive { openness: 0.0, ..Default::default() });
        let open  = Persona::from_traits(BigFive { openness: 1.0, ..Default::default() });
        assert!(open.stance_threshold(1.5) > rigid.stance_threshold(1.5), "open minds need more evidence to harden a view");
    }

    #[test]
    fn conscientiousness_builds_stickier_habits_and_slower_decay() {
        let lax  = Persona::from_traits(BigFive { conscientiousness: 0.0, neuroticism: 0.0, ..Default::default() });
        let firm = Persona::from_traits(BigFive { conscientiousness: 1.0, neuroticism: 1.0, ..Default::default() });
        assert!(firm.habit_gain(1.0) > lax.habit_gain(1.0));
        assert!(firm.stance_decay(0.9) > lax.stance_decay(0.9), "rumination keeps old feelings around longer");
    }

    #[test]
    fn drifts_back_toward_temperament() {
        let mut p = Persona::from_traits(BigFive { neuroticism: 0.2, ..Default::default() });
        p.traits.neuroticism = 0.9; // a transient spike away from the baseline 0.2
        p.relax_toward_temperament(0.5);
        assert!(p.traits.neuroticism < 0.9 && p.traits.neuroticism > 0.2, "should relax toward, not past, the baseline");
    }

    #[test]
    fn pairs_round_trip() {
        let mut p = Persona::from_traits(BigFive { openness: 0.8, conscientiousness: 0.3, extraversion: 0.7, agreeableness: 0.9, neuroticism: 0.6 });
        p.temperament.reactivity = 1.4;
        p.values = vec![("fairness".into(), 0.9), ("truth over comfort".into(), 0.7)];
        let back = Persona::from_pairs(&p.to_pairs());
        assert_eq!(back, p, "persona must survive a durable round-trip");
    }

    #[test]
    fn describe_reads_the_dominant_axes() {
        let p = Persona::from_traits(BigFive { openness: 0.9, agreeableness: 0.1, ..Default::default() });
        let d = p.describe();
        assert!(d.contains("curious") && d.contains("blunt"), "got {d}");
    }
}
