# The two LDAs — Fisher discriminants and latent topics

> **Status: IMPLEMENTED** (features `fisher` / `topics`, off by default). Design deltas found
> during the build and the measured numbers are in **[What shipped](#what-shipped-measured)**
> at the end; the rest of this document is the proposal as designed.

Two classical statistical methods, one acronym, and a store that already holds everything
they need:

- **Linear Discriminant Analysis** (Fisher, two-class): find the one direction in the
  semantic space that best separates two classes of facts, and score along it.
- **Latent Dirichlet Allocation**: find the abstract topics a scope's facts are drawn
  from, and index facts by topic.

Both are **ranking-and-inspection tiers**: they re-order and describe what recall already
surfaces, and they never gate truth, never mint a fact, never touch the lexical fast path.
Both are pure std, zero dependencies, deterministic, and feature-gated off by default —
the same contract as `semantic`, `trust`, and `quantum`.

## The gap this closes

Three measured gaps, all documented elsewhere in this guide:

1. **The fuzzy path is window-blind.** `recall_blended` / the semantic fallback rank only
   the most-recent 4,000 facts (`BLENDED_CAP`) because ranking is O(candidates). A
   paraphrase whose target sits 5,000 facts back is unreachable by meaning today —
   SEMANTIC.md lists "index the semantic fallback" as the highest-leverage remaining work.
   A topic index is that index: recall by meaning becomes **scope-wide** at bounded cost,
   with no ANN structure and no new dependency.
2. **Outcome learning is per-class scalar only.** The trust ledger learns *which
   tag-classes* precede grounded gain, but it is blind to *what helpful facts look like*.
   A two-class discriminant (helped vs hurt) learns the **direction in meaning space**
   that grounded gain points along — trust generalized from a lookup table to a geometry,
   under the same learning-rule guardrails (bounded, relaxes, earns only from outcomes).
3. **No abstract view of a scope.** `stats` counts facts; `semantic_neighbors` inspects
   one word. Nothing answers "what is this scope about" or "how much of this question
   does the store's subject matter even cover" — the knowledge-gap signal (STATUS.md)
   currently measures lexical hit density, not topical coverage. Topics answer both.

## What the store already provides

No new data collection is needed; every input is already resident:

| need | already there |
|---|---|
| per-fact bag of words | `Episode.raw` — sorted, interned content words |
| document order | insertion order (`neighbors`/`Passage` stitching relies on it) |
| a shared vector space | `SemanticSpace` — 256-d Random Indexing, `embed()` per fact, int8 cache |
| class labels | `trust::class_of` (open tag taxonomy), scope names, strengthen/reward events |
| an eigen-solver idiom | power iteration + deflation (`SemanticSpace::project`), k-means, quantize |
| deterministic hashing | `fnv` / `splitmix` |
| rank fusion | `recall_blended`'s Reciprocal Rank Fusion — a new signal is just a third rank list |
| lazy-refresh idiom | `emb_cache` epoch bound, `ensure_index` dirty flag |
| side-table persistence | `trust_kv` / `quantum_kv` — lazily created, schema untouched when unused |

## Design

Two modules, two features, layered exactly like the quantum tier: a std-only core generic
over plain vectors/tokens, with the durable wiring compiled only when the store has it.

```
fisher = []   # two-class Fisher discriminant heads; ranking + inspection only
topics = []   # latent Dirichlet allocation over fact word-bags; topic index + inspection
```

`src/fisher.rs` and `src/topics.rs` compile with no other feature. The `NeuronDB` hooks
are `#[cfg(all(feature = "fisher", feature = "semantic"))]` (the discriminant consumes
embeddings) and `#[cfg(all(feature = "topics", feature = "sqlite"))]` for persistence.
A build without the features is byte-identical in schema and ranking.

### 1. `fisher.rs` — the two-class discriminant head

The classical closed form, kept deliberately two-class (one linear solve, no
eigenproblem, no linalg crate):

```
w   ∝  S_λ⁻¹ (μ₊ − μ₋)                      the discriminant direction
c   =  w·(μ₊ + μ₋)/2 + ln(n₋/n₊)            the threshold (priors folded in)
z(x)=  (w·x − c)  with  wᵀ S_λ w = 1        score in within-class sigma units
S_λ =  (1−λ) S_w + λ (tr S_w / d) I         shrinkage toward the scaled identity
```

Shrinkage is what makes this safe in a 256-d space where a class may hold eight samples:
`S_λ` is positive-definite by construction, so the Cholesky solve always succeeds and the
head degrades toward a nearest-mean classifier rather than exploding.

**State is class-agnostic and O(d²) once, not per class.** The head maintains one global
second-moment matrix `M2 = Σ x xᵀ` (packed upper triangle, f64 accumulators, ~263 KB at
d = 256) plus per-class first moments `(Σx, n)` (~2 KB per class, open taxonomy exactly
like the trust ledger). `S_w` for *any* class pair materializes by subtraction, so one
head answers every pairing — helped-vs-hurt, scope-vs-rest, tag-vs-tag — without storing
a matrix per pair.

```rust
pub struct FisherHead {
    dim: usize,
    m2: Vec<f64>,                       // packed upper triangle of Σ x xᵀ
    n: f64,
    classes: HashMap<String, ClassMoment>,  // open taxonomy; nothing privileged a priori
    cached: Option<Axis>,               // lazy: recomputed when counts double (emb_cache idiom)
}
impl FisherHead {
    pub fn observe_labeled(&mut self, class: &str, x: &[f32]);       // rank-1 update, O(d²/2)
    pub fn axis(&mut self, pos: &str, neg: &str) -> Option<&Axis>;   // None until both n ≥ 8
    pub fn score(axis: &Axis, x: &[f32]) -> f32;                     // z, clamped to ±Z_CAP
    pub fn dump(&self) -> String;  pub fn load(blob: &str) -> Self;  // tab-line format
}
```

**Where labels come from — live signals only, nothing hardcoded.** The head never names a
class; classes arrive from the same places the trust ledger's do:

- **The outcome axis.** When `strengthen` fires or the trust ledger is rewarded, the
  recalled fact's embedding is observed as `+` (delta > 0) or `−` (delta < 0). Same
  event stream trust consumes; volume-deduped per round the same way. This axis is
  "what helped minus what hurt", as a direction.
- **Scope moments.** At observe time, 1-in-16 sampled facts contribute their embedding
  under their scope's class name. Any scope-vs-rest axis is then available on demand —
  "does this fact even look like this scope" — which is pollution triage as a score,
  not a rule.
- Any tag-class `class_of` extracts is equally usable; the taxonomy stays open.

**The learning-rule guardrails carry over from trust.rs verbatim in spirit:** the head is
inert (returns `None`, contributes nothing) until both classes clear a sample floor;
scores are clamped at the read (`Z_CAP`, the `STRENGTH_CAP` idiom); moments decay by an
exponential forgetting factor on each update so the axis tracks drift and cannot lock in;
and there is no path by which restatement moves it — only grounded outcome events feed
the outcome axis.

**Consumers (all fail-open):**

- `recall_blended` gains a **third RRF list**: candidates ranked by `z` on the outcome
  axis join the lexical and semantic rankings in the existing fusion sum. Head inert →
  list absent → ranking byte-identical to today. RRF is why this composes cleanly: the
  discriminant contributes rank evidence, never a raw score on a mismatched scale.
- A store-time **hint** (never a block): an incoming fact scoring strongly negative on
  its scope-vs-rest axis is flagged in the op result for the caller to act on.
- `fisher_axis` inspection: the nearest vocabulary words to +w and −w (reusing
  `SemanticSpace::nearest`'s cosine machinery) — a human-readable answer to "what does
  the helpful direction look like".

### 2. `topics.rs` — latent Dirichlet allocation

Collapsed Gibbs sampling over the store's own word-bags, with the standard conditional

```
P(z_i = k | rest)  ∝  (n_dk + α) · (n_kw + β) / (n_k + Vβ)
```

K = 64 topics by default, α = 50/K, β = 0.01 — the textbook defaults, exposed as knobs.

**Determinism is non-negotiable** (the whole store is): every sampling draw is seeded
`splitmix(fnv(fact_text) ^ position ^ sweep)`, sweeps run single-threaded over bounded
batches, so the same corpus always reaches the same state — same-input-same-output, and
the tests can assert exact count tables.

**Short facts need aggregation.** A fact is one sentence; LDA on 8-token documents is
noisy. Refits therefore treat consecutive insertion-order runs of W = 8 episodes as one
pseudo-document for the doc-topic side (insertion order *is* document order — the same
property `neighbors`/stitching already exploits), while topic assignments still land per
fact. New facts **fold in** at observe: 3–5 Gibbs sweeps against frozen counts,
O(tokens × K × sweeps) ≈ a few thousand multiplies — noise next to `train()`.

```rust
pub struct TopicModel {
    k: usize, alpha: f32, beta: f32,
    vocab: HashMap<String, u32>,   // df-capped: hubs (>25% df, the candidates() gate) and hapax excluded
    nkw: Vec<u32>, nk: Vec<u64>,   // K×V topic-word counts + topic totals
}
impl TopicModel {
    pub fn refit(&mut self, docs: &[&[String]], sweeps: usize);          // bounded, deterministic
    pub fn fold_in(&self, tokens: &[String]) -> SmallVec<(u16, f32)>;    // top topics of one fact/query
    pub fn top_words(&self, topic: u16, m: usize) -> Vec<(String, f32)>; // inspection
    pub fn dump(&self) -> String;  pub fn load(blob: &str) -> Self;
}
```

Vocabulary is df-gated with the same logic `candidates()` uses for hubs: a stem in >25%
of documents carries no topical signal and is excluded, as are hapax legomena; the cap
keeps `nkw` at ~6 MB for a 24k-word vocabulary at K = 64 (u32 counts). Assignments are a
sidecar cache keyed by fact text with an epoch stamp (the `emb_cache` pattern — bounded,
evictable, never stored on `Episode`); a refit is triggered lazily when `tokens_seen`
doubles, the same drift bound the embedding cache uses. Facts store zero new bytes.

**Per-scope topic postings** (`topic -> episode indices`, top-1 assignment) live beside
the inverted index in `Neuron` under the feature, and are invalidated by exactly the
events that null the stem index (forget / eviction / reload), rebuilt lazily by the
`ensure_index` idiom.

**Consumers (all fail-open):**

- **Topic-gated fuzzy recall — the headline.** The query folds in (it's short; this is
  microseconds), its top topics select their postings, and `rank_cached` ranks that
  union instead of the recent-4,000 window. Fuzzy recall becomes **scope-wide**: the
  candidate set is O(scope/K)-ish regardless of where the fact sits in history. Query
  folds to nothing, postings stale, or the union thin → fall back to today's windowed
  scan unchanged.
- `topics` op per scope: top-m topics as (share, top words) — "what does this scope
  know". The knowledge-gap signal can then measure *topical* coverage of a question,
  not just lexical hit density.
- `recall_topic` op: page a scope's facts by topic — a thematic read the stem index
  cannot express (recall by abstraction, not by keyword).
- **The two LDAs compose.** A fact's K-dim topic mixture θ is a compact feature vector,
  and a Fisher head at d = 64 over θ is ~100× cheaper to solve than at 256 — and its
  axis is *interpretable*, because each loading is a topic with visible top words. The
  outcome axis over topics literally reads out as "recalls about topic 12 (whale, ship,
  sea…) precede gains; topic 31 precedes losses."

### 3. Wiring (the quantum precedent, minus new read semantics)

Quantum needed `quantum-db` because it changed read semantics (burn-on-read). These tiers
only re-rank and describe, so plain feature conjunctions suffice:

- **db.rs**: a lazily-created `stats_kv (kind, scope, k, v)` side table (the `quantum_kv`
  shape) persists `FisherHead::dump` and `TopicModel::dump`; created on first write, so
  an unused store keeps a byte-identical schema. Observe-path hooks: fold-in after
  `train()`; sampled scope-moment update. Outcome hooks: inside `strengthen` and the
  trust reward path.
- **op.rs / transports**: three new inspection ops (`topics`, `recall_topic`,
  `fisher_axis`) rendered by CLI / MCP / HTTP arms under the features; `recall_blended`'s
  RRF gains its third list internally, no wire change.
- **wasm**: `fisher.rs` compiles as-is (263 KB, no RNG). Topic counts (~6 MB) are heavy
  for the browser default; the lab export ships later behind the same feature, beside
  the existing PCA exports.

## Migration

Nothing migrates. Both features default off; enabling them changes no stored bytes until
first use (`stats_kv` is lazy), and disabling them again leaves a store every older build
reads. The one behavioral change when enabled — blended recall consulting a third rank
list and a scope-wide candidate set — is bounded by RRF (rank evidence only) and by the
inertness floor (no signal until the data has earned one).

## Costs and risks

- **Observe-path overhead.** Fold-in ≈ 3k multiplies per fact; sampled scope moments
  amortize the O(d²/2) rank-1 update to 1-in-16 facts. Budget: <5% of the measured
  ~25k/s durable append rate, asserted by bench before merge.
- **Memory.** Topics ~6 MB at book scale (u32 counts, df-capped vocab) + ~4 bytes/fact
  of postings; Fisher ~263 KB + 2 KB/class. Both bounded, both evictable, both dwarfed
  by the semantic space's ~1 KB/word.
- **Short-text instability.** Sentence-facts are thin documents; the W = 8 pseudo-doc
  window and fold-in averaging are the standard mitigations, and the acceptance tests
  pin recovery on a known two-cluster corpus. If K is badly oversized for a small scope,
  empty topics are harmless (never selected, never posted).
- **Stale topics.** A scope can drift after a refit; postings and assignments carry
  epoch stamps and the doubling bound forces a refit before staleness compounds. Gated
  recall falls back to the windowed scan whenever the gate looks thin, so the failure
  mode is today's behavior, not a miss.
- **Tiny classes.** Shrinkage keeps the solve positive-definite at any n; the sample
  floor keeps the head silent until it has seen enough to say anything.
- **Determinism.** No wall-clock, no thread-order dependence, no unseeded RNG anywhere
  in either module — required for the dump/load round-trip tests and the store's
  same-input-same-output posture.

## Verification

In-module tests (the house pattern — tests live beside the code):

- `fisher.rs`: recovers a known two-Gaussian discriminant (cosine to the analytic
  `Σ⁻¹Δμ` > 0.95); solves positive-definite with n < d via shrinkage; inert below the
  sample floor; relearns a flipped labeling (the `trust_is_relearnable` mirror); scores
  bounded; dump/load round-trips exactly.
- `topics.rs`: on the existing two-cluster test corpus in `semantic.rs` (networking vs
  cooking), K = 4 puts wifi/internet/router and garlic/basil/soup in different topics
  with the expected top words; two identical runs produce identical count tables;
  folding in "browse the web wirelessly" lands in the networking topic.
- Integration: with features on but heads/models empty, `recall_blended` output is
  byte-identical to a featureless build; topic-gated recall returns the same top-1 as
  the full windowed scan on the test corpus.
- Bench: a `stats_bench` example measuring fold-in µs, refit ms per 1k facts, axis-solve
  ms, observe overhead %, and gated-vs-windowed fuzzy recall latency at book scale
  (target: the ~29 ms capped fallback becomes low-single-digit ms, scope-wide).

## Phasing

1. **Cores** — `fisher.rs` + `topics.rs`, std-only, fully tested, wired to nothing.
2. **Store wiring** — outcome/observe hooks, RRF third list, topic-gated fallback,
   `stats_kv` persistence, the three inspection ops across CLI/MCP/HTTP.
3. **Later** — wasm/lab exports (topic view beside the PCA demo), Fisher-over-topics as
   the default outcome-axis feature space, topic-modulated spreading-activation weights.

## What shipped (measured)

Phases 1 and 2 landed: `src/fisher.rs` + `src/topics.rs` (std-only cores, 13 in-module
tests), the `NeuronDB` wiring (streaming absorb at observe, strengthen→"+" / targeted
forget→"−" outcome hooks, sampled scope moments, the topic gate in `recall_semantic` /
`recall_blended`, the fisher RRF third list, lazy `stats_kv` persistence), the `neuron
topics` / `neuron axis` CLI verbs, and `tests/stats_tier.rs` end-to-end. Every transport
(CLI/MCP/HTTP/op) inherits the upgraded recall through `recall_block` with no wire change.
248 tests pass across the full feature matrix; both wasm builds are unaffected.

**Design deltas discovered during the build** (the proposal as-designed had three gaps):

- **Postings are multi-topic.** A sentence's vocabulary can straddle topics, and a 9-token
  fact folds to its *majority* topic while a 2-word query folds to its *words'* topic —
  top-1 postings lost exactly the facts the gate exists to reach. A fact now posts under
  every topic in its mixture (≤3).
- **Query topics take the word-level view too.** The gate unions the query's folded
  mixture with each query word's own strongest topic (`word_topic`), so a short query
  reaches a topic its words dominate even when the joint fold tips elsewhere.
- **The gate's cap trims the no-topic bucket first,** then the oldest topical facts —
  trimming most-ancient-first would have cut precisely the beyond-window facts on a
  polluted union.
- **Learning is streaming absorb, not periodic refit** — the Random-Indexing posture
  (accumulate forever); `refit` remains for batch corpora and the tests. The known trade:
  the earliest facts are absorbed by a cold model and lean on the no-topic bucket.

**Measured** (`examples/stats_bench.rs` + `db_bench` A/B, release, 20,200-fact scope):

| number | value |
|---|---|
| gated blended recall, theme buried 16,000+ facts beyond the window | **hit at episode idx 17** (window floor was idx 16,200), 34.6 ms cold |
| windowed fail-open path (the old behavior, still the fallback) | 14.6 ms |
| outcome axis solve + read (256-d Cholesky, then cached) | **1.75 ms** |
| axis readability | helpful → *nominal, looked, telemetry*; harmful → *experiment, zzdead, abandoned* — exactly the strengthened/forgotten themes |
| single observe, fresh scope (`db_bench` 1) | 14,917 → 14,812 writes/s (**−0.7%**) |
| single observe, growing scope (`db_bench` 2) | ~30k → ~27k writes/s (**~5 µs/fact** for absorb + sampled moments) |
| batch `observe_many` 50k (`db_bench` 3) | 178k → ~85k writes/s (**~2×** — still ≫ the documented ~25k/s durable rate) |
| lexical recall (`db_bench` 4, hit path) | unchanged (the tier never runs on a lexical hit) |
| `scope_topics`, postings warm | 58 µs/call |

**Known boundary:** the outcome hooks embed through the semantic space, which is
resident-only (SEMANTIC.md future-work #3) — so the axis *learns* in long-lived mounts
(MCP/HTTP/embedded) and only accumulates scope moments across one-shot CLI invocations.
The head and the topic model themselves persist and reload exactly (`stats_kv`).
Persisting the semantic space would light the CLI path up with no further change here.
