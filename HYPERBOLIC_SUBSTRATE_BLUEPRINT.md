# hyper: a feature-gated, per-scope Poincaré-ball re-rank layer over neuron-db's stem co-occurrence graph (v1 = assoc re-rank only, gated behind a real-data audit)

> An opt-in `hyper` feature that, PER SCOPE, lays that scope's stem co-occurrence graph (the same per-Neuron index recall_spreading walks) into an 8-D Poincaré ball by ONE deterministic order-invariant pass built lazily alongside ensure_index, then uses Poincaré distance ONLY to re-rank the already-widened assoc pool (recall_assoc_trusted / recall_associative) — bounded to that pool, degrading to identity when a scope has no coords; the lexical recall() path and the recall_spreading inner loop stay byte-identical, and the whole feature is gated behind a real-data audit that can kill it before Phase 0.

## hyper: a per-scope Poincaré re-rank layer for neuron-db (final, blocking fixes folded, critic cuts applied)

### Verdict first (honest)
Three of the four critics returned `needs-revision` and the fourth's blocking issues are all real against file:line. The original draft's THREE load-bearing decisions are each structurally false in this core:
1. "side-table keyed by the stable `Episode.id: i64`" — `encode()` hardcodes `id: -1` (lib.rs:263) and `load()` never restores an id (lib.rs:631); EVERY fact's id is -1. There is no stable per-fact handle. DELETED.
2. "one global `hyp` table on NeuronDB mirroring `sem` (db.rs:124)" — the co-occurrence graph, df, neighbors, and `dfcap` are strictly PER-SCOPE on `Neuron` (`index` field lib.rs:314; `recall_spreading` lib.rs:510-556; `dfcap` lib.rs:527). The intern POOL is process-global (lib.rs:168), so a global stem→coord map fuses every scope's "server"/"depends"/"fail" into one averaged point = distortion WORSE than Euclidean. The "mirror sem" instruction is WRONG for this feature (sem is global on purpose — paraphrase is cross-scope; hierarchy is not). MOVED to per-scope.
3. "Phase 3 chain tie-break = gated one-liner at recall_chain (db.rs:634), else byte-identical" — `recall_chain` resolves each hop via a single `self.recall` returning ONE `Recall` (db.rs:642); the candidate pool is computed and discarded inside `Neuron::recall` (lib.rs:379-402, single `best`). There is no candidate set to tie-break. CUT from v1.

After folding the fixes, the defensible v1 is ONE integration (assoc re-rank), gated behind a real-data frequency audit. If that audit finds depth≥4 lexically-disjoint hierarchies are rare in real scopes (the likely outcome), the honest call is **build only Phases -1/0/1 as a research artifact and do NOT ship the re-rank** — the existing per-scope `0.5/df` graph already wins the realistic workloads.

### The thesis, scoped to what survives
neuron-db's hierarchies are real (scope→entity→fact; `contains`/`caused_by` trees; the shared-stem graph `recall_spreading` walks via `w=0.5/df`, lib.rs:537). A negatively-curved Poincaré ball packs trees at distortion →1 (exponential ball volume matches exponential node growth) where Euclidean crowds at O(log n). BUT — and this is the critic's sharpest point — the layout DERIVES position from the SAME co-occurrence the graph already propagates. If two levels never co-occur, placement has no signal to put them near each other either, except transitively through shared ancestors — which is exactly what the graph's multi-hop spread already does (A→B→C reaches C with no A–C shared stem, lib.rs:528-548). So the geometry can only beat the graph where the graph's `0.5/df` activation DECAY starves a deep-but-real leaf that the ancestor chain still connects. That is a narrow, measurable case, not a general win.

### THE unifying decisions (all five sub-designs collapse to these)
- **Coords are PER-SCOPE.** New field `#[cfg(feature="hyper")] hyp: Option<HyperSpace>` on `Neuron` (lib.rs:319, beside `index`). NOT on NeuronDB. No global Mutex, no `hyp_guard` de-poison twin (it lives inside the shard lock already). Delete the "mirror sem db.rs:124/212/374" instruction.
- **The learned primitive is the per-scope STEM** (`Arc<str>`, the same interned key `index` uses, lib.rs:314). FACT coords are DERIVED transiently (Einstein midpoint of the fact's stems) and never stored. QUERY anchor is derived transiently from cue stems. No per-fact id needed → the `Episode.id` problem evaporates.
- **Layout is ONE deterministic, ORDER-INVARIANT pass, built lazily from the completed index — NOT online EMA streaming.** This is the single biggest fold: the critics independently flagged that online placement (einstein-midpoint of "already-placed neighbors") is order-dependent and the periodic relayout "is idempotent smoothing, NOT convergence" — so dump→load would re-rank. The fix removes the entire streaming-SGD surface (mobius_lerp settle, η steps, vocab-growth triggers, EMA): build `hyp` inside an `ensure_hyper()` twin of `ensure_index()` (lib.rs:335) as a pure function of the scope's FINAL per-stem neighbor-set + df. Idempotent, reload-stable, no persistence, no dump-format change. Invalidate it exactly where `index` is invalidated (lib.rs:361 front-drain, lib.rs:649 reinforce remove, lib.rs:680 forget_prefix, db.rs:676/691 forget).
- **`df`-as-depth is read AT LAYOUT TIME from the completed index, not at observe time.** Because layout is lazy-from-completed-index, every stem's df is its FINAL posting length (lib.rs:535 `posting.len()`), not the wrong-at-placement-time growing value. The online-df soundness gap is gone by construction.
- **Bounded to the existing recall budget.** Layout and re-rank touch only the candidate pool the assoc path already widened to (clamp 64, db.rs:168) plus those facts' stems — NOT the whole scope. Gate behind the scope-size floor (reuse 64) so flat/small scopes build no HyperSpace and pay zero. On a cold (evicted-then-reloaded) scope the one-time layout is capped like ROOT_SCAN_CAP/SEM_FALLBACK_CAP=4000 (lib.rs:573, db.rs:396).

### The geometry (model 𝔹⁸, curvature c=−1; recall calls ONE function)
Distance (robust acosh form, f64 internal / f32 store):
```
d(u,v) = acosh( 1 + 2·‖u−v‖² / ((1−‖u‖²)(1−‖v‖²)) ),  acosh(x)=ln(x+√(x²−1))
```
Möbius add / exp-log maps / Einstein midpoint are needed only for LAYOUT, not recall (math in `math_core`). Re-rank uses `d` only.

### The ONE deterministic per-scope layout pass (ensure_hyper)
Over the candidate-pool stems S (or the whole scope's stems if scope ≤ pool, gated by the 64 floor):
1. compute each stem's df from the completed index posting length (lib.rs:535).
2. seed the lowest-df ("rarest", deepest-leaf-like) and highest-df ("hub", root-like) by df-rank.
3. one pass: `coord[s] = project( rescale_radius( einstein_midpoint({coord[n] : n∈neighbors(s), placed}, w=1/df_n), r_target(df_s) ) )`, where `r_target=tanh(α·(−ln(df/|S|))/2)` and neighbors(s) = stems co-occurring with s in the pool facts. Process stems in df-DESCENDING order (hubs→origin first, leaves→boundary last) so every stem's neighbor-midpoint references already-placed lower-radius ancestors → deterministic, order-invariant given the completed graph.
4. NO settle, NO EMA, NO periodic relayout. Re-running on the same completed index yields byte-identical coords. (One optional in-pass smoothing sweep MAY be added if the Phase-1 Spearman gate needs it, but it too is a pure function of the completed graph — still order-invariant.)

### How recall USES it (v1 = assoc re-rank ONLY)
- **recall_assoc_trusted (db.rs:167) / recall_associative (db.rs:485):** after the existing widen-to-64 + sort, fold a THIRD multiplicative factor into the existing `sort_by` (db.rs:171-175): `score = act · trust · exp(−β·d(query_anchor, fact_coord))`. Composes orthogonally with trust (geometry rides inside the act term; trust re-ranks the geometry output). **DEGRADE-SAFE: if query_anchor is None (no cue stem placed) or the scope has no HyperSpace, multiply by 1.0 → byte-identical act·trust order.** An empty anchor must NEVER zero-out act. Bounded to the clamp-64 pool, so the hot-path budget holds.
- **recall() lexical hot path (lib.rs:366): UNTOUCHED.**
- **recall_spreading inner loop (lib.rs:528-548): UNTOUCHED in v1.**

### Numerical stability (Poincaré is float-fragile at ‖x‖→1)
House style (cortex.rs ln 1e-5; semantic.rs `.max(1e-9)`): EPS_B=1e-5 boundary clamp on EVERY produced point; denom floor `(b·g).max(1e-12)`; `acosh(arg.max(1.0))`; `√((x²−1).max(0.0))`; atanh arg ≤ 1−1e-7; zero-norm guard 1e-12. **f64 internal accumulation; the catastrophic-cancellation subtraction `1−‖x‖²` is done in f64.** STORAGE: the f32 radial ceiling is ~5–6 clean levels (atanh(1−1e-5)≈6), and the payoff lives at depth≥4 — right at the cliff. Resolve in Phase 0 by a monotonicity unit test across 6 planted levels in the chosen storage type; if level-5 vs level-6 distances invert under f32, store coords as **f64** (side-table doubles to ~64 B/stem; stems are 10⁴–10⁵ per scope, still small) and keep the f64-everywhere guard consistent. NaN anywhere → fall back to the existing act·trust order (de-poison, never panic), the `sem_guard` philosophy (db.rs:212).

### Dimensionality: D=8, c=1 (locked for v1, swept only if v1 wins)
D=8 = one `chunks_exact(8)` SIMD lane (cortex.rs:108) for ‖u−v‖² and the two 1−‖x‖² terms; gives angular room for the substrate's near-orthogonal sub-hierarchies (scope ∥ caused_by ∥ contains) past the ~6-level radial ceiling. The D=8-vs-16 / curvature sweep is bench tooling — DEFERRED until v1 clears adopt-or-kill.

### Honest payoff (folded with the circularity critique)
**Where it CAN beat (must be measured on REAL scopes, not the synthetic fixture):** (i) hierarchy faithfulness — measured as Spearman ρ(tree-distance, d) vs the control ρ(tree-distance, graph-hop-count); it is a real win ONLY if hyperbolic-d correlates with tree distance MORE than the graph's own hop-count already does. If they tie, the geometry is re-encoding what the graph has = overhead. (ii) A deep-but-real leaf that the `0.5/df` activation decay starves across 3–4 hops but the ancestor chain still connects.
**Where it does NOT beat (do not oversell):** (i) flat keyword/exact recall — `recall()` is already optimal, geometry stays out. (ii) paraphrase/synonym — the 256-d Euclidean semantic space (semantic.rs) owns it; hyperbolic angle ≈ Euclidean cosine locally, no win. So the recall() semantic-fallback-slot ranker is CUT — it adds a path with no expected win. (iii) local 1–2 hop assoc — the per-scope `0.5/df` graph already nails it; expect a TIE and REPORT it as a tie. (iv) lexically-DISJOINT levels where parent and child never co-occur — placement has NO signal to co-locate them that the graph's transitive spread doesn't already have; the claimed "sharpest win" is largely CIRCULAR and must be proven against the graph head-to-head, not against the Euclidean strawman. (v) flat bag-of-facts scopes (the common chat case) — no hierarchy, pure overhead, gated out by the 64 floor. (vi) speed — never shrinks the candidate set; one extra O(pool·D) re-rank pass, improves ordering not latency.

## math_core
## Poincaré-ball math, Möbius ops, deterministic per-scope layout, stability guards (D=8, c=1)

### Distance (recall calls ONLY this; f64 internal, f32-or-f64 storage per Phase-0 test)
d(u,v) = acosh( 1 + 2·a / (b·g) ), with
  a = ‖u−v‖²,  b = (1 − ‖u‖²),  g = (1 − ‖v‖²),  acosh(x) = ln( x + √(x²−1) )
GUARDS: arg = (1 + 2a / (b·g).max(1e-12)).max(1.0); √ over (arg²−1).max(0.0).
The 1−‖x‖² terms are accumulated in f64 (catastrophic cancellation near the boundary).
For c≠1: d = (1/√c)·acosh(1 + 2c·a/(b·g)); c kept as a tunable field, default 1.

### Möbius addition (c=1; used by exp/log/midpoint, NOT by recall)
x ⊕ y = [ (1 + 2⟨x,y⟩ + ‖y‖²)·x + (1 − ‖x‖²)·y ] / (1 + 2⟨x,y⟩ + ‖x‖²‖y‖²)

### exp / log maps (layout primitives), λ_p = 2/(1−‖p‖²)
exp_p(v) = p ⊕ ( tanh(λ_p·‖v‖/2) · v/‖v‖ )            [exp_0(v)=tanh(‖v‖)·v/‖v‖]
log_p(y) = (2/λ_p)·atanh(‖−p⊕y‖) · (−p⊕y)/‖−p⊕y‖      [log_0(y)=atanh(‖y‖)·y/‖y‖]
GUARD: clamp atanh argument ‖·‖ ≤ 1−1e-7; if ‖v‖<1e-12 return p unchanged.

### Einstein/Klein gyro-midpoint of {pᵢ} with weights {wᵢ} (the layout workhorse)
γᵢ = 1/√(1−‖pᵢ‖²);   m = ( Σ wᵢ·γᵢ·pᵢ ) / ( Σ wᵢ·γᵢ )   then project(m).
(Cheaper + stabler than iterated Möbius sums; weights wᵢ = 1/df_neighborᵢ to mirror the 0.5/df graph link, so geometry == graph.)

### THE deterministic per-scope layout (no online EMA, order-invariant)
Inputs: the COMPLETED per-scope index → for every stem s its final df_s = posting.len() and its neighbor set N(s) = stems co-occurring with s in pool facts.
Process stems in df-DESCENDING order (hub→origin first):
  known = { coord[n] : n ∈ N(s), already placed (higher df) }
  base  = if known empty  then  R0·unit_dir(splitmix(fnv(s))),  R0=0.05
          else            einstein_midpoint(known, weights = 1/df_n)
  r_target = tanh( α · (−ln(df_s / |S|)) / 2 ),   α≈0.6      # rare→boundary, hub→origin
  coord[s] = project( rescale_radius(base, r_target) )
Re-running on the same completed index ⇒ identical coords (the only data-dependence is the
fixed df-descending order + the completed neighbor sets). NO mobius_lerp, NO η, NO settle.

### Derived fact coord (never stored) + transient query anchor
fact_coord(e)     = einstein_midpoint({coord[s] : s ∈ e.s, placed}, uniform)
query_anchor(cue) = einstein_midpoint({coord[s] : s ∈ cue, placed}, uniform)   # None if empty

### The ONLY integration term (v1)
assoc: score = act · trust · exp(−β · d(query_anchor, fact_coord)),  β≈1.0
       DEGRADE: query_anchor None OR scope has no HyperSpace ⇒ factor = 1.0 (byte-identical act·trust)
(chain min-dist tie-break and spread edge-damp are CUT from v1; see payoff.)

### Stability constants (house style)
EPS_B = 1e-5 (clamp ‖x‖ ≤ 1−EPS_B after EVERY produced point); denom floor 1e-12;
acosh arg .max(1.0); atanh arg ≤ 1−1e-7; zero-norm 1e-12.
f64 accumulation; coord storage f32 IF the 6-level monotonicity test passes, else f64.
NaN anywhere ⇒ fall back to act·trust order (de-poison, never panic).

### Why this packs trees at distortion →1 (load-bearing claim) AND its limit
Ball of hyperbolic radius R has volume ~ e^{(D−1)R} (exponential) — matches a b-ary tree's bʰ
leaves, so bʰ leaves fit at radius ~h with mutual hyperbolic distance ≈ tree distance; Euclidean
volume is polynomial → leaves crowd → distortion ~O(diameter)=O(log n). LIMIT: f32 EPS_B=1e-5 caps
clean radial depth at atanh(1−1e-5)≈6 levels — the payoff at depth≥4 sits near that cliff, hence
the Phase-0 monotonicity test that may force f64 storage. CIRCULARITY CAVEAT: the layout's only
signal is co-occurrence; it cannot place two never-co-occurring levels near each other except via
shared ancestors — which the graph's transitive spread already does. So the faithfulness gate is
ρ(tree-dist, d) vs ρ(tree-dist, graph-hop), head-to-head, not vs Euclidean.

## honest payoff
WHERE IT CAN BEAT (real, but each must be measured on REAL scopes before claiming): (1) hierarchy faithfulness — `contains`/`caused_by`/scope→entity→fact trees embed at distortion ~1 vs Euclidean O(log n) crowding (polynomial volume cannot hold an exponential tree without distortion). BUT the honest gate is the HEAD-TO-HEAD ρ(tree-dist, d) vs ρ(tree-dist, graph-hop-count): the geometry only adds value if it tracks tree distance BETTER than recall_spreading's own hop-count already does. If they tie, it is overhead. (2) A deep-but-real leaf that the `0.5/df` activation decay (lib.rs:537) starves across 3–4 hops while the ancestor chain still connects it — a narrow, measurable case.

WHERE IT DOES NOT BEAT (do not oversell): (1) flat keyword/exact recall — recall() (lib.rs:366) is already optimal; geometry stays out. (2) paraphrase/synonym — the 256-d Euclidean semantic space (semantic.rs) owns it; hyperbolic angle ≈ Euclidean cosine locally → the recall() semantic-fallback ranker is CUT (no expected win, just a new path). (3) local 1–2 hop assoc — the per-scope `0.5/df` graph already nails it; expect a TIE and report it AS a tie. (4) the originally-claimed "sharpest win," lexically-DISJOINT levels where parent and child never co-occur — LARGELY CIRCULAR: the layout's ONLY signal is co-occurrence, so it cannot co-locate two never-co-occurring levels except via shared ancestors, which the graph's transitive multi-hop spread (A→B→C, no A–C shared stem, lib.rs:528-548) ALREADY does. The win, if any, is only where the graph's weight-decay starves a leaf the layout's single-radius neighborhood still reaches — prove it against the GRAPH, not the Euclidean strawman. (5) flat bag-of-facts scopes (the common chat case) — no hierarchy → pure overhead → gated out by the 64 scope-size floor. (6) speed — one extra O(pool·D) re-rank, ordering only, never latency; and the cold (evicted→reloaded) scope pays a one-time capped (≤4000) layout that MUST be a kill-criterion metric in the bench, since ensure() eviction (db.rs:277/122) makes cold-load a production path, not a boot cost.

CUTS THE CRITICS FORCED (folded): (a) Phase 3 chain tie-break — recall_chain (db.rs:642) takes a SINGLE recall() best; no candidate set exists to tie-break; the "operate oscillation" payoff was asserted, not measured. CUT from v1. (b) Phase 4 spreading-edge damp — sits on the deepest hot loop (lib.rs:528-548), needs two derived fact_coords per edge per hop, damp-only (can only ever REMOVE recall the graph surfaces). CUT from v1. (c) online EMA/settle/periodic-relayout — order-dependent, would re-rank across dump→load, breaking reload-stability; replaced by the single deterministic lazy-from-completed-index pass. (d) global NeuronDB `hyp` table + hyp_guard — fuses scopes via the process-global intern pool; moved per-scope onto Neuron.

HONEST RISK FRAMING + ADOPT-OR-KILL: the per-scope `0.5/df` graph is genuinely strong on realistic (lexically-linked, shallow) workloads, so the plausible outcome is "win only on the narrow deep-decay case, if that case even occurs." Gate ORDER: run Phase -1 (real-scope frequency audit) FIRST — if depth≥4 hierarchies with a starved-but-connected deep leaf are near-zero in real data, KILL before Phase 0 (the synthetic bench would pass regardless and mislead). Then ship the assoc re-rank ONLY if, on REAL held-out scopes: recall@10 lift ≥15% at depth≥4 AND ρ(tree-dist,d) > ρ(tree-dist,graph-hop) by a clear margin AND <2% delta on the flat control AND cold-load overhead under the latency bar. If the audit says rare, the honest verdict is: BUILD Phases -1/0/1 as a faithfulness research artifact, DO NOT ship the re-rank, keep it opt-in for hierarchy-heavy operate/causal workloads only — or recommend not building it at all.

## rust diff
NEW FILE rust/neuron-core/src/hyper.rs — all `#[cfg(feature="hyper")]`, std-only, cortex.rs hand-indexed f64-internal kernel style:
  consts: D=8, C=1.0, EPS_B=1e-5, R0=0.05, ALPHA=0.6, BETA=1.0, DENOM_FLOOR=1e-12.
  struct HyperSpace { stem: HashMap<Arc<str>,[F;D]> }  // F = f32 or f64 per the Phase-0 monotonicity test
  fns: dist (guarded acosh), mobius_add, expmap/logmap (+_0), einstein_midpoint(points,weights),
       project (boundary clamp), unit_dir(fnv/splitmix seed, reuse semantic.rs:17-27), rescale_radius,
       build(index_postings, pool_stems) -> HyperSpace  (THE deterministic order-invariant pass),
       fact_coord(stems)->[F;D], query_anchor(cue)->Option<[F;D]>.
  NO mobius_lerp / settle / relayout (online EMA surface CUT).

MODIFIED rust/neuron-core/Cargo.toml — add `hyper = []` near line 24 (std-only, no deps, independent of `semantic`/`trust`; all off by default).

MODIFIED rust/neuron-core/src/lib.rs:
  `#[cfg(feature="hyper")] pub mod hyper;` near line 726 (beside the semantic mod decl).
  NEW field on Neuron (lib.rs:319, beside `index`): `#[cfg(feature="hyper")] hyp: Option<crate::hyper::HyperSpace>`, init None in Neuron::new (lib.rs:320).
  NEW method `#[cfg(feature="hyper")] fn ensure_hyper(&mut self)` — twin of ensure_index (lib.rs:335): calls ensure_index, then if hyp is None builds it from the completed index over the pool/scope stems (gated by the 64 scope-size floor). Pure function of the completed index → order-invariant.
  Invalidate hyp = None EXACTLY where `index` is nulled: observe front-drain (lib.rs:361), reinforce_prefix remove (lib.rs:649), forget_prefix (lib.rs:680). (These already set index=None; add `#[cfg(feature="hyper")] self.hyp=None;` beside each.)
  Episode (184), encode (216), dump_from (605), load (616), recall (366), recall_spreading (510): ALL UNCHANGED. No Episode.id dependence anywhere.

MODIFIED rust/neuron-core/src/db.rs:
  invalidate_index (lib.rs:685) is called from db forget (db.rs:676) and the sub-scope wipe (db.rs:691); extend invalidate_index to also null hyp (so db.rs needs no edit there).
  recall_associative (db.rs:485) / recall_assoc_trusted (db.rs:167): gated — before/inside the existing sort_by (db.rs:171-175), call ensure_hyper on the scope's Neuron, compute query_anchor once, and fold `· exp(−β·dist(query_anchor, fact_coord))` into the score; if query_anchor is None OR hyp is None, factor = 1.0 (byte-identical). Bounded to the clamp-64 pool (db.rs:168).
  NO global `hyp` field on NeuronDB, NO hyp_guard, NO change to observe_many (367), NO change to load/ensure (252).

NEW FILE rust/neuron-core/examples/hyper_audit.rs (gated `hyper`) — PHASE -1 read-only analyzer over real dumped scopes (operate telemetry, chat-lab events, cortex examples): counts scopes with a contains/caused_by/scope-entity-fact tree of depth≥4, and of those how many have ≥1 fully lexically-disjoint level (zero shared stem parent→child, the df<2 condition lib.rs:536). KILL the feature if that count is near-zero. Runs BEFORE any kernel.

NEW FILE rust/neuron-core/benches/hyper_hierarchy.rs (gated `hyper`) — planted b-ary tree (V1 shared-stem / V2 lexically-disjoint / V3 wide-flat-control) + REAL-scope replay; metrics recall@k-by-depth, Spearman ρ(tree-dist,d) AND ρ(tree-dist,graph-hop) head-to-head, cold-recall latency delta; 3 arms (graph / Euclidean semantic / hyper) with the adopt-or-kill rule.

GATE INVARIANT: `hyper` OFF ⇒ Neuron/Episode size, dump format, recall(), recall_spreading order, observe path ALL byte-identical; zero new instructions on the hot path. ON ⇒ only the assoc re-rank changes, bounded to the clamp-64 pool, degrade-safe to act·trust. Independent of `semantic` (different call site, per-scope vs global, never share a vector).

## build phases

### Phase -1 — real-scope frequency audit (read-only, gates EVERYTHING, one afternoon, BEFORE any kernel)
NEW examples/hyper_audit.rs: load real dumped scopes (operate telemetry, chat-lab events.jsonl, cortex examples) via the existing load() path; for each scope reconstruct the per-Neuron index and measure (a) how many scopes have a contains/caused_by/scope→entity→fact tree of depth≥4, and (b) of those, how many have at least one deep leaf that recall_spreading's 0.5/df decay starves (low activation across 3–4 hops) yet an ancestor chain still connects. If that count is near-zero, KILL the feature here — the synthetic bench in later phases would pass regardless and ship dead weight. This is the blueprint's own open question turned into the FIRST gate, per the payoff critic.

### Phase 0 — kernel + bench fixture (no integration, no behavior change)
Add `hyper = []` to Cargo.toml and `#[cfg(feature="hyper")] pub mod hyper;` to lib.rs:726. Write hyper.rs kernels ONLY: dist (guarded acosh), mobius_add, exp/log maps, einstein_midpoint, project, unit_dir, rescale_radius — f64-internal, cortex.rs hand-indexed style. NO mobius_lerp/settle/relayout. Unit tests: d(x,x)=0, triangle inequality, boundary-clamp keeps ‖x‖<1, acosh never NaN at ‖x‖→1, AND the decisive monotonicity-across-6-planted-levels test in BOTH f32 and f64 storage — if f32 inverts level-5 vs level-6 distance, lock storage to f64. Build the planted b-ary fixture (benches/hyper_hierarchy.rs) V1/V2/V3 + the metrics incl. the head-to-head ρ(tree-dist,d) vs ρ(tree-dist,graph-hop) and cold-load latency. Gate OFF ⇒ zero change.

### Phase 1 — per-scope deterministic layout + lazy ensure_hyper (LAYOUT ONLY, no recall use; the go/no-go gate)
Add `#[cfg(feature="hyper")] hyp: Option<HyperSpace>` to Neuron (lib.rs:319), init None (lib.rs:320). Implement build()/fact_coord()/query_anchor() in hyper.rs and ensure_hyper() as the ensure_index twin (lib.rs:335), built from the COMPLETED index, gated by the 64 scope-size floor, bounded to the pool/cap (4000). Add `#[cfg(feature="hyper")] self.hyp=None;` beside each index=None (lib.rs:361/649/680) and extend invalidate_index (lib.rs:685) to null hyp (covers db.rs:676/691). VERIFY reload-stability: dump→load→ensure_hyper twice and assert byte-identical coords (the order-invariance proof). Run the head-to-head Spearman gate on V1/V2 AND on the Phase -1 real scopes — ρ(tree-dist,d) MUST beat ρ(tree-dist,graph-hop) by a clear margin or the whole feature dies here.

### Phase 2 — assoc re-rank (the ONLY v1 integration) + adopt-or-kill on REAL data
Gate the `· exp(−β·d(query_anchor,fact_coord))` factor into the existing sort_by of recall_assoc_trusted (db.rs:171-175) / recall_associative (db.rs:485), bounded to the clamp-64 pool (db.rs:168), DEGRADE-SAFE (query_anchor None or hyp None ⇒ ×1.0 ⇒ byte-identical act·trust). Run the 3 arms (graph / Euclidean semantic / hyper) on V1/V2/V3 AND replay the Phase -1 REAL scopes through assoc hyper on/off, measuring recall@10 lift on held-out leaf facts + cold-load latency delta. ADOPT-OR-KILL on REAL data: ship only if recall@10 lift ≥15% at depth≥4 AND <2% delta on the flat control AND cold-load under the latency bar. If it only wins on synthetic V2 and not real scopes, DO NOT ship the re-rank — keep Phases 0/1 as a research artifact.

### Phase 3 (CONDITIONAL, only if Phase 2 ships) — D/curvature sweep + opt-in CLI inspector
ONLY if Phase 2 cleared adopt-or-kill on real data. Run the D=8-vs-16 and curvature-c sweep on the planted taxonomy + real scopes to lock the consts. Optionally expose a read-only `neuron hyper` CLI op (op.rs vocabulary entry like chain/assoc) that prints a stem's coord/df/radius for inspection — NOT a relayout op (there is no online relayout). Document the f32/f64 ~6-level radial ceiling and the Phase -1 depth-≥4 frequency finding as the honest value boundary. CUT for v1 (do NOT build before Phase 2 wins): chain tie-break (no candidate set at db.rs:642), spreading-edge damp (deepest hot loop lib.rs:528-548), online EMA/relayout, the recall() semantic-fallback ranker (no expected win).
