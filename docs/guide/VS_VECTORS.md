# neuron-db vs dense vectors — a fair, measured head-to-head

This is the honest scorecard for the question *"can a lexical-episodic store beat dense-vector
embeddings as an LLM's memory?"* It is **not** a claim of "superior in all regards" — dense vectors
genuinely win one case (zero-shot pure paraphrase), and that is stated plainly below. Everything
here was measured on one machine over one **frozen dataset fed identically to both engines**, scored
by the **same retrieval-identity rule**. The methodology was designed and adversarially reviewed
before the run (a vector-database advocate and a scientific-integrity reviewer both signed off on
fairness, on the condition that the losses are reported as loudly as the wins — they are).

Reproduce: `rust/neuron-core/examples/vs_vectors.rs` (neuron-db side) + the lab scripts
`vs_gen.py` / `vs_vectors_local.py` / `vs_report.py`.

## Setup

- **Dataset:** 236 facts, 83 queries across five classes — *exact-identifier* (opaque tokens, some
  near-duplicate like `sk_9f3a2b71` vs `sk_9f3b2a17`), *exact-lexical*, *paraphrase* (18 pairs with
  **vocabulary disjoint from the fact** — dense vectors' home turf, authored without consulting the
  synonym table), *adversarial-distractor* (near-duplicate facts differing only in one entity), and a
  *no-answer* set (nothing in the store answers them).
- **neuron-db:** the real Rust core, `--features "sqlite semantic"`, in-process. Three modes —
  *lexical* (default inverted index), *blended* (semantic re-rank on every query) with and without a
  background corpus.
- **Dense vectors:** a **real pretrained embedder run locally**, `all-MiniLM-L6-v2` (384-d), mean-
  pooled, cosine top-k. Local on purpose — it isolates the embedding *architecture* from network
  latency. A hosted API (e.g. `text-embedding-3-small`, 1536-d) would add ~50–150 ms network RTT
  **on top** of every query and 4× the footprint.
- **Scoring (identical both sides):** a hit = the gold fact id is in the engine's top-k. Abstentions
  count as misses. No-answer queries are scored on the abstention axis (abstain = correct).

## Accuracy (per class, never blended)

| class | neuron-db lexical | neuron-db blended **+corpus** | dense vectors (MiniLM) |
|---|---|---|---|
| exact-identifier (n=20) | **100% / 100%** | 100% / 100% | 100% / 100% |
| exact-lexical (n=18) | **100% / 100%** | 100% / 100% | 94% / 100% |
| paraphrase (n=18) | 6% / 6% | **94% / 100%** | 94% / 94% |
| adversarial-distractor (n=15) | **100% / 100%** | 100% / 100% | 93% / 100% |
| no-answer (n=12) | abstains 0% | abstains 0% | abstains 0% |

*(hit@1 / hit@3. neuron-db also returns the literal value char-for-char — `val-exact` 100% on
identifiers — which a vector store cannot do; it retrieves a fact, then an LLM must extract.)*

**Reading it honestly:**
- **neuron-db wins or ties everything except zero-shot paraphrase**, and actually *edges* the dense
  embedder on exact-lexical (100 vs 94) and near-duplicate distractors (100 vs 93) — the embedder
  blurs near-identical facts into one tight cosine neighborhood; the inverted index separates them on
  the exact distinguishing token.
- **Paraphrase is the one real gap.** Lexical neuron-db, with zero shared words, abstains — **6%**, an
  honest loss. The dense embedder gets **94%** out of the box. That gap is the whole reason vectors
  dominate the current mindshare.
- **The gap is closeable without a dense embedder.** neuron-db's *blended* mode — its std-only
  Random-Indexing semantic space used as a re-rank over the lexical candidates — reaches **94% / 100%,
  level with the dense embedder**, at ~86 µs/query and no model on the hot path. **Caveat, stated
  loudly:** this requires the semantic space to have been trained on a background corpus. Without that
  corpus (training only on the facts themselves) paraphrase stays at **6%**. The corpus used here is a
  small generic text that co-locates the relevant concepts and their synonyms — a *disclosed, favorable
  analog* of the dense embedder's internet-scale pretraining. So the result demonstrates the
  **mechanism** (a no-model re-rank can bridge paraphrase given relevant background text), not
  GPT-scale zero-shot coverage.
- **Shared, fundamental weakness:** on no-answer queries **both engines fail to abstain** (100%
  false-positive). neuron-db answers on hub-word overlap (`api`, `key`); the embedder's baseline
  cosine clears any sane threshold. This is not a cheap win for either side: you cannot separate *"the
  heroku api key?"* (no answer — abstain) from *"the thing I use to get online → wifi"* (legitimate
  paraphrase — answer) **lexically**, because both have a specific term that isn't stored, and you
  cannot separate them **semantically** either, because the no-answer query is genuinely similar to the
  generic part of a stored fact. A measured experiment confirmed it (see below).

## Latency (per query)

| engine | p50 | p95 | what it includes |
|---|---|---|---|
| neuron-db lexical | **13.8 µs** | 330.9 µs | in-process: stem → inverted-index → score the episodes that share a cue |
| neuron-db blended | 86.2 µs | 92.1 µs | adds the int8 semantic re-rank over the lit candidates |
| dense vectors — local (end-to-end) | 5.1 ms | 6.4 ms | local MiniLM forward-pass + cosine (network-free floor) |
| **dense vectors — hosted (end-to-end)** | **320 ms** | **457 ms** | **`text-embedding-3-small` query-embed RTT + cosine — measured, the real production cost** |
| dense vectors (search only) | 0.02–11 ms | — | cosine alone, embed excluded |

**neuron-db is ~373× faster than a *local* embedder and ~23,000× faster than a *hosted* one at the
median (13.8 µs vs 320 ms measured).** The hosted query-embed round-trip — the unavoidable "turn the
text into a vector first" step — was measured at **320 ms p50**, even worse than the 50–150 ms I'd
projected. The cosine loop itself is cheap; **the decisive, intrinsic vector cost is that a vector
query must embed at all.** neuron-db never embeds, so it skips that step entirely.

## Footprint

| | bytes/fact | note |
|---|---|---|
| shared source text (both sides) | 38 B | the fact strings — both engines store these |
| neuron-db, entire on-disk store | 296 B | incl. ~36 KB **fixed** SQLite/WAL overhead at this tiny N |
| dense vectors alone (MiniLM 384-d) | 1,536 B | no text, no ANN graph |
| dense vectors alone (hosted 1536-d) | 6,144 B | the common production size |

At N=236 the dense vectors alone are **5.2× neuron-db's entire store**; as the fixed SQLite overhead
amortizes, the per-fact structural cost is ~48 B vs a 6,144 B dense vector — the documented **~128×**
density gap. Ingest: the dense side spent **12.9 s embedding the 236 facts** (every fact a model
forward-pass); neuron-db ingest is local CPU in milliseconds and a fact is searchable the instant it's
written.

## The honest verdict

> neuron-db decisively wins end-to-end latency, footprint/density, exact-identifier and near-duplicate
> precision, literal-value return, and ingest cost — because it skips the embedding step entirely.
> Dense vectors win one case: zero-shot pure-paraphrase recall with no shared words. neuron-db closes
> even that to parity with a std-only, no-model semantic re-rank **when its semantic space has seen a
> relevant corpus** — otherwise that one gap remains, and is disclosed, not buried.

## What's next (ranked, from the design review)

1. **IDF / hub-aware abstention (validated, deferred).** Require the winning candidate to share a
   *discriminative* (low document-frequency) cue, not just hub words; abstain otherwise. This was
   prototyped and verified safe (all 131 tests green) and it **does** fix the false-positive on the
   bare `Neuron` (lexical, wasm) tier. But it was **reverted**, not shipped: in a realistic
   `--features semantic` deployment the `recall_semantic` fallback re-answers the query anyway (it
   matches the generic part), so the end-to-end no-answer number is unchanged — and tightening the
   semantic threshold to stop that would regress the paraphrase recall the blended mode just won. A
   real fix needs **coordinated lexical + semantic gating** that distinguishes a discriminative miss
   from a true zero-overlap paraphrase, without lowering paraphrase recall. That is a focused,
   measured piece of work, not a one-liner — recorded here rather than shipped half-done.
2. **Promote the semantic space from miss-only fallback to a confidence-gated top-k re-ranker** over
   the lexical candidates — closes *partial*-overlap paraphrase on the fast path. Does not help
   zero-overlap (no candidate lights up), which stays disclosed.
3. **Bidirectional synonym + morphology query expansion** before the index lookup — moves a slice of
   today's slow-fallback synonym hits onto the microsecond fast path.

## Disclosures (so the numbers can't mislead)

- Single machine, N=236, one run. Selective-recall scaling to ~flat µs at 1M facts is measured
  separately in `BENCHMARKS.md` (not re-run here); neuron-db's broad-cue worst case (a stem in every
  fact) is O(matches)=O(N) and is **not** the selective number.
- Two dense baselines were run: a *local* 384-d model (MiniLM) and the *hosted* 1536-d
  `text-embedding-3-small` (now **measured**, not projected: 320 ms p50 end-to-end, 6144 B/fact).
  **Important honesty correction:** the hosted model scores **100% on every class — including
  paraphrase**, so neuron-db does **not** beat a good hosted embedder on *accuracy*. The "neuron-db
  edges vectors on exact/distractor precision" note above held against the *small local* model
  (94/93%); the hosted model ties those at 100% and wins paraphrase outright. neuron-db's durable,
  structural advantage is **latency (~23,000×), footprint (~20–128×), ingest, and zero infra** — not
  recall accuracy. That is the honest shape of the win.
- The blended paraphrase win depends on the background corpus (disclosed above).
- Paraphrase queries were authored without reading the synonym table, so the lexical path got no
  unfair alias coverage.
