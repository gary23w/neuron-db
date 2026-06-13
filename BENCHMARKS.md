# Benchmarks

Measured in the dev sandbox: single core, Python 3.10 and Rust 1.96 (release, LTO).
Numbers are indicative, not a tuned lab result. Reproduce with `python tests/bench.py`
(Python) and `cargo run --release --bin bench` (Rust core, on the `rust` branch).

## How many neurons in 1000 ms

| path | neurons/sec (3 facts each) |
|---|---|
| Python, SQLite-backed (`NeuronDB.turn`) | ~1,200 |
| Python, pure in-memory `Neuron` | ~30,000 |
| **Rust core, in-memory** | **~215,000** |

The SQLite number is the realistic API rate (durable write per neuron). In-memory is the
ceiling. Rust is ~7× the Python in-memory rate.

## Recall latency vs neuron size

| facts in neuron | Python (indexed) | Rust core |
|---|---|---|
| 500 | ~245 µs | ~50 µs |
| 5,000 | ~56 µs (rare cue) | — |

Recall is sub-linear: a stem→fact inverted index means only facts sharing a cue stem are
scored, so latency tracks the number of *candidate* facts, not the total. Rust is ~5× the
Python constant factor; the bigger Rust gains are true multi-core concurrency (no GIL) and
a single static binary.

## Capacity and accuracy — the honest part

A neuron's recall accuracy depends on whether its keys are **lexically distinct**, not on
how many facts it holds.

| keys | facts | recall@1 |
|---|---|---|
| distinct (`the north wifi password`, `the spare gate code`, …) | 400 | **400/400 (100%)** |
| colliding stems (`project0`, `project1`, … all stem to `projec`) | 500 | ~1/500 |

This is the defining property of the engine. Recall matches on stemmed cue overlap, and
the stemmer truncates to 6 characters, so keys that differ only past that prefix
(`project0` vs `project1`) collapse to the same stem and become indistinguishable — recall
then returns the most-recent collider. Real-world keys (names, relations, attributes) are
distinct and recall cleanly at 400+ facts. Near-duplicate keys are the failure mode.

**Implication for a memory bank:** an LLM writing many similar facts (e.g. ten "meeting on
{date}" entries) will collide. The harness design (`docs/MEMORY_HARNESS.md`) addresses this
with explicit-key entries, full-token disambiguation, and a dedup/supersede policy, rather
than relying on fuzzy stems for near-duplicates.

## Storage

| form | bytes/fact |
|---|---|
| plaintext fact (text + flag; index recomputed) | ~30–38 |
| plaintext, gzipped | ~9–28 (corpus-dependent) |
| encrypted entry (AES-GCM + keyed index) | ~99 (77 gzipped) |

A 500-fact neuron is ~15–20 KB. The cap is recall quality (cue collisions grow with size),
not storage; `MAX_FACTS` defaults to 500 for that reason.

## What the tests prove vs what the benchmarks prove

The 17 unit tests prove correctness (recall, isolation, abstention, encryption, isolation
between neurons). These benchmarks prove capability and bound it honestly: fast creation,
sub-linear recall, 100% on distinct keys, and a clear collision boundary on ne
---

# Comprehensive run (v0.4) — `python tests/bench_full.py`

36 unit tests pass (`test_neuron_db` 10, `test_secure` 7, `test_capabilities` 13, `test_plastic` 6). Three axes. Dev sandbox, Python 3.10, single core.

## A) Speed

| operation | result |
|---|---|
| neuron creation, in-memory (3 facts) | ~30,000 / sec |
| neuron creation, SQLite (durable) | ~920 / sec |
| recall latency, N=100 | ~66 µs |
| recall latency, N=1,000 | ~380 µs |
| recall latency, N=5,000 | ~630 µs |
| recall latency, N=10,000 | ~1.4 ms |
| write throughput (`observe`) | ~55,000 facts / sec |
| recall: static vs plastic | 380 µs vs 530 µs (1.4×) |
| secure put (AES-GCM + keyed index) | ~390 / sec |
| secure get | ~270 µs |
| router recall (2,000 facts / 16 shards) | ~0.6 ms |
| arithmetic op | ~12 µs |

Recall grows with N (a cue with many candidates scores more of them); keep a hot neuron to hundreds of facts or shard with the router. Plastic adds ~40%, still sub-ms.

## B) Performance over time (10,000-turn plastic session)

| turn | facts | recall@1 (rotating probe) | latency | store |
|---|---|---|---|---|
| 2,000 | 1,015 | 50% | 317 µs | 38 KB |
| 4,000 | 1,724 | 72% | 539 µs | 64 KB |
| 6,000 | 1,630 | 82% | 454 µs | 61 KB |
| 8,000 | 1,383 | 90% | 416 µs | 52 KB |
| 10,000 | 1,262 | 74% | 413 µs | 48 KB |

Latency stays flat (~0.3–0.5 ms) across 10k turns; consolidation holds facts bounded (~1,300, not unbounded) under continuous writes. Accuracy dips happen when consolidation prunes a sampled fact — correct forgetting, not a regression.

## C) Neural plasticity — measured effects

**Adaptation** — two facts collide on "meeting"; reinforcing "monday" overtakes recency after **2 uses**:

```
reinforce(monday)   winner   w(mon)  w(fri)
        0           friday    1.00    2.00
        2           monday    4.00    1.00
       40           monday   42.00    1.00
```

**Forgetting** — untouched fact decays cleanly (half_life=50): `0.99, 0.70, 0.49, 0.25, 0.06` at 0/25/50/100/200 ticks idle.

**Association** — co-activating two unrelated facts grows their Hebbian link 0.5 → 8.0 over 5 rounds; spreading activation then surfaces the associate.

**Consolidation (sleep)** — 5 duplicates + 1 decayed fact consolidate 6 → 1 (4 merged, 1 pruned), recall preserved.

None of these are visible to a static recall@1 test — the reason the measurement was reframed (`docs/PLASTICITY.md`).
