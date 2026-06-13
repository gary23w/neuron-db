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
sub-linear recall, 100% on distinct keys, and a clear collision boundary on near-duplicate
keys that the next design phase is built to handle.
