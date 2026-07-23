# Quantum tier vs normal — benchmarks

Head-to-head measurements of the quantum-teleportation tier against the normal (main-branch)
store, at three levels: the engine (in-process, release build), the CLI (spawn-inclusive —
the model nl-veil actually runs, forking `neuron.exe` per op), and the real nl-veil pipeline
(`veil rag ingest` + recall through the veil-shaped verbs). Reproduce the engine numbers with:

```sh
cargo run --release --features quantum-db --example quantum_bench
```

**Method.** The engine comparison runs both paths inside one binary (the plain path is the
identical code the normal branch ships, so feature-off = main behavior). The CLI and pipeline
comparisons use two release binaries: one built from `main` in a clean worktree
(`--features secure,server,semantic`) and one from this branch (`+ quantum-db`). Same
workloads, separate databases, two rounds each. Windows x64, release profile, temp-dir DBs.

---

## Engine (µs/op, 5,000-fact scope, 2,000 reads, 300 pairs, 10,000 links)

| Op | Normal path | Quantum path | Verdict |
|---|---:|---:|---|
| observe (bulk) | 4.9 | 4.9 (same code) | identical |
| recall, indexed cue | 5.2 | 4.7 (dormant) | **parity** — the dormant fast-path is one atomic load |
| recall, quantum state present in db | 5.2 | 10.8 | +5.6 µs absolute, only paid while write-once/superposition state exists |
| recall, hub-cue full-scan regime | 1291.3 | 1289.7 | parity (−0.1%) |
| move an association | 1222.8 (recall+forget+observe) | **184.2 (teleport)** | **6.6× faster**, and atomic — no copies-both-exist window |
| one-shot secret cycle | 1009.6 (observe+get+forget) | **156.6 (write_once + burning read)** | **6.5× faster**, and the cleanup can never be forgotten |
| hold/resolve an ambiguous value | store N facts, forget losers by hand | 38.4 store_super + 300.4 recall_super | no classical equivalent is atomic |
| entangle (write a link) | — | 181.7 | includes both endpoint-index maintenances |
| link lookup @ 10k links | — | 6.5 | the teleport hot path |
| relay cascade (64-hop chain) | — | 162.1 /hop | **unbounded** — settles at the chain's end, not at a cap |

The classical "move" and "one-shot" columns are the honest equivalents a caller performs
today: three separate verbs, including `forget`'s secure-delete WAL checkpoint — a cost the
tier's atomic rewrite never pays, and a window in which both copies of the association exist.

## Optimizations applied for fairness (before → after)

| Tweak | Effect |
|---|---|
| Endpoint indexes on `entanglements` (`idx_ent_src`/`idx_ent_dst`) + two indexed probes instead of one OR-scan | find_entanglements @10k links **519.7 → 6.5 µs (~80×)**; teleport 253 → 184 µs. Cost: entangle 149 → 182 µs (index upkeep — the right trade; links are read far more than written) |
| Dormant fast-path (`quantum_dormant`, a cached per-process hint) | quantum-aware read on a store with no quantum state: **2 SQL probes → 1 atomic load**; parity with plain recall |
| `prepare_cached` on every quantum read statement | no re-parse per lookup on the hot path |

**A base-engine win the benchmark surfaced** (applies to the NORMAL tier too, shipped in the
same branch): single `recall` gathered candidates from every cue-stem posting ungated, so any
query sharing a common word with a large scope scored the whole scope — `recall_many` already
had the df-gate; `recall` never got it. Applying the shared gated gather took the indexed-cue
read from **1,455.8 → 5.2 µs (~280×)** on a 5,000-fact scope with recurring schema words, with
zero behavioral drift (the full 31-suite matrix stays green, and answer parity vs the main
binary held 6/6 on a 2,800-fact veil-ingested document).

Also shipped alongside: **hops are unbounded by default** everywhere. `hops = 0` (now the
default on CLI/MCP/HTTP/wasm) spreads until the frontier drains — convergence, not a budget —
with an activation floor so decayed-out branches stop propagating. The quantum mirror is
`teleport_cascade`: relays until the graph settles, terminated by e-bit conservation (a cycle
drains its finite budget; verified by test).

## CLI level (ms/op, spawn-inclusive — nl-veil's execution model; two rounds)

| Phase | main | quantum | main #2 | quantum #2 |
|---|---:|---:|---:|---:|
| observe ×120 | 21.43 | 20.97 | 18.84 | 19.34 |
| get ×120 | 17.28 | 17.01 | 16.96 | 16.54 |
| veil KV cycle ×40 (forget+observe+export+forget) | 71.74 | 70.70 | 69.29 | 68.50 |
| assoc ×30 | 16.90 | 16.78 | 16.66 | 16.28 |

Process spawn dominates (~17 ms floor); the tier adds **no measurable CLI overhead** — the
quantum binary matched or edged main in 7 of 8 cells (all within noise).

## nl-veil pipeline (real `veil rag ingest`, 174 KB document → 2,800 facts)

| | main binary | quantum binary |
|---|---:|---:|
| ingest wall time (r1 / r2) | 0.26 s / 0.22 s | 0.22 s / 0.22 s |
| facts distilled / stored | 2800 / 2800 | 2800 / 2800 |
| recall answers on the ingested doc | — | **6/6 identical to main** |

## Unit suites

- Branch, full native matrix `secure,server,mcp,http,compress,quantum-db`: **31 suites green** (includes 12 quantum protocol/durability tests, 2 cascade tests, the HTTP-surface pin, and the unbounded-spread convergence test).
- Branch with quantum OFF and default features: green (the tier compiles away completely).
- `main` (clean worktree), its full matrix: green.

## Verdict

**Quantum beats normal overall.** Criteria and outcomes:

1. **No base-op regression** — engine parity when dormant, +5.6 µs absolute only while quantum
   state exists; CLI and pipeline parity; 6/6 answer equivalence. ✓
2. **Workflow efficiency** — the tier's ops are 6.5–6.6× faster than their classical
   multi-verb equivalents, and atomic (1 spawn instead of 3 in nl-veil's per-op model). ✓
3. **Net effect of merging** — the branch also carries the df-gated recall (~280× on
   schema-heavy scopes) and unbounded-hops convergence, which speed the NORMAL tier. ✓

Merged to main on these results; nl-veil migrated to the quantum binary.
