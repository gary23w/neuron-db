# NeuronDB — Benchmarks & Test Findings

This document records a comprehensive unit-testing and benchmarking pass over the
**NeuronDB** tier (the durable SQLite-backed store, `rust/neuron-core/src/db.rs`),
the bugs it surfaced and the patches applied, the measured performance
characteristics, and recommendations.

NeuronDB is the durable tier: a database of neurons in one SQLite file
(`rusqlite`, bundled), thread-safe via one connection plus an in-memory LRU cache
behind a `Mutex`. It is feature-gated behind `sqlite`.

---

## 1. Methodology

| | |
|---|---|
| SQLite | bundled via `rusqlite` 0.31, `journal_mode=WAL` |
| Build | `--release` (`opt-level=3`, `lto=true`) for benchmarks; debug for tests |

Numbers below are from a single run on one modern multi-core desktop; treat them as
relative characteristics (how things scale), not absolute guarantees. Re-run the
benchmarks on your own target for hardware-specific figures.

Reproduce:

```sh
# tests
cargo test --features "sqlite secure server"
cargo test --features sqlite --test db_comprehensive

# benchmarks (release)
cargo run --release --bin bench                                  # in-memory core
cargo run --release --features sqlite --example db_bench         # NeuronDB micro-benchmark
cargo run --release --features sqlite --example scenario_bench   # user-testing benchmark
cargo run --release --features sqlite --example llm_memory_bench # LLM-memory needle-in-haystack
```

> **Build tip:** point `CARGO_TARGET_DIR` at a path outside any cloud-synced folder.
> Synced folders can lock or block freshly-built build-script executables and slow
> every compile.

---

## 2. Test coverage

**107 tests pass** across the workspace (`--features "sqlite secure server mcp semantic"`).

| Suite | Tests | Area |
|---|---:|---|
| lib unit tests | 13 | stemmer, recall, encode internals |
| `tests/db_comprehensive.rs` | 53 | the full NeuronDB surface + edge cases |
| `tests/recall.rs` | 8 | core recall semantics |
| `tests/turn.rs` | 6 | conversation routing brain |
| `tests/plastic.rs` | 6 | plastic neuron |
| `tests/db_tier.rs` | 5 | original NeuronDB smoke tests |
| `tests/scale.rs` (new) | 5 | incremental-index correctness + recall at scale |
| `tests/router.rs` | 3 | sharded router |
| `tests/secure_tier.rs` | 3 | encrypted store |
| `tests/semantic_tier.rs` | 3 | distributional semantic recall |
| `tests/inference.rs` | 2 | cortex |
| **Total** | **107** | |

The new `db_comprehensive.rs` covers:

- **observe / recall / get** — value isolation (pick the value nearest the cue),
  abstention on no match, `get` == `recall().value`.
- **numeric recall** — `how many …?` isolates the number; short numeric values
  (`17`) bypass the 3-char floor.
- **observe_many (batch)** — equivalence to N individual `observe`s; empty-slice no-op.
- **recall_many (top-k)** — returns multiple hits, respects `k`.
- **forget** — by substring (case-insensitive), forget-all, no-op on miss, counts.
- **stats / neurons** — capacity, turn counter, scope listing.
- **durability** — WAL survival across reopen, turn-counter persistence,
  1000-fact reload, tab-in-text roundtrip (the `\t` field-delimiter collision).
- **LRU cache** — data survives eviction + cold reload; writes after eviction
  append (not replace); concurrent writes past the 256-slot cap don't corrupt.
- **max_facts capacity** — evicts oldest, keeps newest; `turn` flags `capacity_reached`.
- **unicode / robustness** — CJK and accented values roundtrip; emoji/CJK/curly-quote
  input never panics.
- **turn brain** — statement→ack, question→recall, yes/no, arithmetic, idk, smalltalk.
- **scope isolation** — no cross-scope leakage.
- **concurrency** — distinct scopes in parallel, no lost writes on one hot scope,
  readers + writer with no torn reads, eviction under contention.

---

## 3. Bugs found & patched

Two genuine bugs were found and fixed during this pass. Both are in shared core
code (`lib.rs`, `turn.rs`), so the fixes benefit every tier (core, router, server,
wasm), not just NeuronDB.

### 3.1 UTF-8 panic in the stemmer (severity: high — crash / DoS)

`stem1()` truncated stems by **byte** index (`w[..6]` / `w[..4]`). On multibyte
UTF-8 (e.g. `東京`, byte 4 lands inside the second character) this **panics the
process**. A second instance: `w1()` stripped a possessive with `t[..t.len()-2]`,
which cuts mid-character when the apostrophe is a multibyte curly quote
(`’`, U+2019, 3 bytes) — i.e. ordinary smart-quoted input like `Sarah’s`.

Impact: observing or recalling arbitrary non-ASCII text could crash the server /
demo. Since the demo streams arbitrary user text, this was remotely triggerable.

Fix: truncate by **chars** (`chars().take(n)`) and strip the possessive with
`strip_suffix` (char-safe). For ASCII the behavior is identical.

Regression tests: `unicode_cjk_value_roundtrips`, `accented_value_roundtrips`,
`curly_apostrophe_does_not_panic`, `emoji_and_cjk_input_do_not_panic`.

### 3.2 Arithmetic question with trailing `?` fell through (severity: medium)

`find_math()` parsed operands directly, so `"what is 12 * 11?"` tokenized the
second operand as `"11?"`, which fails to parse — the query dropped to "i don't
know" instead of answering `132`. Natural arithmetic questions end with `?`.

Fix: strip surrounding punctuation from operand tokens before parsing.

Regression test: `turn_arithmetic`.

### 3.3 Over-aggressive stemming caused false recalls (severity: medium — precision)

`stem1()` truncated every 4–7 char word to its first **4 chars**, so `plan`,
`plane`, `plant`, and `planet` all collapsed to `plan`. The user-testing benchmark
(§5.3) caught this: asking *"what is my favorite planet?"* of a store holding
*"my plan is pro"* wrongly returned `pro` on **100% of users** — a recall where it
should have abstained.

Fix: keep 5–6 char words at **5 chars** (8+ still truncate to 6). This eliminated
the false positive while keeping all real recall categories at 100% and the full
test suite green — i.e. precision improved with no loss of intended recall.

Regression test: `stemming_does_not_overmatch_planet_to_plan`; quantified by the
"no-collision check" probe in `scenario_bench` (0% → 100%).

> Residual: words sharing a 6-char prefix still collide (`token` vs `token9` both
> stem to `token`), and 4-char words are unchanged. This is inherent to a prefix
> stemmer; alias maps (§ recall) and selective cues remain the mitigations.

---

## 4. Known limitations (asserted, not bugs)

These are intentional design behaviors of the language layer. They are now pinned
by tests so any change is visible. They are limitations worth knowing when wiring
NeuronDB into an app.

| Limitation | Cause | Test |
|---|---|---|
| `observe()` drops any text containing `?` | questions must not be stored as facts | `limitation_observe_drops_text_with_question_mark` |
| 2-char alpha values (e.g. `CA`, `US`) not retrievable | <3-char non-numeric tokens are dropped as candidates; some are stopwords | `limitation_two_char_alpha_value_not_retrievable`, `limitation_single_content_word_fact_dropped` |
| Stem collisions (residual) | stems are 6-char prefixes, so words sharing a 6-char prefix still collide (`token` vs `token9`); a value sharing the cue's stem is treated as the cue word and dropped from the value pool. The worst 4-char collisions (`planet`→`plan`) were fixed in §3.3. | documented in `tab_in_text_survives_reopen`; `stemming_does_not_overmatch_planet_to_plan` |

If you need to store country/currency/state codes or URLs with query strings,
store the full name, or store the value where the asked-about word differs from it.

---

## 5. Benchmark results

### 5.1 In-memory core (`bench` bin)

```
creation: 138,639 neurons (3 facts each) / sec
recall accuracy (N=500, distinct keys): 500/500
```

### 5.2 NeuronDB tier (`db_bench` example, release)

```
1) single observe(), fresh scope each ............ 3,326 writes/sec
2) single observe() into ONE growing scope:
     first  2,000 facts ........................... 2,125 writes/sec
     second 2,000 facts ........................... 1,314 writes/sec   (blob-rewrite cost)
3) batch observe_many() 50,000 facts ............. 266,570 writes/sec
4) recall latency vs scope size:
     scope  1,000 facts: selective 3.30 us/call | broad/shared cue   263 us/call
     scope 10,000 facts: selective 3.42 us/call | broad/shared cue 2,647 us/call
     scope 50,000 facts: selective 3.54 us/call | broad/shared cue 13,228 us/call
5) cache hit 53 us/call ; cold reload (200 facts from sqlite) 687 us/call
6) reopen + first-recall a 50,000-fact scope (load + index) ... 0.206 s
7) concurrent observe, 8 threads x 2,000 writes .. 2,185 writes/sec (lock-serialized)
```

### 5.3 User-testing benchmark (`scenario_bench` example, release)

Simulates **1,000 users**, each with a per-user scope (`user:{i}`) holding a
7-fact profile (name, plan, city, manager, editor, timezone, seat count), then
asks each a battery of natural questions and scores accuracy + latency. This tests
NeuronDB the way an app actually uses it.

```
ingest: 7,000 facts across 1,000 scopes in 2.0s (3,493 writes/sec)

recall accuracy by category:
  direct lookup      100.0%  (3000/3000)   "what is my name / plan / city?"
  alias paraphrase   100.0%  (3000/3000)   "subscription"->plan, "boss"->manager, "ide"->editor
  numeric            100.0%  (1000/1000)   "how many seats do i have?"
  abstention         100.0%  (2000/2000)   unstored, non-colliding -> correctly None
  OVERALL            100.0%  (9000/9000)

stemming precision probe (separate, post-fix): 100.0% (1000/1000)

recall latency over 10,000 queries: p50 3.7 us | p95 28.6 us | p99 32.0 us | max 96 us
```

Takeaway: for the intended per-user memory workload, NeuronDB is **100% accurate**
across direct lookups, alias paraphrases, numeric extraction, and abstention, at
**single-digit-microsecond p50** latency. (Before the §3.3 fix, the stemming probe
scored 0%.)

### 5.4 LLM-memory: needle-in-a-haystack (`llm_memory_bench` example, release)

The use case is external LLM memory: the store holds far more than fits in a context
window, and each turn injects only the top-k relevant facts. So the test plants 50
known needles among a growing haystack of distractors and measures whether recall
still finds them.

```
haystack | single-recall | top-8 block | needle p50/p95   | broad cue (worst case, O(N))
  1,000  | 50/50  100%   | 50/50  100% | 16.4 / 17.4 us    |    247 us
  5,000  | 50/50  100%   | 50/50  100% | 16.5 / 17.5 us    |  1,224 us
 20,000  | 50/50  100%   | 50/50  100% | 16.5 / 17.5 us    |  5,162 us
 50,000  | 50/50  100%   | 50/50  100% | 16.7 / 17.4 us    | 14,348 us
```

Takeaway: with 50 needles hidden among up to **50,000 facts**, both single recall and
the top-8 injectable block stay **100% accurate**, and needle latency is **flat at
~16 µs** — independent of store size. The LLM only ever sees the top-k block, never the
haystack, which is what lets external memory sidestep the context-window limit. The
"broad cue" column (a word shared by every fact) is the one O(N) failure mode; give
each memory a distinct subject to avoid it.

---

## 6. Performance analysis

**Recall cost is O(matching candidates), not O(scope size).** This is the single
most important characteristic:

- A **selective cue** (a word unique to one fact) recalls in a **flat ~3.4 µs**
  whether the scope holds 1k or 50k facts — the stem→fact inverted index hits ~1
  episode. ~290k recalls/sec, independent of size.
- A **broad cue** (a word present in *every* fact) makes the candidate set the
  whole scope, so recall is **linear** in N (263 µs → 13 ms from 1k → 50k).

So a scope's recall speed is governed by the *rarity of the queried words*, not the
fact count. Per-user scopes (the intended model) are small and diverse, so recall
stays in the microsecond range.

**Writes commit immediately and rewrite the whole scope blob.** Each `observe`
serializes the entire scope and does an `INSERT … ON CONFLICT` under WAL:

- ~3.3k single writes/sec into fresh scopes; this *degrades as a scope grows*
  (2,125 → 1,314 writes/sec across the first vs second 2,000 facts) because the
  blob re-serialize is O(current size) — i.e. filling one large scope one fact at
  a time is quadratic.
- **`observe_many` is ~80× faster** (267k/sec): one load, many appends, one save.
  For event streams / bulk import, batch is mandatory.

**The LRU cache (256 scopes) hides re-parse cost** on hot scopes (53 µs warm). A
cold scope (evicted, or first touch after reopen) costs a reload + re-encode +
index rebuild — 687 µs for 200 facts, scaling with scope size; reopening and
first-recalling a 50k-fact scope is ~0.2 s.

**Concurrency buys correctness, not write parallelism.** A single connection
behind one `Mutex` serializes all writes (~2.2k/sec contended). Tests confirm: no
lost writes on a hot scope, no torn reads with concurrent readers + a writer, no
corruption when writes force cache eviction. For higher write throughput, batch or
shard across files (the per-user model lends itself to this).

---

## 7. Recommendations

Ordered by leverage:

1. **Keep scopes small and shard by entity** (`user:`, `session:`). This keeps both
   recall (candidate set) and per-write blob cost bounded. `NeuronRouter` exists
   for this. Avoid putting tens of thousands of facts in one scope.
2. **Batch writes** with `observe_many` for any stream of more than a few facts;
   single `observe` is for interactive, one-at-a-time updates.
3. **Prefer selective cues.** Recall is fast when the queried words are rare in the
   scope. Storing a distinct key per fact (vs a shared template word) gives flat
   microsecond recall at any size.
4. **Mitigate the write-scaling cost for large scopes** (future work): an append-or
   row-per-fact persistence instead of one re-serialized blob, or a write-behind
   commit so bursty single writes amortize.
5. **Short categorical values** (country/state/currency codes) and **URLs with `?`**
   are not first-class today — see §4. A flag on `observe` to keep `?` text, and a
   relaxed length floor for known-categorical values, would close the analytics gap.

---

## 8. Recall optimization & scale to 1,000,000 facts (infinite-context pass)

A later pass benchmarked the recall hot path at scale, found two O(N) costs, fixed them
(results unchanged — **107/107 tests pass**, incl. an `incremental == full-rebuild`
equivalence test), and measured how far the *total* addressable memory can grow while the
LLM's per-turn context stays flat. (These numbers supersede §5 for the recall paths.)

### 8.1 Two optimizations

**Recall hot loop.** Per candidate, `recall()` allocated a `HashSet` from the sorted stem
vector and re-tokenized + re-stemmed the whole fact for the subject-position tiebreak. Both
replaced with a `binary_search` over the already-sorted `Episode.s` plus a precomputed
per-stem position array (`Episode.pos`, built once in `encode()`).

| path | before | after | speedup |
|---|---|---|---|
| core recall @ 500 facts | 214 µs | 55 µs | ~3.9× |
| broad cue @ 50k facts | 44 ms | 10.6 ms | ~4.2× |
| `recall_chain` | 39 µs/hop | 12.6 µs/hop | ~3.1× |

**Incremental index.** `build_index()` rebuilt the entire stem→fact map (O(N)) on the first
recall after any write — ~0.94 s per turn at 1M facts. `ensure_index()` now incrementally
indexes only appended facts (O(new)); any removal nulls the index so the incremental path
only ever sees pure appends.

| pattern (1,000,000-fact store) | before | after | speedup |
|---|---|---|---|
| append 1 fact + recall it (per turn) | ~0.94 s | 10.5 µs | ~90,000× |

Trade: `encode()` now precomputes `pos`, so creation drops ~140k → ~125k facts/s — a good
trade since reads dominate writes in a memory store.

### 8.2 How big can the total context grow? (`examples/context_scale.rs`)

| total facts | selective recall | per-turn injected block | store (serialized) |
|---|---|---|---|
| 10,000 | ~6.9 µs | 15 facts / ~120 tok | 0.4 MB |
| 50,000 | ~5.5 µs | 15 facts / ~121 tok | 2 MB |
| 200,000 | ~5.8 µs | 15 facts / ~122 tok | 7 MB |
| 1,000,000 | ~5.8 µs | 15 facts / ~127 tok | 38 MB |

Selective recall is **flat at ~6 µs from 10k to 1,000,000 facts**, and the per-turn block
injected into the model stays **flat (~120 tokens)** while total memory grows 100×. That is
the "effectively infinite context": total addressable memory is bounded only by storage
(~26–33M facts/GiB), while the model's window cost is constant.

### 8.3 Serialized byte-size / density (`examples/metrics.rs`)

`dump()` stores flag + text only — **no embedding bytes**. Measured **32.5 bytes/fact →
~33M facts/GiB** for short facts (≈ text length; longer facts scale with their text). For
contrast, a single 1,536-d float32 embedding is ~6 KB/item.

### 8.4 With the LLM in the loop (`bench_compare.py`, gpt-4o-mini, recall_chain)

Same interlinked multi-hop question set, two memories, at growing sizes:

| total facts | neuron-db ctx/turn | markdown ctx/turn | neuron acc | markdown acc | neuron $/1k-q | markdown $/1k-q |
|---|---|---|---|---|---|---|
| 1,000  | 1,122 tok | 9,898 tok | 100% | 100% | $0.19 | $1.49 |
| 6,000  | 1,122 tok | 67,067 tok | 100% | 92% | $0.19 | $10.06 |
| 50,000 | 1,122 tok | ~446,863 tok (won't fit a 128k window) | 100% | — | $0.19 | — |

neuron-db's injected context is **flat at 1,122 tokens** at every size, 100% accurate (incl.
all 1/2/3-hop), exactly 2 model calls per answer. The markdown baseline grows linearly
(8.8× → 59.8×), degrades (92% at 6k — lost-in-the-middle), and at 50,000 facts (~447k
tokens) **cannot be loaded at all** — while neuron-db answers at the same flat cost.
Reproduce: `python examples/mcp-chat/bench_compare.py --sizes 1000,6000 --chain` (and
`--sizes 50000 --no-markdown --chain` for the scale point).

---

## 9. Status

- **131/131 tests pass** (`--features "sqlite secure server mcp semantic"`), 0 warnings on the
  full and wasm32 builds.
- 3 original issues fixed (UTF-8 panic, arithmetic `?`, stemming false positives), plus a later
  adversarial-testing round (§10); recall hot path + index made ~4× / O(1)-incremental (§8).
- Selective recall flat ~µs to 1,000,000 facts; per-turn LLM context flat (~1.1k tokens)
  while total memory grows unbounded; serialized density 32.5 bytes/fact (no embeddings).
- wasm32 core still builds with no features (all changes are std-only, dependency-free).
- Reproducible via `bench`, `db_bench`, `metrics`, `context_scale`, `scenario_bench`,
  `doc_storage_space`, and `examples/mcp-chat/bench_compare.py`. Project roadmap: **[STATUS.md](STATUS.md)**.

## 10. Latest run — 2026-06-15 (release)

Fresh `db_bench` + `doc_storage_space` on the current build, plus the capabilities added since §1–8.

**Throughput & recall (`db_bench`, release)**

| metric | result |
|---|---|
| single `observe`, fresh scope | 3,252 writes/s |
| single `observe`, growing scope | 1,655 → 862 writes/s (blob-rewrite cost; opt-in write-behind closes it) |
| batch `observe_many` (50k) | 234,515 writes/s |
| recall, **selective cue** | 4.18 µs (1k) · 4.34 µs (10k) · 4.44 µs (50k) — **flat** |
| recall, broad/hub cue | 205 µs · 2.07 ms · 10.6 ms (O(N); hub-filter is on the roadmap) |
| `recall_chain` | ~12.2 µs/hop (flat to 50 hops) |
| warm cache hit | 43.6 µs ; cold reload (200 facts) 0.80 ms |
| reopen + load + index (50k) | 0.196 s |

**Footprint with int8 (`doc_storage_space`, 100 docs / 4,000 facts)**

| component | value |
|---|---|
| SQLite store (text) | 1.80 MB (~451 B/fact) |
| semantic space + embedding cache | 11.38 MB → **4.11 MB** after `compact()` (int8) |
| **total resident** | **13.18 MB → 5.91 MB** after compact (~2.2×) |
| ingest | 4.2 MB/s, ~33k facts/s |

**Scale (recall miss-path).** A lexical *miss* runs the morphological + semantic fallbacks; both are
now capped to the most-recent 4,000 facts, so miss cost is **flat** as a scope grows (measured 1.06×
over 2.4× the facts), while exact recall via the inverted index stays scope-independent.

**Capabilities added since §1–8:** `recall_associative` (spreading activation); typed neurons via
`note` (`fact`/`user`/`instruction`/`var`) + `recall_var`; semantic demoted to an opt-in ranking
signal; int8 quantization; opt-in write-behind (`NEURON_FLUSH_EVERY`); allocation-free `load()` and
index-on-load. Verified live with `gpt-4o-mini`, `o4-mini`, and `o3` driving the MCP tools.
