# neuron-db — project status

Where the project stands, what just shipped, and what's next on the menu. Updated **2026-06-15**.

## Where we are

neuron-db is a working associative-memory core (pure Rust, std-only, compiles to WebAssembly) with
durable, encrypted, HTTP, and MCP tiers layered on top. As an LLM's external **hippocampus** the
core loop is feature-complete: a model mounts `neuron-mcp` over stdio and gets a
**passive-capture → associative-recall → typed-write** memory that is window-independent and
storage-bound.

It's been verified end-to-end with real models — `gpt-4o-mini`, `o4-mini`, and `o3` — driving the
MCP tools, including a reasoning model deliberating over which tool to call. And it holds at scale:
**selective recall stays flat (~µs) as a scope grows**, exact recall is unbounded by total size, and
the fuzzy/semantic fallback is bounded so a large scope never makes a miss expensive.

### Public MCP tools
| tool | what it does |
|---|---|
| `recall` | top-k relevant facts as a memory block (lexical/associative default; `rank="semantic"` opt-in) |
| `recall_associative` | spreading activation over the shared-entity graph — surfaces connections plain recall misses |
| `recall_value` | the single isolated value for a direct question |
| `recall_chain` | multi-hop relations resolved server-side in one call, flat per-hop cost |
| `remember` | store durable facts in plain language |
| `note` | mint a **typed** neuron — `fact` / `user` / `instruction` / `var` |
| `recall_var` | exact read of a named variable set with `note(kind=var)` |
| `forget` / `stats` | delete (cascades to typed sub-scopes) / report scope size |

### Tiers
`Neuron` (in-memory) · `PlasticNeuron` (adaptive: strength/decay/Hebbian links/spreading) ·
`NeuronRouter` (sharded) · `NeuronDB` (durable SQLite) · `SecureNeuronDB` (AES-256-GCM) ·
HTTP server + `serve` · `neuron-mcp` (stdio MCP). Default build is zero-dependency and wasm-ready.

## Latest verified numbers (release, 2026-06-15)

- **Unit suite:** 131 tests, 0 failures (21 binaries).
- **Write:** single `observe` 3.3k/s; batch `observe_many` 235k/s; opt-in write-behind closes the
  single-observe gap on a growing scope (default stays immediately durable).
- **Recall:** selective cue **~4.2 µs, flat 1k→50k facts**; `recall_chain` ~12 µs/hop (flat to 50 hops);
  warm cache hit 44 µs; cold reload (200 facts) 0.8 ms; reopen + index a 50k-fact scope 0.20 s.
- **Footprint (int8):** 100 documents / 4,000 facts → **13.2 MB resident, 5.9 MB after `compact()`**
  (~2.2×); SQLite store ~451 B/fact; embedding cache int8 ~397 B/fact.
- **Scale:** the lexical-**miss** fallback is now bounded — flat across 5k→12k facts — while exact
  recall via the inverted index is scope-independent.

See **[BENCHMARKS.md](BENCHMARKS.md)** for methodology and the full tables.

## Recently shipped

- **Neuron-first redesign.** `recall_associative` (spreading activation over the shared-entity graph)
  is a first-class recall path; the semantic space was demoted from the silent default to an
  **opt-in ranking signal** — it ranks, it doesn't create memory.
- **Typed neurons** via `note`: `fact` / `user` / `instruction` / `var`, plus `recall_var` (named
  variables with exact read + upsert). A harness can re-inject `instruction` neurons every turn so
  a standing rule survives the short context window.
- **int8 quantization** of the semantic space and embedding cache (~2–3× smaller), with a
  train-then-serve `compact()`.
- **Opt-in write-behind** (`NEURON_FLUSH_EVERY`) for high single-observe throughput; flushed on
  eviction, `flush_all()`, and shutdown. Default = immediate per-write durability.
- **Performance:** allocation-free `load()`, inverted index built during load, sparse
  spreading-activation, and a cap on the O(N) recall miss-path so it stays flat as a scope grows.
- **Reliability:** an adversarial-testing round fixed real bugs — variable key collision,
  multi-word value truncation, `dump`/`load` tab/newline corruption, non-atomic var upsert, `forget`
  not cascading to typed sub-scopes, unknown-`kind` silent drop, and a JSON-array parser edge.
- **Reference harness** (`neuron-chat-lab`): live two-pane lab — passive capture, a per-document
  register, and optional **reasoning-model** support so a deliberating model drives the tools itself.
- **Measured head-to-head vs dense vectors** (`docs/guide/VS_VECTORS.md`): on one frozen dataset with
  identical scoring, neuron-db is ~360× faster end-to-end and ties or wins every class except
  zero-shot paraphrase (which the semantic re-rank closes to parity given a corpus). MCP round-trip
  measured at ~0.5 ms over real stdio — the store is never the bottleneck; the model is.
- **Cleanup:** a `cargo clippy` pass to industry standard (autofix + justified numeric-kernel allows)
  and a fix for terse colon-delimited entries the min-word filter was dropping on first insert.

## What's next (the menu)

Grounded in a read of the real crate (`lib.rs` / `db.rs` / `router.rs` / `server.rs`):

- **Persist the inverted index (the next build).** `dump()` discards the derived stems/positions, so
  `Neuron::load` re-runs `encode()` and rebuilds the index on every cold scope reload — a cost the
  "flat µs" story hides across the 256-scope LRU. Append an opt-in (`NEURON_PERSIST_INDEX`) V2 dump
  section, backward-compatible via the existing tolerant trailing-field parser; pure std, ~2–3× disk
  on the durable tier only.
- **Scale out to a swarm — feasible today, no new deps.** Scopes are already the partition unit, so a
  ~10-line consistent-hash placement + a `FederatedRouter` (a coordinator over remote `serve` nodes,
  reusing the HTTP recall response's existing merge key over `std::net::TcpStream`) gives a real
  cross-host fleet. Writes route to the deterministic owner → per-scope sequential consistency for
  free. Two load-bearing rules: typed sub-scopes must co-locate with their parent (so a cascade
  delete stays node-local), and chain/associative queries route by the start entity (the spreading
  graph lives in one `Neuron`).
- **Cross-restart persistence of the plastic graph + semantic space.** Hebbian links and the
  Random-Indexing space are rebuilt per session; `dump`/`load` via the existing int8 quantize path
  makes adaptation and meaning durable (true long-term memory, not a session cache).
- **Recall explanations + confidence (S, high-trust).** The `coverage`/`overlap`/`exact` signals are
  already computed and dropped at the MCP boundary — surface them as an opt-in verbose suffix and a
  coarse high/med/low confidence label to back the honest-abstention story.
- **Hub-cue recall.** Broad cues that match most facts are still O(N); an IDF/hub pre-filter
  (mirroring the spreading `dfcap`) would collapse them toward the selective ~µs path.
- **Onboarding + ops.** Export/import + backup, per-call hybrid-rank tuning knobs, an atomic-counter
  metrics endpoint, and more one-command MCP client configs.

**North star:** a genuinely *infinite context* — the AI remembers everything, storage-bound, with
recall that stays fast at any size. The items above are the path there.
