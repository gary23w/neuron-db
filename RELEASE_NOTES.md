# neuron-db v0.4.0

Everything merged since v0.3.0. The headline is keyed facts: a fact can be stored under a key,
replaced by that key and removed by that key, in the core, the durable store, the WebAssembly
build and the CLI.

## What's new in 0.4.0

### Keyed facts
- **Core.** `Neuron::upsert_keyed(key, text)` stores the text under a key and drops whatever the
  key held before. `Neuron::forget_key(key)` removes it. The key matches exactly and never enters
  the fact text, so replacing one record cannot remove another record whose text happens to
  contain the same substring.
- **One vocabulary.** `Upsert` and `ForgetKey` are in the shared op vocabulary (`op::apply`) and
  implemented for both the durable `NeuronDB` and the in-memory WebAssembly store.
- **CLI.** `neuron upsert <scope> <key> <text...|->` and `neuron forgetkey <scope> <key>`, both with
  `--json`. A missing key or text is a usage error (exit 2); a keyless `forgetkey` never falls
  through to a scope-wide forget.
- **WebAssembly.** New `mem()` ops `upsert`, `forgetkey` and `recallkeyed`. `recallkeyed` returns
  `fact`, `coverage`, `overlap` and `key` per hit; `recallscored` keeps its three fields.
- **JS binding.** `upsert`, `forgetKey` and `recallKeyed`, with types.
- **Persistence.** A keyed fact carries its key in the dump's flag field (`<flag>:<key>`).
  Existing dumps load unchanged.
- An upsert whose text stores nothing leaves the key's existing facts in place.
- Not yet: the interactive shell, the MCP server and the HTTP server do not expose the keyed ops,
  and keyed recall is WebAssembly-only.

### Fixes
- **JS binding: `forget()` no longer trims its match.** A match ending in a space (`"rec12 "`) was
  sent without the space and also removed `rec120`, `rec1200` and so on. The match is now sent as
  given; only tabs and newlines are replaced.
- **WebAssembly: results larger than 1 MiB are no longer cut off.** A `dump` of a large scope used
  to return a partial blob with no error. Oversized results now go to a heap buffer and
  `answer_ptr()` points at whichever buffer holds the last result. A host must call
  `answer_ptr()` after each call, as the binding and the demos already do.

### Also since 0.3.0
- **Scored recall on the CLI.** New verb `recallscored <scope> <query…> [--k N] [--semantic]
  [--across]`: top-k facts with their numbers (`coverage`/`overlap`/`exact`/`idx`) as JSON —
  the CLI face of the wasm tier's op, so a spawn-per-op host carries numeric confidence across
  its own seams instead of re-deriving it from prose.
- **Consolidation-time contradiction marks, on by default.** `observe` now scans the nearest
  stored facts for conservative polarity conflicts (negation asymmetry; same-head "X is Y" with
  fully divergent values) before storing. The new fact still stores — write cheap, adjudicate at
  recall — and both sides are marked in a `{scope}::contested` sidecar scope that `recallscored`
  surfaces as `contested` plus the disagreeing sibling (`with`). `--no-check` opts out; when
  nothing conflicts the output stays byte-identical. Default-on is deliberate: capability rides
  the binary, not the argv, so no host flag can ever be mistaken for fact text by an older build.
- **Sharded concurrency.** The durable cache + its connection are partitioned by top-level scope
  family, so independent tenants recall in parallel: cache-hot recall scales ~8× across 16 cores
  (it was flat under one global lock). `examples/concurrency_bench`.
- **Append-log durable writes.** A single `observe()` is now one O(new) INSERT, not a whole-scope
  blob rewrite — single durable writes go ~3,300 → ~25,000/s and stay **flat as a scope grows to
  millions**; batch ingest ~235k/s.
- **Hybrid recall, fused with RRF.** `recall_blended` now fuses lexical + semantic rankings via
  Reciprocal Rank Fusion: best hit@3 (97%) and MRR (0.84) in a coder-memory A/B vs grep and a
  pure-vector ranker, at ~4× lower latency. `examples/coder_bench`.
- **Bounded semantic recall** (windowed, capped embed cache), **stem interning** (~2× less per-fact
  stem RAM), and a **df-aware + parallel** recall path keep large-scope recall flat.
- **Opt-in personality layer** (`personality` feature, OFF by default): a Big-Five + temperament
  `Persona` that, when explicitly attached, modulates the affect/stance dynamics. A neutral persona
  is byte-identical to none. New CLI: `stance` / `mood` / `affect`.
- **Quantum tier** (experimental, `quantum` feature; durable state and CLI/HTTP ops with
  `quantum-db`): entangled cross-scope links, teleport recall, burn-after-reading facts and
  superposed values. A structural analogy, not hardware. See `docs/guide/QUANTUM_TELEPORTATION.md`.
- **Statistics tier** (opt-in): `fisher` adds two-class Fisher linear-discriminant heads, and
  `topics` adds deterministic latent-Dirichlet-allocation topics with topic-gated scope-wide
  fuzzy recall.
- **Durable semantic space** (`semantic-db` feature): per-word context vectors persist in a lazy
  `sem_kv` side table, so fuzzy recall survives process restarts.
- **Trust floor** (`trust` feature): learned per-tag-class trust and trust-weighted associative
  recall, with `trust_get` to read a class's trust.
- **Document recall.** Hits carry their episode index; `context` stitches top hits into in-order
  passages; `assoc --across` reaches document sub-scopes; `read` pages a whole document in order.
  The WebAssembly build has `context` and `readpage`.
- **Strengthen-only plasticity.** `strengthen <scope> <match...>` bumps facts containing the
  match and never creates a fact.
- **`stats-batch`.** Read-only statistics for several scopes in one database process.
- **More CLI verbs.** `assoc`, `chain`, `reinforce`, `why`, `route` and `caps`, plus fact-pack
  preload across the CLI, MCP and WebAssembly.
- **JS binding as a package.** The binding ships under `js/` as `@gary23w/neuron-db`. The npm
  package has not been republished with this release.

### Tests
269 tests pass under the CI feature set
(`secure,server,mcp,http,compress,quantum-db,fisher,topics,semantic-db`).

---

# neuron-db v0.3.0

An associative memory you can run anywhere — and the flat-cost long-term memory for an LLM.
Pure-Rust core (zero deps, compiles to WebAssembly); durable storage, encryption, an HTTP
server, and an MCP server are opt-in. This is the first tagged release.

## Highlights

### gary-neuron v3: a ~7M-parameter dispatcher baked into the build
gary-neuron is a 6,973,952-parameter int8 transformer (E=256, H=8, L=8, vocab=2048, 512-token
context). It ships inside the WebAssembly/binary build via `include_bytes`, runs on CPU with no
GPU and no download. It is not a selectable chat model. It is the always-on middle layer between
a host model (the reasoning neocortex it escalates to) and neuron-db: each turn the hippocampus
emits exactly one route — `ANSWER`, `ESCALATE`, `FETCH <topic>`, or `STORE <fact>`. On `ANSWER`
the literal value comes from neuron-db's deterministic recall: the hippocampus decides the route,
the store grounds the bytes.

Held-out results: routing triage (ANSWER vs ESCALATE vs FETCH) is 100% on each class; grounded
ANSWER accuracy is 88–98% across working sets from 1 to 18 facts; two-hop chaining is 100%.
Numeric comparison is the acknowledged limit, sitting near chance.

A browser/WASM dispatch is about 54 ms after a SIMD128 pass over the matmuls, down from ~172 ms;
the hippocampus is also fast natively. neuron-db recall under that load measures p50 ~3.9 µs and
p99 ~36 µs over 10k queries against 7000 facts across 1000 scopes (indexed recall plus a
write-behind store).

### The in-browser lab is now *reasoned-routed* — the model thinks, neuron-db grounds
The live lab (`docs/lab.html`) runs a WebLLM model entirely in your tab with neuron-db
(as WASM) as its long-term memory. Each turn:
- **Perceive** — neuron-db computes a **knowledge-gap signal** (coverage: how much of the
  question it already holds). Phrasing-independent, so it generalizes across any wording.
- **Reason** — the model picks **one action** from the gap signal + recalled facts: answer /
  web_search / deep_research / store / set_rule. No brittle regex router decides intent.
- **Act** — neuron-db executes (fetch, store, chain) and the model answers grounded in the result.

This dissolves the classic failure modes: it searches the web when it doesn't know instead of
confabulating, commands can't be mis-stored as facts, and a knowledge gap forces a fetch.
**Working memory** (recent dialogue) plus a persistent per-chat **focus** keep a multi-turn task
on subject across scroll.

### Live web access from WASM — no proxy, no worker
The http-enabled build declares a `host_http` import the browser fulfills with `fetch`, so the
WASM owns the request and reaches public **CORS** APIs directly (Wikipedia, DuckDuckGo,
open-meteo). A recursive **deep-research** crawl follows Wikipedia's link graph into neuron-db.
Nothing leaves your browser; the production build omits the import entirely.

### New WASM ops + a ~100× recall fix
- New `mem()` ops: `assess` (the gap signal in the core), `recallscored`, `vars`, `delvar`,
  `dump`/`load` (serialize + rehydrate a scope), `scopes`.
- `recall()` no longer does an eager O(N) `root_scan` for ordinary questions — `value`/`assess`/
  `chain` drop from ~6 ms to ~60 µs on a 5k-fact scope (full recall suite still green).

### Site + demos
- Balanced two-column hero on desktop; fixed a page-scroll regression.
- All eight interactive demos made responsive across phone → desktop (no overflow, no overlap).
- Faster first model load: defaults to the smallest fast model (~0.9 GB), one-time and cached.

## Core (recap)
- Tiers: `Neuron` (in-memory) · `PlasticNeuron` (adaptive) · `NeuronRouter` (sharded) ·
  `NeuronDB` (durable SQLite) · `SecureNeuronDB` (AES-256-GCM) · HTTP `serve` · `neuron-mcp` (stdio MCP).
- Measured vs a markdown-dump LLM memory: 100% multi-hop accuracy at flat ~1.1k tokens/turn;
  selective recall stays ~µs out to 1,000,000 facts; ~130× denser than a 1536-d vector store.

## Tests
135 tests pass with all features; each feature now also compiles and tests in isolation.

## Build
```sh
./build.sh                                                  # native (sqlite + secure + server)
cargo install --path rust/neuron-core --features mcp        # the neuron-mcp stdio server
docs/demos/build-wasm.sh                                    # the in-browser wasm
```
