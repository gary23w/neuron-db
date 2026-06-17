# neuron-db v0.3.0

An associative memory you can run anywhere — and the flat-cost long-term memory for an LLM.
Pure-Rust core (zero deps, compiles to WebAssembly); durable storage, encryption, an HTTP
server, and an MCP server are opt-in. This is the first tagged release.

## Highlights

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
