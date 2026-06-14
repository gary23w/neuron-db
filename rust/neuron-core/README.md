# neuron-core (Rust)

The fast reimplementation of the neuron — the associative memory at the heart of
[neuron-db](https://github.com/gary23w/neuron-db). Standard library only, no
dependencies, builds offline.

The Rust core is the **canonical implementation** (on `main`); a Python reference — including
the gary-neuron cortex bridge and training tooling — is preserved on the `legacy-python`
branch. Same behavior, ~7× faster creation and ~5× faster recall, a single static binary, and
true multi-core concurrency (no GIL).

```bash
cargo test                                                   # default (std-only) store tests
cargo test --features "sqlite secure server mcp semantic"    # every tier
cargo run --release --bin bench                              # microbenchmark
```

## Tiers (Cargo features)

| feature | tier | what it adds |
|---|---|---|
| *(default)* | `Neuron` · `PlasticNeuron` · `NeuronRouter` | in-memory associative store, plasticity, sharding — std-only, wasm-clean |
| `sqlite` | `NeuronDB` + `neuron` CLI | durable scopes in one SQLite file |
| `secure` | `SecureNeuronDB` | AES-256-GCM values; per-scope secret supplied per call, never stored |
| `server` | HTTP server + `serve` | one endpoint per scope over a std `TcpListener` |
| `mcp` | `neuron-mcp` | stdio MCP server: `recall` / `recall_chain` / `remember` / `forget` / `stats` |
| `semantic` | `SemanticSpace` | corpus-distributional fuzzy recall (Random Indexing, no model, no deps) |

Tests live in `tests/` (recall, db_tier, secure_tier, router, turn, plastic, inference,
db_comprehensive, semantic_tier). Server-side multi-hop (`recall_chain`), the book-ingestion
test (`examples/book_ingest.rs`), and the LLM memory-bank harness are documented in
[`docs/guide/MEMORY_HARNESS.md`](../../docs/guide/MEMORY_HARNESS.md).

## API

```rust
use neuron_core::Neuron;
let mut n = Neuron::new(500);
n.observe("the wifi password is hunter2");
let r = n.recall("what is the wifi password?").unwrap();
assert_eq!(r.value, "hunter2");
```

MIT.
