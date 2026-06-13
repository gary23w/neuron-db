# neuron-core (Rust)

The fast reimplementation of the neuron — the associative memory at the heart of
[neuron-db](https://github.com/gary23w/neuron-db). Standard library only, no
dependencies, builds offline.

This is the **`rust` branch refactor** of the Python prototype (preserved on the
`python-prototype` branch). Same behavior, ~7× faster creation and ~5× faster recall, and
a single static binary with true multi-core concurrency (no GIL).

```bash
cargo test            # 6 parity tests: recall, value isolation, abstention,
                      # relation binding, 400/400 distinct-key scale, dump/load
cargo run --release --bin bench
```

## Status

| ported | not yet |
|---|---|
| store: observe / recall / value-isolation / abstention | SQLite database layer |
| stem index, relation-binding, multi-number isolation | encrypted tier (SecureNeuronDB) |
| minimal serialize / load | HTTP server, MCP tools |

Roadmap to parity and the LLM memory-bank harness is in
[`docs/MEMORY_HARNESS.md`](../docs/MEMORY_HARNESS.md) on `main`.

## API

```rust
use neuron_core::Neuron;
let mut n = Neuron::new(500);
n.observe("the wifi password is hunter2");
let r = n.recall("what is the wifi password?").unwrap();
assert_eq!(r.value, "hunter2");
```

MIT.
