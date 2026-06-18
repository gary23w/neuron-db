---
license: mit
tags:
  - associative-memory
  - tiny-language-model
  - micro-llm
  - neuron-db
  - gary-neuron
  - on-device
---

# gary-neuron

The micro **dispatcher cortex** at the center of [neuron-db](https://github.com/gary23w/neuron-db). It sits between a host model (or app) and the store: given the working set neuron-db recalled plus the turn, it **routes**. Tiny (a few-million-parameter int8 transformer), baked into the binary (`include_bytes!`) — no GPU, no runtime download, no network. Built by **[gary23w](https://github.com/gary23w)**.

## Role — the middle layer

gary-neuron is the always-on middle, not a chat model. For each turn it emits exactly one route:

| route | meaning |
|---|---|
| `ANSWER` | memory has it — serve from the store (no host-model round trip) |
| `ESCALATE` | memory can't — hand the turn up to the larger host model |
| `FETCH <topic>` | live-world question — go to the web |
| `STORE <fact>` | a declarative — remember it |

It holds across realistic recalled working sets and chains multi-fact lookups. On the `ANSWER` route the **exact value comes from neuron-db's deterministic recall** — the cortex decides the route; the store grounds the bytes. (For pure recall you don't need the model at all; the store returns values deterministically.)

## Use (Rust)

The neuron-db crate bundles the model — nothing to load at runtime:

```rust
use neuron_core::model::{GaryModel, Dispatch};

let m = GaryModel::embedded();
match m.dispatch(&working_set, "what is the api key?") {
    Dispatch::Answer(_) => { /* serve the store's recalled value */ }
    Dispatch::Escalate  => { /* hand up to the host model */ }
    Dispatch::Fetch(t)  => { /* fetch `t` from the web */ }
    Dispatch::Store(f)  => { /* remember `f` */ }
}
```

## Files

```
cortex.bin          packed int8/f32 weights (layout in manifest.tsv)
manifest.tsv        tensor manifest the loader reads
vocab.tsv / petite_merges.txt   byte-level BPE tokenizer
config.json         compact runtime config
```

Served by the neuron-db Rust crate (and its WASM build), not by `transformers`.

MIT. Part of the gary-neuron family by [gary23w](https://github.com/gary23w).
