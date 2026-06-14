---
license: mit
library_name: numpy
tags:
  - associative-memory
  - tiny-language-model
  - pure-numpy
  - neuron-db
  - gary-neuron
---

# gary-neuron-emergent

The emergence-trained cortex behind [neuron-db](https://github.com/gary23w/neuron-db) — a
1.13M-parameter, pure-NumPy GPT that learned to **read its context window and copy values
out of it**. It is the optional "thinking" tier that sits on top of the neuron-db store:
the store retrieves a small working set, this cortex generates the answer over it.

## What it is

- **arch:** gpt-numpy — 8 layers, 4 heads, dim 96, 384-token context, vocab 2048
- **params:** 1,128,384
- **trained:** step 33,597 (curriculum: copy-from-window + masked answer loss + abstention)
- **emergence:** in-context QA probe reached 5/6 — it copies unseen values from the window
  (`how many participants? -> 84,512`, `what is the wifi password? -> vekam73`) and learned
  to abstain ("i don't know right now.") when the answer isn't present.
- **val:** answer-token loss ~0.20; chat perplexity held (~val_soda 2.2)

## Files

```
cortex.bin            packed weight tensors (gpt-numpy layout; see manifest.tsv)
manifest.tsv          weight tensor manifest (name/shape/offset) for cortex.bin
vocab.tsv             byte-level BPE vocab (2048)
petite_merges.txt     BPE merges
config.json           E/H/L/BLK, vocab, param count, trained step
```

## Use

The neuron-db Rust crate **bundles this model into the binary** (`include_bytes!`), so there
are no files to load at runtime. Build it from the working set the store retrieved:

```rust
use neuron_core::model::GaryModel;

let m = GaryModel::embedded();                 // cortex + tokenizer, baked in at compile time
let facts = vec!["the launch is on Friday".to_string()];
let answer = m.think(&facts, "when is the launch?", 10);   // -> "Friday"
```

The `think` binary (`rust/neuron-core/src/think.rs`) is a thin CLI over the same call.

> **Legacy (removed):** the original prototype loaded these files in Python via
> `gpt_numpy` and a `neuron_db.bridge` helper (`from neuron_db.bridge import
> GaryNeuronBridge`). That Python code is gone from the main tree — it's preserved only on
> the `legacy-python` branch. The Rust crate above is the supported path.

## Honest notes

- It was trained on a ~2k everyday-token vocabulary in a `U:/G:` fact format. It excels at
  copying normalized facts out of a bounded window; it is not a general chatbot.
- For exact recall you don't need it at all — neuron-db's store returns the value
  deterministically. This cortex is for generation/association over the working set.

MIT. Part of the gary-neuron family by [gary23w](https://github.com/gary23w).
