---
license: mit
tags:
  - associative-memory
  - tiny-language-model
  - micro-llm
  - neuron-db
  - gary-neuron
  - grounded-qa
  - on-device
---

# gary-neuron

A **~6.9M-parameter** reasoning cortex that sits **between [neuron-db](https://github.com/gary23w/neuron-db) and the user**: the store retrieves a small working set of facts, and this tiny model reads that window and produces a grounded answer over it. Built by **[gary23w](https://github.com/gary23w)**.

It is deliberately *micro* — small enough to bake straight into a binary (`include_bytes!`, no runtime download, no GPU) yet large enough to **select the right fact among distractors, chain two facts, abstain when the answer isn't present, and copy arbitrary values it has never seen.**

## Architecture

A pre-LN decoder-only transformer, served in **int8**:

| | |
|---|---|
| params | **6,908,416** (~6.9M) |
| dim (E) | 256 |
| layers | 8 |
| heads | 8 |
| context (BLK) | 256 tokens |
| vocab | 2048 byte-level BPE |
| activation | tanh-GELU |
| norm | LayerNorm (eps 1e-5), pre-LN |
| positions | learned |
| output | weight-tied to token embedding |
| weights | per-row **int8** (matmuls) + f32 LayerNorm/bias |

The forward pass is **byte-identical** to the Rust runtime in the neuron-db crate, so a checkpoint exported from training swaps into `cortex.bin` with **zero code changes**.

## How it was trained

A 7M model can't be a general LLM — but it *can* master one **narrow** distribution: *read a working set + a query → grounded answer* (knowing what to retrieve is neuron-db's job, not the model's). So it is trained on exactly that:

- **Distillation seed** — a strong teacher generates natural `(facts, query) → ideal answer` examples across the target tasks, plus a **gary23w identity layer** so the model always knows *it is gary-neuron, built by gary23w*.
- **Synthetic copy-from-window curriculum** (~16k examples) — keys/values/relations randomized every instance, with the **value shape decoupled from the key**, so the model learns the *mechanism* ("emit the span attached to the asked key") rather than memorizing values. This is what makes copy/select **generalize to values it has never seen**.
- Answer-masked next-token cross-entropy; AdamW; cosine LR + warmup.

## Evaluation

Held-out exact-match accuracy, **N=80 fresh random cases per task** (values never seen in training), greedy decoding, measured in the real int8 runtime:

| task | gary-neuron ~6.9M | prior 1.13M cortex |
|---|---|---|
| copy (arbitrary value shapes) | **90.0%** | 11.2% |
| number-copy (comma integers) | **100%** | 8.8% |
| select among 3–5 distractors | **70.0%** | 2.5% |
| abstain (answer absent) | **87.5%** | 0% |
| **multi-hop (2 facts)** | **91.2%** | 0% |
| compare (which is more/fewer) | 53.8% | 0% |
| **mean** | **~82%** | **~3.8%** |

## Use

The neuron-db Rust crate **bundles this model into the binary** (`include_bytes!`) — nothing to load at runtime:

```rust
use neuron_core::model::GaryModel;

let m = GaryModel::embedded();                       // cortex + tokenizer, baked in
let facts = vec![
    "the api key is zeta-9931".to_string(),
    "the launch is on Friday".to_string(),
];
let answer = m.think(&facts, "what is the api key?", 16);   // -> "zeta-9931"
```

The `think` binary (`rust/neuron-core/src/think.rs`) is a thin CLI over the same call.

## Files

```
cortex.bin          packed int8/f32 weight tensors (layout in manifest.tsv)
manifest.tsv        tensor manifest: "#cfg E H L BLK VOCAB", then name/qtype/offset/bytelen/rows/cols
vocab.tsv           byte-level BPE vocab (2048)
petite_merges.txt   BPE merges
config.json         E/H/L/BLK, vocab, param count, author
```

`config.json` is a compact custom config (not a `transformers` config) — the weights are served by the neuron-db Rust crate, not by `AutoModel`.

## Honest limits

- **compare** (deciding which of two numbers is larger) sits at ~54% — numeric magnitude reasoning is genuinely hard for a char/BPE model this small. Treat it as a stretch task, not a guarantee.
- **paraphrase** beyond copy-from-window is weak; the 2048-token vocab fragments rare words.
- Decoding is **greedy**, so **ungrounded open-ended generation** is not a goal — with no facts the model tends to fall back to its identity line. In the application it always receives a retrieved working set.
- For exact recall you don't need the model at all — neuron-db's store returns the value deterministically. The cortex is for generation/association over the working set.

MIT. Part of the gary-neuron family by [gary23w](https://github.com/gary23w).
