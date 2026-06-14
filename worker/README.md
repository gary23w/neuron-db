# neuron-db on Cloudflare Workers

neuron-db's Rust core — store + the emergence cortex — compiled to a single WebAssembly
module and served from a Worker. No Python, no external model, no network at request time;
the model is baked into the `.wasm`.

## Verified

The WASM runs in the JS WebAssembly VM (the same engine Workers use). Self-test in Node:

```
selftest code = 3   store_recall = YES   cortex_copied_value = YES   answer = "vekam73"   (~155 ms cold)
POST /v1/think {"query":"how many participants?","facts":["only the first 84,512 ... badges"]} -> {"answer":"84,512"}
```

## Deploy

```
./build.sh            # rust/neuron-core -> neuron_core.wasm (needs the rust toolchain once)
npx wrangler login    # your Cloudflare account
npx wrangler deploy
```

Then:

```
curl https://neuron-db.<you>.workers.dev/                       # health + selftest
curl -X POST https://neuron-db.<you>.workers.dev/v1/think \
  -d '{"query":"what is the wifi password?","facts":["the wifi password is vekam73"]}'
# -> {"answer":"vekam73"}
```

## How it works

`index.mjs` imports `neuron_core.wasm` (a CompiledWasm module per `wrangler.jsonc`),
instantiates it once per isolate, and calls three C-ABI exports:

- `selftest()` — store recall + cortex generation, returns a bitmask.
- `alloc(n)` / `run(ptr,len)` — host writes `"query\nfact1\nfact2..."` into wasm memory,
  `run` builds a store from the facts, recalls the working set, and the cortex answers;
  the answer is read back via `answer_ptr()` / `answer_len()`.

## Honest notes

- This Worker is **stateless per request** (facts come in the request body). For a durable,
  multi-tenant store, back it with a Durable Object per neuron (see `docs/guide/MEMORY_HARNESS.md`)
  and persist `Neuron::dump()` in DO storage between calls.
- A real deploy needs your Cloudflare account; it was verified in a local WASM VM here, not
  on Cloudflare's edge.
- The bundled cortex is the ~2k-vocab emergence model — great at copying values from a
  bounded working set, not a general chatbot. For exact recall you don't need it at all
  (`Neuron::recall` returns the value deterministically).
