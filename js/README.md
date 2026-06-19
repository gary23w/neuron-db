# neuron-db

The official host binding for **[neuron-db](https://github.com/gary23w/neuron-db)** — a pure-Rust
associative memory plus the **gary-neuron** dispatcher cortex, compiled to WebAssembly. One
dependency-free ES module that turns the raw `mem(ptr, len)` byte-FFI into a typed, self-validating
API. Runs anywhere WebAssembly does: Cloudflare Workers, the browser, Node, Deno, Bun.

```bash
npm i @gary23w/neuron-db
```

The package bundles a lean **store-only** `neuron_core.wasm` (~352 KB) — recall, associative recall,
multi-hop chains, variables, dump/load. For the **cortex** (`route()` / `dispatch()`), mount a cortex
build (see [Cortex](#cortex) below).

## Quickstart

```js
import { NeuronDB } from "@gary23w/neuron-db";

// Node:
const db = await NeuronDB.forNode(new URL("@gary23w/neuron-db/wasm", import.meta.url).pathname);
// Browser / Deno:
const db = await NeuronDB.forBrowser(new URL("@gary23w/neuron-db/wasm", import.meta.url));
// Cloudflare Workers (CompiledWasm import — see wrangler config):
import wasm from "@gary23w/neuron-db/wasm";
const db = NeuronDB.fromModule(wasm);

db.observeMany("session:42", [
  "the deploy region is us-west-2",
  "the api key is zeta-9931",
]);

db.recall("session:42", "what is the deploy region?");   // ["the deploy region is us-west-2", ...]
db.recallScored("session:42", "api key", 5);             // [{ fact, coverage, overlap }, ...]
db.assess("session:42", "api key");                      // { coverage, hits, hasValue, ... }

const blob = db.dump("session:42");                      // persist anywhere (KV, localStorage, fs)
db.load("session:42", blob);                             // restore in a later request
```

It bakes in the things every embedder otherwise rediscovers the hard way: the alloc → write → mem →
answer_ptr → dealloc memory dance (once), the per-op encode/decode/parse, the **robust recall default**
(`recall()` uses the broad `recall_many` set, never the single-hit op that abstains on multi-word
queries), and a **self-describing surface** — it reads the build's `ops` at load and *fails loud* on an
op the build doesn't expose, instead of the silent empty string the raw FFI returns.

## Cortex

The cortex (gary-neuron) is the dispatcher: each turn it routes `answer` / `escalate` / `fetch` /
`store` over the recalled working set. On a **cortex build**, `route()` is the headline — recall +
dispatch in one call:

```js
const turn = db.route("session:42", userMessage);
// turn => { type: "answer" | "escalate" | "fetch" | "store", value, facts }
switch (turn.type) {
  case "answer":   reply(turn.value); break;                 // memory answered it — no big-model call
  case "escalate": reply(await yourLLM(userMessage, turn.facts)); break;
  case "fetch":    reply(await search(turn.value)); break;
  case "store":    db.observe("session:42", userMessage); break;
}
```

Raw model text never reaches the user — a degenerate generation resolves to `escalate`, not garbage.

A cortex build is ~7.6 MB (the int8 model is baked in with `include_bytes!`). It's available from the
[neuron-db repo](https://github.com/gary23w/neuron-db) (`worker/neuron_core.wasm`) or build it:
`cargo build --lib --release --target wasm32-unknown-unknown` in `rust/neuron-core`. Load it like any
other wasm; `db.hasCortex` tells you whether the build you mounted has it.

## API

`NeuronDB.fromModule(mod, opts?)` · `fromBytes(bytes, opts?)` · `forNode(path, opts?)` ·
`forBrowser(url, opts?)` — `opts` is `{ httpHandler?, imports? }`.

| | |
|---|---|
| **write** | `observe(scope, fact)` · `observeMany(scope, facts)` |
| **read** | `recall(scope, q, k?)` · `recallScored(scope, q, k?)` · `recallValue(scope, q)` · `assess(scope, q)` · `assoc(scope, q, k?, hops?)` · `chain(scope, start, path)` |
| **vars** | `setVar` · `getVar` · `vars(scope)` · `delVar` |
| **lifecycle** | `forget(scope, match?)` · `stats(scope)` · `scopes()` · `dump(scope)` · `load(scope, blob)` |
| **cortex** | `route(scope, q, {k?})` · `dispatch(q, facts?)` · `hasCortex` |
| **introspect** | `listOps()` · `supports(op)` · `caps()` · `resolveCaps(hostCaps?)` |

Full TypeScript types ship with the package (`index.d.mts`).

## License

MIT © gary23w
