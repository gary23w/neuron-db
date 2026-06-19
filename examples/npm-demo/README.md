# neuron-db npm demo

A small TypeScript app that installs **[`@gary23w/neuron-db`](https://www.npmjs.com/package/@gary23w/neuron-db)**
from npm and uses it as its database — a login flow plus a memory console. Accounts *and* every memory
live in neuron-db itself; nothing else is involved.

```bash
cd examples/npm-demo
npm install        # pulls vite + typescript and @gary23w/neuron-db from npm
npm run dev        # copies the cortex wasm, then serves on http://localhost:8076
```

> This example depends on the **published** package (`"@gary23w/neuron-db": "^0.1.0"`), so it's a real
> `npm install` of what anyone else would get. The binding is the same code as the in-repo
> [`js/`](../../js) source.

## What it shows

- **neuron-db as the user store.** Sign up / log in: each account is a PBKDF2-hashed record in the
  `system:users` scope (`src/auth.ts`). *(Demo only — it's all client-side, so it isn't real auth.)*
- **A per-user associative memory.** Your facts live in `user:<name>`, persisted to `localStorage`
  via `dump()` / `load()`, so they survive a reload.
- **The console** (`src/main.ts`) — `remember`, `recall` (with coverage + µs timing), `assoc`
  (spreading activation), `chain` (multi-hop), `forget`, `stats`, and **`ask`** which routes the turn
  through the gary-neuron **cortex** (`answer` / `escalate` / `fetch` / `store`).

The entire neuron-db surface used here is the typed binding:

```ts
import { NeuronDB } from "@gary23w/neuron-db";
const db = await NeuronDB.forBrowser("/neuron_core.wasm");
db.observeMany("user:ada", facts);
db.recallScored("user:ada", "deadline", 6);   // [{ fact, coverage, overlap }]
db.route("user:ada", "what is the api key?");  // { type, value, facts }  (cortex)
```

`ask` needs a cortex build of the wasm; `copy-wasm.mjs` grabs one from the repo (the npm package bundles
only the lean store-only build). On a store-only build `db.hasCortex` is false and `ask` says so.
