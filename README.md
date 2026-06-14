# neuron-db

An associative memory you can run anywhere — and the flat-cost **long-term memory for an
LLM**. Write facts in plain language, recall them by meaning, and **link neurons across
arbitrarily deep chains at no extra model cost**. No tables, no schema, no embeddings, no
model required. The core is pure Rust with zero dependencies and compiles to WebAssembly;
durable storage, encryption, an HTTP server, and an MCP server are opt-in features.

```sh
./build.sh
neuron --db app.db turn me 'my plan is pro'
neuron --db app.db get  me 'what plan am i on?'      # -> pro
```

▶ **[Watch the synapse fire in 3D](https://gary23w.github.io/neuron-db/demos/synapse-3d.html)** — the real Rust core, in your browser.

## LLM memory: link infinite neurons, at flat cost

An LLM's context window is small; neuron-db is the memory that lives *outside* it. A
relational question — *"the timezone of the manager of the owner of Aurora"* — normally
makes the model recall a fact, wait, recall the next, wait… N hops = N+1 model calls.
**`recall_chain`** collapses that: the model sends one path, and the synapse walks the
whole chain server-side, each hop a microsecond recall. **Depth is paid in microseconds,
not model turns.**

Measured live against the memory most LLMs use today (a markdown file of all facts dumped
into context every turn), `gpt-4o-mini`, a 700-fact user:

| | neuron-db | markdown-dump |
|---|---|---|
| multi-hop accuracy (1/2/3 hops) | **100%** | 83–100% |
| context cost / turn | **~1.1k tokens (flat)** | 2.7k → 67k (linear) |
| cost at 6,000 facts | **$0.19 / 1k-q** | $10.06 / 1k-q |
| model calls per answer, any depth | **2** | 1 |
| needle recall to 50k facts | **100% · ~16 µs** | context-bound |

The markdown-dump reinjects the whole memory every turn and eventually overruns the window;
neuron-db injects only what it recalled — flat cost, no ceiling, matching or beating
accuracy. Full numbers: **[docs/COMPARISON.md](docs/COMPARISON.md)** · how fast recall fires:
**[docs/SYNAPSE.md](docs/SYNAPSE.md)**.

**Mount it in one line.** `neuron-mcp` is a native std-only stdio MCP server — point any MCP
client (Claude Desktop/Code, Cursor) at the binary and your model gets `remember` / `recall`
/ `recall_chain` as tools. No Node, no Python, no HTTP process. See
**[docs/MEMORY_HARNESS.md](docs/MEMORY_HARNESS.md)** and **[examples/mcp_chat/](examples/mcp_chat/)**.

```sh
cargo build --release --features mcp --bin neuron-mcp
```

## What it is

A **fact** is a sentence (`"the api key is zeta-9931"`); neuron-db keeps the surprising word
as the retrievable value and indexes the rest as cues. A **scope** is a named bag of facts
(`user:42`), and a database is a file of scopes. You insert by stating things and read by
asking questions — retrieval is associative (cue overlap), so you never declare a column or
write SQL. Full model and every operation: **[docs/API.md](docs/API.md)**.

```rust
use neuron_core::db::NeuronDB;
let db = NeuronDB::open("app.db", 500);
db.observe("user:42", "the plan is pro");
db.get("user:42", "what plan?");            // Some("pro")
db.forget("user:42", Some("plan"));         // delete by substring
```

## Tiers

- **`Neuron`** — in-memory associative store (default, std-only). Recall in microseconds.
- **`PlasticNeuron`** — recall adapts: strength on use, decay on disuse, Hebbian links, and
  a neurotransmitter-style spreading-activation recall.
- **`NeuronRouter`** — shard across many small neurons and fan a query out (`--features` none).
- **`NeuronDB`** — durable database of scopes in one SQLite file (`--features sqlite`).
- **`SecureNeuronDB`** — AES-256-GCM values, per-scope secret never stored (`--features secure`).
- **HTTP server + `serve` binary** — one endpoint per scope (`--features server`).
- **`neuron-mcp`** — stdio MCP server so any LLM mounts neuron-db as memory (`--features mcp`).

## Why it's interesting

- **Tiny.** A fact's retrieval state is stems and scalars, not a dense vector — about 48
  bytes/fact serialized, roughly **130× more facts per GiB** than a 1536-dim float vector
  store. See **[docs/STORAGE.md](docs/STORAGE.md)**.
- **Fast and dependency-free.** Microsecond recall, no GPU, no model. The default build runs
  in a 1 MB WebAssembly worker.
- **Adaptive.** The plastic tier learns from use with O(1) scalar updates — no re-embedding,
  no re-indexing.

The trade: it's scalar-first, lexical recall — not learned semantic similarity. It bridges
morphology (`owner`/`owned`/`owns`) and a curated synonym ontology (`reports to`↔`manager`,
`lives in`↔`city`) for free, but open-vocabulary paraphrase (`"the thing I use to get
online"` → `"wifi password"`) would still want an embedding tier. In exchange you get
microsecond recall, ~130× the density of a vector store, and no model on the hot path.

## Build

```sh
./build.sh                                            # sqlite + secure + server
cargo build --release --features "sqlite secure server"
cargo install --path rust/neuron-core --features "sqlite secure server"
```

Default build is zero-dependency and targets `wasm32-unknown-unknown`; the native tiers are
opt-in features so they never touch the wasm build. Running it as a service (and Docker):
**[docs/DEPLOY.md](docs/DEPLOY.md)**.

## Security

Embedded SQLite has no login — control access by filesystem permissions, the HTTP server's
`NEURON_DB_KEY` bearer token, or per-scope encryption with `SecureNeuronDB`. Details:
**[SECURITY.md](SECURITY.md)**.

## Implementations

The store and service tiers are canonical in **Rust** (`rust/neuron-core/`). A Python
reference implementation — including the gary-neuron cortex bridge and training tooling —
is preserved on the **`legacy-python`** branch.

## Examples

Runnable code and integration guides are in **[examples/](examples/)** — quickstart, a
chatbot-memory loop, per-user profiles, sharding, encrypted secrets, HTTP clients
(curl/browser/Node/Python), and guides for wiring neuron-db into a **[chatbot](examples/guides/CHATBOT.md)**
or an **[existing API](examples/guides/EXISTING_API.md)**.

## Docs

- [docs/SYNAPSE.md](docs/SYNAPSE.md) — how fast recall fires for an LLM, measured
- [docs/COMPARISON.md](docs/COMPARISON.md) — multi-hop + neuron-db vs the markdown-dump memory
- [docs/MEMORY_HARNESS.md](docs/MEMORY_HARNESS.md) — mount as LLM memory; the MCP server & tools
- [docs/API.md](docs/API.md) — data model and every operation (library / CLI / HTTP)
- [docs/DEPLOY.md](docs/DEPLOY.md) — build, install, Docker, env, backups
- [docs/STORAGE.md](docs/STORAGE.md) — storage density vs vector databases
- [SECURITY.md](SECURITY.md) — encryption and access model

MIT licensed. Author: gary23w.
