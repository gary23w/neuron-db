# neuron-db

An associative memory you can run anywhere. Write facts in plain language, recall them by
meaning. No tables, no schema, no embeddings, no model required. The core is pure Rust with
zero dependencies and compiles to WebAssembly; durable storage, encryption, and an HTTP
server are opt-in features.

```sh
./build.sh
neuron --db app.db turn me 'my plan is pro'
neuron --db app.db get  me 'what plan am i on?'      # -> pro
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

## Why it's interesting

- **Tiny.** A fact's retrieval state is stems and scalars, not a dense vector — about 48
  bytes/fact serialized, roughly **130× more facts per GiB** than a 1536-dim float vector
  store. See **[docs/STORAGE.md](docs/STORAGE.md)**.
- **Fast and dependency-free.** Microsecond recall, no GPU, no model. The default build runs
  in a 1 MB WebAssembly worker.
- **Adaptive.** The plastic tier learns from use with O(1) scalar updates — no re-embedding,
  no re-indexing.

The trade: it does cue and association recall, not semantic similarity. `"the thing I use to
get online"` won't match `"wifi password"` without an embedding. It's scalar-first by design.

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

- [docs/API.md](docs/API.md) — data model and every operation (library / CLI / HTTP)
- [docs/DEPLOY.md](docs/DEPLOY.md) — build, install, Docker, env, backups
- [docs/STORAGE.md](docs/STORAGE.md) — storage density vs vector databases
- [SECURITY.md](SECURITY.md) — encryption and access model

MIT licensed. Author: gary23w.
