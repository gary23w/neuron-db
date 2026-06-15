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
| multi-hop accuracy (1/2/3 hops) | **100%** | 92–100% (degrades) |
| context cost / turn | **~1.1k tokens (flat)** | 9.9k → 447k (linear, overflows) |
| cost at 6,000 facts | **$0.19 / 1k-q** | $10.06 / 1k-q |
| model calls per answer, any depth | **2** | 1 |
| selective recall in 1,000,000 facts | **100% · ~6 µs** | context-bound |

The markdown-dump reinjects the whole memory every turn and eventually overruns the window;
neuron-db injects only what it recalled — flat cost, no ceiling, matching or beating
accuracy. Measured to **50,000 facts**, neuron-db answers at **100%** on ~1.1k tokens of
context while the equivalent markdown memory (~447k tokens) can't fit a 128k window at all;
selective recall stays **flat ~6 µs out to 1,000,000 facts** (38 MB), and appending a fact
then recalling it costs ~10 µs/turn even at that size (incremental indexing). Full numbers:
**[docs/guide/COMPARISON.md](docs/guide/COMPARISON.md)** · how fast recall fires:
**[docs/guide/SYNAPSE.md](docs/guide/SYNAPSE.md)** · scale + raw metrics:
**[docs/guide/BENCHMARKS.md](docs/guide/BENCHMARKS.md)**.

**Mount it in one line.** `neuron-mcp` is a native stdio MCP server — point any MCP client
(Claude Desktop/Code, Cursor) at the binary and your model gets the full memory toolset:
`recall` / `recall_associative` (spreading activation) / `recall_chain` / `recall_value` /
`remember` / `note` (typed neurons: fact·user·instruction·var) / `recall_var` / `forget` / `stats`.
No Node, no Python, no HTTP process. The binary prints its own paste-ready client config, so
setup is two commands:

```sh
cargo install --path rust/neuron-core --features mcp   # builds + installs neuron-mcp
neuron-mcp --config                                    # prints config for Claude Desktop / Cursor / Claude Code
```

See **[docs/guide/DEPLOY.md](docs/guide/DEPLOY.md)**, **[docs/guide/MEMORY_HARNESS.md](docs/guide/MEMORY_HARNESS.md)**, and **[examples/mcp-chat/](examples/mcp-chat/)**.

## What it is

A **fact** is a sentence (`"the api key is zeta-9931"`); neuron-db keeps the surprising word
as the retrievable value and indexes the rest as cues. A **scope** is a named bag of facts
(`user:42`), and a database is a file of scopes. You insert by stating things and read by
asking questions — retrieval is associative (cue overlap), so you never declare a column or
write SQL. Full model and every operation: **[docs/guide/API.md](docs/guide/API.md)**.

```rust
use neuron_core::db::NeuronDB;
let db = NeuronDB::open("app.db", 500);
db.observe("user:42", "the plan is pro");
db.get("user:42", "what plan?");            // Some("pro")
db.forget("user:42", Some("plan"));         // delete by substring
```

## How it works (in plain words)

Think about how you remember a conversation. You don't keep a perfect transcript, and you don't
re-grow your brain on every sentence — you keep **discrete little memories**, and when something
reminds you of one, it comes back. neuron-db works the same way: each fact is a small **episode**,
and a *neuron* is just the bag of episodes for one user or agent.

- **Storing a fact.** You hand it a sentence in plain language (*"the API key is zeta-9931"*). It
  does **not** turn that into a giant numeric vector or update any model weights. It files the
  sentence away as one memory, tagged with its key words (its *cues*) and the one surprising word
  worth fetching back (`zeta-9931`). A memory costs about as much as the text itself — a few dozen
  bytes — because there's no embedding and no model to retrain.
- **Recalling it.** You ask *"what's my API key?"* It pulls the meaningful words out of your
  question, looks them up in a small index that maps each word to the memories that mention it, and
  the matching memories **light up**. It scores them by how well they fit, picks the best, and hands
  back the value. Because it jumps straight to the handful of memories that share your cue — instead
  of scanning everything — recall stays in **microseconds whether you have ten facts or ten million**.
- **Why it's cheap and scales.** A vector database spends 1–12 KB per fact on a dense embedding so
  it can search by meaning; neuron-db spends roughly the size of the text, because its search key is
  just the words plus a couple of numbers. The same disk holds **~100× more facts**, and recall cost
  depends on how many memories match your cue — not how many you've ever stored.

That's the whole idea: **memory as cheap, discrete episodes plus a word-to-memory index** — much
closer to how a brain files away a moment and recalls it on a cue than to how a search engine
indexes documents or a model bakes facts into its weights. The result is a model with effectively
unlimited, storage-bound memory that stays fast at any size. Full mechanism:
**[docs/guide/DESIGN.md](docs/guide/DESIGN.md)**.

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
  store. See **[docs/guide/STORAGE.md](docs/guide/STORAGE.md)**.
- **Fast and dependency-free.** Microsecond recall, no GPU, no model. The default build runs
  in a 1 MB WebAssembly worker.
- **Adaptive.** The plastic tier learns from use with O(1) scalar updates — no re-embedding,
  no re-indexing.

Recall is scalar-first and layered: exact/stem cues, morphology (`owner`/`owned`/`owns`), a
curated synonym ontology (`reports to`↔`manager`), and — with `--features semantic` — a
**corpus-distributional semantic space** (Random Indexing, std-only, no model) that grounds
meaning in co-occurrence so open-vocabulary paraphrase resolves too: trained on text,
`"the thing I use to get online"` recalls the `wifi` fact. The fuzzy tier is a fallback, so
the lexical path keeps its microsecond, ~130×-denser, no-model-on-the-hot-path profile. See
**[docs/guide/SEMANTIC.md](docs/guide/SEMANTIC.md)** (incl. the book-ingestion test: 600k words in ~0.5s,
~3 ms lexical recall over 29k facts, and a semantic space that learns `whale`→`ship/sea/sperm`
from the text alone).

## Build

```sh
./build.sh                                            # sqlite + secure + server
cargo build --release --features "sqlite secure server"
cargo install --path rust/neuron-core --features "sqlite secure server"
```

Default build is zero-dependency and targets `wasm32-unknown-unknown`; the native tiers are
opt-in features so they never touch the wasm build. Running it as a service (and Docker):
**[docs/guide/DEPLOY.md](docs/guide/DEPLOY.md)**.

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

- [docs/guide/STATUS.md](docs/guide/STATUS.md) — **where the project is, what just shipped, what's next**
- [docs/guide/SYNAPSE.md](docs/guide/SYNAPSE.md) — how fast recall fires for an LLM, measured
- [docs/guide/SEMANTIC.md](docs/guide/SEMANTIC.md) — the fuzzy semantic space + the book-ingestion test
- [docs/guide/COMPARISON.md](docs/guide/COMPARISON.md) — multi-hop + neuron-db vs the markdown-dump memory
- [docs/guide/MEMORY_HARNESS.md](docs/guide/MEMORY_HARNESS.md) — mount as LLM memory; the MCP server & tools
- [docs/guide/API.md](docs/guide/API.md) — data model and every operation (library / CLI / HTTP)
- [docs/guide/DEPLOY.md](docs/guide/DEPLOY.md) — build, install, Docker, env, backups
- [docs/guide/STORAGE.md](docs/guide/STORAGE.md) — storage density vs vector databases
- [docs/guide/DESIGN.md](docs/guide/DESIGN.md) — how it works: write · recall · abstain
- [docs/guide/PLASTICITY.md](docs/guide/PLASTICITY.md) — the memory that adapts, decays, and associates
- [docs/guide/BENCHMARKS.md](docs/guide/BENCHMARKS.md) — speed, recall, and capacity numbers
- [SECURITY.md](SECURITY.md) — encryption and access model

MIT licensed. Author: gary23w.
