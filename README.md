```
NEURON-DB(7)                  neuron-db manual                  NEURON-DB(7)
```

# NAME

**neuron-db** — a plastic associative memory in a single file. Write facts in plain
language, recall them by meaning, watch the memory adapt with use. No model, no
embeddings, no dependencies.

# SYNOPSIS

```
from neuron_db import NeuronDB, Neuron, PlasticNeuron, NeuronRouter, SecureNeuronDB

db = NeuronDB("memory.db")
db.turn(scope, message)            # store a fact, or ask one back
db.get(scope, query)               # -> exact value | None

python -m neuron_db serve  [--db FILE] [--host H] [--port P]
python -m neuron_db chat   [--db FILE] [--neuron ID]
python -m neuron_db demo
```

# DESCRIPTION

A **neuron** is a small associative memory: it stores declarative facts (the surprising
word becomes the value), retrieves them by cue overlap with relation-binding, isolates the
value nearest the asked-about word, and abstains when it knows nothing. A **neuron-db** is
a database of such neurons — one row each, addressable by id, isolated from one another.

Recall needs no embedding model and no neural network; it is string and set logic over a
stem index, so it runs in microseconds, in-process, anywhere Python (or, on the `rust`
branch, anywhere a static binary) runs. On top of that, the **plastic** tier makes the
memory change with use: facts gain strength when recalled, decay when ignored, and wire
together when recalled together — all as cheap scalar updates (see **WHY SCALAR**).

## ARCHITECTURE — neurons, hippocampus, neocortex in sync

Three tiers, borrowed from how a brain splits memory from reasoning. Only the bottom tier
is required; the upper two are optional layers you add when you want the store to *reason*
or *grow*.

```
   +------------------------------------------------------------------+
   |  NEOCORTEX        an LLM - or YOUR APP - the reasoner / decider   |
   |                   (decides what to ask and what to keep)          |
   +-----^------------------------------------------------+-----------+
         | grounded value / association                   | query . new fact
         | (injected into context)                        v
   +-----+------------------------------------------------------------+
   |  HIPPOCAMPUS      gary-neuron (cortex + plastic hippocampus)      |
   |                   surprise-gates writes . reads the WORKING SET   |
   |                   . consolidates during /sleep  (OPTIONAL tier)   |
   +-----^------------------------------------------------+-----------+
         | working set: a handful of facts                | reinforce . consolidate
         | (bounded window, never the whole DB)           v
   +-----+------------------------------------------------------------+
   |  NEURONS          neuron-db - the store, at scale                |
   |  scalar plasticity:  strength . decay . Hebbian links . spreading|
   |  [n0][n1][n2][n3] ............... millions of facts, us recall    |
   +------------------------------------------------------------------+

   in sync:  the neocortex asks  ->  the store retrieves a small working set
             ->  the hippocampus thinks/associates over that bounded window
             ->  answer flows up;  reinforcement + new facts flow down
             ->  /sleep folds experience into weights (the system GROWS)
```

The cost contract that makes this work: the neural tier (hippocampus) only ever runs over
the bounded working set the store hands it — never over the whole database. So plasticity
and "thinking" stay fast no matter how large the store grows.

# GETTING STARTED

You have an app and you want it to remember things. neuron-db is the database.

```
pip install -e .                       # or just import from the repo (stdlib only)
from neuron_db import NeuronDB
db = NeuronDB("app.db")                # one file; one "neuron" per user/scope

db.turn("user:42", "my plan is pro")           # write a fact in plain language
db.turn("user:42", "the trial ends on March 3")
db.get("user:42", "what is my plan?")          # -> "pro"      (exact value, no prose)
db.get("user:42", "when does the trial end?")  # -> "March 3"
```

Your app decides what to store and what to ask; the database remembers and answers.
Pick the class for the job:

```
NeuronDB        durable app storage, never forgets    <- start here
PlasticNeuron   memory that adapts / ranks by use       (optional)
SecureNeuronDB  encrypted values, key never stored      (for secrets)
NeuronRouter    chain shards past one neuron's size      (for scale)
```

Not a Python app? Run `python -m neuron_db serve` and POST to one HTTP endpoint (see
HTTP SERVER). You never need a model or an LLM for any of this.

# INSTALLATION

```
git clone https://github.com/gary23w/neuron-db && cd neuron-db
python -m neuron_db demo          # zero setup, stdlib only
pip install -e .                  # optional: installs the `neuron-db` console script
pip install -e ".[crypto]"        # optional: AES-256-GCM for the encrypted tier
```

Requires Python >= 3.9. No third-party dependencies for the core.

# COMPONENTS

**Neuron** — one associative memory, in memory. Permanent (no decay).

**NeuronDB** — a database of neurons in one SQLite file; durable; never decays. The default
for an app backend. The gary-neuron cortex is enabled by default and lazy-loads on the first
`think()` call; `get()`/`recall()` stay the pure microsecond store path, and `think()` falls
back to the store value when no model is present. Pass `model=False` to disable.

**PlasticNeuron** — a Neuron whose recall adapts: usage strength, lazy decay, Hebbian
association, spreading activation, and `consolidate()`. The store tier of the plastic
design. Recall can run as `recall_spreading()`, a neurotransmitter-style pass that releases
activation at the cued facts, gates conflicting relations off, and spreads across synapses
with reuptake decay, all in scalar microseconds. See **NEUROTRANSMITTER RECALL**.

**NeuronRouter** — chains many neuron shards into one memory; auto-spills past a per-shard
cap and fans recall out. Holds far more than one neuron with sharp recall.

**SecureNeuronDB** — encrypted neurons: AES-256-GCM values + keyed-hash index, per-neuron
secret never stored. A stolen file is opaque. See **SECURITY**.

# WHERE IS THE MODEL?

There isn't one in this repo — by design. neuron-db is the *store*: it remembers and
recalls with no model, no embeddings, no GPU. The trained gary-neuron brain (the
"hippocampus / neocortex" tiers in the diagram) is OPTIONAL and lives in its own project
(`gary-neuron-chat`; the cortex is also ported to TypeScript in `neuron-cloud`).

To let the store *generate / reason* over what it recalls, connect the model with
`neuron_db.bridge` — it loads the cortex from a checkout, it does not bundle weights here:

```
pip install neuron-db[model]                 # numpy + tokenizers
export NEURON_MODEL_DIR=/path/to/gary-neuron-chat
from neuron_db.bridge import GaryNeuronBridge
brain = GaryNeuronBridge()
brain.think(store, "when do we launch?")     # store retrieves the working set; cortex answers
```

**The emergence model is published and bundled.**
- On HuggingFace: [`gary23w/gary-neuron-emergent`](https://huggingface.co/gary23w/gary-neuron-emergent) (step 33597, 384-ctx). The Python bridge auto-downloads it if you `pip install huggingface_hub numpy tokenizers`, or set `NEURON_MODEL_DIR` to a local checkout.
- In this repo: `rust/neuron-core/model/` (int8, ~1.2 MB) is baked into the Rust binary via `include_bytes!`, so `cargo run --bin think` runs the model with no files and no network.

The store hands the model only a small working set (the bounded window in the diagram); the
model never touches the whole database. Without the bridge, neuron-db is a complete, fast,
model-free database on its own.

# API

## Neuron(max_facts=500)

```
observe(text) -> [episode]      store fact(s); a multi-sentence paste becomes several
recall(query) -> {fact,value,coverage,overlap,echo} | None
dump() -> str                   minimal serialization (text + flag, ~30 B/fact)
Neuron.load(blob, max_facts)    rebuild (index recomputed; migration-free)
fact_count
```

## NeuronDB(path="neurons.db", max_facts=500, cache_size=256)

```
turn(id, message) -> {reply, kind, wrote, facts}    conversational: store or answer
get(id, query)    -> value | None                   exact value, no prose (store, microseconds)
recall(id, query) -> {..} | None
think(id, query)  -> {answer, source, model}        compose an answer with the cortex (auto-loaded)
forget(id, match=None) -> {forgot, remaining}
stats(id) -> {facts, max_facts, created, updated, turns}
neurons() -> [id]
```

## PlasticNeuron(max_facts=500, half_life=200.0, link_window=3)

```
observe / recall                as Neuron, but recall reinforces and re-ranks by strength
reinforce(eid, amount=1.0)      strengthen a fact by id
pin(eid)                        protect a fact from ever being pruned
recall_related(query, k=3)      the hit PLUS its strongest associates (one-hop spreading)
recall_spreading(query, hops=2, k=10)   neurotransmitter recall: excitatory cue release,
                                inhibitory relation gating, multi-hop spread with reuptake
consolidate(prune_below=0.05) -> {merged, pruned, facts}   "sleep": merge dups, prune decayed
```

`half_life` is in ticks (smaller = forgets faster). `half_life=None` disables decay while
keeping the rest of the plasticity. **Decay only changes recall ranking; it never deletes a
fact.** Only `consolidate()` removes anything, and it protects `self` facts and pinned ids.

## NeuronRouter(per_shard=128)

```
observe(text)         route to current shard; spill to a new shard when full
recall(query) / get(query)    fan out across shards, return the best
fact_count / shard_count
```

## SecureNeuronDB(path="secure.db")

```
put(id, secret, key_phrase, value)     encrypt + store; secret is NOT persisted
get(id, secret, query) -> value | None  decrypt iff the secret + cue match
```

## turn(neuron, message) -> {reply, kind, wrote}

The routing brain behind `NeuronDB.turn`. `kind` in {ack, recall, idk, smalltalk, math,
self}. Statements are stored and acknowledged; questions are answered from memory or with
"i don't know right now."; arithmetic (`+ - * /`, word forms too) is evaluated.

# HTTP SERVER

`python -m neuron_db serve` exposes one endpoint. Set `NEURON_DB_KEY` to require
`Authorization: Bearer <key>`.

```
POST /v1/{neuron}        {"message": "..."}        -> {"reply","kind","facts"}
POST /v1/{neuron}/get    {"query": "..."}          -> {"value": ... | null}
POST /v1/{neuron}/forget {"match": "..."}          -> {"forgot","remaining"}
GET  /v1/{neuron}                                   -> stats
```

# EXAMPLES

Plain app backend, no model (`examples/app_backend.py`):

```
db = NeuronDB("app.db")
db.turn("user:42", "my plan is pro")
db.get("user:42", "what is my plan?")          # -> "pro"
```

Plastic memory that adapts:

```
n = PlasticNeuron()
n.observe("the meeting is on monday"); n.observe("the meeting is on friday")
n.recall("when is the meeting?")["value"]       # "friday" (recency)
for _ in range(3): n.reinforce(n.episodes[0]["_id"])
n.recall("when is the meeting?")["value"]       # "monday" (adapted to use)
```

Encrypted secret:

```
v = SecureNeuronDB("vault.db")
v.put("alice", "alice-secret", "wifi password", "hunter2")
v.get("alice", "alice-secret", "what is the wifi password?")   # "hunter2"
v.get("alice", "WRONG",        "what is the wifi password?")   # None
```

# NEUROTRANSMITTER RECALL

The brain does not run a forward pass to remember. A cue releases transmitter at the
neurons it matches, activation spreads across synapses, fades with distance as the
transmitter is taken back up, and the neurons left most active are the memory.
`recall_spreading()` models that directly:

- **Excitatory release.** A cue fires the facts it matches, graded by how well it matches
  and how strong that synapse already is.
- **Inhibitory gate.** A relation conflict suppresses release, so a dog-name cue does not
  fire a brother-name memory.
- **Spread and reuptake.** Activation travels to linked facts and loses a fixed fraction
  each hop, so it falls off with distance and only the strongest few synapses of a neuron
  fire.

Because it is multi-hop, it surfaces facts that share no word with the query but are wired
to one that does. Because it is scalar arithmetic over a sparse graph, a selective query
runs in microseconds and a worst-case full scan over 2,000 facts still finishes far under a
millisecond. It is a pure read and changes nothing in the store.

```
n = PlasticNeuron(half_life=1e9)
n.observe("the phoenix project is owned by Dana")
n.observe("the rollout ships in the third quarter")
# co-activate so they wire together, then a phoenix cue reaches the rollout fact:
for _ in range(4): n.recall("who owns phoenix?"); n.recall("when does the rollout ship?")
n.recall_spreading("who owns phoenix?")     # returns Dana's fact AND the wired rollout fact
```

The point this makes about latency: the synapse layer here is microseconds. The slow part
of an LLM-memory stack is the model that composes language on top, not the recall. Route
retrieval through the synapses and call the model only when you need it to write a sentence.

# WHY SCALAR

The defining design decision: a fact's *memory state* is a few **scalars** — a strength
float, a last-used timestamp, and scalar link weights — not a dense embedding vector.

| | scalar plasticity (neuron-db) | vector memory (embeddings / RAG) |
|---|---|---|
| per-fact state | a few numbers (~16 B) | a 384-1536-dim float vector (~1-6 KB) |
| write cost | O(1) - bump a number | a model forward pass (ms, often a GPU) |
| recall cost | O(candidates) set ops, us | nearest-neighbor over all vectors (ANN) |
| adaptation | `w += 1` - free | re-embed or re-rank; not native |
| forgetting | `w * 0.5^(age/h)` - one multiply | not native; needs bookkeeping |
| interpretable | yes - read the strength (e.g. 42.0) | no - opaque coordinates |
| dependencies | none; runs in SQLite / a Worker / edge | embedding model + vector DB + (often) GPU |
| determinism | exact, testable | approximate (ANN, float drift) |

Why this is the better overall choice for a *plastic* memory:

1. **Plasticity becomes free.** Hebbian learning — strengthen on use, decay on disuse, link
   on co-use — is literally scalar arithmetic. Expressing it as numbers means every plastic
   update is O(1), which is the whole answer to "adaptation without lag." A vector store
   would have to re-embed or re-index to adapt; a scalar just increments.
2. **It fits anywhere.** Bytes per fact, no model to load, no GPU — so the same memory runs
   in a 1 MB Cloudflare Worker, inside SQLite, or in-process, at microsecond latency. A
   50,000-fact neuron is 2.3 MB; the same in 1 KB embeddings would be ~75 MB plus an index.
3. **You can audit it.** "Why did this fact win?" has a numeric answer — its strength, its
   overlap, its link weight. Memory you can read is memory you can trust, which matters when
   it backs an app or an agent.
4. **It composes with the symbolic store.** Stems are discrete, strengths are scalar; they
   live in the same cheap, inspectable representation. No impedance mismatch.

The honest trade: scalars can't do *semantic similarity* — "the thing I use to get online"
will not find "wifi password" without an embedding. So the design is **scalar-first**: the
fast, cheap, interpretable scalar tiers handle exact and cue recall plus all the plasticity
(the common case), and a vector/semantic tier is an *optional* fallback added only where it
earns its latency. You pay for vectors only when you actually need meaning-matching, never
for the everyday lookup-and-adapt path.

# STORAGE CAPACITY

Because a fact's retrieval state is stems and scalars rather than a dense embedding,
neuron-db holds far more facts per byte than the vector stores usually used for LLM memory.
Measured over 20,000 plain facts, the serialized footprint is 48 bytes per fact, about 22.4
million facts per GiB. Sized for the same job (store a fact and retrieve it by content):

```
store                                   bytes/item   items/GiB    vs neuron-db
neuron-db (serialized)                          48   22,351,461        1.0x
vector, binary quant, 1536-d (lossy)           259    4,145,097        5.4x
vector, int8 quant, 1536-d                    1603      669,816       33.4x
vector, f32, 768-d                            3139      342,061       65.3x
vector, f32, 1536-d                           6211      172,876      129.3x
pgvector / Qdrant, f32 1536-d + HNSW          9283      115,667      193.2x
vector, f32, 3072-d (OpenAI large)           12355       86,907      257.2x
```

So neuron-db fits roughly 33x to 257x more facts in the same disk than a float32 vector
store, about 130x at the common 1536-dim default, and still about 5x more than a
binary-quantized index that has already lost most of its accuracy. The trade is real:
neuron-db does cue and association recall, not cosine-similarity semantic search. Full
method, per-product detail, and caveats are in `docs/STORAGE.md`.

# PERFORMANCE

Dev sandbox, Python 3.10, single core. Reproduce: `python tests/bench_full.py`,
`python tests/test_stress.py`.

```
creation (in-memory, 3 facts)            ~30,000 /sec
creation (SQLite, durable)               ~920 /sec
write throughput                         ~50,000 facts/sec
recall, single neuron, N=100             ~66 us
recall, single neuron, N=1,000           ~380 us
recall, single neuron, N=10,000          ~6 ms      (common stems => big posting lists)
recall, single neuron, N=50,000          ~11 ms     accuracy 1000/1000 on distinct keys
recall, plastic vs static                1.3-1.4x   (88 us vs 67 us @ N=100)
50,000 Hebbian links built               0.03 s
spreading activation (one hop)           ~9 us
consolidate 20,000 dup facts -> 1        0.02 s
secure get (AES-256-GCM)                 ~270 us
```

**Scaling note (honest):** a single neuron's recall grows with size because a common cue
word can match thousands of facts. Keep a hot neuron to hundreds-low-thousands of facts and
shard with `NeuronRouter` beyond that; recall stays 100% on distinct keys at 100k facts via
the router (naive fan-out is then O(shards) ~ 25 ms — a routing index to pick the shard is
the documented next optimization).

## Plasticity, measured

```
adaptation   reinforced fact overtakes recency after 2 uses; strength climbs 1->42
forgetting   strength 0.99 -> 0.06 over 0..200 idle ticks (half_life=50); 4.7e-302 at hl=1
             ...and STILL recallable: decay changes ranking, never deletes
association  co-activation link grows 0.5 -> 8.0 over 5 rounds; spreading surfaces it
neurotransmit recall_spreading reaches a 2-hop fact sharing no cue word; selective query in us
sleep        20,000 duplicate facts consolidate to 1, recall preserved
```

# SECURITY

Two tiers. `NeuronDB` stores plaintext (fast fuzzy recall) but has **no bulk-export** —
values leave only one cue at a time, never a dump. `SecureNeuronDB` stores only ciphertext
and keyed hashes; the per-neuron secret is supplied per call and never persisted, so a
stolen database file is opaque and changing the neuron id reads nothing without that
neuron's key. It is *not* encryption against a compromised running process, and it is
probeable with the right secret. Full model: `THREATS.md`.

# CAVEATS

- **Near-duplicate keys are disambiguated.** Candidate gathering still uses 6-char stems
  (so fuzzy recall keeps working), but ranking now applies an exact full-token tier first, so
  `project170` beats the `project17` stem collision when the query names it exactly. Stem
  matching remains the fallback when no exact token matches.
- **Multi-word values come back whole.** A run of consecutive capitalized tokens
  ("Search Console") is returned as one value rather than a single clipped token. Lowercase
  multi-word answers still return their head token.
- **Decay is opt-in and ranking-only** (`PlasticNeuron`); plain `NeuronDB` never forgets. This is by design: a durable store should not silently drop data.
- **The plastic store re-weights; it does not learn.** It cannot invent a fact you never
  stored. Genuine "getting smarter" is offline sleep-consolidation into model weights.

# FILES

```
neuron_db/neuron.py     the associative store + recall
neuron_db/plastic.py    PlasticNeuron (scalar plasticity + neurotransmitter recall)
neuron_db/db.py         NeuronDB (SQLite, a database of neurons)
neuron_db/router.py     NeuronRouter (chained shards)
neuron_db/secure.py     SecureNeuronDB (encrypted)
neuron_db/server.py     one-endpoint HTTP server
tests/                  50+ unit tests + bench_full.py + test_stress.py
docs/PLASTICITY.md      the two-tier plastic design + decay semantics
docs/STORAGE.md         storage density vs commercial vector stores (measured)
docs/MEMORY_HARNESS.md  mounting as an LLM's memory; where gary-neuron fits
rust/neuron-core/       the Rust port: core store + plastic.rs (PlasticNeuron) + cortex/wasm
```

# ENVIRONMENT

```
NEURON_DB_KEY    if set, the HTTP server requires Authorization: Bearer <key>
```

# SEE ALSO

`docs/PLASTICITY.md`, `docs/STORAGE.md`, `docs/MEMORY_HARNESS.md`, `BENCHMARKS.md`, `THREATS.md`,
`rust/neuron-core/README.md`.

# AUTHOR

gary23w. MIT licensed.
