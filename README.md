# neuron-db

**Memory you talk to, in a single file. No model, no embeddings, no dependencies.**

A *neuron* is a small associative memory: you write facts in plain language and ask
for them back later by meaning. `neuron-db` keeps many neurons in one SQLite file and
serves them over one HTTP endpoint. It is pure Python standard library — clone it and
run it.

```python
from neuron_db import NeuronDB
db = NeuronDB("memory.db")

db.turn("alice", "the wifi password is hunter2")     # 'got it -- hunter2.'
db.turn("alice", "what is the wifi password?")        # 'hunter2.'
db.turn("alice", "what is my blood type?")            # "i don't know right now."
```

## What makes it different

A normal table answers `SELECT value WHERE key = ?` and a single bad read hands over
every row. A neuron has **no operation that dumps its values** — facts go in by being
stated and come out only when a cue retrieves the right one. That is the headline
property: you can write secrets into a neuron and they can't be bulk-exported, only
asked for. (Read [`SECURITY.md`](SECURITY.md) — it's honest about the limits too.)

It also isolates values out of messy text:

```python
db.turn("x", "only the first 1,000 users score 150,000 coins")
db.turn("x", "how many users?")    # '1,000.'
db.turn("x", "how many coins?")    # '150,000.'   <- the right number, by proximity
```

And it abstains instead of guessing. A memory that makes things up is worse than one
that admits the gap.

## Install & run

No dependencies. Python 3.9+.

```bash
git clone https://github.com/gary23w/neuron-db
cd neuron-db

python -m neuron_db demo               # scripted tour, no setup
python -m neuron_db chat               # talk to a neuron in your terminal
python -m neuron_db serve --port 8088  # one-endpoint HTTP server
```

Optional install: `pip install -e .` then `neuron-db serve`.

## The one endpoint

```
POST /v1/{neuron}        {"message": "..."}   -> {"reply", "kind", "facts"}
GET  /v1/{neuron}                              -> {"facts", "turns", ...}
POST /v1/{neuron}/forget {"match": "wifi"}     -> prune (omit match to clear)
```

```bash
curl -X POST localhost:8088/v1/alice -d '{"message":"my name is Marisol"}'
curl -X POST localhost:8088/v1/alice -d '{"message":"what is my name?"}'   # -> {"reply":"Marisol.", ...}
```

Set `NEURON_DB_KEY` to require `Authorization: Bearer <key>`.

### Exact values, not prose

`POST /v1/{neuron}/get {"query":"..."}` returns the value itself — `{"value":"hunter2"}`
or `{"value":null}` — with no wrapper text and no punctuation. (The `turn` endpoint is the
conversational one; `get` is the machine one.)

### Encrypted neurons (for secrets)

For sensitive data, use `SecureNeuronDB`: the value is encrypted and the database never
stores the key. You pass a per-neuron secret on each call; the database keeps only
ciphertext and keyed hashes.

```python
from neuron_db.secure import SecureNeuronDB
v = SecureNeuronDB("vault.db")
v.put("alice", "alice-secret", "wifi password", "hunter2")
v.get("alice", "alice-secret", "what is the wifi password?")   # -> "hunter2"
v.get("alice", "WRONG-secret", "what is the wifi password?")   # -> None
v.get("bob",   "alice-secret", "wifi password")                # -> None (key bound to neuron id)
```

A stolen database file is opaque — no values, cues, or keys. Changing the neuron id in a
request reads nothing without that neuron's secret. AES-256-GCM with `pip install
neuron-db[crypto]`, sound stdlib fallback otherwise. Full model and limits in
[`THREATS.md`](THREATS.md).

### Performance

Recall is sub-linear (a stem→fact index) and hot neurons are cached in memory, so the
re-parse cost is paid once, not per call.

| operation | latency |
|---|---|

## Benchmarks, branches & roadmap

- **Capability numbers** (creation rate, recall latency, the honest stem-collision boundary): [`BENCHMARKS.md`](BENCHMARKS.md)
- **Memory-bank design** (mount neuron-db as an LLM's memory; 3-tier exact/cue/semantic retrieval; MCP tools): [`docs/MEMORY_HARNESS.md`](docs/MEMORY_HARNESS.md)
- **`rust` branch** — the faster reimplementation in progress (`rust/neuron-core`, built + tested)
- **`python-prototype` branch** — this Python build, preserved
