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

## How a neuron works

| step | what happens |
|---|---|
| **write** | the most surprising content word of a statement becomes its value; names, ages, relations (dog, sister, boss) and coreference ("her name is Mochi") are recognized |
| **read** | a question's words become a stemmed cue; the best fact is found under relation-binding; the value nearest the asked-about word is returned |
| **abstain** | no match → `i don't know right now.` |

A neuron persists as just its raw facts plus one flag each — about **30 bytes per
fact** — and the whole index is recomputed on load, so improving the recall logic
never requires a data migration. See [`docs/DESIGN.md`](docs/DESIGN.md).

## Why "database"

The store is pure logic, so it drops into anything: this repo is the SQLite build;
the same engine runs per-row in Postgres (a `pgrx` extension exposing `neuron_observe()`
/ `neuron_recall()`), in SQLite as a user-defined function, or per-object at the edge
(see the hosted **neuron-cloud** variant, which adds an optional language model for
conversational replies). A neuron is a new kind of column: written in language,
queried by meaning, and impossible to dump.

## Tests

```bash
python tests/test_neuron_db.py        # 10/10, no pytest required
```

## License

MIT — see [`LICENSE`](LICENSE). Built by [gary23w](https://github.com/gary23w).
