# How neuron-db works

A neuron is an associative memory with three moves: write a fact, recall by cue,
or abstain. None of it uses a model or embeddings — it is string and set logic.

## Write

A statement is split into sentences (so a paste of several facts becomes several
entries). For each, the **most surprising content word** becomes the value. Surprise
is a cheap heuristic: digits score high, mid-sentence capitalized words score next,
long words a little. Introductions ("i'm Aiko"), ages ("i'm 28"), relations
(dog, sister, boss) and coreference ("her name is Mochi" → binds to a puppy mentioned
just before) are recognized. Commands ("count to ten"), questions, and short
second-person chit-chat are not stored.

## Recall

A question's words become a **stemmed cue**. Every stored fact is scored by how many
cue stems it shares, under **relation-binding**: a fact about your dog can't answer a
question about your sister, and vice-versa. The best fact wins by overlap, then by
self-name priority, subject match, specificity, and recency.

From the winning fact, the **value nearest the asked-about word** is returned. That is
why "the first 1,000 users score 150,000 coins" answers `how many users? → 1,000` and
`how many coins? → 150,000`. Numbers are returned crisply; a lone capitalized word from
a long sentence is quoted instead of clipped (so "Search Console" doesn't come back as
just "Console").

If nothing clears the binders, recall returns nothing and the reply is
`i don't know right now.`

## Storage — ~30 bytes per fact

A neuron persists only its raw facts, each with one flag bit:

```json
[["my name is Marisol", 1], ["the door code is 4452", 0]]
```

Everything the recall engine indexes on — the value, the candidate words, the stems,
the subject — is **recomputed from the text on load**. Two consequences:

1. A fact costs about 30 bytes (≈28 gzipped). A million neurons holding 20 facts each
   is on the order of 500 MB before column compression.
2. The recall logic can be improved and re-released without migrating any stored data,
   because the index isn't stored.

The floor is the facts themselves — you can't recall what you never stored.

## The database

`NeuronDB` keeps one row per neuron in SQLite: `(id, facts, created, updated, turns)`.
Each call loads the neuron's blob, rebuilds it, runs, and saves. Access is serialized
with a lock so the threaded HTTP server shares one connection safely. There is no query
that returns all the values of a neuron — only `recall`, which returns one.

## Porting the engine elsewhere

`neuron.py` has no dependencies and no I/O, so the same logic runs:

- **SQLite** (this repo) — a row per neuron.
- **Postgres** — a `pgrx` (Rust) extension: a `neuron` column type plus
  `neuron_observe(col, text)` and `neuron_recall(col, query)` functions, queryable in
  SQL alongside ordinary data, still with no bulk-dump operator.
- **Edge** — one object per neuron (see the neuron-cloud variant), with an optional
  small language model for conversational replies on top of the same store.
