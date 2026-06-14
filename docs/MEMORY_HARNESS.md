# Design: mounting neuron-db as an LLM's memory bank

This is the design for the work that comes *after* the Rust refactor: a harness that lets
a language model use neuron-db as long-term memory, plus an MCP server that exposes
retrieval as a tool. The goal is recall that is fast, reliable, and honest about what it
knows — and that does not collide when the model writes many similar facts.

The benchmarks (`BENCHMARKS.md`) set the constraint: neuron-db recalls distinct keys at
100% and near-duplicate keys poorly. The harness is built around that fact.

## 0. Where gary-neuron fits: the LLM's hippocampus

Three parts, three roles, borrowed straight from how a brain splits the job:

| part | brain analog | role |
|---|---|---|
| the LLM (GPT-class) | neocortex | reasoning and language; big, general, stateless per call |
| **gary-neuron** (cortex + plastic hippocampus) | **hippocampus** | what to remember, what to recall, consolidation; small, trained, plastic |
| neuron-db (PlasticNeuron + store) | engram store | durable substrate; cheap, scales to millions |

The LLM never talks to the raw store. It talks through gary-neuron, which is a **memory
co-processor** sitting between the LLM and the database:

- **Write path** — the LLM produces an exchange; gary-neuron's encoder surprise-gates it
  (it was trained to write the surprising token), and the salient facts land in neuron-db.
  The LLM doesn't decide what's worth keeping; the hippocampus does.
- **Read path** — the LLM asks a question (an MCP `memory.recall` call); neuron-db pulls a
  small working set in microseconds; gary-neuron's cortex reads that bounded window — the
  thing it was trained to emergence on — and returns the isolated value or the associative
  completion, which is injected into the LLM's context as grounded evidence.
- **Consolidation** — offline, gary-neuron replays buffered episodes into its own weights
  (`/sleep`). It grows. The LLM is untouched and pays nothing for it.

So the connection is: **the LLM is the reasoner; gary-neuron is its memory organ.** The MCP
tools below are the wire — `memory.recall`/`memory.write` are implemented as
store-retrieve-then-gary-neuron-read and gary-neuron-encode-then-store. The LLM just calls
the tools; gary-neuron is the engine behind them.

Why a trained model here instead of the LLM doing its own memory? Cost and fit. gary-neuron
is ~1.1M params — it runs in-process at the edge in milliseconds and burns no LLM tokens or
API calls per memory op, it's deterministic, and it does associative completion the
symbolic store can't. Honest limit: it was trained on a ~2k-token everyday vocabulary in a
`U:/G:` fact format, so it shines on normalized facts over a bounded window, not on
arbitrary technical prose. The practical split: the symbolic store handles out-of-vocab
exact recall; gary-neuron handles in-vocab associative recall and consolidation. This is
literally the gary-neuron-chat architecture with the human user replaced by an LLM.

## 1. Where memory sits in the loop

```
        ┌──────────────────────── turn ─────────────────────────┐
 user → │  RETRIEVE: pull relevant memories for the prompt       │
        │  GENERATE: model answers, grounded in those memories   │
        │  EXTRACT:  pull new durable facts out of the exchange  │
        │  WRITE:    store them (dedup / supersede old ones)      │
        └────────────────────────────────────────────────────────┘
```

Two model-facing operations matter: **retrieve before generating** and **write after**.
Everything else is policy around those two.

### Scoping

One neuron per memory scope, addressed by id: `user:{id}`, `agent:{id}`, `session:{id}`,
`org:{id}`. The model never sees another scope's neuron — isolation is the neuron boundary
(and, for sensitive scopes, the encrypted tier with a per-scope key the harness holds).

## 2. Retrieval — fast and reliable

Retrieval is the hot path and the reliability risk. Three tiers, cheapest first:

1. **Exact key (KV).** Facts the model writes with an explicit key (`set("plan_tier","pro")`)
   are stored in a hash-indexed table and returned verbatim. O(1), never collides. This is
   the right tier for profile fields, settings, IDs — anything the model knows the key for.
2. **Cue recall (the neuron).** Natural-language questions hit the stem-indexed store:
   sub-linear, returns the isolated value or abstains. Best for "what's my X" style recall
   over distinct facts.
3. **Semantic fallback (optional).** For paraphrased queries with no stem overlap
   ("the thing I use to get online" → wifi password), an optional embedding index (a local
   MiniLM-class model, or the host LLM's own embeddings) ranks candidates. Used only when
   tiers 1–2 return nothing, so the common case never pays the embedding cost.

A retrieve call runs 1, then 2, then 3, and stops at the first hit. Returns a small,
ranked list with provenance (which fact, which tier, confidence), never a guess.

### Killing the collision problem

Near-duplicate keys are the known weakness. The harness handles them before they reach the
store:

- **Explicit keys win.** The extractor prefers `set(key, value)` over free-text for
  anything that looks like a field, so "ten meetings" become ten exact keys
  (`meeting:2026-06-13`), not ten colliding "meeting on …" sentences.
- **Full-token disambiguation.** When the store would tie on a 6-char stem, the retriever
  re-ranks the tied candidates on the full surface tokens and recency, so `project17` and
  `project170` are separable even though their stems match.
- **Supersede on write.** Writing a fact whose key already exists updates in place rather
  than appending a near-duplicate (`plan_tier` pro → enterprise replaces, doesn't stack).
- **Dedup pass.** A periodic consolidation merges entries with identical keyed hashes,
  keeping the newest. (This mirrors the prototype's "sleep" consolidation.)

### Performance targets

| operation | target | basis |
|---|---|---|
| exact-key get | < 50 µs | hash lookup, Rust core |
| cue recall (≤ 1k facts) | < 100 µs | measured ~50 µs in Rust |
| semantic fallback | < 10 ms | only on miss; small local index |
| retrieve (p99, warm) | < 2 ms | tiers 1–2 only for the common case |

Reliability: retrieval is deterministic for tiers 1–2 (same input → same output), the
neuron is cached in memory (no per-call re-parse), and a miss returns an explicit "no
memory" rather than a hallucinated value. The store has no network dependency, so retrieval
can't fail on an external service.

## 3. The MCP server

A thin MCP server over the Rust core exposes four tools. Small surface on purpose — an LLM
should not need to learn a query language.

| tool | input | output |
|---|---|---|
| `memory.recall` | `{scope, query, k?}` | ranked `[{value, fact, tier, confidence}]` or `[]` |
| `memory.get` | `{scope, key}` | exact value or null |
| `memory.write` | `{scope, text}` or `{scope, key, value}` | `{stored, superseded}` |
| `memory.forget` | `{scope, match}` | `{removed}` |

Conventions th