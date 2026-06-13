# Design: mounting neuron-db as an LLM's memory bank

This is the design for the work that comes *after* the Rust refactor: a harness that lets
a language model use neuron-db as long-term memory, plus an MCP server that exposes
retrieval as a tool. The goal is recall that is fast, reliable, and honest about what it
knows — and that does not collide when the model writes many similar facts.

The benchmarks (`BENCHMARKS.md`) set the constraint: neuron-db recalls distinct keys at
100% and near-duplicate keys poorly. The harness is built around that fact.

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

Conventions that make it reliable in an agent loop:

- **Recall returns evidence, not prose.** The model decides how to use `value`/`fact`;
  the tool never editorializes. An empty result is a first-class answer.
- **Confidence is surfaced.** Tier-1 exact = 1.0; tier-2 cue = coverage; tier-3 semantic =
  similarity. The model can be told to ignore anything below a threshold.
- **Writes are idempotent on key.** Re-writing the same key is a supersede, so retries and
  duplicate tool calls don't bloat the neuron.
- **Auth + scope are server-side.** The model passes a logical `scope`; the server maps it
  to the real (possibly encrypted) neuron with the right key. The model never holds keys.

### RAG vs this

Classic RAG chunks documents, embeds them, and pulls text back for the model to read.
This design returns **values and facts**, not chunks — tiers 1–2 need no embedding model
at all, which is why retrieval is microseconds, not tens of milliseconds, and why it runs
with no GPU and no vector database. The optional semantic tier is RAG-style, but it is the
fallback, not the foundation. For document-heavy use, tier 3 can front a normal vector
store; the neuron stays the fast, exact, abstaining core.

## 4. Reliability and safety

- **Abstention over hallucination.** Every tier returns "no memory" rather than a guess;
  the model is instructed to say it doesn't know rather than invent.
- **Provenance.** Each recall carries the stored fact it came from, so the model (and a
  human) can audit why an answer was given.
- **Encrypted scopes.** Sensitive memory uses `SecureNeuronDB`; the harness holds the
  per-scope key, the store holds ciphertext, and there is no bulk-export (see `THREATS.md`).
- **Bounded growth.** `MAX_FACTS` per neuron plus the dedup pass keep recall sharp and
  storage predictable; overflow is reported, not silently dropped.

## 5. Build order

1. **Rust core** (`rust` branch): the store, exact-key table, dump/load. (Started.)
2. **Retriever**: tiers 1–2 with full-token disambiguation and the supersede/dedup policy.
3. **MCP server**: the four tools over the core, with scope→neuron mapping and auth.
4. **Semantic tier**: optional embedding fallback behind the same `recall` tool.
5. **Harness adapters**: drop-in middleware for common agent frameworks (retrieve-before /
   write-after hooks), plus the existing one-endpoint HTTP API for everything else.

The throughline: keep the fast, exact, abstaining core doing the common case in
microseconds, and add cleverness (semantic search) only at the edges where it earns its
latency.
