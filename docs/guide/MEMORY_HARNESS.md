# Design: mounting neuron-db as an LLM's memory bank

This is the design for the work that comes *after* the Rust refactor: a harness that lets
a language model use neuron-db as long-term memory, plus an MCP server that exposes
retrieval as a tool. The goal is recall that is fast, reliable, and honest about what it
knows — and that does not collide when the model writes many similar facts.

The benchmarks (`BENCHMARKS.md`) set the constraint: neuron-db recalls distinct keys at
100% and near-duplicate keys poorly. The harness is built around that fact.

## 0. Where gary-neuron fits: the dispatcher

Three parts, three roles, split roughly the way a brain divides the work:

| part | brain analog | role |
|---|---|---|
| the host model (GPT-class) | neocortex | reasoning and language; big, general, stateless per call |
| **gary-neuron** (dispatcher cortex) | **hippocampus** | routes the turn: decides whether to answer from memory, escalate, fetch, or store; small, trained, runs locally |
| neuron-db (PlasticNeuron + store) | engram store | durable substrate; cheap, scales to millions |

gary-neuron is a ~7M-parameter (6,973,952) int8 transformer (E=256, H=8, L=8, vocab 2048,
512-token context). It is baked into the WebAssembly/binary build with `include_bytes`, runs
on CPU with no GPU and no download.

It is not a selectable chat model. It is the always-on middle layer between the host model and
neuron-db, and each turn it emits exactly one route:

- **`ANSWER`** — the question is covered by memory. The cortex picks the route; the literal value
  comes from neuron-db's deterministic recall. The cortex decides, the store grounds the bytes.
- **`ESCALATE`** — the turn needs the host model's reasoning or language. gary-neuron hands it up.
- **`FETCH <topic>`** — the answer isn't in memory yet and should be pulled in (web/tool) before
  proceeding.
- **`STORE <fact>`** — the exchange contains something durable to write to neuron-db.

So the host model is the reasoner and gary-neuron is the router in front of memory. The MCP tools
below are the wire; gary-neuron is the engine that decides which of them a turn needs.

Why a trained model here instead of the host model routing its own memory? Cost and fit.
gary-neuron runs in-process at the edge and burns no host-model tokens or API calls per turn, and
its routing is deterministic. Held-out results: routing triage (ANSWER vs ESCALATE vs FETCH) is
100% on each class; grounded ANSWER accuracy is 88–98% across working sets from 1 to 18 facts;
two-hop chaining is 100%. The one acknowledged limit is numeric comparison, which is near chance.
A browser/WASM dispatch is about 54 ms after a SIMD128 pass over the matmuls (down from ~172 ms),
and the cortex is also fast natively. neuron-db recall underneath is p50 ~3.9 µs / p99 ~36 µs,
measured over 10k queries on 7000 facts across 1000 scopes.

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

### Passive encoding — don't let the model gate storage

The biggest failure mode in practice is leaving *what to store* up to the model. Given only a
`remember` tool, an LLM stores erratically — it skips most of what the user says and will tell
you "I can only store what you explicitly ask me to remember." A real hippocampus doesn't work
that way: it encodes the experience stream automatically, and salience/consolidation curate
later.

So the **harness — not the model — should auto-observe each user message** into the scope
before the model runs. `observe` already drops questions and dedups exact restatements, so it
captures statements and skips noise on its own. The model's job then narrows to **retrieval**
(`recall` / `recall_chain`), with `remember` reserved for emphasis or a model-authored
conclusion. Storage becomes continuous and reliable regardless of whether the model "bothers."

This is the recommended wiring; the reference implementation is `examples/mcp-chat/`, and a
two-pane live harness (chat on the left, the neuron-db memory firing on the right, with passive
auto-capture on by default) demonstrates it end to end.

### Documents — one scope each, plus a register (categorization at ingest)

Passive-encoding a long pasted document into the conversational scope flattens it into hundreds
of sentence-facts mixed with everything else — and the user refers to it by *position* ("the
first blog", "that article"), which shares no words with its content, so recall can't find it.
The harness should treat a document as a first-class object: detect a large paste, store it in
its **own scope** (`{scope}__doc{n}`), and keep an ordered **register** of `{n, title}` it
injects into the system prompt each turn. Give the model a `recall_document(n, query)` tool the
harness routes to that document's scope. Then "summarize the first blog" → doc #1 → a
**semantic-ranked** block recall from that scope alone — coherent, no cross-document bleed. Keep
the document OUT of the context window (a stub) so the answer comes from recall, not free-ridden
context. (Implemented in the two-pane lab; uses `recall_blended` for semantic-ranked blocks.)

### Recall ranking — neurons first, semantic as an opt-in signal

The default `recall` ranks by **associative cue-overlap** over the stem index — the neuron
substrate, which is what actually builds structure. For broad/narrative queries that tie on
`ov=1` keyword hits, two stronger paths exist:

- **`recall_associative`** — spreading activation over the shared-entity graph. Seeds on cue
  matches, then flows activation along *discriminative* shared stems (rare entities link strongly,
  hub words are ignored), so it surfaces facts that share no words with the query but are wired to
  a match. This is association, not ranking — it traverses structure the raw text never stated.
- **`recall(rank:"semantic")`** — `recall_blended` (feature `semantic`): semantic cosine over
  cached int8 embeddings plus a lexical boost. Honest framing: the semantic space is a *ranking
  signal*, not memory — it creates no neurons and can be dropped without losing a fact. It earns
  its keep only as an opt-in tie-smoother for broad recall; it is no longer the silent default.

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

A thin MCP server over the Rust core exposes a small set of tools — an LLM should not need
to learn a query language. This is **implemented** as `neuron-mcp` (`rust/neuron-core/src/mcp.rs`,
`rust/neuron-core/src/mcp_bin.rs`), a native std-only stdio JSON-RPC 2.0 server embedding `NeuronDB`. No Node
or Python runtime, no separate HTTP process — one binary the client launches.

| tool | input | returns (MCP text content) |
|---|---|---|
| `recall` | `{scope, query, k?, rank?}` | top-k facts as a memory block (`- fact` lines). Default rank is lexical/associative; `rank:"semantic"` opts into semantic ranking for broad/narrative queries |
| `recall_associative` | `{scope, query, k?, hops?}` | spreading-activation block: facts that match the cue (`*`) plus associates reached by shared-entity links (`-`) — surfaces connections keyword recall misses |
| `recall_value` | `{scope, query}` | the single isolated value for a direct question, or `(no memory)` |
| `recall_chain` | `{scope, start, path:[…]}` | walks a chain of relations server-side and returns the final value + trail (see §3a) |
| `remember` | `{scope, text}` or `{scope, facts:[…]}` | `Stored N fact(s)` |
| `note` | `{scope, kind, text, key?}` | mint a TYPED neuron — `kind` ∈ fact/user/instruction/var; returns the stored address (so a save can't be hallucinated). `var` requires `key` and upserts |
| `recall_var` | `{scope, key}` | the exact value of a named variable set via `note(kind=var)`, or `(unset: key)` |
| `forget` | `{scope, match?}` | `Forgot N; M remain` (omit `match` to clear the scope) |
| `stats` | `{scope}` | `scope holds N fact(s) …` |

> Typed neurons are scope conventions, not a schema change: `note` routes to `{scope}::instr` and
> `{scope}::var` sub-scopes. A harness re-injects the instruction neurons into the system prompt each
> turn (the "instructions batch"), so a standing rule survives the rolling window. `note(kind=var)`
> returns the stored address, which is what stops a model from claiming a save it never performed.
> Everything maps onto `NeuronDB` methods (see `BENCHMARKS.md`).

### 3a. recall_chain — infinite hops at no model cost

A relational question ("the timezone of the manager of the owner of Aurora") normally
forces the LLM to chain recalls: recall the owner, *wait for it*, recall that person's
manager, *wait*, recall their timezone — **N hops = N+1 model round-trips.** That's the only
real cost of depth (see `COMPARISON.md` §3).

`recall_chain` collapses that into **one** model call. The LLM passes the starting entity
and the ordered relations; the server walks the chain itself, resolving each
`"<current> <relation>"` by recall in microseconds, and returns the final value plus the
trail:

```
recall_chain(scope, start="Aurora", path=["owner","manager","timezone"])
  -> "WET  (via Aurora -> Marisol -> Dana -> WET)"
```

Each hop is one neuron recall (tens of µs, see `SYNAPSE.md`) with **zero model round-trips
between hops**, so a 3-hop or a 30-hop answer costs the LLM the same: one call to form the
path, one to phrase the answer. A hop only advances if the relation actually appears in the
recalled fact (root-normalized), so a broken chain reports where it stopped instead of
silently drifting. This is "infinite hops at no LLM cost": depth is paid in microseconds by
the synapse, not in model turns.

### 3b. Fuzzy recall — closing the lexical gap

Recall is lexical: a query whose words don't match the stored facts can miss
(`COMPARISON.md` §4, where misaligned vocabulary scored 17%). neuron-db closes that with a
two-part fallback that runs *only when the relation doesn't fully bind*, so the warm fast
path keeps its flat microsecond cost:

1. **Morphological root** — both query and facts normalize to a suffix-stripped root, so
   `owner`/`owned`/`owns` → `own` and `dependency`/`depends` → `depend`.
2. **Synonym canonicalization** — a curated synonym→canonical ontology applied to *both*
   sides, so a fact stored as `reports to` and a query asking for `manager` both canonicalize
   to the same word (likewise `lives in`↔`city`, `boss`/`supervisor`/`lead`↔`manager`).

This is not a learned embedder — it's a deterministic, zero-cost ontology — but it closes the
relation-synonym class that the gap was made of. With it, the misaligned-vocabulary benchmark
goes from 17% to **100%** through `recall_chain` (`COMPARISON.md` §6). Open-vocabulary
paraphrase ("the thing I use to get online" → wifi) would still want an embedding tier; the
relation synonyms that matter for entity memory are handled.

### Conventions the LLM follows

- **Recall before generating.** When the user refers to something they may have said
  before, call `recall` (or `recall_value` for a direct field) and inject the returned block
  into context as grounded evidence — don't guess.
- **Write after the turn.** For anything durable the user stated, call `remember`. Keep each
  fact a short plain-language statement (`my plan is pro`); store distinct subjects so cues
  stay selective (see the O(candidates) result in `BENCHMARKS.md`).
- **One scope per memory owner.** Pass `user:{id}` / `session:{id}` / `agent:{id}` — scopes
  are fully isolated, so multi-tenant memory needs no extra plumbing.
- **Abstention is a feature.** A miss returns "No memories found" / `(no memory)`, never a
  fabricated value. Treat that as "I don't have this stored," not an error.

## 4. Mounting it

Build the server:

```sh
cargo build --release --features mcp --bin neuron-mcp
```

Register it with any MCP client (Claude Desktop, Claude Code, Cursor, …). Example client
config:

```json
{
  "mcpServers": {
    "neuron-memory": {
      "command": "/abs/path/to/neuron-mcp",
      "env": { "NEURON_MCP_DB": "/abs/path/to/memory.db", "NEURON_MAX_FACTS": "100000" }
    }
  }
}
```

Environment:

| var | default | meaning |
|---|---|---|
| `NEURON_MCP_DB` | `neuron-memory.db` | SQLite file the memory persists to |
| `NEURON_MAX_FACTS` | `100000` | per-scope fact cap (oldest evicted past it) |

Transport is newline-delimited JSON-RPC 2.0 on stdin/stdout; the server handles
`initialize`, `tools/list`, `tools/call`, `ping`, and ignores notifications. Smoke-test it
without a client by piping messages:

```sh
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"remember","arguments":{"scope":"user:1","text":"my plan is pro"}}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"recall_value","arguments":{"scope":"user:1","query":"what plan am i on?"}}}' \
  | neuron-mcp
```

For non-MCP / API-only agents, the same recall→inject→write loop is available over the HTTP
server (`/v1/{scope}/recall_many`, `/observe`, `/forget`) — see `API.md`.

## 5. Status

Implemented today: the `neuron-mcp` stdio server with `recall` / `recall_associative` /
`recall_value` / `recall_chain` / `remember` / `note` (typed neurons) / `recall_var` / `forget` /
`stats`, backed by the durable `NeuronDB`, with
unit + end-to-end tests. Verified live (gpt-4o-mini, `COMPARISON.md` §6): **100% accuracy at
1, 2, and 3 hops, on both aligned and synonym-misaligned vocabulary**, each in a constant 2
LLM calls at any depth, flat ~1.1k tokens at any memory size. The needle benchmark holds at
100% and ~16 µs to 50k facts (`BENCHMARKS.md` §5.4).

Future work: an embedding tier for open-vocabulary paraphrase (the relation-synonym class is
already handled, §3b), an exact-key KV tier, and supersede-on-write / dedup (`/sleep`).