# Synapse: how fast can an LLM recall from neuron-db on the fly?

This documents an end-to-end measurement of the **synapse** — the speed at which
neuron-db fires and returns recalled neurons to a live LLM through the MCP server,
under a realistic, data-heavy session. The question: when a user is talking to an
assistant backed by neuron-db, how quickly does memory come back, and what does the
model actually return?

TL;DR: over a **700-fact** per-user store, a recall fires in a **median of ~20–40 µs**
(pure neuron recall) and **~0.2 ms** end-to-end including stdio IPC, while the LLM call
itself takes **~1.8 s**. Memory is **~40,000–86,000× faster than the model** — the
synapse is effectively free; the model is the only thing you wait on.

---

## 1. Setup

A simulated returning user ("Garrett", a founder/engineer) with a large memory, talking
to an OpenAI model that has neuron-db mounted as memory over MCP.

| | |
|---|---|
| Store | **700 facts** in one scope: ~50 meaningful profile/project/people facts + 650 historical "ticket" facts (the haystack) |
| Transport | `neuron-mcp` (native stdio MCP server) ↔ Python client ↔ OpenAI |
| Model | `gpt-4o-mini`, temperature 0 |
| Session | 11 turns: direct lookups, multi-field rundowns, a full-profile summary, an update, and an abstention |
| Harness | `examples/mcp-chat/app_sim.py` (seeds the store, runs the session, measures) |

The 700 facts live in neuron-db, **not** in the model's context. Each turn the model
pulls only the handful it needs via `recall`/`recall_value`. That is the whole point:
the context window stays tiny no matter how much is remembered.

### How the three latencies are measured

| latency | what it is | measured |
|---|---|---|
| **synapse** | pure neuron recall inside the server | server-side, `Instant` around the recall, emitted to stderr (`NEURON_MCP_LOG=1`) |
| **round-trip** | recall **+** stdio JSON-RPC IPC | client wall-clock around the MCP call |
| **LLM** | the OpenAI request | client wall-clock around the HTTP call |

Reproduce:

```sh
cargo build --release --features mcp --bin neuron-mcp     # build the server
export OPENAI_API_KEY=sk-...
python examples/mcp-chat/app_sim.py                       # seed + run + measure
```

---

## 2. Synapse performance

Per-turn timing from a representative run (700-fact store):

| turn | tool | neurons returned | synapse (µs) | round-trip (ms) | LLM (ms) |
|---:|---|---:|---:|---:|---:|
| 1 | recall_value (coffee) | 1 | 394\* | 0.64 | 1821 |
| 1 | recall_value (diet) | 1 | 8 | 0.61 | 1821 |
| 2 | recall_value (Aurora status) | 1 | 34 | 0.19 | 1817 |
| 3 | recall (teammates) | 4 | 49 | 0.25 | 2112 |
| 5 | recall (region/cloud/db) | 5 | 54 | 0.19 | 1604 |
| 6 | recall (Beacon) | 11 | 77 | 0.25 | 1733 |
| 7 | recall_value (manager) | 1 | 10 | 0.09 | 1896 |
| 9 | recall_value (Beacon status) | 1 | 293\* | 0.44 | 2116 |
| 10 | recall (project block) | 12 | 60 | 0.18 | 6256 |

\* The slow outliers are explained in §4 (cold index / rebuild after a write).

**Aggregate (recall calls):**

```
store size (neurons fired through): 701 facts
pure neuron recall (synapse): min 3 / median 21 / p95 394 / max 394 us
MCP round-trip (recall + stdio): median 0.19 / p95 0.64 ms
stdio/IPC overhead (rtt - synapse): ~0.17 ms
LLM call latency: median 1817 / p95 6256 ms
=> memory recall is ~86,000x faster than the model
```

Two things stand out:

- The **synapse is microseconds**, even firing through 700 stored neurons and returning
  blocks of up to 12. Recall cost tracks the number of *matching candidates*, not the
  store size — see `BENCHMARKS.md` §5.4, where needle recall stays flat to **50,000**
  facts. So this stays fast as the user accumulates years of memory.
- The **end-to-end cost is dominated entirely by the model.** Of a ~1.8 s turn, neuron-db
  is ~0.0002 s. Memory is not a latency concern; it is free relative to the LLM.

---

## 3. What the LLM returns

The model used the tools naturally and answered from memory. Examples (verbatim):

**Multi-fact block** — *"Who are my teammates and what does each lead?"* → one `recall`
(4 neurons, 49 µs):
> - Mateo: Frontend · Priya: Infrastructure · Lena: Design · Bjorn: Data

**Update then recall** — *"project Beacon is now unblocked and in progress"* → the model
stored `project Beacon status is in progress`; the next turn's
`recall_value("project Beacon status")` returned **progress** (293 µs).

**Full-profile summary** — *"A new assistant is joining — summarize my profile and
projects"* → the model issued several `recall` calls and synthesized a structured
rundown of all four projects (status, deadline, stack) plus the team.

**Abstention** — *"Do you know my blood type?"* → `recall_value` → `(no memory)` → the
model said it doesn't have it, no fabrication.

---

## 4. Findings & what was patched on the fly

### Index rebuild after writes (the timing outliers)

The first recall in a session (~390 µs) and the first recall after a `remember`
(turn 9, ~293 µs) are ~10× slower than warm recalls. neuron-db builds its stem→fact
inverted index **lazily**, and a write invalidates it, so the next recall rebuilds it
(O(facts), ~0.3 ms for 700). Every warm recall after that is **3–77 µs**. Net: even the
worst case is sub-millisecond; steady-state is tens of microseconds.

### Update phrasing (fixed via the harness prompt)

In the first run the model stored the update as *"Beacon is now unblocked and in
progress"* — which contains no word "status", so `recall_value("Beacon status")` matched
the **old** "status is blocked on payment provider" and returned `provider`
(stale). Because neuron-db appends rather than supersedes, recall ranks by overlap then
recency, and the old fact won on the word "status".

**Patch:** the system prompt now tells the model to phrase an update with the field's own
wording (`project Beacon status is in progress`). After the patch, the update is found
correctly and recall returns **progress**. (The deeper fix — supersede-on-write — is
listed as future work in `MEMORY_HARNESS.md`.)

### Lexical recall gap (the real limitation)

Recall matches on **overlapping words, not meaning**. When the model queried
`recall("dev environment")`, it shared no words with the stored facts (`my primary editor
is neovim`, `my laptop is a Framework 16`, …), so recall returned nothing even though the
facts exist. Prompting the model to use concrete nouns helped only partially —
`gpt-4o-mini` still reached for the abstract category and missed.

This is the honest boundary of a scalar, lexical store: it is excellent at cued recall
(microsecond, accurate) and weak at *"tell me everything about this broad category"* when
the query doesn't share words with how facts were stored. Mitigations, in order:

1. **Use concrete query terms** that match how facts are stored (the model's job; helped
   by the prompt).
2. **Semantic fallback** — an optional embedding tier consulted only on a lexical miss
   (future work, `MEMORY_HARNESS.md` §2).
3. **Structured keys** — a KV tier so a profile summary can fetch a known set of fields
   rather than cue-recall a vague category (future work).

---

## 5. Conclusion

For an LLM using neuron-db as memory:

- **Speed is a non-issue.** Recall fires in tens of microseconds and round-trips in
  ~0.2 ms — ~40,000–86,000× faster than the model call, and flat as the store grows into
  the tens of thousands of facts (`BENCHMARKS.md` §5.4).
- **Accuracy is high on cued recall** — direct fields, multi-fact blocks, numeric values,
  updates (with field-matched phrasing), and abstention all worked.
- **The limitation is lexical, not performance.** Abstract/paraphrased queries that don't
  share words with stored facts can miss; semantic fallback and structured keys are the
  paths to close that, and they don't change the speed story.

The synapse is effectively instantaneous; the work left is recall *reach*, not recall
*speed*.
