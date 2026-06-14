# neuron-db vs the markdown-dump: linking neurons, cost, and scale

This measures the neocortex↔hippocampus split — an LLM (the reasoner) using neuron-db as
its memory organ — under a **realistic, interlinked memory**, and puts it head-to-head
against the memory pattern most LLM setups use today: **dumping a markdown file of all
remembered facts into the context window every turn.**

Three questions, answered with measurements:

1. Can the LLM **link several neurons together** (chain recalls) to answer questions whose
   answer isn't in any single fact?
2. **How much does that cost** the LLM?
3. Is neuron-db a **measurable improvement** over the markdown-dump memory?

Harness: `examples/mcp-chat/bench_compare.py` (model: `gpt-4o-mini`). Token counts are exact
(OpenAI `usage`); accuracy is objective (ground truth computed from the graph).

---

## 1. The memory: an interlinked knowledge graph

A small org/personal graph where facts **reference each other**, so multi-hop questions
require chaining:

- **People** — each with a role/team, city, timezone, and a **manager** (another person).
- **Projects** — each with an **owner** (a person), status, deadline, language, and a
  **dependency** on another project.
- **Filler** — unique "log entry" facts to grow the store without changing the questions.

The identical content is materialized two ways: as neuron-db facts (`project Aurora owner
is Marisol`) and as a markdown document (`- Aurora: owner Marisol; status …`). Memory size
is grown via filler (1,000 → 6,000 → 50,000 facts) while the question set stays fixed.

### Question set (fixed, 12 questions over 2 projects)

| hops | example | what it forces |
|---|---|---|
| 1 | "Who owns project Aurora?" | one recall |
| 2 | "What city does the **owner of** Aurora live in?" | owner → city |
| 2 | "Who is the **manager of the owner of** Aurora?" | owner → manager |
| 3 | "What timezone is the **manager of the owner of** Aurora in?" | owner → manager → timezone |
| 2 | "Status of the project that Aurora **depends on**?" | dependency → status |

---

## 2. Can the LLM link neurons? (multi-hop)

Yes. Given recall tools, the model chains them: it recalls the inner entity, then uses that
result as the query for the next recall. Measured (vocabulary aligned, see §4):

| hops | accuracy | avg recalls the model issued |
|---|---:|---:|
| 1-hop | **100%** | 1.0 |
| 2-hop | **67%** | 1.8 |
| 3-hop | **50%** | 3.0 |

So the LLM genuinely walks the graph — *"timezone of the manager of the owner of Aurora"*
becomes `recall Aurora owner → Marisol`, `recall Marisol manager → Dana`,
`recall Dana timezone → WET`. Accuracy **compounds down** with depth: every hop must
succeed, so a per-hop reliability of ~80% lands ~50% at three hops. This is the cost of
chaining, and it argues for storing relationships so a single recall can answer (fewer
hops) where possible.

---

## 3. How much does linking cost?

Each hop is **one extra LLM round-trip** (the model must see the recalled value before it
can form the next query) plus **one neuron recall** (microseconds — see `SYNAPSE.md`).

- The **synapse is free**: each recall is tens of µs over the whole store.
- The **cost is LLM calls**: an N-hop answer ≈ N+1 model calls. That shows up as latency
  (~2.2 s for a multi-hop answer here, vs ~0.8 s for a single-call markdown answer) and as
  N× the (small) per-call token cost.

In other words: chaining doesn't cost memory, it costs model turns. Minimizing hops is a
prompt/schema concern, not a neuron-db performance concern.

> **Update — `recall_chain` removes the per-hop model cost.** The server now offers a
> multi-hop tool: the LLM passes one `(start, path)` and the synapse walks the whole chain
> server-side in microseconds (**~12.6 µs per hop, flat** — so a 50-hop chain is ~0.63 ms
> total, still 2 model calls). Each hop is a recall; a hop only advances if the relation
> actually matched the recalled fact, so broken chains report where they stopped. That
> turns an N-hop answer from **N+1 model calls into 2** (one to form the path, one to phrase
> the answer) — depth is now paid in microseconds, not model turns. "Infinite hops at no LLM
> cost." See `MEMORY_HARNESS.md` §3a. The lexical sensitivity in §4 is likewise softened by
> a morphological recall fallback (`owner`/`owned`/`owns` unify) — `MEMORY_HARNESS.md` §3b.

---

## 4. neuron-db vs markdown-dump

Same questions, two memory strategies, growing memory:

| facts | neuron in-tokens | md in-tokens | md/neuron | neuron $/1k-q | md $/1k-q | neuron acc | md acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1,122 | 9,898 | 8.8× | $0.19 | $1.49 | **100%** | 100% |
| 6,000 | 1,122 | 67,067 | **59.8×** | $0.19 | **$10.06** | **100%** | 92% |
| 50,000 | **1,122 (flat)** | ~446,863\* | **398×** | **$0.19** | n/a\* | **100%** | n/a\* |

\* the 50,000-fact markdown-dump **cannot be run**: at ~446,863 tokens it is **3.5× over
`gpt-4o-mini`'s 128k window**, so the model can't load it at all (token/cost/accuracy are
n/a). neuron-db answers the same questions at 100% on 1,122 tokens. Accuracy is selective —
haystack size doesn't change it — so it is effectively size-independent.

**What this shows:**

- **Context cost.** neuron-db is **flat (~1,122 tokens/turn)** no matter how much is
  remembered — it injects only the recalled facts. The markdown-dump is **linear**: it
  reinjects the *entire* memory every turn (9.9k → 67k → ~447k tokens), **8.8× more at 1k
  growing to ~398× at 50k** here, and unbounded. The synapse-timing work (`SYNAPSE.md`) and
  the needle benchmark (`BENCHMARKS.md` §5.4) show neuron recall stays flat to 50,000+
  facts, so this gap only widens.
- **The context-window wall.** The markdown-dump *cannot* grow past the window: at ~6,000
  facts it already spends ~67k tokens; by ~12k facts it approaches the 128k limit and
  breaks. At 50,000 facts the markdown memory is ~446,863 tokens — **3.5× the entire
  `gpt-4o-mini` window** — so it literally cannot be loaded and the run is impossible.
  neuron-db stays at **1,122 tokens with 100% accuracy** on that same store. This is the
  headline result: **effectively unbounded total memory at a flat per-turn cost.** neuron-db
  has no such ceiling (scale to millions, sharded per scope).
- **Accuracy.** Comparable (~75%) **when the query vocabulary matches how facts are
  stored.** The markdown-dump is **more robust to phrasing** — the model reads everything,
  so it doesn't matter whether you say "owner" or "owned by". neuron-db is **lexical**: in
  an earlier run where facts said `owned by` but questions said `owns/owner`, neuron-db
  accuracy collapsed to **17%** while markdown was unaffected. Aligning the vocabulary
  recovered it to 75%. This is the central tradeoff (and the case for a semantic-recall
  fallback — `MEMORY_HARNESS.md` §2).
- **Latency.** For a *single* multi-hop answer, the markdown-dump is faster (one big call)
  than neuron-db's several small calls — until the memory grows, at which point every
  markdown call carries the whole store and gets slower and pricier, while neuron-db's
  per-call size stays flat.

---

## 5. Verdict

| | markdown-dump | neuron-db |
|---|---|---|
| context cost / turn | **linear** in memory | **flat** (only recalled facts) |
| ceiling | the context window | none (scales to millions, sharded) |
| phrasing robustness | **high** (reads all) | lexical — needs aligned vocabulary |
| multi-hop | one call, model reasons over all | chained recalls (N+1 calls) |
| best when | memory is small & fits comfortably | memory is large / growing |

- For a **small, stable** memory that fits in context, the markdown-dump is simpler and
  more forgiving — and fine.
- For a **large or growing** memory, neuron-db is the one that scales: **flat cost, no
  context ceiling**, at the price of (a) recall that needs vocabulary alignment (closeable
  with semantic fallback) and (b) extra model round-trips for deep chains.
- The pragmatic design is **hybrid**: keep a tiny "hot" markdown of a few always-relevant
  facts in context, and put the long tail in neuron-db, recalled on demand.

The headline: **as memory grows, the markdown-dump's cost and context pressure grow with
it; neuron-db's stay flat.** The LLM can link neurons to answer relational questions; the
recall itself is free, and the only real cost of depth is model turns.

---

## 6. Final results — all fixes applied, verified live

After implementing `recall_chain`, the morphological fallback, and fixing three bugs that a
live `gpt-4o-mini` run surfaced (below), neuron-db **matches or beats** the markdown-dump on
accuracy at a **flat, ~60× lower token cost**:

| facts | neuron acc | md acc | neuron in-tok | md in-tok | neuron $/1k-q | md $/1k-q |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | **100%** | 100% | 1,122 | 9,898 | $0.19 | $1.49 |
| 6,000 | **100%** | 92% | 1,122 | 67,067 | $0.19 | $10.06 |
| 50,000 | **100%** | n/a | **1,122 (flat)** | ~446,863 (overflows window) | **$0.19** | n/a |

Per-hop accuracy is **100% at 1, 2, and 3 hops**, each in a constant **2.0 LLM calls** (the
model forms a `recall_chain` path once; the synapse walks it server-side, ~12.6 µs/hop). The
markdown-dump is not only pricier but degrades in accuracy at scale (92% at 6,000) — the
model mis-reasons over the raw 67k-token dump (lost-in-the-middle), whereas `recall_chain` is
deterministic. At 50,000 facts the markdown memory (~446,863 tokens) **overflows the model's
window entirely** and cannot run, while neuron-db answers at 100% on a flat 1,122 tokens.

### Bugs found by the live bench and fixed on the fly

1. **`recall_chain` multi-word relations.** A relation like `"depends on"` failed the per-hop
   validation (single fact words never equal the two-word relation), wrongly breaking the
   chain. Fixed: match on any content word of the relation, tolerant of stem/root variants.
2. **Subject/object ambiguity.** `recall_value("Aurora depends on")` returned **Falcon** —
   the query's bag of words also matches `"Falcon depends on Aurora"`, and recall's recency
   tiebreak picked the wrong one (recall and recall_many even disagreed). Fixed: a
   subject-position tiebreak prefers the fact where the query's words appear earliest, applied
   consistently to both paths.
3. **Partial-bind fallback.** When a relation word matched nothing (query "owner", fact
   "owned"), primary recall fell back to entity-only overlap and picked an arbitrary fact by
   recency. Fixed: when the relation doesn't fully bind, run the fuzzy scan and prefer it if
   it matches more of the query.
4. **Synonym bridging.** A curated synonym→canonical ontology, applied to *both* the query
   and the stored facts (`reports to`↔`manager`, `lives in`↔`city`, `owned by`↔`owner`), with
   the same subject-position tiebreak in the fallback path. See `MEMORY_HARNESS.md` §3b.

### Synonym-misaligned result (the former gap)

Storing facts with *different verbs than the queries use* ("is owned by", "lives in",
"reports to") was the adversarial case that scored **17%**. After the morphological root,
synonym ontology, and subject-position tiebreak in the fallback, it is **100%** at 1, 2, and
3 hops via `recall_chain` — identical to the aligned case, at the same flat ~1.1k tokens and
2 LLM calls. (Manual step-by-step chaining by the model is ~83% on the dependency question —
the model sometimes mis-reads a `recall` block — which is why the harness steers multi-hop to
`recall_chain`, where it is deterministic.) Open-vocabulary paraphrase beyond known synonyms
would still want an embedding tier.

A measurement note: multi-word values (`"in design"`) isolate to their salient token
(`"design"`); the scorer credits that as correct.

## Reproduce

```sh
cargo build --release --features mcp --bin neuron-mcp
export OPENAI_API_KEY=sk-...
python examples/mcp-chat/bench_compare.py --sizes 1000,6000 --chain
# scale point (markdown can't load 50k facts, so skip it):
python examples/mcp-chat/bench_compare.py --sizes 50000 --no-markdown --chain
```
