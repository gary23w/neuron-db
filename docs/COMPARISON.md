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

Harness: `examples/mcp_chat/bench_compare.py` (model: `gpt-4o-mini`). Token counts are exact
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
is grown via filler (300 → 1,500 → 6,000 facts) while the question set stays fixed.

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

---

## 4. neuron-db vs markdown-dump

Same questions, two memory strategies, growing memory:

| facts | neuron in-tokens | md in-tokens | md/neuron | neuron $/1k-q | md $/1k-q | neuron acc | md acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 1,183 | 2,656 | 2.2× | $0.20 | $0.40 | 75% | 83% |
| 1,500 | 1,183 | 15,412 | 13× | $0.20 | $2.31 | 75% | 75% |
| 6,000 | ~1,219 | 67,075 | **55×** | $0.21 | **$10.06** | —\* | —\* |

\* the 6,000-fact accuracy run was cut short by an API quota limit; token/cost figures are
from a complete prior run. Accuracy was identical at 300 and 1,500 facts (recall is
selective, so haystack size doesn't change it), so it is effectively size-independent.

**What this shows:**

- **Context cost.** neuron-db is **flat (~1.2k tokens/turn)** no matter how much is
  remembered — it injects only the recalled facts. The markdown-dump is **linear**: it
  reinjects the *entire* memory every turn (2.7k → 15k → 67k tokens), **2–55× more** here
  and unbounded. The synapse-timing work (`SYNAPSE.md`) and the needle benchmark
  (`BENCHMARKS.md` §5.4) show neuron recall stays flat to 50,000 facts, so this gap only
  widens.
- **The context-window ceiling.** The markdown-dump *cannot* grow past the window: at
  ~6,000 facts it already spends ~67k tokens; by ~12–15k facts it approaches a 128k limit
  and breaks. neuron-db has no such ceiling (scale to millions, sharded per scope).
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

## Reproduce

```sh
cargo build --release --features mcp --bin neuron-mcp
export OPENAI_API_KEY=sk-...
python examples/mcp_chat/bench_compare.py --sizes 300,1500,6000
```
