# The plastic database: a memory that thinks and grows

## What we are actually trying to achieve

Every benchmark in `BENCHMARKS.md` asks a key-value question: *given a cue, did the store
return the exact fact I wrote?* That measures a lookup table. It is the wrong target for
what we're building.

A plastic memory's value shows up across a *sequence* of uses, over time:

- **Adaptation** — what you use often surfaces first and wins ambiguous cues.
- **Association** — recalling A makes related B reachable, building structure you never
  explicitly wrote ("fire together, wire together").
- **Forgetting** — unused facts decay so the memory stays sharp instead of bloating.
- **Growth** — the memory consolidates experience into its own weights and gets better.

So the metrics change. Not recall@1 on a frozen set, but: does the right fact win *after
use*? Does a cue surface its associates? Do stale facts fall away? `tests/test_plastic.py`
measures exactly these.

## Why this is built *in gary-neuron*, not around it

gary-neuron was trained to emergence for a reason: the cortex learned to read its context
window and copy values out of it, and the plastic hippocampus learned to adapt *during
use* without gradient descent. That is the thinking-and-growing engine. A symbolic
re-derivation that ignores it would waste the training.

The catch is performance. The plastic hippocampus is a neural attention pass — O(N) over
whatever is in context. Run it across a million-fact database per query and you get exactly
the lag we want to avoid. The resolution is to **split plasticity across two tiers**, so
the neural part only ever sees a bounded working set.

## Two tiers

```
   cue ─▶  STORE TIER  (neuron-db, PlasticNeuron)         ── scales to millions ──
           cheap scalar plasticity decides WHAT is relevant:
             • strength (bumped on use), lazy exponential decay
             • Hebbian association graph, one-hop spreading activation
           returns a small working set (a handful of facts)
              │
              ▼
           MODEL TIER  (gary-neuron: cortex + plastic hippocampus)  ── bounded ──
           runs only over that working set (a 192–384-token window):
             • cortex reads the context and answers / completes
             • plastic hippocampus does surprise-gated, in-the-moment adaptation
           cost is O(working set), never O(database)
              │
              ▼
           SLEEP  (consolidation)  ── off the hot path ──
           folds new episodes into cortex weights; merges/prunes the store.
           the model literally grows; the store stays lean.
```

The store is the bloodstream; gary-neuron is the brain. The brain never has to think about
the whole body at once — the store feeds it only what matters, already ranked by use and
association.

## The store tier (shipped: `PlasticNeuron`)

Pure standard library, every update O(1) or O(neighbors), no background threads:

| mechanism | cost | how |
|---|---|---|
| strength | O(1) | each fact has a weight, bumped on recall |
| decay | O(1), lazy | `w · ½^(age / half_life)`, computed at read time — no sweeps |
| association | O(1) | facts recalled within a window get linked |
| spreading activation | O(neighbors) | recall returns the hit plus its strongest associates |
| consolidate | off hot path | merge duplicate-stem facts, prune decayed ones |

Measured overhead: plastic recall is **1.3× the static store** (88 µs vs 67 µs on a
100-fact neuron) — the same order of magnitude. Adaptation, association, and forgetting all
work (`test_plastic.py`, 6/6). This is the substrate that makes the model tier cheap: it
narrows millions of facts to a working set without any neural cost.

## The model tier (design: wiring in gary-neuron)

The trained model lives in `Garrett/gary-neuron-chat` (numpy) and `neuron-cloud` (the
TypeScript port of the cortex forward pass). To keep neuron-db itself zero-dependency, the
model tier is an **optional adapter**, not a core dependency:

1. **Retrieve** — `PlasticNeuron.recall_related(cue, k)` returns the working set: the best
   fact plus its associates, ranked by strength × link weight.
2. **Think** — format the working set as the `U:/G:` context the cortex was trained on, run
   the forward pass (TS at the edge via neuron-cloud, or numpy locally). The cortex copies
   the answer out of the window; the plastic hippocampus reinforces what mattered. This is
   where "trained to emergence" pays off — it only ever sees the bounded window.
3. **Write back** — surprise-gated facts the exchange produced go back into the store, and
   the hippocampus's in-the-moment weights bias the next few turns.
4. **Sleep** — periodically, replay buffered episodes mixed with base corpus to consolidate
   them into the cortex weights (the gary-neuron-chat `/sleep` mechanism), then run
   `PlasticNeuron.consolidate()` to merge and prune the store. The model grows; the store
   stays lean. Both happen off the query path.

## Performance contract

| where | cost | scales with |
|---|---|---|
| store recall (ranking + decay) | ~90 µs | candidate facts (sub-linear via index) |
| association spread | ~µs | neighbors of the hit |
| model forward (cortex) | ~ms | working-set tokens (fixed window), **not** db size |
| sleep / consolidate | seconds, async | total facts, off the hot path |

No query ever runs a neural net over the whole database. That is the entire point of the
split, and it is why a plastic, thinking, growing memory can stay fast.

## Status

- **Shipped:** `PlasticNeuron` — store-tier plasticity (strength, decay, association,
  spreading, consolidation), tested, ~1.3× overhead.
- **Next:** the model-tier adapter — feed `recall_related` working sets to the gary-neuron
  cortex (numpy locally / TS in neuron-cloud) and wire `/sleep` consolidation back into both
  the weights and the store.

## Decay: where it lives and what it can (and can't) do

Can your data silently vanish? No. Guarantees, tested in `tests/test_plastic_limits.py`:

- **Decay is a store-tier feature added in `PlasticNeuron`** — not inherited from the gary-neuron hippocampus. In the original model the episodic store was permanent; only the hippocampus was transient (fast weights, reset per conversation).
- **Decay only changes recall ranking.** A heavily decayed fact (effective strength ~0) is still returned by `recall` — it just loses ties to fresher facts. Decay never deletes.
- **Only `consolidate(prune_below=...)` deletes**, explicit/opt-in, and it **protects `self` facts plus anything you `pin()`**.
- **Plain `Neuron` / `NeuronDB` do not decay at all** — the default for a database that must never forget. `PlasticNeuron(half_life=None)` keeps plasticity with decay off.

So "factual decay" is opt-in, ranking-only, and reversible by pinning — not a leak.

## Do you need the "neocortex" (an LLM)?

No. The three-tier brain picture is only for the *memory-for-an-LLM-agent* case. For a plain app or website database, **your app is the neocortex** — it decides what to store and ask; neuron-db is just the database (no LLM, no model, no dependencies — see `examples/app_backend.py`). Use `NeuronDB` for durable storage, or `PlasticNeuron` for usage-weighted ranking and associations. The LLM and the gary-neuron hippocampus are optional layers added only when you want the store to *reason* or *consolidate*.

## Does the hippocampus "get smarter"? — the honest answer

Three different things; only one is learning:

- **`PlasticNeuron` re-weights.** Use strengthens, disuse decays, co-use associates — it changes *which* stored fact wins a cue, but never invents a fact you didn't store (`test_plasticity_does_not_invent_unstored_facts`) and never alters a stored value (`test_reinforcement_does_not_change_the_value`). Adaptation, not learning; no generalization.
- **The trained hippocampus adapts within a conversation** (fast weights) then resets — also not permanent learning.
- **Only sleep consolidation into the cortex weights is true "getting smarter"** — offline gradient training (`/sleep`), not a runtime effect.

The memory gets better-*tuned* to your usage at runtime, cheaply and safely; it gets genuinely *smarter* only when experience is consolidated into the model's weights offline.
