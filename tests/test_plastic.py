"""Plasticity tests: the memory adapts with use, associates, and forgets.
These measure what a STATIC store can't -- behavior changing over a sequence of uses.
Run: python tests/test_plastic.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db.plastic import PlasticNeuron
from neuron_db import Neuron


def test_use_strengthens_recall():
    # two facts match "the meeting" equally; recency alone would pick the newer (friday)
    n = PlasticNeuron()
    n.observe("the meeting is on monday")
    n.observe("the meeting is on friday")
    assert n.recall("when is the meeting?")["value"] == "friday"   # recency baseline
    # use the monday fact a lot -> its strength should win the ambiguous cue
    mid = n.episodes[0]["_id"]
    for _ in range(30): n.reinforce(mid)
    assert n.recall("when is the meeting?")["value"] == "monday"   # adapted to usage


def test_association_forms_from_coactivation():
    n = PlasticNeuron()
    n.observe("Alice leads the Phoenix project")
    n.observe("the Falcon initiative ships in Q3")
    # recall them together repeatedly -> Hebbian link forms (fire together, wire together)
    for _ in range(4):
        n.recall("who leads Phoenix?")
        n.recall("when does Falcon ship?")
    rel = n.recall_related("who leads Phoenix?")
    linked = [r for r in rel[1:]]
    assert any("Falcon" in r["fact"] for r in linked), "co-activation should link Phoenix<->Falcon"


def test_decay_and_consolidation_forgets_unused():
    n = PlasticNeuron(half_life=5)
    n.observe("the temporary code is 1111")     # never used again
    n.observe("my name is Gary")
    for _ in range(60):                          # advance time; Gary stays in use, code decays
        n.recall("what is my name?")
    before = n.fact_count
    rep = n.consolidate(prune_below=0.1)
    assert rep["pruned"] >= 1 and n.fact_count < before
    assert n.recall("what is the temporary code?") is None        # forgotten
    assert n.recall("what is my name?")["value"] == "Gary"        # kept


def test_consolidate_merges_duplicates():
    n = PlasticNeuron()
    n.observe("the door code is 4452")
    n.observe("the door code is 4452")           # said twice
    rep = n.consolidate()
    assert rep["merged"] >= 1
    assert n.recall("what is the door code?")["value"] == "4452"


def test_plasticity_does_not_break_base_recall():
    n = PlasticNeuron()
    n.observe("my name is Marisol")
    n.observe("only the first 1,000 users score 150,000 coins")
    assert n.recall("what is my name?")["value"] == "Marisol"
    assert n.recall("how many users?")["value"] == "1,000"
    assert n.recall("how many coins?")["value"] == "150,000"
    assert n.recall("what is my blood type?") is None


def test_overhead_is_negligible():
    # plastic recall must be the same order of magnitude as the static store
    facts = [f"the {a} {b} key is V{i}" for i, (a, b) in enumerate(
        [(x, y) for x in "north south east west main spare old new blue red".split()
                for y in "wifi door bank vault gate shed desk badge gym car".split()])]
    base = Neuron(max_facts=5000); plas = PlasticNeuron(max_facts=5000)
    for f in facts: base.observe(f); plas.observe(f)
    q = "what is the spare vault key?"
    base.recall(q); plas.recall(q)
    t = time.perf_counter()
    for _ in range(2000): base.recall(q)
    tb = (time.perf_counter() - t) / 2000 * 1e6
    t = time.perf_counter()
    for _ in range(2000): plas.recall(q)
    tp = (time.perf_counter() - t) / 2000 * 1e6
    print(f"   base {tb:.0f} us/recall  |  plastic {tp:.0f} us/recall  ({tp/tb:.2f}x)")
    assert tp < tb * 3.0, f"plastic overhead too high: {tp:.0f} vs {tb:.0f} us"


if __name__ == "__main__":
    fns = [x for k, x in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try: fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e: print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
