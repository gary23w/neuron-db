"""Pushing the plasticity to its limits + nailing down the guarantees honestly.
Run: python tests/test_plastic_limits.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db.plastic import PlasticNeuron
from neuron_db import NeuronDB


def v(n, q): return (n.recall(q) or {}).get("value")


# ---- the decay guarantee: ranking only, never deletion ----
def test_decay_never_deletes_without_consolidate():
    n = PlasticNeuron(half_life=5)
    n.observe("the archive code is 7777")
    for _ in range(200): n.tick += 1          # let it decay hard (eff ~ 0)
    assert n._eff(n.episodes[0]["_id"]) < 0.01  # decayed to nearly nothing...
    assert v(n, "what is the archive code?") == "7777"  # ...but STILL fully recallable

def test_no_decay_mode_is_permanent():
    n = PlasticNeuron(half_life=None)          # decay disabled
    n.observe("the archive code is 7777")
    for _ in range(10000): n.tick += 1
    assert n._eff(n.episodes[0]["_id"]) == 1.0  # strength never falls
    assert v(n, "what is the archive code?") == "7777"

def test_plain_db_never_decays():
    db = NeuronDB(":memory:")
    db.turn("u", "the archive code is 7777")
    for _ in range(1000): db.turn("u", "hello")   # lots of unrelated activity
    assert db.get("u", "what is the archive code?") == "7777"

def test_consolidate_protects_pinned_and_self():
    n = PlasticNeuron(half_life=3)
    n.observe("my name is Gary")                     # self-fact
    n.observe("the launch is on Tuesday")            # pin this one
    pid = n.episodes[1]["_id"]; n.pin(pid)
    n.observe("a throwaway note about nothing much")  # let this decay + be pruned
    for _ in range(50): n.tick += 1
    n.consolidate(prune_below=0.2)
    assert v(n, "what is my name?") == "Gary"        # self protected
    assert v(n, "when is the launch?") == "Tuesday"  # pinned protected


# ---- adaptation under heavy interference ----
def test_adaptation_under_500_competitors():
    # 500 facts all collide on the cue 'budget' (same stem); reinforce one; it must win.
    n = PlasticNeuron(half_life=1e9, max_facts=2000)
    for i in range(500): n.observe(f"the budget option is choice{i:04d}")
    target = n.episodes[123]["_id"]
    for _ in range(60): n.reinforce(target)
    hit = n.recall("what is the budget option?")
    assert hit["value"] == "choice0123", f"got {hit['value']}"

def test_reinforced_fact_survives_catastrophic_writes():
    # reinforce an old fact, then bury it under 1000 new facts; it still wins its cue
    n = PlasticNeuron(half_life=1e9, max_facts=5000)
    n.observe("the master key is GOLDEN")
    mid = n.episodes[0]["_id"]
    for _ in range(50): n.reinforce(mid)
    for i in range(1000): n.observe(f"the side note number {i} is junk{i}")
    assert v(n, "what is the master key?") == "GOLDEN"


# ---- association capacity ----
def test_association_picks_strongest_among_many():
    n = PlasticNeuron(half_life=1e9)
    n.observe("Alice owns the Phoenix project")
    # link Phoenix to 10 facts, one much more strongly
    for i in range(10):
        n.observe(f"detail{i} the thing{i} is item{i}")
    aid = n.episodes[0]["_id"]
    for i in range(1, 11):
        bid = n.episodes[i]["_id"]
        n._link(aid, bid, float(i))            # detail9 strongest
    rel = n.recall_related("who owns Phoenix?", k=3)
    assert any("thing9" in r["fact"] for r in rel[1:]), "strongest associate should surface"


# ---- the honest boundary: re-weighting is NOT learning ----
def test_plasticity_does_not_invent_unstored_facts():
    # no amount of reinforcement/association makes it answer something never stored
    n = PlasticNeuron()
    n.observe("the wifi password is hunter2")
    wid = n.episodes[0]["_id"]
    for _ in range(100): n.reinforce(wid)        # hammer it
    assert n.recall("what is my bank pin?") is None   # it cannot generalize -> abstains

def test_reinforcement_does_not_change_the_value():
    # adaptation re-ranks WHICH fact wins; it never alters the stored value itself
    n = PlasticNeuron()
    n.observe("the door code is 4452")
    for _ in range(100): n.recall("what is the door code?")
    assert v(n, "what is the door code?") == "4452"   # still exactly what was written


if __name__ == "__main__":
    fns = [x for k, x in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try: fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e: print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
