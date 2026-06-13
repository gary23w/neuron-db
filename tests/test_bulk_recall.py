"""Recalling LARGE sums of data: when the consumer needs many facts (many tokens) back,
not a single value. Uses recall_many (top-k by overlap) and the synapse graph (spreading
activation pulls connected facts that don't even share the cue). Model-free, fast.
Run: python tests/test_bulk_recall.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import Neuron
from neuron_db.plastic import PlasticNeuron

ADJ = "north south east west main spare old new gold iron jade onyx ruby teal".split()
NOUN = "server log report ticket build deploy alert metric trace span query cache".split()


def build(n_total=1000, cluster=40, topic="phoenix"):
    n = Neuron(max_facts=10 ** 9); members = []
    for i in range(n_total):
        if i % (n_total // cluster) == 0 and len(members) < cluster:
            f = f"the {topic} {NOUN[i % len(NOUN)]} {i} status is value{i}"; members.append(f)
        else:
            f = f"the {ADJ[i % len(ADJ)]} {NOUN[i % len(NOUN)]} {i} note is x{i}"
        n.observe(f)
    return n, members, topic


def test_bulk_recall_coverage_and_speed():
    n, members, topic = build()
    n.recall_many(topic)  # warm
    t = time.perf_counter(); got = n.recall_many(f"everything about {topic}", k=50)
    ms = (time.perf_counter() - t) * 1e3
    hits = sum(1 for g in got if topic in g)
    assert hits == len(members), f"cluster coverage {hits}/{len(members)}"
    assert ms < 5.0, f"bulk recall too slow: {ms:.2f} ms"

def test_bulk_latency_flat_in_k():
    n, _, topic = build()
    def lat(k):
        t = time.perf_counter()
        for _ in range(300): n.recall_many(topic, k=k)
        return (time.perf_counter() - t) / 300 * 1e6
    l10, l200 = lat(10), lat(200)
    # pulling 20x more facts should not cost 20x more time (it's the candidate scan that dominates)
    assert l200 < l10 * 3, f"k=10:{l10:.0f}us k=200:{l200:.0f}us"

def test_synapses_pull_connected_facts():
    # spreading activation over the Hebbian graph surfaces an associate that shares NO cue stem
    n = PlasticNeuron(half_life=1e9)
    n.observe("Alice owns the Phoenix project")
    n.observe("the rollout ships in the third quarter")   # no shared stem with "phoenix"
    for _ in range(4):                                     # co-activate -> wire together
        n.recall("who owns Phoenix?"); n.recall("when does the rollout ship?")
    rel = n.recall_related("who owns Phoenix?", k=3)
    assert any("rollout" in r["fact"] for r in rel[1:]), "synapse should pull the connected fact"


if __name__ == "__main__":
    n, members, topic = build()
    got = n.recall_many(f"everything about {topic}", k=50)
    hits = sum(1 for g in got if topic in g)
    toks = sum(len(g.split()) for g in got)
    print(f"store 1000 facts, '{topic}' cluster of {len(members)}")
    print(f"  recall_many -> {len(got)} facts ({hits}/{len(members)} cluster), ~{toks} tokens")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try: fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e: print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
