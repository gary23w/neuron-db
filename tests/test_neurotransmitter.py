"""Neurotransmitter-style recall: spreading activation over the synapse graph.

The brain doesn't run a forward pass to remember -- a cue releases transmitter,
activation spreads across synapses, fades with distance (reuptake), and the
most-activated neurons are the memory. recall_spreading() does exactly this in
microseconds: excitatory release at cue matches, inhibitory gating on relation
conflicts, multi-hop spread that surfaces associated facts sharing NO cue word.

Run: python tests/test_neurotransmitter.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db.plastic import PlasticNeuron


def _wire(a, b, store, n=4):
    """Co-activate two facts so they wire together (fire together -> wire together)."""
    for _ in range(n):
        store.recall(a); store.recall(b)


def test_multi_hop_surfaces_unshared_fact():
    # A shares the cue; C shares nothing with the cue but is wired A->B->C.
    n = PlasticNeuron(half_life=1e9)
    n.observe("the phoenix project is owned by Dana")
    n.observe("the rollout ships in the third quarter")
    n.observe("the budget was approved by finance")
    _wire("who owns phoenix?", "when does the rollout ship?", n)   # A-B
    _wire("when does the rollout ship?", "who approved the budget?", n)  # B-C
    res = n.recall_spreading("who owns phoenix?", hops=2, k=5)
    facts = [r["fact"] for r in res]
    assert any("phoenix" in f for f in facts), "seed (excitatory match) must fire"
    # the 2-hop fact has no shared word with the query yet should be reachable
    assert any("budget" in f for f in facts), f"2-hop associate not surfaced: {facts}"
    # and it must be marked as reached-by-spreading, not a direct seed
    budget = next(r for r in res if "budget" in r["fact"])
    assert budget["seed"] is False


def test_reuptake_attenuates_with_distance():
    n = PlasticNeuron(half_life=1e9)
    n.observe("the alpha signal is strong")
    n.observe("the relay forwards traffic")
    _wire("what about alpha?", "what does the relay do?", n)
    res = {r["fact"]: r["activation"] for r in n.recall_spreading("what about alpha?", hops=1)}
    seed = next(v for f, v in res.items() if "alpha" in f)
    relay = next((v for f, v in res.items() if "relay" in f), 0.0)
    assert relay > 0, "neighbour should receive some activation"
    assert relay < seed, "reuptake must make a 1-hop neighbour weaker than the seed"


def test_inhibitory_relation_gate():
    # a dog-name cue must NOT fire a brother-name memory (relation mismatch = inhibition)
    n = PlasticNeuron(half_life=1e9)
    n.observe("my dog's name is Rex")
    n.observe("my brother's name is Sam")
    res = n.recall_spreading("what is my dog's name?", hops=1, k=5)
    facts = " ".join(r["fact"].lower() for r in res)
    assert "rex" in facts, "correct relation should fire"
    assert "sam" not in facts, "conflicting relation must be inhibited (gated off)"


def test_latency_far_under_cortex():
    # Latency is O(candidates that match the cue), not O(store) and never a forward pass.
    # Distinct vocab -> a selective cue matches one cluster (true microseconds). A broad
    # cue that matches every fact is the worst case -- still far under the 65 ms cortex.
    ADJ = "north south east west gold iron jade onyx ruby teal main spare".split()
    NOUN = "server log report ticket build deploy alert metric trace span".split()
    n = PlasticNeuron(half_life=1e9, max_facts=10**9)
    for i in range(2000):
        n.observe(f"the {ADJ[i % len(ADJ)]} {NOUN[(i // 12) % len(NOUN)]} {i} reads code{i}")
    def lat(q, reps=500):
        n.recall_spreading(q)
        t = time.perf_counter()
        for _ in range(reps): n.recall_spreading(q)
        return (time.perf_counter() - t) / reps * 1e6
    selective = lat("jade trace 1450")     # narrow cue -> few candidates
    broad = lat("reads", reps=100)         # matches every fact -> full scan (worst case)
    print(f"    [latency] selective cue {selective:.0f} us | worst-case scan {broad:.0f} us "
          f"(vs ~65000 us cortex)")
    assert selective < 600, f"selective spreading should be microseconds: {selective:.0f} us"
    assert broad < 8000, f"even a full-scan cue must stay well under the cortex: {broad:.0f} us"


def test_pure_read_does_not_mutate():
    n = PlasticNeuron(half_life=1e9)
    for i in range(20):
        n.observe(f"the node{i} link is up{i}")
    before_facts = n.fact_count
    before_w = dict(n._w); before_tick = n.tick
    n.recall_spreading("node7 link", hops=2)
    assert n.fact_count == before_facts
    assert n._w == before_w, "spreading recall must not change strengths"
    assert n.tick == before_tick, "spreading recall must not advance the clock"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try: fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e: print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
