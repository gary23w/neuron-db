"""Maximum-potential stress tests for the episodic store and the plastic (hippocampus)
tier. Prints real numbers; asserts correctness where it matters.
Run: python tests/test_stress.py
"""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import Neuron, NeuronRouter
from neuron_db.plastic import PlasticNeuron

ADJ = "north south east west main spare old new blue red work home beach city lake hill park river farm studio gold iron jade onyx ruby teal coral amber slate ivory".split()
NOUN = "wifi door bank email garage locker safe router vault gate shed cabin boat desk phone badge gym car attic porch mill barn dock loft cave well pond tower bridge crypt forge".split()
THING = "password code pin key combo token secret hash cipher seal".split()
ZONE = "alpha bravo delta echo foxtrot golf hotel india juliet kilo lima mike nova oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()

def distinct(n):
    # 4-component keys (adj x noun x thing x zone ~ 2.3M unique stem-sets); dedup guarantees
    # each fact has a unique stem-set, which is what the store needs to recall it uniquely.
    out = []; seen = set(); i = 0
    while len(out) < n:
        a = ADJ[i % len(ADJ)]; no = NOUN[(i // len(ADJ)) % len(NOUN)]
        th = THING[(i // (len(ADJ) * len(NOUN))) % len(THING)]
        zo = ZONE[(i // (len(ADJ) * len(NOUN) * len(THING))) % len(ZONE)]
        i += 1
        key = (a, no, th, zo)
        if key in seen: continue
        seen.add(key)
        out.append((f"the {a} {no} {th} {zo}", f"what is the {a} {no} {th} {zo}?", f"V{len(out):06d}"))
    return out

def head(t): print(f"\n{'='*60}\n{t}\n{'='*60}")
def kv(k, v): print(f"  {k:<40} {v}")


def episodic_at_scale():
    head("EPISODIC STORE AT SCALE (distinct keys)")
    for N in (10_000, 50_000):
        facts = distinct(N); n = Neuron(max_facts=N + 10)
        t0 = time.perf_counter()
        for ph, _, v in facts: n.observe(f"{ph} is {v}")
        build = time.perf_counter() - t0
        probes = random.Random(1).sample(facts, 1000)
        hits = sum(1 for ph, q, v in probes if (n.recall(q) or {}).get("value") == v)
        q = probes[0][1]; n.recall(q)
        t0 = time.perf_counter()
        for _ in range(1000): n.recall(q)
        lat = (time.perf_counter() - t0) / 1000 * 1e6
        dump = len(n.dump())
        kv(f"N={N:,}", f"build {build:.1f}s | recall@1 {hits}/1000 | {lat:.0f} us/recall | {dump/1e6:.1f} MB")
        assert hits >= 995, f"distinct recall dropped: {hits}/1000 at N={N}"


def router_at_max():
    head("ROUTER (chained neurons) AT 100k FACTS")
    facts = distinct(100_000)
    r = NeuronRouter(per_shard=256)
    t0 = time.perf_counter()
    for ph, _, v in facts: r.observe(f"{ph} is {v}")
    build = time.perf_counter() - t0
    probes = random.Random(2).sample(facts, 500)
    hits = sum(1 for ph, q, v in probes if r.get(q) == v)
    t0 = time.perf_counter()
    for ph, q, v in probes[:200]: r.get(q)
    lat = (time.perf_counter() - t0) / 200 * 1e3
    kv("100,000 facts", f"build {build:.1f}s | {r.shard_count} shards | recall@1 {hits}/500 | {lat:.1f} ms/recall (fan-out)")


def hippocampus_max():
    head("PLASTIC (HIPPOCAMPUS TIER) AT MAX")
    # adaptation must hold under massive interference
    n = PlasticNeuron(half_life=1e9, max_facts=60000)
    facts = distinct(20_000)
    t0 = time.perf_counter()
    for ph, _, v in facts: n.observe(f"{ph} is {v}")
    kv("plastic build, 20k facts", f"{time.perf_counter()-t0:.1f}s")
    target = n.episodes[12345]["_id"]
    for _ in range(80): n.reinforce(target)
    ph, q, v = facts[12345]
    assert n.recall(q)["value"] == v
    kv("adaptation holds under 20k facts", "OK")

    # association graph stress: many links, spreading still O(neighbors)
    p = PlasticNeuron(half_life=1e9)
    for i in range(5000): p.observe(f"node{i} the item{i} is x{i}")
    rng = random.Random(3); ids = [e["_id"] for e in p.episodes]
    t0 = time.perf_counter()
    for _ in range(50000): p._link(rng.choice(ids), rng.choice(ids), 1.0)
    kv("50k Hebbian links built", f"{time.perf_counter()-t0:.2f}s")
    hub = ids[0]
    t0 = time.perf_counter()
    for _ in range(2000): sorted(p._links.get(hub, {}).items(), key=lambda kv2: -kv2[1])[:3]
    kv("spreading activation lookup", f"{(time.perf_counter()-t0)/2000*1e6:.0f} us/hop")

    # consolidation on a large store with heavy duplication
    c = PlasticNeuron(half_life=8, max_facts=100000)
    for _ in range(10):
        for i in range(2000): c.observe(f"the report{i} status is ok")   # massive duplication
    before = c.fact_count
    t0 = time.perf_counter(); rep = c.consolidate(prune_below=0.3); dt = time.perf_counter() - t0
    kv("consolidate 20k->? (dedupe)", f"{before} -> {rep['facts']} in {dt:.2f}s ({rep['merged']} merged)")

    # sustained throughput
    s = PlasticNeuron(half_life=1e9, max_facts=10**9)
    t0 = time.perf_counter()
    for i in range(50000): s.observe(f"the {ADJ[i%30]} {NOUN[i%30]}{i} key is K{i}")
    kv("sustained write throughput", f"{int(50000/(time.perf_counter()-t0)):,} facts/sec")


def decay_extremes():
    head("DECAY EXTREMES")
    n = PlasticNeuron(half_life=1)
    n.observe("the code is 9999")
    for _ in range(1000): n.tick += 1
    eff = n._eff(n.episodes[0]["_id"])
    kv("strength after