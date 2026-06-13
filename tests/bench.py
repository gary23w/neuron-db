"""Capability benchmarks. Run: python tests/bench.py"""
import os, sys, time, random, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import Neuron, NeuronDB


def creation_rate():
    with tempfile.TemporaryDirectory() as d:
        db = NeuronDB(os.path.join(d, "b.db"))
        n = 0; t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            db.turn(f"u{n}", "my name is U%d. my plan is pro. my city is Halifax" % n); n += 1
        print(f"creation (SQLite, 3 facts): {n}/sec")
    t0 = time.perf_counter(); m = 0
    while time.perf_counter() - t0 < 1.0:
        Neuron().observe("my name is X. my plan is pro. my city is Halifax"); m += 1
    print(f"creation (in-memory):       {m}/sec")


def accuracy_distinct(N=400):
    random.seed(2)
    adjs = ["north","south","east","west","main","spare","old","new","blue","red","work","home","beach","city","lake","hill","park","river","farm","studio"]
    nouns = ["wifi","door","bank","email","garage","locker","safe","router","vault","gate"]
    things = ["password","code","pin","key","number","combo","token","secret"]
    n = Neuron(max_facts=2000); probes = []; seen = set(); i = 0
    while len(probes) < N:
        a, no, th = random.choice(adjs), random.choice(nouns), random.choice(things)
        p = f"the {a} {no} {th}"
        if p in seen: continue
        seen.add(p); v = f"{random.randint(1000,9999)}{chr(65+i%26)}"
        n.observe(f"{p} is {v}"); probes.append((f"what is the {a} {no} {th}?", v)); i += 1
    hits = sum(1 for q, a in probes if (n.recall(q) or {}).get("value") == a)
    print(f"recall@1, {N} distinct keys: {hits}/{N} ({100*hits/N:.0f}%)")


def latency():
    for N in (100, 500, 5000):
        n = Neuron(max_facts=20000)
        for i in range(N): n.observe(f"the alpha{i} beta gamma is val{i}")
        q = "what is the alpha7 beta gamma?"; n.recall(q)
        t = time.perf_counter()
        for _ in range(500): n.recall(q)
        print(f"recall latency N={N}: {(time.perf_counter()-t)/500*1e6:.0f} us")


if __name__ == "__main__":
    creation_rate(); accuracy_distinct(); latency()
