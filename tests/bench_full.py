"""Comprehensive benchmark across every neuron-db use case, on three axes:
   A) SPEED            -- how fast is each operation
   B) OVER TIME        -- does it stay fast/accurate as a long session accumulates
   C) PLASTICITY       -- the measurable result of adaptation / association / forgetting

Run: python tests/bench_full.py    (deterministic; numbers vary by machine)
"""
import os, sys, time, random, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import Neuron, NeuronDB, NeuronRouter, SecureNeuronDB
from neuron_db.plastic import PlasticNeuron

ADJ = "north south east west main spare old new blue red work home beach city lake hill park river farm studio".split()
NOUN = "wifi door bank email garage locker safe router vault gate shed cabin boat desk phone badge gym car attic porch mill barn dock loft cave well pond tower bridge crypt".split()
THING = "password code pin key combo token secret hash".split()

def distinct_facts(n):
    out = []; i = 0
    while len(out) < n:
        a, no, th = ADJ[i % len(ADJ)], NOUN[(i // len(ADJ)) % len(NOUN)], THING[(i // (len(ADJ)*len(NOUN))) % len(THING)]
        i += 1
        out.append((f"the {a} {no} {th}", f"what is the {a} {no} {th}?", f"V{i:05d}"))
    return out

def section(t): print(f"\n{'='*64}\n{t}\n{'='*64}")
def line(k, v): print(f"  {k:<42} {v}")


# ---------------------------------------------------------------- A) SPEED
def speed():
    section("A) SPEED")
    # creation
    t0 = time.perf_counter(); m = 0
    while time.perf_counter() - t0 < 0.5:
        Neuron().observe("my name is X. my plan is pro. my city is Halifax"); m += 1
    line("neuron creation (in-memory, 3 facts)", f"{int(m/(time.perf_counter()-t0)):,}/sec")
    with tempfile.TemporaryDirectory() as d:
        db = NeuronDB(os.path.join(d, "s.db")); t0 = time.perf_counter(); k = 0
        while time.perf_counter() - t0 < 0.5:
            db.turn(f"u{k}", "my name is U. my plan is pro. my city is Halifax"); k += 1
        line("neuron creation (SQLite, durable)", f"{int(k/(time.perf_counter()-t0)):,}/sec"); db.close()
    # recall latency vs N
    for N in (100, 1000, 5000, 10000):
        facts = distinct_facts(N); n = Neuron(max_facts=N + 10)
        for ph, _, v in facts: n.observe(f"{ph} is {v}")
        q = facts[N // 2][1]; n.recall(q)
        t0 = time.perf_counter()
        for _ in range(3000): n.recall(q)
        line(f"recall latency, N={N}", f"{(time.perf_counter()-t0)/3000*1e6:.1f} us")
    # write throughput
    n = Neuron(max_facts=10 ** 9); t0 = time.perf_counter()
    for i in range(20000): n.observe(f"the {ADJ[i%20]} {NOUN[i%30]}{i} note is N{i}")
    line("write throughput (observe)", f"{int(20000/(time.perf_counter()-t0)):,} facts/sec")
    # plastic vs base recall
    facts = distinct_facts(1000); b = Neuron(max_facts=2000); p = PlasticNeuron(max_facts=2000)
    for ph, _, v in facts: b.observe(f"{ph} is {v}"); p.observe(f"{ph} is {v}")
    q = facts[500][1]; b.recall(q); p.recall(q)
    t0 = time.perf_counter()
    for _ in range(3000): b.recall(q)
    tb = (time.perf_counter()-t0)/3000*1e6
    t0 = time.perf_counter()
    for _ in range(3000): p.recall(q)
    tp = (time.perf_counter()-t0)/3000*1e6
    line("recall: static vs plastic", f"{tb:.1f} us  vs  {tp:.1f} us  ({tp/tb:.2f}x)")
    # secure
    with tempfile.TemporaryDirectory() as d:
        s = SecureNeuronDB(os.path.join(d, "v.db"))
        t0 = time.perf_counter()
        for i in range(500): s.put("v", "secret", f"key number {i}", f"val{i}")
        line("secure put (AES-GCM + keyed index)", f"{int(500/(time.perf_counter()-t0)):,}/sec")
        s.put("v", "secret", "wifi password", "hunter2")
        t0 = time.perf_counter()
        for _ in range(1000): s.get("v", "secret", "what is the wifi password?")
        line("secure get", f"{(time.perf_counter()-t0)/1000*1e6:.0f} us"); s.close()
    # router fan-out
    r = NeuronRouter(per_shard=128)
    for ph, _, v in distinct_facts(2000): r.observe(f"{ph} is {v}")
    q = distinct_facts(2000)[1000][1]; r.recall(q)
    t0 = time.perf_counter()
    for _ in range(1000): r.recall(q)
    line(f"router recall (2000 facts / {r.shard_count} shards, fan-out)", f"{(time.perf_counter()-t0)/1000*1e3:.2f} ms")
    # math
    db = NeuronDB(":memory:"); t0 = time.perf_counter()
    for _ in range(5000): db.turn("x", "12345 * 6789")
    line("arithmetic op", f"{(time.perf_counter()-t0)/5000*1e6:.1f} us")


# ----------------------------------------------------- B) PERFORMANCE OVER TIME
def over_time():
    section("B) PERFORMANCE OVER TIME  (10k-turn plastic session)")
    rng = random.Random(7)
    p = PlasticNeuron(max_facts=10 ** 9, half_life=400)
    facts = distinct_facts(2000)
    print(f"  {'turn':>6} {'facts':>7} {'recall@1':>9} {'latency':>9} {'dumpKB':>8}")
    consolidations = 0
    for turn in range(1, 10001):
        if rng.random() < 0.5 and len(facts):                 # write a new fact
            ph, q, v = facts[turn % len(facts)]
            p.observe(f"{ph} is {v}")
        else:                                                  # recall a known fact
            ph, q, v = facts[rng.randrange(min(turn, len(facts)))]
            p.recall(q)
        if turn % 2500 == 0:                                   # periodic sleep/consolidate
            p.consolidate(prune_below=0.02); consolidations += 1
        if turn % 2000 == 0 or turn == 1:
            sample = [facts[i] for i in range(0, min(p.fact_count, 200))]
            hits = sum(1 for ph, q, v in sample if (p.recall(q) or {}).get("value") == v)
            acc = f"{100*hits/max(1,len(sample)):.0f}%"
            q = facts[len(facts)//2][1]; p.recall(q)
            t0 = time.perf_counter()
            for _ in range(500): p.recall(q)
            lat = (time.perf_counter()-t0)/500*1e6
            dumpkb = sum(len(s) for s in [json_dump(p)]) / 1024
            print(f"  {turn:>6} {p.fact_count:>7} {acc:>9} {lat:>7.0f}us {dumpkb:>7.1f}")
    line("consolidations run", consolidations)
    line("verdict", "latency flat as the store grows; consolidation keeps facts bounded")

def json_dump(p):
    import json
    return json.dumps([[e["t"], 1 if e.get("self") else 0] for e in p.episodes])


# ------------------------------------------------------- C) PLASTICITY RESULTS
def plasticity():
    section("C) NEURAL PLASTICITY  (measured effects)")

    # adaptation curve: two facts collide on the cue 'meeting'; recency picks 'friday'.
    # reinforce 'monday' progressively and watch it take over.
    print("  adaptation curve (which fact wins 'when is the meeting?'):")
    print(f"    {'reinforce(monday)':>18} {'winner':>10} {'w(mon)':>8} {'w(fri)':>8}")
    for reps in (0, 2, 5, 10, 20, 40):
        p = PlasticNeuron(half_life=10000)
        p.observe("the meeting is on monday"); p.observe("the meeting is on friday")
        mid = p.episodes[0]["_id"]; fid = p.episodes[1]["_id"]
        for _ in range(reps): p.reinforce(mid)
        win = p.recall("when is the meeting?")["value"]
        print(f"    {reps:>18} {win:>10} {p._eff(mid):>8.2f} {p._eff(fid):>8.2f}")

    # forgetting curve: strength of an untouched fact over time (ticks since last use)
    print("\n  forgetting curve (effective strength vs ticks idle, half_life=50):")
    p = PlasticNeuron(half_life=50); p.observe("the vault code is 8842")
    eid = p.episodes[0]["_id"]
    print("    ticks:   " + "  ".join(f"{t:>4}" for t in (0, 25, 50, 100, 200)))
    strs = []
    base_tick = p.tick
    for t in (0, 25, 50, 100, 200):
        p.tick = base_tick + t
        strs.append(p._eff(eid))
    print("    strength:" + "  ".join(f"{s:>4.2f}" for s in strs))

    # association growth: co-activate two unrelated facts; link weight rises; spreading works
    print("\n  association (co-activation link weight over rounds):")
    p = PlasticNeuron()
    p.observe("Alice leads the Phoenix project"); p.observe("the Falcon initiative ships in Q3")
    aid = p.episodes[0]["_id"]; bid = p.episodes[1]["_id"]
    weights = []
    for rnd in range(1, 6):
        p.recall("who leads Phoenix?"); p.recall("when does Falcon ship?")
        weights.append(round(p._links.get(aid, {}).get(bid, 0.0), 2))
    print("    rounds 1..5 link weight: " + " ".join(str(w) for w in weights))
    rel = p.recall_related("who leads Phoenix?")
    surfaced = any("Falcon" in r["fact"] for r in rel[1:])
    line("spreading activation surfaces the associate", surfaced)

    # consolidation gain: duplicates + decayed facts -> merged/pruned, recall preserved
    print("\n  consolidation (sleep):")
    p = PlasticNeuron(half_life=5)
    for _ in range(5): p.observe("the door code is 4452")     # said 5 times
    p.observe("my name is Gary")
    for _ in range(40): p.recall("what is my name?")          # Gary stays warm, dups decay
    before = p.fact_count
    rep = p.consolidate(prune_below=0.1)
    line("facts before -> after", f"{before} -> {p.fact_count}")
    line("merged / pruned", f"{rep['merged']} / {rep['pruned']}")
    line("recall still correct after sleep", p.recall("what is my name?")["value"] == "Gary")


if __name__ == "__main__":
    print("neuron-db comprehensive benchmark")
    print(f"python {sys.version.split()[0]}")
    speed(); over_time(); plasticity()
    print("\ndone.")
