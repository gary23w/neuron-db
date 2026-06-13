"""THE FINAL FRONTIER: can the emergence cortex find a value buried in a LARGE context?
Fills the model's 384-token window with many facts and asks for one buried among them
(needle-in-the-window), at different fill levels and needle positions. Measures in-context
retrieval under a full window -- the hardest thing we ask of the model.

Needs the model (NEURON_MODEL_DIR or HF auto-download). Run:
  NEURON_MODEL_DIR=/path python tests/test_large_context.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ADJ = "north south east west main spare old new blue red work home beach city lake hill".split()
NOUN = "wifi door bank email garage locker safe router vault gate shed cabin desk phone gym attic".split()

def build_prompt(b, n_facts, needle_pos, needle_val="73218"):
    facts = []
    for i in range(n_facts):
        a, no = ADJ[i % len(ADJ)], NOUN[(i // len(ADJ)) % len(NOUN)]
        facts.append((f"the {a} {no} code is {1000+i}", f"the {a} {no} code"))
    # the needle: a distinctive target inserted at needle_pos
    ta, tno = "secret", "panel"
    facts.insert(needle_pos, (f"the {ta} {tno} code is {needle_val}", f"the {ta} {tno} code"))
    lines = "".join(f"U: {f}\nG: noted.\n" for f, _ in facts)
    query = f"what is the {ta} {tno} code?"
    return lines + f"U: {query}\nG:", query

def gen(b, prompt, n=10):
    ids = b.tok.encode(prompt).ids
    ntok = len(ids); ids = ids[-b.CFG["BLK"]:]
    out = []
    for _ in range(n):
        import numpy as np
        lg, _ = b._G.forward(b.P, np.array([ids[-b.CFG["BLK"]:]]), b.CFG)
        nx = int(lg[0, -1].argmax())
        if nx == 0 or b.tok.decode([nx]) == "\n": break
        ids.append(nx); out.append(nx)
    return b.tok.decode(out).strip(), ntok

def main():
    try:
        from neuron_db.bridge import GaryNeuronBridge
        b = GaryNeuronBridge(max_new=10)
    except Exception as e:
        print(f"SKIP (model unavailable): {str(e)[:90]}"); return 0
    print(f"hippocampus ENABLED, window = {b.CFG['BLK']} tokens\n")
    print(f"  {'facts':>5} {'pos':>6} {'tokens':>7} {'found':>6}  answer")
    results = []
    for n_facts in (3, 8, 15, 25, 35):
        for pos_name, pos in [("first", 0), ("middle", n_facts // 2), ("last", n_facts)]:
            prompt, q = build_prompt(b, n_facts, pos)
            ans, ntok = gen(b, prompt)
            found = "73218" in ans
            results.append((n_facts, pos_name, ntok, found))
            print(f"  {n_facts:>5} {pos_name:>6} {ntok:>7} {('YES' if found else 'no'):>6}  {ans!r}")
    ok = sum(1 for *_, f in results if f)
    print(f"\n  needle found: {ok}/{len(results)}")
    print("  (window is 384 tokens; prompts beyond that are truncated to the tail, so an")
    print("   early needle in an over-long context falls out of view -- that boundary is the point.)")
    return 0

if __name__ == "__main__":
    sys.exit(main())


def main_with_store():
    """The architecture's answer: don't make the cortex find the needle in a big context.
    Let the STORE narrow 35 similar facts to the right one, THEN the cortex generates."""
    try:
        from neuron_db.bridge import GaryNeuronBridge
        from neuron_db.plastic import PlasticNeuron
        b = GaryNeuronBridge(max_new=10)
    except Exception as e:
        print(f"SKIP: {str(e)[:80]}"); return
    n = PlasticNeuron()
    for i in range(35):
        a, no = ADJ[i % len(ADJ)], NOUN[(i // len(ADJ)) % len(NOUN)]
        n.observe(f"the {a} {no} code is {1000+i}")
    n.observe("the secret panel code is 73218")          # the needle, among 35 distractors
    q = "what is the secret panel code?"
    ws = n.recall(q)
    print("\n  STORE-NARROWED (35 facts in the store):")
    print(f"    store.recall -> {ws['value'] if ws else None}  (the store finds the needle)")
    print(f"    cortex over the narrowed working set -> {b.think(n, q)!r}")

if __name__ == "__main__":
    pass
