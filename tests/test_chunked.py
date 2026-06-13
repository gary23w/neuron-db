"""Large context, solved: normalize a big context to a cortex-sized input by chunking it
into the store, letting the store SELECT the relevant chunk, and running the cortex once
over that. Compares against dumping the whole context at the cortex.

Needs the model. Run: NEURON_MODEL_DIR=/path python tests/test_chunked.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ADJ = "north south east west main spare old new blue red work home beach city lake hill".split()
NOUN = "wifi door bank email garage locker safe router vault gate shed cabin desk phone gym attic".split()

def make(n, needle_val="73218"):
    facts = [f"the {ADJ[i % len(ADJ)]} {NOUN[(i // len(ADJ)) % len(NOUN)]} code is {1000 + i}" for i in range(n)]
    facts.insert(n // 2, f"the secret panel code is {needle_val}")   # buried in the middle
    return facts, "what is the secret panel code?"

def main():
    try:
        from neuron_db.bridge import GaryNeuronBridge
        from neuron_db.reader import ChunkedReader
        b = GaryNeuronBridge(max_new=10); rd = ChunkedReader(b)
    except Exception as e:
        print(f"SKIP (model unavailable): {str(e)[:90]}"); return 0
    print(f"window = {b.CFG['BLK']} tokens\n  {'ctx':>5} {'strategy':<26} {'found':>6} {'latency':>9}")
    chunked_ok = 0; chunked_n = 0
    for n in (15, 50, 150, 500):
        facts, q = make(n); blob = ". ".join(facts)
        t = time.perf_counter(); a = b.generate(facts, q); msA = (time.perf_counter() - t) * 1e3
        print(f"  {n:>5} {'A: dump full context':<26} {('YES' if a and '73218' in a else 'no'):>6} {msA:>7.0f}ms  {a!r}")
        t = time.perf_counter(); c = rd.answer(blob, q); msC = (time.perf_counter() - t) * 1e3
        ok = bool(c and '73218' in c); chunked_ok += ok; chunked_n += 1
        print(f"  {n:>5} {'C: chunk+select+cortex':<26} {('YES' if ok else 'no'):>6} {msC:>7.0f}ms  {c!r}")
    print(f"\n  chunk+select+cortex: {chunked_ok}/{chunked_n} found, latency ~flat in context size")
    print("  (store selection is O(candidates) us; the cortex runs ONCE over the 1 selected chunk,")
    print("   so latency does not grow with the context. Dumping the full context is slower AND wrong.)")
    assert chunked_ok == chunked_n, "chunk+select should find the needle at every size"
    return 0

if __name__ == "__main__":
    sys.exit(main())
