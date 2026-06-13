"""Run every known recall scenario with the HIPPOCAMPUS (emergence cortex) ENABLED:
the store retrieves a working set, the model generates the answer over it. This measures
the FULL stack, not just the model-free store.

Needs the model: set NEURON_MODEL_DIR to a checkout with cortex.npz + tokenizer + gpt_numpy.py
(or let bridge auto-download gary23w/gary-neuron-emergent). Skips cleanly if unavailable.

Run: NEURON_MODEL_DIR=/path/to/model python tests/test_model_enabled.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db.plastic import PlasticNeuron

# (category, facts[], query, expected-substring-in-answer)
CASES = [
    ("number",     ["only the first 84,512 participants will receive badges"], "how many participants?", "84,512"),
    ("number",     ["the room holds 200 guests"], "how many guests?", "200"),
    ("code",       ["the wifi password is vekam73"], "what is the wifi password?", "vekam73"),
    ("code",       ["the door code is 4452"], "what is the door code?", "4452"),
    ("code",       ["the access code is 7781"], "what is the access code?", "7781"),
    ("multi-num",  ["the first 1,000 users score 150,000 coins"], "how many coins?", "150,000"),
    ("multi-num",  ["the first 1,000 users score 150,000 coins"], "how many users?", "1,000"),
    ("date",       ["the launch is on Friday"], "when is the launch?", "Friday"),
    ("date",       ["the meeting is on Tuesday"], "what day is the meeting?", "Tuesday"),
    ("name",       ["my name is Marisol"], "what is my name?", "Marisol"),
    ("name",       ["my dog is called Biscuit"], "what is my dog's name?", "Biscuit"),
    ("relation",   ["my sister is called Dana"], "what is my sister's name?", "Dana"),
    ("attr",       ["my favorite color is teal"], "what is my favorite color?", "teal"),
    ("attr",       ["i work as a plumber"], "what is my job?", "plumber"),
    ("city",       ["i live in Halifax"], "where do i live?", "Halifax"),
    ("abstain",    ["the wifi password is vekam73"], "what is my blood type?", "i don't know"),
]


def run(bridge):
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0]); rows = []
    for cat, facts, q, exp in CASES:
        n = PlasticNeuron()
        for f in facts: n.observe(f)
        ans = bridge.think(n, q)
        ok = exp.lower() in ans.lower()
        per[cat][0] += int(ok); per[cat][1] += 1
        rows.append((cat, q, exp, ans, ok))
    return per, rows


def main():
    try:
        from neuron_db.bridge import GaryNeuronBridge
        b = GaryNeuronBridge(max_new=12)
    except Exception as e:
        print(f"SKIP (model unavailable): {str(e)[:90]}"); return 0
    print(f"hippocampus ENABLED: {b.CFG}\n")
    per, rows = run(b)
    print(f"  {'category':<10} {'query':<34} {'want':<10} {'cortex answer'}")
    for cat, q, exp, ans, ok in rows:
        print(f"  {'OK' if ok else '..'} {cat:<8} {q[:33]:<34} {exp:<10} {ans!r}")
    tot_ok = sum(v[0] for v in per.values()); tot = sum(v[1] for v in per.values())
    print("\n  per category (cortex generation over the store's working set):")
    for cat in sorted(per):
        ok, t = per[cat]; print(f"    {cat:<10} {ok}/{t}")
    print(f"\n  FULL STACK (store + cortex): {tot_ok}/{tot} = {100*tot_ok/tot:.0f}%")
    print(f"  (note: the store alone recalls these deterministically; this measures the model's"
          f"\n   generation quality over the bounded window — a ~2k-vocab emergence cortex.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
