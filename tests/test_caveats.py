"""Caveat fixes ported alongside the Rust core: near-duplicate keys disambiguate via an
exact-token tier, and multi-word values come back whole. Run: python tests/test_caveats.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import Neuron

def test_exact_key_disambiguation():
    n = Neuron()
    n.observe("the project17 token is aaa111")
    n.observe("the project170 token is bbb222")
    assert n.recall("what is the project170 token?")["value"] == "bbb222"
    assert n.recall("what is the project17 token?")["value"] == "aaa111"

def test_multi_word_value():
    n = Neuron()
    n.observe("my favorite tool is Search Console")
    assert n.recall("what is my favorite tool?")["value"] == "Search Console"

if __name__ == "__main__":
    fns=[v for k,v in sorted(globals().items()) if k.startswith("test_")]; ok=0
    for fn in fns:
        try: fn(); ok+=1; print("PASS",fn.__name__)
        except Exception as e: print("FAIL",fn.__name__,e)
    print(f"\n{ok}/{len(fns)} passed"); sys.exit(0 if ok==len(fns) else 1)
