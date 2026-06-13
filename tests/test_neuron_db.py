"""neuron-db tests -- recall, value isolation, abstention, persistence, isolation.
Run:  python -m pytest -q   (or: python tests/test_neuron_db.py)
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import Neuron, NeuronDB


def _v(db, nid, q): return db.turn(nid, q)["reply"].rstrip(".").strip().lower()


def test_basic_recall():
    n = Neuron()
    n.observe("my name is Marisol")
    n.observe("the wifi password is hunter2")
    assert n.recall("what is my name?")["value"] == "Marisol"
    assert n.recall("what is the wifi password?")["value"] == "hunter2"


def test_value_isolation_multi_number():
    n = Neuron()
    n.observe("only the first 1,000 users score 150,000 coins")
    assert n.recall("how many users?")["value"] == "1,000"
    assert n.recall("how many coins?")["value"] == "150,000"


def test_word_number():
    n = Neuron()
    n.observe("there are four real bugs in the transcript")
    assert n.recall("how many bugs?")["value"].lower() == "four"


def test_abstention():
    n = Neuron()
    n.observe("my name is Gary")
    assert n.recall("what is my blood type?") is None


def test_relation_binding():
    n = Neuron()
    n.observe("my dog is called Biscuit")
    n.observe("my sister is called Dana")
    assert n.recall("what is my dog's name?")["value"] == "Biscuit"
    assert n.recall("what is my sister's name?")["value"] == "Dana"


def test_coreference():
    n = Neuron()
    n.observe("i adopted a puppy. her name is Mochi")
    assert n.recall("what is my puppy's name?")["value"] == "Mochi"


def test_minimal_serialization_lossless():
    n = Neuron()
    for f in ["my name is Marisol", "the door code is 4452", "i live in Halifax"]:
        n.observe(f)
    blob = n.dump()
    assert len(blob) / n.fact_count < 60        # ~30-40 B/fact
    n2 = Neuron.load(blob)
    assert n2.recall("what is my name?")["value"] == "Marisol"
    assert n2.recall("what is the door code?")["value"] == "4452"


def test_db_persistence_and_isolation():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.db")
        db = NeuronDB(path)
        db.turn("alice", "my name is Marisol")
        db.turn("bob", "my name is Viktor")
        db.close()
        db2 = NeuronDB(path)                      # reopen = "restart"
        assert _v(db2, "alice", "what is my name?") == "marisol"
        assert _v(db2, "bob", "what is my name?") == "viktor"   # neurons are isolated
        assert set(db2.neurons()) == {"alice", "bob"}
        db2.close()


def test_db_forget():
    with tempfile.TemporaryDirectory() as d:
        db = NeuronDB(os.path.join(d, "t.db"))
        db.turn("x", "the wifi password is hunter2")
        assert db.forget("x", "wifi")["forgot"] == 1
        assert db.turn("x", "what is the wifi password?")["kind"] == "idk"


def test_math_and_smalltalk():
    db = NeuronDB(":memory:")
    assert "42" in db.turn("x", "17 + 25")["reply"]
    assert db.turn("x", "hello")["kind"] == "smalltalk"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
