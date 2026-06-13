"""The model bridge is optional and import-safe. These tests run WITHOUT the model
(neuron-db ships model-free); the full generation path is exercised only when
NEURON_MODEL_DIR points at a gary-neuron-chat checkout.
Run: python tests/test_bridge.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_is_safe_without_model():
    # importing the bridge never forces numpy or the model
    from neuron_db.bridge import GaryNeuronBridge
    assert GaryNeuronBridge is not None


def test_clear_error_when_model_absent():
    from neuron_db.bridge import GaryNeuronBridge
    try:
        GaryNeuronBridge(model_dir="/no/such/model/dir")
        assert False, "should have raised"
    except RuntimeError as e:
        assert "model-free by design" in str(e)


def test_core_works_without_bridge():
    # the whole database is usable with no model at all
    from neuron_db import NeuronDB
    db = NeuronDB(":memory:")
    db.turn("u", "the launch is on Friday")
    assert db.get("u", "when is the launch?") == "Friday"


if __name__ == "__main__":
    fns = [x for k, x in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try: fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e: print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
