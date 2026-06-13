"""The model bridge is optional and import-safe. Core works with no model at all.
Run: python tests/test_bridge.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import_is_safe_without_model():
    from neuron_db.bridge import GaryNeuronBridge
    assert GaryNeuronBridge is not None
    assert GaryNeuronBridge.HF_REPO == "gary23w/gary-neuron-emergent"

def test_core_works_without_bridge():
    from neuron_db import NeuronDB
    db = NeuronDB(":memory:")
    db.turn("u", "the launch is on Friday")
    assert db.get("u", "when is the launch?") == "Friday"

def test_bridge_resolves_or_errors_cleanly():
    # With huggingface_hub present the bridge auto-downloads the published model, so we don't
    # construct it here (no network in a unit test). Without hf_hub and no local dir, it must
    # raise the documented RuntimeError rather than some opaque crash.
    from neuron_db.bridge import GaryNeuronBridge
    try:
        import huggingface_hub  # noqa
        return  # reachable -> fine; integration path tested elsewhere
    except ImportError:
        pass
    try:
        GaryNeuronBridge(model_dir="/no/such/dir"); assert False, "should raise"
    except RuntimeError as e:
        assert "model-free by design" in str(e)

if __name__ == "__main__":
    fns = [x for k, x in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try: fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e: print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
