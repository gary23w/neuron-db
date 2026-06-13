"""Security tests for the encrypted neuron layer.
Run: python tests/test_secure.py
"""
import os, sys, sqlite3, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db.secure import SecureNeuronDB, SecureNeuron, derive_key, aead_encrypt, aead_decrypt


def test_exact_value_right_key():
    with tempfile.TemporaryDirectory() as d:
        db = SecureNeuronDB(os.path.join(d, "s.db"))
        db.put("alice", "pass-123", "wifi password", "hunter2")
        # exact value, no prose, no punctuation
        assert db.get("alice", "pass-123", "what is the wifi password?") == "hunter2"


def test_wrong_key_denied():
    with tempfile.TemporaryDirectory() as d:
        db = SecureNeuronDB(os.path.join(d, "s.db"))
        db.put("alice", "pass-123", "wifi password", "hunter2")
        assert db.get("alice", "WRONG", "what is the wifi password?") is None


def test_id_bump_denied():
    # using one neuron's secret against another's id must fail (key is bound to id)
    with tempfile.TemporaryDirectory() as d:
        db = SecureNeuronDB(os.path.join(d, "s.db"))
        db.put("alice", "alice-secret", "wifi password", "hunter2")
        db.put("bob", "bob-secret", "wifi password", "swordfish")
        assert db.get("bob", "alice-secret", "wifi password") is None
        assert db.get("bob", "bob-secret", "wifi password") == "swordfish"


def test_unknown_cue_abstains():
    with tempfile.TemporaryDirectory() as d:
        db = SecureNeuronDB(os.path.join(d, "s.db"))
        db.put("alice", "pass-123", "wifi password", "hunter2")
        assert db.get("alice", "pass-123", "what is my blood type?") is None


def test_dump_is_ciphertext_only():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.db")
        db = SecureNeuronDB(path)
        db.put("alice", "pass-123", "wifi password", "hunter2")
        db.put("alice", "pass-123", "door code", "4452")
        db.close()
        # an attacker who steals the file sees no plaintext: not values, cues, or keys
        for _id, blob in sqlite3.connect(path).execute("SELECT id, blob FROM secure"):
            for secret in ("hunter2", "4452", "wifi", "password", "door", "code", "pass-123"):
                assert secret not in blob, f"LEAK: {secret} found in stored blob"


def test_aead_roundtrip_and_tamper():
    k = derive_key("secret", "neuron-1")
    ct = aead_encrypt(k, b"top secret value")
    assert aead_decrypt(k, ct) == b"top secret value"
    # tampering with the ciphertext is detected
    bad = bytearray(ct); bad[-1] ^= 1
    try:
        aead_decrypt(k, bytes(bad)); assert False, "tamper not detected"
    except Exception:
        pass
    # wrong key fails
    try:
        aead_decrypt(derive_key("other", "neuron-1"), ct); assert False, "wrong key accepted"
    except Exception:
        pass


def test_fuzzy_key_match():
    # the KEY phrase can be fuzzy (the value stays encrypted); "wifi key" finds "wifi password"
    n = SecureNeuron(derive_key("s", "n"))
    n.put("the wifi password", "hunter2")
    assert n.get("what's the wifi key?") == "hunter2"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try: fn(); ok += 1; print(f"PASS {fn.__name__}")
        except Exception as e: print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
