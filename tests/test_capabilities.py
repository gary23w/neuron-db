import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import Neuron, NeuronDB, NeuronRouter

def v(n, q): return (n.recall(q) or {}).get("value")

def test_age_and_intro():
    n = Neuron(); n.observe("i'm 34"); n.observe("i'm Aiko")
    assert v(n, "what is my age?") == "34"
    assert v(n, "what is my name?") == "Aiko"

def test_coreference():
    n = Neuron(); n.observe("i adopted a puppy. her name is Mochi")
    assert v(n, "what is my puppy's name?") == "Mochi"

def test_accented_names():
    n = Neuron(); n.observe("my friend is named Tomás"); n.observe("i live in Zürich")
    assert v(n, "what is my friend's name?") == "Tomás"
    assert v(n, "where do i live?") == "Zürich"

def test_multi_fact_paste():
    n = Neuron()
    wrote = n.observe("the wifi is hunter2. the door code is 4452. checkout is at 11am.")
    assert len(wrote) == 3
    assert v(n, "what is the door code?") == "4452"
    assert v(n, "what is the wifi?") == "hunter2"

def test_correction_supersede():
    n = Neuron(); n.observe("my favorite color is blue")
    assert v(n, "what is my favorite color?") == "blue"
    n.observe("my favorite color is teal")
    assert v(n, "what is my favorite color?") == "teal"

def test_number_isolation_three_values():
    n = Neuron(); n.observe("the room holds 50 chairs, 8 tables and 200 guests")
    assert v(n, "how many guests?") == "200"
    assert v(n, "how many chairs?") == "50"
    assert v(n, "how many tables?") == "8"

def test_word_numbers():
    n = Neuron(); n.observe("there are seven wonders and twelve months")
    assert v(n, "how many wonders?").lower() == "seven"
    assert v(n, "how many months?").lower() == "twelve"

def test_does_not_store_questions_or_commands():
    n = Neuron()
    assert n.observe("what is the wifi password?") == []
    assert n.observe("summarize the news") == []
    assert n.fact_count == 0

def test_abstains_on_empty_cue():
    n = Neuron(); n.observe("my name is Gary")
    assert n.recall("hi") is None and n.recall("ok") is None

def test_large_distinct_capacity():
    n = Neuron(max_facts=5000); probes = []
    adjs=["north","south","east","west","main","spare","old","new","blue","red","work","home","beach","city","lake","hill","park","river","farm","studio"]
    nouns=["wifi","door","bank","email","garage","locker","safe","router","vault","gate","shed","cabin","boat","desk","phone","badge","gym","car","attic","porch"]
    things=["password","code","pin","key"]; seen=set(); i=0
    while len(probes)<300:
        a,no,th=adjs[i%20],nouns[(i//20)%20],things[(i//400)%4]; i+=1
        if (a,no,th) in seen: continue
        seen.add((a,no,th)); val=f"V{i:04d}"
        n.observe(f"the {a} {no} {th} is {val}"); probes.append((f"what is the {a} {no} {th}?",val))
    hits=sum(1 for q,a in probes if v(n,q)==a)
    assert hits>=295, f"only {hits}/300"

def test_db_restart_persistence():
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"t.db")
        db=NeuronDB(p); db.turn("u","my name is Marisol"); db.turn("u","the access code is 9921"); db.close()
        db2=NeuronDB(p)
        assert db2.get("u","what is my name?")=="Marisol"
        assert db2.get("u","what is the access code?")=="9921"; db2.close()

def test_math_ops():
    db=NeuronDB(":memory:")
    assert "42" in db.turn("x","17 + 25")["reply"]
    assert "= 9" in db.turn("x","12 - 3")["reply"]
    assert "= 56" in db.turn("x","8 * 7")["reply"]
    assert "= 4" in db.turn("x","20 / 5")["reply"]
    assert "100" in db.turn("x","what is 50 plus 50?")["reply"]

def test_router_scales_past_one_neuron():
    r=NeuronRouter(per_shard=64); probes=[]
    adjs=["north","south","east","west","main","spare","old","new","blue","red","work","home","beach","city","lake","hill","park","river","farm","studio"]
    nouns=["wifi","door","bank","email","garage","locker","safe","router","vault","gate","shed","cabin","boat","desk","phone","badge","gym","car","attic","porch","mill","barn","dock","loft","cave"]
    i=0
    while len(probes)<500:
        a,no=adjs[i%20],nouns[(i//20)%25]; i+=1
        r.observe(f"the {a} {no} key is K{i:04d}"); probes.append((f"what is the {a} {no} key?",f"K{i:04d}"))
    assert r.shard_count>=7 and r.fact_count==500
    hits=sum(1 for q,a in probes if r.get(q)==a)
    assert hits>=495, f"router recall {hits}/500"

if __name__=="__main__":
    fns=[x for k,x in sorted(globals().items()) if k.startswith("test_")]
    ok=0
    for fn in fns:
        try: fn(); ok+=1; print(f"PASS {fn.__name__}")
        except Exception as e: print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed"); sys.exit(0 if ok==len(fns) else 1)
