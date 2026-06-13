"""NeuronDB.think(): cortex is enabled by default and exposed via think(), while get()/recall()
stay the pure store path. Model-agnostic: passes whether or not the cortex is installed.
Run: python tests/test_think_default.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_db import NeuronDB

P=F=0
def ck(name, cond, info=""):
    global P,F; ok=bool(cond); P+=ok; F+=(not ok); print(("PASS " if ok else "FAIL ")+name+(("  | "+info) if info else ""))

# get()/recall() are always pure store (no model needed, microseconds)
db=NeuronDB(tempfile.mktemp(suffix=".db"), model=False)
db.observe("u:1","my plan is pro")
ck("get() is store-only and exact", db.get("u:1","what plan?")=="pro")
r=db.think("u:1","what plan?")
ck("think() degrades to store with no model", r["model"] is False and r["source"]=="store" and r["answer"]=="pro", str(r))
db.close()

# default constructor: cortex auto-loads IF present; think() returns a well-formed result either way
db=NeuronDB(tempfile.mktemp(suffix=".db"))   # model=True by default
db.observe("u:2","the deploy region is us-west-2")
ck("get() unaffected by default model flag", db.get("u:2","what is the deploy region?")=="us-west-2")
r=db.think("u:2","what is the deploy region?")
ck("think() returns {answer,source,model}", set(r)=={"answer","source","model"}, str(r))
ck("source matches model availability", (r["source"]=="cortex")==db.model_enabled, f"enabled={db.model_enabled} source={r['source']}")
db.close()

print(f"\n{P}/{P+F} passed")
sys.exit(0 if F==0 else 1)
