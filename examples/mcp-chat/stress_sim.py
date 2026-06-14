#!/usr/bin/env python3
"""Long stress simulation: a deep org graph (k-ary management tree + project dependency DAG)
with thousands of facts, and DEEP multi-hop questions (up to 5-6 hops) answered through the
real neuron-mcp recall_chain. Ground truth is computed from the graph, so accuracy is
objective. Every miss is printed with the path the model used, so bugs can be diagnosed.

Run: set OPENAI_API_KEY, then  python stress_sim.py [--people 121] [--projects 40] [--filler 1500]
"""
import argparse, json, os, statistics, sys, time, tempfile
from chat import Mcp, openai_chat, to_openai_tools

CITIES = ["Halifax","Lisbon","Nairobi","Osaka","Bogota","Tallinn","Austin","Oslo","Lima","Cairo","Perth","Riga"]
TZS    = ["AST","WET","EAT","JST","COT","EET","CST","CET","PET","EST","AWST","MSK"]
STATUS = ["shipped","reviewing","blocked","designing","building","planned","paused","archived"]
SYL1 = ["Ka","Ma","Ze","Ri","Lo","Na","Ta","Vi","Su","Ko","Da","Pe","Ju","Ha","Bo","Fe","Gi","Wo","Yu","Ne","Sa","Mo","Lu","Ti","Qa","Xi","Ru","Ba"]
SYL2 = ["ron","lin","dar","mira","sho","vik","tara","len","nor","sai","del","mon","ric","bel","gus","wen","far","zon","kai","ven","sol","tem","pho","quin"]
PSYL = ["Aur","Bea","Cit","Del","Eve","Fal","Gra","Hel","Ion","Jad","Kit","Lyr","Mer","Nox","Orb","Pyx","Qua","Rho","Sol","Tau","Umb","Vex","Wis","Xan","Yon","Zeph"]

def names(n):
    out, seen = [], set()
    for a in SYL1:
        for b in SYL2:
            nm = a + b
            if nm not in seen: seen.add(nm); out.append(nm)
            if len(out) >= n: return out
    return out

def pnames(n):
    out = []
    for s in PSYL:
        out.append(s + "Project") if False else out.append(s + "us")  # e.g. Aurus, Beaus...
        if len(out) >= n: return out
    # extend if needed
    i = 0
    while len(out) < n:
        out.append(PSYL[i % len(PSYL)] + "x" + str(i)); i += 1
    return out

def build(n_people, n_projects, filler, K=3):
    nm = names(n_people)
    people = []
    for i in range(n_people):
        mgr = None if i == 0 else nm[(i - 1) // K]
        people.append({"name": nm[i], "mgr": mgr, "city": CITIES[i % len(CITIES)], "tz": TZS[i % len(TZS)]})
    pidx = {p["name"]: p for p in people}
    pn = pnames(n_projects)
    projects = []
    for j in range(n_projects):
        dep = pn[j - 3] if j >= 3 else None
        projects.append({"name": pn[j], "owner": nm[(j * 7 + 5) % n_people],
                         "status": STATUS[j % len(STATUS)], "dep": dep})
    prj = {p["name"]: p for p in projects}

    facts = []
    for p in people:
        if p["mgr"]: facts.append(f"{p['name']} manager is {p['mgr']}")
        facts.append(f"{p['name']} city is {p['city']}")
        facts.append(f"{p['name']} timezone is {p['tz']}")
    for pr in projects:
        facts.append(f"project {pr['name']} owner is {pr['owner']}")
        facts.append(f"project {pr['name']} status is {pr['status']}")
        if pr["dep"]: facts.append(f"project {pr['name']} depends on project {pr['dep']}")
    def code(x):
        s="";
        for _ in range(5): s+=chr(97+x%26); x//=26
        return s
    for i in range(filler):
        facts.append(f"log entry {code(i)} recorded routine event number {i}")

    # ---- helpers over ground truth ----
    def ancestor(name, k):
        cur = name
        for _ in range(k):
            m = pidx.get(cur, {}).get("mgr")
            if not m: return None
            cur = m
        return cur
    def dep_chain(pname, k):
        cur = pname
        for _ in range(k):
            d = prj.get(cur, {}).get("dep")
            if not d: return None
            cur = d
        return cur

    # ---- questions with computed answers + the canonical path ----
    qs = []
    deep_people = [p["name"] for p in people if (len(people) > 40)][-30:]  # the leaves, deepest in tree
    import itertools
    # 1) k-level manager chains
    for k in range(1, 6):
        for nme in deep_people[:6]:
            anc = ancestor(nme, k)
            if anc:
                ladder = "the manager of " * k
                qs.append({"q": f"Who is {ladder}{nme}? Answer with just the name.",
                           "expect": [anc], "hops": k, "path": ["manager"] * k, "start": nme, "cat": f"mgr^{k}"})
    # 2) owner -> attr
    for pr in projects[:10]:
        o = pidx[pr["owner"]]
        qs.append({"q": f"What city does the owner of project {pr['name']} live in?",
                   "expect": [o["city"]], "hops": 2, "path": ["owner","city"], "start": pr["name"], "cat":"own->city"})
        m = pidx.get(o["mgr"]) if o["mgr"] else None
        if m:
            qs.append({"q": f"What timezone is the manager of the owner of project {pr['name']} in?",
                       "expect": [m["tz"]], "hops": 3, "path": ["owner","manager","timezone"], "start": pr["name"], "cat":"own->mgr->tz"})
    # 3) dependency chains (deep) — nested relative clauses ("the project that … depends on")
    def nested_dep(name, k):
        return "the project that " * k + name + " depends on" * k
    for pr in projects:
        d1 = dep_chain(pr["name"], 1)
        if d1:
            qs.append({"q": f"What is the status of the project that {pr['name']} depends on?",
                       "expect": [prj[d1]["status"]], "hops": 2, "path":["depends on","status"], "start": pr["name"], "cat":"dep->status"})
        d2 = dep_chain(pr["name"], 2)
        if d2:
            qs.append({"q": f"Who owns {nested_dep(pr['name'],2)}?",
                       "expect": [prj[d2]["owner"]], "hops": 3, "path":["depends on","depends on","owner"], "start": pr["name"], "cat":"dep^2->owner"})
        d3 = dep_chain(pr["name"], 3)
        if d3:
            qs.append({"q": f"What is the status of {nested_dep(pr['name'],3)}?",
                       "expect": [prj[d3]["status"]], "hops": 4, "path":["depends on","depends on","depends on","status"], "start": pr["name"], "cat":"dep^3->status"})
    return facts, qs

SYS = (
    "You answer questions using a long-term memory accessed through tools "
    "(recall_value, recall_chain).\n"
    "For a MULTI-HOP question ('the A of the B of ... of X'), call recall_chain ONCE with "
    "start=X (the innermost named entity) and path = the relations from X outward. The path "
    "for 'the manager of the manager of Kai' is ['manager','manager'] with start='Kai'. "
    "For 'the status of the project that P depends on' use start='P', path=['depends on','status']. "
    "COUNT the repeated relation words exactly: 'the project that the project that the project P "
    "depends on depends on depends on' has THREE 'depends on' steps, so path=['depends on','depends "
    "on','depends on','status'] with start='P'. Match the count exactly. "
    "For a single field use recall_value. Answer with just the value. If a tool returns nothing, say you don't know."
)
_TRIV = {"in","the","a","an","of","on","at","is","just","name"}
def salient(e):
    ws=[w for w in str(e).lower().split() if w not in _TRIV]; return ws[-1] if ws else str(e).lower()
def score(ans, expect):
    a=ans.lower(); return all(str(e).lower() in a or salient(e) in a for e in expect)

def ask(mcp, scope, tools, model, key, q):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":q["q"]}]
    tin=tout=calls=0; used=[]; ans=""
    t0=time.perf_counter()
    for _ in range(10):
        msg,u=openai_chat(msgs,tools,model,key); calls+=1
        tin+=u.get("prompt_tokens",0); tout+=u.get("completion_tokens",0)
        tcs=msg.get("tool_calls"); am={"role":"assistant","content":msg.get("content")}
        if tcs: am["tool_calls"]=tcs
        msgs.append(am)
        if not tcs: ans=msg.get("content") or ""; break
        for tc in tcs:
            try: args=json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError: args={}
            args["scope"]=scope
            shown={k:v for k,v in args.items() if k!="scope"}
            used.append(tc["function"]["name"]+json.dumps(shown))
            text,_,_=mcp.call(tc["function"]["name"],args)
            msgs.append({"role":"tool","tool_call_id":tc["id"],"content":text})
    return {"ok":score(ans,q["expect"]),"ans":ans,"tin":tin,"tout":tout,"calls":calls,"used":used,
            "ms":(time.perf_counter()-t0)*1000}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--people",type=int,default=121); ap.add_argument("--projects",type=int,default=40)
    ap.add_argument("--filler",type=int,default=1500); ap.add_argument("--model",default=os.environ.get("OPENAI_MODEL","gpt-4o-mini"))
    ap.add_argument("--mcp",default=None)
    ap.add_argument("--dry",action="store_true",help="fire canonical paths directly via recall_chain (no LLM) to test neuron-db itself")
    args=ap.parse_args()
    key=None if args.dry else (os.environ.get("OPENAI_API_KEY") or sys.exit("set OPENAI_API_KEY"))
    here=os.path.dirname(os.path.abspath(__file__))
    local=next((p for p in (os.path.join(here,"neuron-mcp.exe"),os.path.join(here,"neuron-mcp")) if os.path.exists(p)),None)
    mcp_bin=args.mcp or os.environ.get("NEURON_MCP_BIN") or local or "neuron-mcp"

    facts,qs=build(args.people,args.projects,args.filler)
    env=dict(os.environ); env["NEURON_MCP_DB"]=os.path.join(here,"stress.db")
    if os.path.exists(env["NEURON_MCP_DB"]): os.remove(env["NEURON_MCP_DB"])
    mcp=Mcp([mcp_bin],env); mcp.handshake()
    tools=None if args.dry else to_openai_tools(mcp.list_tools())
    scope="org"
    for i in range(0,len(facts),500): mcp.call("remember",{"scope":scope,"facts":facts[i:i+500]})
    print(f"seeded {len(facts)} facts ({args.people} people, {args.projects} projects, {args.filler} filler)")
    print(f"mode={'DRY (recall_chain direct)' if args.dry else args.model}  questions={len(qs)}\n{'='*72}")

    def ask_dry(q):
        t0=time.perf_counter()
        text,_,_=mcp.call("recall_chain",{"scope":scope,"start":q["start"],"path":q["path"]})
        # text is "value  (via a -> b -> c)" or "chain broke after: ..."
        val=text.split("  (via")[0].strip() if "(via" in text else ""
        return {"ok":score(val,q["expect"]),"ans":text,"tin":0,"tout":0,"calls":0,"used":["recall_chain"+json.dumps({'start':q['start'],'path':q['path']})],"ms":(time.perf_counter()-t0)*1000}

    results=[]
    try:
        for i,q in enumerate(qs):
            r=ask_dry(q) if args.dry else ask(mcp,scope,tools,args.model,key,q)
            r["q"]=q; results.append(r)
            mark="OK " if r["ok"] else "XX "
            print(f"{mark}[{q['cat']:>12}] {('✓' if r['ok'] else r['ans'][:46])}")
    except SystemExit as e:
        print(f"(stopped early: {e})")
    mcp.close()

    print("\n"+"="*72+"\nACCURACY BY HOP DEPTH")
    byh={}
    for r in results: byh.setdefault(r["q"]["hops"],[]).append(r)
    for h in sorted(byh):
        rs=byh[h]; acc=sum(x["ok"] for x in rs)/len(rs)*100
        print(f"  {h}-hop: {acc:>5.0f}%  ({sum(x['ok'] for x in rs)}/{len(rs)})  median {statistics.median(x['calls'] for x in rs):.0f} llm-calls")
    print("\nACCURACY BY CATEGORY")
    byc={}
    for r in results: byc.setdefault(r["q"]["cat"],[]).append(r)
    for c in sorted(byc):
        rs=byc[c]; print(f"  {c:>14}: {sum(x['ok'] for x in rs)/len(rs)*100:>5.0f}%  ({sum(x['ok'] for x in rs)}/{len(rs)})")
    ov=sum(r["ok"] for r in results); tot=len(results)
    print(f"\nOVERALL: {ov}/{tot} = {ov/tot*100:.1f}%")
    tin=sum(r["tin"] for r in results); print(f"avg input tokens/q: {tin/max(1,tot):.0f}  ·  avg llm-calls/q: {statistics.mean(r['calls'] for r in results):.1f}")

    miss=[r for r in results if not r["ok"]]
    if miss:
        print(f"\n{'='*72}\nMISSES ({len(miss)}) — path the model used vs expected:")
        for r in miss:
            print(f"  [{r['q']['cat']}] {r['q']['q'][:70]}")
            print(f"      expected={r['q']['expect']}  canonical_path={r['q']['path']}")
            print(f"      got={r['ans'][:80]!r}")
            print(f"      tools={r['used']}")
    try: os.remove(env["NEURON_MCP_DB"])
    except OSError: pass

if __name__=="__main__":
    main()
