#!/usr/bin/env python3
"""Deep measurement: neocortex (LLM) <-> hippocampus (neuron-db) under a realistic,
interlinked memory, with multi-hop retrieval, AND a head-to-head against the markdown-dump
memory that most LLM setups use today.

It builds an org/personal knowledge graph (people with managers/teams/cities/timezones;
projects with owners/dependencies/status/deadlines) where facts reference each other, so
questions force the model to LINK neurons (chain several recalls). The same question set is
answered two ways at growing memory sizes:

  - neuron-db : model has recall tools (MCP); pulls facts on demand, chaining as needed.
  - markdown  : the ENTIRE memory is dumped into the system prompt every turn; no tools.

For each we capture exact tokens (OpenAI usage), $ cost, latency, accuracy, and (neuron)
the number of hops the model used. Ground truth is computed from the graph, so accuracy is
objective.

Run: set OPENAI_API_KEY, then  python bench_compare.py
"""
import argparse, json, os, statistics, sys, time, tempfile
from chat import Mcp, to_openai_tools, openai_chat

# gpt-4o-mini approx pricing ($ per token)
PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6
PHRASING = "aligned"   # "aligned" | "misaligned" (set from CLI; read by gen_org)

NAMES = ["Marisol","Dana","Kenji","Amara","Bjorn","Priya","Tariq","Lena","Mateo","Nadia","Owen","Sofia","Hiro","Esther","Diego","Yuki"]
TEAMS = ["frontend","infrastructure","design","data","platform","mobile","security","growth"]
CITIES = ["Halifax","Lisbon","Nairobi","Osaka","Bogota","Tallinn","Austin","Oslo"]
TZS = ["AST","WET","EAT","JST","COT","EET","CST","CET"]
ROLES = ["engineer","designer","analyst","architect","researcher","specialist"]
PNAMES = ["Aurora","Beacon","Citadel","Delta","Everest","Falcon","Granite","Helix"]
STATUS = ["shipped","in review","blocked","in design","in progress","planned","paused","archived"]
MONTHS = ["January","February","March","April","September","October","November","December"]
LANGS = ["Rust","Python","Go","Kotlin","TypeScript","Elixir","Swift","Zig"]

def code(x):
    s = ""
    for _ in range(5): s += chr(ord('a') + x % 26); x //= 26
    return s

def gen_org(n_people, n_projects, total_facts):
    people = []
    for i in range(n_people):
        mgr = None if i == 0 else NAMES[(i - 1) // 4]   # spans of 4 report to one manager
        people.append({"name": NAMES[i], "role": ROLES[i % len(ROLES)], "team": TEAMS[i % len(TEAMS)],
                       "city": CITIES[i % len(CITIES)], "tz": TZS[i % len(TZS)], "mgr": mgr})
    pidx = {p["name"]: p for p in people}
    projects = []
    for j in range(n_projects):
        oi = 1 + (j % (n_people - 1))               # owner is never person 0 (who has no manager)
        dep = PNAMES[(j + 3) % n_projects]
        if dep == PNAMES[j]: dep = PNAMES[(j + 1) % n_projects]
        projects.append({"name": PNAMES[j], "owner": NAMES[oi], "status": STATUS[j % len(STATUS)],
                         "deadline": MONTHS[j % len(MONTHS)], "lang": LANGS[j % len(LANGS)], "dep": dep})

    # Facts use canonical field NOUNS (owner/manager/city/timezone/status/deadline) so the
    # model's recall queries lexically align (the system prompt steers it to those nouns).
    facts = []
    if PHRASING == "misaligned":
        # store with different verbs than the queries use (owned by / lives in / reports to)
        # to exercise the morphological + alias fallback
        for p in people:
            facts.append(f"{p['name']} works as a {p['role']} on the {p['team']} team")
            facts.append(f"{p['name']} lives in {p['city']}")
            facts.append(f"{p['name']} timezone is {p['tz']}")
            if p["mgr"]: facts.append(f"{p['name']} reports to {p['mgr']}")
        for pr in projects:
            facts.append(f"project {pr['name']} is owned by {pr['owner']}")
            facts.append(f"project {pr['name']} status is {pr['status']}")
            facts.append(f"project {pr['name']} deadline is {pr['deadline']}")
            facts.append(f"project {pr['name']} is written in {pr['lang']}")
            facts.append(f"project {pr['name']} depends on project {pr['dep']}")
    else:
        for p in people:
            facts.append(f"{p['name']} role is {p['role']} on the {p['team']} team")
            facts.append(f"{p['name']} city is {p['city']}")
            facts.append(f"{p['name']} timezone is {p['tz']}")
            if p["mgr"]: facts.append(f"{p['name']} manager is {p['mgr']}")
        for pr in projects:
            facts.append(f"project {pr['name']} owner is {pr['owner']}")
            facts.append(f"project {pr['name']} status is {pr['status']}")
            facts.append(f"project {pr['name']} deadline is {pr['deadline']}")
            facts.append(f"project {pr['name']} language is {pr['lang']}")
            facts.append(f"project {pr['name']} depends on project {pr['dep']}")
    core = len(facts)
    for i in range(max(0, total_facts - core)):
        facts.append(f"log entry {code(i)} recorded routine event number {i}")

    # markdown rendering of the SAME memory
    md = ["# Memory", "", "## People"]
    for p in people:
        mgr = f"; manager {p['mgr']}" if p["mgr"] else ""
        md.append(f"- {p['name']}: role {p['role']} on {p['team']}; city {p['city']}; timezone {p['tz']}{mgr}")
    md.append("\n## Projects")
    for pr in projects:
        md.append(f"- {pr['name']}: owner {pr['owner']}; status {pr['status']}; deadline {pr['deadline']}; "
                  f"language {pr['lang']}; depends on {pr['dep']}")
    md.append("\n## Log")
    for i in range(max(0, total_facts - core)):
        md.append(f"- {code(i)}: routine event number {i}")
    md = "\n".join(md)

    # questions with computed ground truth, over 2 projects, spanning 1-3 hops
    qs = []
    for pr in projects[:2]:
        o = pidx[pr["owner"]]
        m = pidx[o["mgr"]] if o["mgr"] else None
        d = next(x for x in projects if x["name"] == pr["dep"])
        qs.append({"q": f"Who owns project {pr['name']}?", "expect": [pr["owner"]], "hops": 1})
        qs.append({"q": f"What is project {pr['name']}'s deadline?", "expect": [pr["deadline"]], "hops": 1})
        qs.append({"q": f"What city does the owner of project {pr['name']} live in?", "expect": [o["city"]], "hops": 2})
        qs.append({"q": f"Who is the manager of the owner of project {pr['name']}?", "expect": [o["mgr"]], "hops": 2})
        if m:
            qs.append({"q": f"What timezone is the manager of the owner of project {pr['name']} in?", "expect": [m["tz"]], "hops": 3})
        qs.append({"q": f"What is the status of the project that {pr['name']} depends on?", "expect": [d["status"]], "hops": 2})
    return facts, md, qs

NEURON_SYS = (
    "You answer questions using a long-term memory accessed through tools (recall, recall_value).\n"
    "Memory matches on the WORDS in stored facts. To answer a question that depends on another "
    "fact (e.g. 'the manager of the owner of project X'), recall STEP BY STEP: first recall the "
    "inner entity, then use that result as the query for the next recall. Example chain: "
    "recall_value 'Aurora owner' -> 'Marisol'; recall_value 'Marisol manager' -> 'Dana'; "
    "recall_value 'Dana timezone' -> 'WET'. Answer with just the value. If recall returns nothing, say you don't know."
)
NEURON_SYS_CHAIN = (
    "You answer questions using a long-term memory accessed through tools.\n"
    "For a single field, call recall_value (e.g. 'Aurora deadline').\n"
    "For a MULTI-HOP question of the form 'the A of the B of X', call recall_chain ONCE with "
    "start=X and path listing relations from X outward. Example: 'the timezone of the manager "
    "of the owner of Aurora' -> recall_chain(start='Aurora', path=['owner','manager','timezone']). "
    "'the status of the project that Aurora depends on' -> recall_chain(start='Aurora', "
    "path=['depends on','status']). Answer with just the value. If a tool returns nothing, say you don't know."
)
MD_SYS = ("You answer questions using the MEMORY provided below. Answer concisely with just the value. "
          "If it is not in the memory, say you don't know.\n")

_TRIVIAL = {"in", "the", "a", "an", "of", "on", "at", "is"}
def _salient(e):
    # last content word of a value, so "in design" credits an answer of "design"
    ws = [w for w in str(e).lower().split() if w not in _TRIVIAL]
    return ws[-1] if ws else str(e).lower()
def score(ans, expect):
    a = ans.lower()
    return all(str(e).lower() in a or _salient(e) in a for e in expect)

def ask_neuron(mcp, scope, tools, model, key, q, sys_prompt):
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": q["q"]}]
    tin = tout = hops = calls = 0
    t0 = time.perf_counter()
    ans = ""
    for _ in range(8):  # bound the chain
        msg, u = openai_chat(msgs, tools, model, key); calls += 1
        tin += u.get("prompt_tokens", 0); tout += u.get("completion_tokens", 0)
        tcs = msg.get("tool_calls")
        am = {"role": "assistant", "content": msg.get("content")}
        if tcs: am["tool_calls"] = tcs
        msgs.append(am)
        if not tcs:
            ans = msg.get("content") or ""; break
        for tc in tcs:
            try: args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError: args = {}
            args["scope"] = scope
            text, _, _ = mcp.call(tc["function"]["name"], args); hops += 1
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
    return {"tin": tin, "tout": tout, "hops": hops, "calls": calls,
            "ms": (time.perf_counter() - t0) * 1000, "ok": score(ans, q["expect"]), "ans": ans}

def ask_markdown(md, model, key, q):
    msgs = [{"role": "system", "content": MD_SYS + "\n# MEMORY\n" + md}, {"role": "user", "content": q["q"]}]
    t0 = time.perf_counter()
    msg, u = openai_chat(msgs, [], model, key)
    ans = msg.get("content") or ""
    return {"tin": u.get("prompt_tokens", 0), "tout": u.get("completion_tokens", 0), "hops": 0,
            "ms": (time.perf_counter() - t0) * 1000, "ok": score(ans, q["expect"]), "ans": ans}

def agg(rs):
    n = len(rs)
    tin = sum(r["tin"] for r in rs); tout = sum(r["tout"] for r in rs)
    return {"acc": sum(r["ok"] for r in rs) / n * 100, "tin": tin / n, "tout": tout / n,
            "ms": statistics.median(r["ms"] for r in rs), "hops": statistics.mean(r["hops"] for r in rs),
            "calls": statistics.mean(r.get("calls", 0) for r in rs),
            "cost": (tin * PRICE_IN + tout * PRICE_OUT) / n}

def main():
    global PHRASING
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--mcp", default=None)
    ap.add_argument("--sizes", default="300,1500,6000")
    ap.add_argument("--chain", action="store_true", help="prompt the model to use recall_chain for multi-hop")
    ap.add_argument("--misaligned", action="store_true", help="store facts with different verbs than the queries (exercises the fallback)")
    ap.add_argument("--no-markdown", action="store_true", help="skip the markdown baseline (saves tokens on focused runs)")
    ap.add_argument("--debug", action="store_true", help="print each missed question + the model's answer")
    args = ap.parse_args()
    PHRASING = "misaligned" if args.misaligned else "aligned"
    neuron_sys = NEURON_SYS_CHAIN if args.chain else NEURON_SYS
    key = os.environ.get("OPENAI_API_KEY") or sys.exit("set OPENAI_API_KEY")
    here = os.path.dirname(os.path.abspath(__file__))
    local = next((p for p in (os.path.join(here, "neuron-mcp.exe"), os.path.join(here, "neuron-mcp")) if os.path.exists(p)), None)
    mcp_bin = args.mcp or os.environ.get("NEURON_MCP_BIN") or local or "neuron-mcp"
    sizes = [int(s) for s in args.sizes.split(",")]

    syn_log = os.path.join(tempfile.gettempdir(), f"cmp_{os.getpid()}.log")
    env = dict(os.environ)
    env["NEURON_MCP_LOG"] = "1"
    env["NEURON_MCP_DB"] = os.path.join(here, "bench_compare.db")
    if os.path.exists(env["NEURON_MCP_DB"]): os.remove(env["NEURON_MCP_DB"])
    mcp = Mcp([mcp_bin], env, stderr_path=syn_log)
    mcp.handshake()
    tools = to_openai_tools(mcp.list_tools())

    cfg = f"phrasing={PHRASING}  multihop={'recall_chain' if args.chain else 'manual'}  markdown={'off' if args.no_markdown else 'on'}"
    print(f"model={args.model}  sizes={sizes}  {cfg}\n")
    rows = []
    for total in sizes:
        facts, md, qs = gen_org(12, 8, total)
        scope = f"org{total}-{PHRASING}"
        for i in range(0, len(facts), 500):
            mcp.call("remember", {"scope": scope, "facts": facts[i:i + 500]})
        md_tokens_est = len(md) // 4
        print(f"--- memory = {len(facts)} facts ({len(qs)} questions; markdown ~{md_tokens_est} tokens) ---")
        try:
            neu = [ask_neuron(mcp, scope, tools, args.model, key, q, neuron_sys) for q in qs]
            mdr = None if args.no_markdown else [ask_markdown(md, args.model, key, q) for q in qs]
        except SystemExit as e:   # e.g. API quota/rate limit mid-run: keep prior results
            print(f"  (stopped at {len(facts)} facts: {e})")
            break
        na = agg(neu)
        ma = agg(mdr) if mdr else None
        rows.append((len(facts), na, ma))
        print(f"  neuron-db: acc {na['acc']:.0f}%  in {na['tin']:.0f} tok  out {na['tout']:.0f}  llm-calls {na['calls']:.1f}  tool-calls {na['hops']:.1f}  {na['ms']:.0f} ms  ${na['cost']*1000:.4f}/1k-q")
        if ma:
            print(f"  markdown : acc {ma['acc']:.0f}%  in {ma['tin']:.0f} tok  out {ma['tout']:.0f}  llm-calls 1.0  tool-calls -    {ma['ms']:.0f} ms  ${ma['cost']*1000:.4f}/1k-q")
        # per-hop accuracy for the neuron mode (shows neuron-linking)
        byhop = {}
        for q, r in zip(qs, neu):
            byhop.setdefault(q["hops"], []).append(r)
        for h in sorted(byhop):
            rs = byhop[h]
            print(f"      neuron {h}-hop: acc {sum(x['ok'] for x in rs)/len(rs)*100:.0f}%  avg llm-calls {statistics.mean(x['calls'] for x in rs):.1f}")
        if args.debug:
            for q, r in zip(qs, neu):
                if not r["ok"]:
                    print(f"      MISS [{q['hops']}h] {q['q']}  expected={q['expect']}  got={r['ans'][:90]!r}")
    mcp.close()

    if any(ma for _, _, ma in rows):
        print("\n================ neuron-db vs markdown-dump ================")
        print(f"{'facts':>6} | {'neuron in_tok':>13} {'md in_tok':>10} | {'neuron $/1k':>11} {'md $/1k':>9} | {'neuron acc':>10} {'md acc':>7} | {'md/neuron tok':>13}")
        for f, na, ma in rows:
            if not ma: continue
            ratio = ma["tin"] / max(1, na["tin"])
            print(f"{f:>6} | {na['tin']:>13.0f} {ma['tin']:>10.0f} | {na['cost']*1000:>11.4f} {ma['cost']*1000:>9.4f} | {na['acc']:>9.0f}% {ma['acc']:>6.0f}% | {ratio:>12.1f}x")
    try: os.remove(syn_log); os.remove(env["NEURON_MCP_DB"])
    except OSError: pass

if __name__ == "__main__":
    main()
