#!/usr/bin/env python3
"""Large app simulation: a returning user with a big memory, an LLM recalling sizable
blocks per turn over the neuron-db MCP server. Measures the *synapse* -- how fast
neuron-db fires and returns recalled neurons on the fly.

It seeds a realistic per-user store (hundreds of facts) directly via the MCP `remember`
tool, then runs a multi-turn conversation through an OpenAI model. Three latencies are
separated:
  - synapse (pure neuron recall, microseconds)  <- server-side, from NEURON_MCP_LOG
  - MCP round-trip (recall + stdio IPC)          <- client-side wall clock
  - LLM call (network)                            <- client-side wall clock

Run: set OPENAI_API_KEY, then
  python app_sim.py [--model gpt-4o-mini] [--mcp <path>] [--scope user:garrett]
"""
import argparse, json, os, statistics, sys, time, tempfile
from chat import Mcp, openai_chat, to_openai_tools, Chat, SYSTEM

# the model should pull larger blocks for broad questions
APP_SYSTEM = SYSTEM + (
    "\n- For broad questions ('summarize', 'give me a rundown', 'everything about X', "
    "'who are my ...'), call `recall` with a larger k (15-30) to pull a full block, then "
    "synthesize. For a single field, use `recall_value`."
)

def persona_facts():
    """A believable founder/engineer persona: ~50 meaningful facts + bulk history."""
    core = [
        "my name is Garrett Stimpson", "my role is founder and staff engineer",
        "my company is Neuronworks", "my home city is Halifax", "my timezone is AST",
        "my primary editor is neovim", "my main language is Rust",
        "my secondary language is Python", "my laptop is a Framework 16",
        "my phone is a Pixel 9", "my coffee order is a flat white",
        "my dietary restriction is no peanuts", "my git host is GitHub",
        "my default deploy region is us-east-1", "my database of choice is SQLite",
        "my dog is named Pixel", "my partner is named Dana",
        "my standup time is 9:30am", "my preferred theme is gruvbox dark",
        "my keyboard is a Moonlander", "my desk setup has two 4k monitors",
        "my note app is Obsidian", "my browser is Firefox",
        "my cloud provider is Fly.io", "my ci system is GitHub Actions",
        # projects
        "project Aurora is a memory database written in Rust",
        "project Aurora deadline is December 12", "project Aurora status is in review",
        "project Aurora repo is github.com/neuronworks/aurora",
        "project Beacon is a billing service written in Go",
        "project Beacon status is blocked on payment provider",
        "project Beacon deadline is January 20",
        "project Citadel is the authentication service in Rust",
        "project Citadel status is shipped",
        "project Delta is a mobile client in Kotlin",
        "project Delta status is in design",
        # people
        "my teammate Mateo leads frontend", "my teammate Priya leads infrastructure",
        "my teammate Lena leads design", "my teammate Bjorn leads data",
        "my manager is Nadia", "my investor contact is Amara at Northvine",
        "my accountant is Tariq", "my lawyer is Kenji",
        # settings / numbers
        "my api rate limit is 1000 requests per minute",
        "my session timeout is 30 minutes", "my backup runs at 2am daily",
        "my on-call rotation is every third week", "my team size is 9 people",
        "my office wifi network is neuronworks-5g",
    ]
    # bulk history (distinct subjects so they don't collide with the core cues)
    bulk = []
    for i in range(1, 651):
        bulk.append(f"ticket TKT{1000+i} is about subsystem-{i} and is closed")
    return core, bulk

# the conversation -- varied recalls, some deliberately broad/large
TURNS = [
    "Morning! Remind me my coffee order and my dietary restriction before standup.",
    "What's the status and deadline on project Aurora?",
    "Who are my teammates and what does each of them lead?",
    "Give me a full rundown of my dev environment: editor, languages, laptop, keyboard, theme, terminal browser.",
    "What deploy region, cloud provider, and database do I use?",
    "Tell me everything you know about project Beacon.",
    "What's my standup time and who's my manager?",
    "Update: project Beacon is now unblocked and in progress.",
    "What's the latest status on Beacon now?",
    "A new assistant is joining - summarize my entire profile and projects for them.",
    "Do you happen to know my blood type?",
]

def pct(xs, q):
    if not xs: return 0.0
    xs = sorted(xs); return xs[min(int(len(xs) * q), len(xs) - 1)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="user:garrett")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--mcp", default=None)
    args = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY") or sys.exit("set OPENAI_API_KEY")

    here = os.path.dirname(os.path.abspath(__file__))
    local = next((p for p in (os.path.join(here, "neuron-mcp.exe"), os.path.join(here, "neuron-mcp")) if os.path.exists(p)), None)
    mcp_bin = args.mcp or os.environ.get("NEURON_MCP_BIN") or local or "neuron-mcp"

    syn_log = os.path.join(tempfile.gettempdir(), f"synapse_{os.getpid()}.log")
    env = dict(os.environ)
    env["NEURON_MCP_LOG"] = "1"                       # enable server-side synapse timing
    env["NEURON_MCP_DB"] = os.path.join(here, "app_sim_memory.db")
    if os.path.exists(env["NEURON_MCP_DB"]): os.remove(env["NEURON_MCP_DB"])

    mcp = Mcp([mcp_bin], env, stderr_path=syn_log)
    mcp.handshake()
    tools = to_openai_tools(mcp.list_tools())

    # ---- seed a large store directly via the MCP remember tool (batched) ----
    core, bulk = persona_facts()
    allf = core + bulk
    t = time.perf_counter()
    stored = 0
    CHUNK = 500
    for i in range(0, len(allf), CHUNK):
        text, _, _ = mcp.call("remember", {"scope": args.scope, "facts": allf[i:i+CHUNK]})
        stored += int(text.split()[1]) if text.split()[1].isdigit() else 0
    seed_ms = (time.perf_counter() - t) * 1000.0
    sz, _, _ = mcp.call("stats", {"scope": args.scope})
    print(f"seeded {stored} facts in {seed_ms:.0f} ms ({stored/seed_ms*1000:.0f} facts/s) -> {sz}")
    print(f"model={args.model}  scope={args.scope}\n{'='*70}")

    # ---- run the conversation ----
    chat = Chat(mcp, args.scope, args.model, key, tools)
    chat.messages[0] = {"role": "system", "content": APP_SYSTEM}
    for u in TURNS:
        print(f"\033[1m> {u}\033[0m")
        reply = chat.turn(u)
        print(f"\033[32m{reply}\033[0m\n")

    mcp.close()

    # ---- correlate client round-trips with server-side synapse timings ----
    syn = []
    try:
        with open(syn_log, encoding="utf-8") as f:
            for line in f:
                if line.startswith("synapse "):
                    syn.append(json.loads(line[len("synapse "):]))
    except FileNotFoundError:
        pass
    # seed calls were the first len(chunks)+1(stats) synapse lines; align recalls by tool
    recall_syn = [s for s in syn if s["tool"] in ("recall", "recall_value")]
    recall_calls = [c for c in chat.tool_calls if c["tool"] in ("recall", "recall_value")]

    print("=" * 70)
    print("PER-TURN TIMING (recall calls)")
    print(f"{'turn':>4} {'tool':<13} {'store':>6} {'ret':>4} {'synapse_us':>10} {'rtt_ms':>8} {'llm_ms':>8}")
    si = 0
    for rec in chat.records:
        for c in rec["calls"]:
            s = recall_syn[si] if c["tool"] in ("recall", "recall_value") and si < len(recall_syn) else None
            if c["tool"] in ("recall", "recall_value"):
                su = s["us"] if s else 0; st = s["store"] if s else 0; rt = s["returned"] if s else 0
                print(f"{rec['turn']:>4} {c['tool']:<13} {st:>6} {rt:>4} {su:>10} {c['rtt_us']/1000:>8.2f} {rec['llm_ms']:>8.0f}")
                si += 1
            else:
                print(f"{rec['turn']:>4} {c['tool']:<13} {'-':>6} {'-':>4} {'-':>10} {c['rtt_us']/1000:>8.2f} {rec['llm_ms']:>8.0f}")

    pure = [s["us"] for s in recall_syn]
    rtts = [c["rtt_us"] for c in recall_calls]
    llms = [r["llm_ms"] for r in chat.records]
    print("\n" + "=" * 70)
    print("SYNAPSE PERFORMANCE SUMMARY")
    print(f"store size (neurons fired through): {syn[-1]['store'] if syn else '?'} facts")
    print(f"recall calls: {len(pure)}")
    if pure:
        print(f"  pure neuron recall (synapse): min {min(pure)} / median {int(statistics.median(pure))} / "
              f"p95 {pct(pure,0.95)} / max {max(pure)} us")
        print(f"  MCP round-trip (recall+stdio): median {statistics.median(rtts)/1000:.2f} / p95 {pct(rtts,0.95)/1000:.2f} ms")
        print(f"  stdio/IPC overhead (rtt-synapse): ~{(statistics.median(rtts)-statistics.median(pure))/1000:.2f} ms")
    if llms:
        print(f"  LLM call latency: median {statistics.median(llms):.0f} / p95 {pct(llms,0.95):.0f} ms")
        if pure:
            print(f"  => memory recall is ~{statistics.median(llms)*1000/max(1,statistics.median(pure)):.0f}x faster than the model")
    tally = {}
    for c in chat.tool_calls: tally[c["tool"]] = tally.get(c["tool"], 0) + 1
    print(f"tool-call tally: {tally}")
    try: os.remove(syn_log)
    except OSError: pass

if __name__ == "__main__":
    main()
