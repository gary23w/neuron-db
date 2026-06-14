#!/usr/bin/env python3
"""CLI chat client that gives an OpenAI model long-term memory via the neuron-db MCP
server. Stdlib only (urllib + subprocess) -- no pip install.

It launches `neuron-mcp` over stdio, performs the MCP handshake, exposes the MCP tools
to the model as OpenAI functions, and routes the model's tool calls through the real MCP
server. The memory *scope* is bound by the client (one scope per session/user), so the
model never has to manage it -- it just calls recall/remember.

Usage:
  set OPENAI_API_KEY, then:
    python chat.py                      # interactive REPL
    python chat.py --demo               # run the built-in memory test transcript
    python chat.py --script turns.txt   # one user turn per line
  options: --scope user:abc  --model gpt-4o-mini  --mcp <path-to-neuron-mcp>
"""
import argparse, json, os, subprocess, sys, time, urllib.request, urllib.error, itertools

# make unicode (emoji, em-dash, CJK) printable on a legacy Windows console
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

SYSTEM = (
    "You are a helpful assistant with a persistent long-term memory exposed as tools "
    "(recall, recall_value, remember, forget, stats).\n"
    "- When the user states durable facts about themselves (name, preferences, plan, "
    "constraints, allergies, etc.), call `remember` to store each as a short plain "
    "statement like 'my plan is pro'.\n"
    "- BEFORE answering anything about the user or something they told you earlier, call "
    "`recall` or `recall_value` to retrieve it. The memory is the source of truth -- do "
    "not rely on chat history alone.\n"
    "- Recall matches facts by overlapping WORDS, not meaning. Query with the concrete "
    "nouns used when a fact was stored ('editor', 'laptop', 'keyboard', 'deploy region'), "
    "not abstract categories ('dev environment', 'my setup'). For a multi-field request, "
    "put all the specific nouns in one recall query (or call recall several times).\n"
    "- If a recall returns nothing, retry once with more specific terms before giving up.\n"
    "- When you store an update to an existing field, reuse that field's wording ('project "
    "Beacon status is in progress', not 'Beacon is now unblocked') so later recalls find it.\n"
    "- If recall still returns no memory, say you don't have it stored rather than guessing.\n"
    "- Keep answers concise."
)

# ---------------- MCP stdio client ----------------
class Mcp:
    def __init__(self, cmd, env, stderr_path=None):
        # optionally capture the server's stderr (the synapse-timing log) to a file
        self._errf = open(stderr_path, "w", encoding="utf-8") if stderr_path else None
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=(self._errf or subprocess.DEVNULL),
                                  text=True, bufsize=1, env=env)
        self.ids = itertools.count(1)

    def _rpc(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if not notify:
            msg["id"] = next(self.ids)
        if params is not None:
            msg["params"] = params
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        if notify:
            return None
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed the connection")
        return json.loads(line)

    def handshake(self):
        self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
        self._rpc("notifications/initialized", notify=True)

    def list_tools(self):
        return self._rpc("tools/list")["result"]["tools"]

    def call(self, name, args):
        t = time.perf_counter_ns()
        r = self._rpc("tools/call", {"name": name, "arguments": args})
        rtt_us = (time.perf_counter_ns() - t) / 1000.0   # client-side round-trip
        content = r.get("result", {}).get("content", [])
        text = "".join(c.get("text", "") for c in content)
        is_err = r.get("result", {}).get("isError", False)
        return text, is_err, rtt_us

    def close(self):
        try:
            self.p.stdin.close(); self.p.terminate()
        except Exception:
            pass
        if self._errf:
            try: self._errf.flush(); self._errf.close()
            except Exception: pass

def to_openai_tools(mcp_tools):
    """MCP tool defs -> OpenAI function tools, with `scope` hidden (bound by the client)."""
    out = []
    for t in mcp_tools:
        sch = json.loads(json.dumps(t["inputSchema"]))
        sch.get("properties", {}).pop("scope", None)
        sch["required"] = [r for r in sch.get("required", []) if r != "scope"]
        out.append({"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": sch}})
    return out

# ---------------- OpenAI ----------------
def openai_chat(messages, tools, model, key):
    body = json.dumps({"model": model, "messages": messages, "tools": tools,
                       "tool_choice": "auto", "temperature": 0}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        sys.exit(f"OpenAI error {e.code}: {e.read().decode()[:500]}")

# ---------------- chat loop ----------------
class Chat:
    def __init__(self, mcp, scope, model, key, tools):
        self.mcp, self.scope, self.model, self.key, self.tools = mcp, scope, model, key, tools
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.tool_calls = []   # [{turn, tool, args, result, rtt_us}]
        self.records = []      # per-turn: {turn, user, reply, llm_ms, calls:[...]}
        self.turn_idx = 0

    def turn(self, user_text):
        self.turn_idx += 1
        self.messages.append({"role": "user", "content": user_text})
        llm_ms = 0.0
        calls = []
        while True:
            t = time.perf_counter()
            msg = openai_chat(self.messages, self.tools, self.model, self.key)
            llm_ms += (time.perf_counter() - t) * 1000.0
            tcs = msg.get("tool_calls")
            am = {"role": "assistant", "content": msg.get("content")}
            if tcs:
                am["tool_calls"] = tcs
            self.messages.append(am)
            if not tcs:
                reply = msg.get("content") or ""
                self.records.append({"turn": self.turn_idx, "user": user_text,
                                     "reply": reply, "llm_ms": llm_ms, "calls": calls})
                return reply
            for tc in tcs:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                args["scope"] = self.scope   # client binds the scope
                result, is_err, rtt_us = self.mcp.call(name, args)
                shown = {k: v for k, v in args.items() if k != "scope"}
                rec = {"turn": self.turn_idx, "tool": name, "args": shown,
                       "result": result, "rtt_us": rtt_us}
                self.tool_calls.append(rec); calls.append(rec)
                print(f"   \033[36m[tool] {name}({json.dumps(shown)}) -> {result[:80]!r}  ({rtt_us/1000:.2f} ms rtt)\033[0m")
                self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="user:demo")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--mcp", default=None, help="path to neuron-mcp (default: $NEURON_MCP_BIN, or a binary beside this script, or 'neuron-mcp' on PATH)")
    ap.add_argument("--demo", action="store_true", help="run the built-in memory test transcript")
    ap.add_argument("--script", help="file with one user turn per line")
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("set OPENAI_API_KEY")

    # resolve the server binary: --mcp, else env, else beside this script, else PATH
    here = os.path.dirname(os.path.abspath(__file__))
    local = next((p for p in (os.path.join(here, "neuron-mcp.exe"), os.path.join(here, "neuron-mcp")) if os.path.exists(p)), None)
    mcp_bin = args.mcp or os.environ.get("NEURON_MCP_BIN") or local or "neuron-mcp"

    env = dict(os.environ)
    env.setdefault("NEURON_MCP_DB", os.path.join(os.getcwd(), "mcp_chat_memory.db"))
    mcp = Mcp([mcp_bin], env)
    mcp.handshake()
    mcp_tools = mcp.list_tools()
    tools = to_openai_tools(mcp_tools)
    print(f"mounted neuron-db MCP: {len(mcp_tools)} tools {[t['name'] for t in mcp_tools]}")
    print(f"model={args.model}  scope={args.scope}  db={env['NEURON_MCP_DB']}\n")

    chat = Chat(mcp, args.scope, args.model, key, tools)

    DEMO = [
        "Hey! My name is Garrett. I'm on the enterprise plan, I use neovim, and I'm allergic to peanuts.",
        "What text editor do I use?",
        "Quick — am I allergic to anything I should worry about at lunch?",
        "What plan am I on?",
        "Actually I just downgraded to the pro plan.",
        "Remind me which plan I'm on now?",
        "What's my favorite movie?",
    ]

    def drive(turns):
        for t in turns:
            print(f"\033[1m> {t}\033[0m")
            reply = chat.turn(t)
            print(f"\033[32m{reply}\033[0m\n")

    if args.demo:
        drive(DEMO)
    elif args.script:
        with open(args.script, encoding="utf-8-sig") as f:  # tolerate a BOM
            drive([ln.strip() for ln in f if ln.strip()])
    else:
        print("interactive chat (Ctrl-C to quit)")
        try:
            while True:
                t = input("\033[1m> \033[0m").strip()
                if t:
                    print(f"\033[32m{chat.turn(t)}\033[0m\n")
        except (EOFError, KeyboardInterrupt):
            print()

    # telemetry summary
    tally = {}
    for rec in chat.tool_calls:
        tally[rec["tool"]] = tally.get(rec["tool"], 0) + 1
    print("---- session telemetry ----")
    print(f"total tool calls: {len(chat.tool_calls)}  by tool: {tally}")
    mcp.close()

if __name__ == "__main__":
    main()
