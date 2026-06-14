# mcp_chat — give an LLM long-term memory via neuron-db (MCP)

A tiny CLI chat client (Python stdlib only) that connects an OpenAI model to the
**neuron-db MCP server** (`neuron-mcp`). It launches the server over stdio, performs the
MCP handshake, exposes the MCP tools to the model as functions, and routes the model's
tool calls through the real server — so it exercises the genuine mount path end to end.

The memory **scope is bound by the client** (one scope per session/user); the model never
manages it — it just calls `recall` / `remember`.

## Run

```sh
# 1) build the server
cargo build --release --features mcp --bin neuron-mcp     # from rust/neuron-core
# place the binary on PATH, beside chat.py, or pass --mcp <path>

# 2) set your key and chat
export OPENAI_API_KEY=sk-...
python chat.py                          # interactive REPL
python chat.py --demo                   # built-in memory test transcript
python chat.py --script turns.txt       # one user turn per line
```

Options: `--scope user:abc` · `--model gpt-4o-mini` · `--mcp <path-to-neuron-mcp>`.
Env: `OPENAI_API_KEY` (required), `OPENAI_MODEL`, `NEURON_MCP_BIN`, `NEURON_MCP_DB`.

## What the model is told

The system prompt instructs it to `remember` durable facts the user states and to
`recall` before answering questions about the user — and to abstain (not guess) when
recall returns nothing.

## Measured (gpt-4o-mini, `--demo`)

A 7-turn session (state facts → recall them → update a fact → recall again → ask an
unknown): **10 tool calls, 100% correct** — the model stored each fact, recalled by direct
and paraphrased queries, the updated plan won via recency, and it abstained on the unknown
instead of fabricating. This session also surfaced and fixed a real MCP framing bug
(multi-line `tools/list` response); see `docs/MEMORY_HARNESS.md`.

## Notes

- Windows + Microsoft Store Python virtualizes `%LOCALAPPDATA%`/`%TEMP%`, so it can't launch
  a binary living there. Keep `neuron-mcp` on a normal path (e.g. beside this script).
- `neuron-mcp.exe` and `*.db` here are gitignored.
