# Deploying neuron-db

neuron-db is a Rust crate plus three binaries (`neuron` CLI, `serve` HTTP server, `neuron-mcp`
MCP server). The store is embedded SQLite, so "deploying" is just running a binary next to a
data volume. None of the tiers below require an LLM — the CLI and HTTP server are a complete
standalone memory database on their own.

## Build from source

```sh
./build.sh                      # checks cargo, builds with sqlite+secure+server, prints next steps
# or pick features yourself:
cd rust/neuron-core
cargo build --release --features "sqlite secure server"
```

Binaries land in `rust/neuron-core/target/release/` (`neuron`, `serve`). Install onto PATH:

```sh
cargo install --path rust/neuron-core --features "sqlite secure server"
```

### Feature flags

| feature | adds | pulls in |
|---|---|---|
| *(default)* | `Neuron`, `PlasticNeuron`, `NeuronRouter`, `turn`, wasm core | nothing (std-only) |
| `sqlite` | `NeuronDB`, the `neuron` CLI | rusqlite (bundled sqlite) |
| `secure` | `SecureNeuronDB`, `secure-*` CLI | aes-gcm, sha2, hmac, base64, getrandom |
| `server` | HTTP server, `serve` binary | — (std TcpListener) |
| `semantic` | semantic-ranked recall (`recall_blended`), int8 space | — (std-only) |
| `mcp` | `neuron-mcp` MCP server (pulls in `semantic`) | rusqlite |

The default build has zero dependencies and compiles to `wasm32-unknown-unknown` for the
edge/worker target — the native tiers above are opt-in so they never touch that build.

## Run the server

```sh
serve /data/neurons.db 8088           # path, port
# or drive it from env:
NEURON_DB=/data/neurons.db NEURON_HOST=0.0.0.0 NEURON_PORT=8088 serve
```

Environment:

| var | meaning | default |
|---|---|---|
| `NEURON_DB` | database file path | `neurons.db` |
| `NEURON_HOST` | bind address (`0.0.0.0` to expose) | `127.0.0.1` |
| `NEURON_PORT` | port | `8088` |
| `NEURON_DB_KEY` | if set, require `Authorization: Bearer <key>` | unset (auth off) |
| `NEURON_LOG` | per-request access logs to stderr (`off`/`0` to disable) | on |

## Docker

```sh
docker build -t neuron-db .
docker run -d -p 8088:8088 -v neuron-data:/data \
  -e NEURON_DB_KEY=$(openssl rand -hex 16) neuron-db
```

Or with Compose (persists to a named volume, restarts on failure):

```sh
NEURON_DB_KEY=$(openssl rand -hex 16) docker compose up -d
```

The image binds `0.0.0.0` inside the container and stores the db on the `/data` volume. Put
it behind your own TLS-terminating proxy for public exposure.

## CLI against a running file

The `neuron` CLI opens the SQLite file directly — no server required. It's the quickest way
to inspect or seed a database:

```sh
neuron --db /data/neurons.db list
neuron --db /data/neurons.db stats user:42
neuron --db /data/neurons.db get   user:42 "what plan?"
```

See `docs/guide/API.md` for the full command and endpoint reference.

## Deploy as LLM memory (MCP)

`neuron-mcp` gives any MCP client (Claude Desktop, Claude Code, Cursor, …) a persistent
`recall → inject → remember` memory loop over stdio. It ships with semantic-ranked recall by
default, so recall is topically coherent out of the box — no harness or extra wiring required.

Build it, then let it write its own client config:

```sh
cargo install --path rust/neuron-core --features mcp   # installs `neuron-mcp` onto PATH
neuron-mcp --config                                    # prints paste-ready config for THIS machine
```

`--config` fills in the binary's absolute path and the db location for you. Paste the JSON into
your client's config file (Claude Desktop: `claude_desktop_config.json`; Cursor: `~/.cursor/mcp.json`),
or for Claude Code run the one-liner it prints:

```sh
claude mcp add neuron --env NEURON_MCP_DB=neuron.db -- /path/to/neuron-mcp
```

Restart the client and the `recall` / `remember` / `forget` / `stats` tools appear. Environment:

| var | meaning | default |
|---|---|---|
| `NEURON_MCP_DB` | database file path | `neuron.db` |
| `NEURON_MAX_FACTS` | per-scope fact cap | unbounded |
| `NEURON_MCP_LOG` | `1` logs per-call synapse timing to stderr | off |

The DB is the same SQLite file the CLI and server use, so you can inspect a model's memory with
`neuron --db neuron.db list` while it runs. For the passive auto-capture / document-register
harness patterns layered on top of this server, see `docs/guide/MEMORY_HARNESS.md`.

## Backups

The database is a single SQLite file (plus `-wal`/`-shm` while running). Back it up with
`sqlite3 neurons.db ".backup backup.db"` or just copy the file when the server is stopped.
