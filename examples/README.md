# Examples

Runnable code and integration guides for wiring neuron-db into apps, websites, APIs, and
chatbots.

## Runnable Rust examples (`cargo run --example <name>`)

Located in `rust/neuron-core/examples/`:

| example | what it shows | run |
|---|---|---|
| `quickstart` | insert / read / update / delete on a durable `NeuronDB` | `cargo run --example quickstart --features sqlite` |
| `chatbot_memory` | the recall → inject → write chatbot loop (embedded) | `cargo run --example chatbot_memory --features sqlite` |
| `user_profiles` | per-user memory for an app backend; isolated scopes | `cargo run --example user_profiles --features sqlite` |
| `plastic_adaptive` | strength / decay / neurotransmitter spreading recall | `cargo run --example plastic_adaptive` |
| `sharded_scale` | `NeuronRouter` holding 5k facts with fan-out recall | `cargo run --example sharded_scale` |
| `encrypted_secrets` | `SecureNeuronDB` — encrypted values, per-scope secret | `cargo run --example encrypted_secrets --features secure` |

## Live demo (`examples/browser-demo/`)

A two-pane proof of concept: a memory chatbot on the left, and a live feed of **everything
neuron-db is collecting** from the browser on the right (mouse/clicks/scroll/keys/session).
Open `examples/browser-demo/index.html`, or run `examples/browser-demo/run.sh` / `run.ps1` /
`start-demo.bat` to back it with the real local server. See [browser-demo/README.md](browser-demo/README.md).

## HTTP clients (`examples/http/`)

For talking to a running `serve` instance from any language:

- `curl.sh` — shell
- `web_app.html` — a browser page using `fetch()` (a per-user memo app)
- `node_client.js` — Node 18+ (built-in fetch)
- `python_client.py` — Python `requests` (the DB is Rust; your app just speaks HTTP)

Start a server first: `serve neurons.db 8088` (or `docker compose up -d`).

## Guides (`examples/guides/`)

- **[CHATBOT.md](guides/CHATBOT.md)** — give a chatbot durable per-user memory: recall facts
  on the fly and inject them into the model's context. Includes the write/recall/inject loop
  and a Node + LLM example.
- **[EXISTING_API.md](guides/EXISTING_API.md)** — add neuron-db to an existing API, either by
  embedding the Rust crate or running it as a sidecar HTTP service.

## See also

- [docs/API.md](../docs/guide/API.md) — the full data model and operation reference.
- [docs/DEPLOY.md](../docs/guide/DEPLOY.md) — build, install, Docker, env, backups.
