# Two-pane live demo

Left pane: a chatbot with durable memory. Right pane: a live feed of **everything neuron-db is
collecting** from the browser (mouse, clicks, scroll, keys, focus, tab switches, session
profile, page timing) — streamed as it happens and rolled up into recallable facts. You get a
random session id on join.

## Run it

**Quickest (no setup):** open `index.html` in a browser. It runs against an in-page mirror of
neuron-db, so the whole PoC works immediately. The moment a real server is reachable it
switches over automatically.

**With the real Rust database (recommended):**

- macOS / Linux:  `./run.sh`
- Windows:        double-click `start-demo.bat` (or right-click `run.ps1` → Run with PowerShell)
- Docker (any OS): from the repo root, `docker compose up -d`, then open `index.html`

Each starts the `serve` binary on `http://localhost:8088`, creates `demo.db`, and opens the
page. The page auto-detects the server (header shows "neuron-db server"), writes every
collected fact to it, and answers the chat by recalling from it. Building the server from
source needs a C compiler for bundled SQLite (Rust + MSVC build tools on Windows); the Docker
route needs none.

## What it demonstrates

- **Client-side collection** — the analytics an SDK would gather, captured live and visible.
- **Memory** — collected facts (`the click count is 14`, `the timezone is …`) become
  recallable; ask the chat "what is the click count?" and it answers from neuron-db.
- **Per-user scoping** — the random `user:<id>` keeps every session isolated.

Nothing leaves your machine: the page talks only to your local server (or the in-page mirror).
