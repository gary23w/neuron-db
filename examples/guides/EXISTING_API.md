# Adding neuron-db to an existing API or app

Two ways to bolt neuron-db onto something you already run. Pick by language and topology.

## Option A — embed the crate (Rust services)

If your backend is Rust, link the crate and call it in-process. No network hop, no extra
service.

```toml
[dependencies]
neuron-core = { git = "https://github.com/gary23w/neuron-db", features = ["sqlite"] }
```

```rust
use neuron_core::db::NeuronDB;
use std::sync::Arc;

// open once at startup, share across handlers (it's internally thread-safe)
let mem = Arc::new(NeuronDB::open("data/memory.db", 500));

// in a request handler, scope memory by the authenticated user:
fn handle(mem: &NeuronDB, user_id: &str, body: &Inbound) {
    if let Some(fact) = &body.remember { mem.observe(user_id, fact); }
    if let Some(q)    = &body.ask      { let _ = mem.get(user_id, q); }
}
```

Where to put the calls: write (`observe`/`turn`) when the user tells you something durable
(preferences, settings, profile updates); read (`get`/`recall`) when you need that context
to personalize a response. Scope every call by the authenticated user id so memories stay
isolated.

## Option B — run it as a sidecar service (any language)

Run the `serve` binary (or the Docker image) alongside your app and call it over HTTP. This
works from Node, Python, Go, Ruby, PHP — anything that can make an HTTP request. Clients for
several languages are in `examples/http/`.

```
your API  ──HTTP──▶  neuron-db `serve` (sidecar)  ──▶  memory.db (SQLite volume)
```

```js
// node: a thin memory middleware
async function withMemory(req, res, next) {
  const user = req.user.id;                 // from your auth
  req.recall = (q) => fetch(`http://memory:8088/v1/${user}/get`,
      { method:"POST", headers:{ "content-type":"application/json", authorization:`Bearer ${process.env.NEURON_DB_KEY}` },
        body: JSON.stringify({ query: q }) }).then(r => r.json()).then(r => r.value);
  next();
}
```

Set `NEURON_DB_KEY` on both the server and your clients to require a bearer token. Put the
sidecar on a private network; terminate TLS at your gateway.

## Endpoints (sidecar mode)

```
POST /v1/{scope}          {message}  -> {reply, kind, wrote, facts}   # store or answer
POST /v1/{scope}/get      {query}    -> {value}                       # value only
POST /v1/{scope}/forget   {match}    -> {forgot, remaining}           # delete by substring
GET  /v1/{scope}                     -> {facts, turns, ...}           # stats
```

`{scope}` is your partition key — usually the user id. URL-encode it.

## Patterns that come up a lot

- **Personalization** — recall the user's preferences before rendering a response or building
  a prompt; store updates when they change a setting.
- **Form pre-fill** — `observe` answers as the user types; `get` them back on the next visit.
- **Support/CRM notes** — one scope per customer; agents `observe` notes, `recall` history.
- **Feature flags / entitlements per user** — `observe("user:1","the plan is pro")`, then
  `get` it at request time.
- **Right to be forgotten** — wire your delete endpoint to `POST /v1/{user}/forget` with no
  match (wipes the scope).

## Operational notes

- The store is a single SQLite file; back it up by copying it (or `sqlite3 .backup`).
- Writes serialize behind one connection — fine for typical API load; for very high write
  rates, batch or shard by user across multiple files.
- It is exact/associative recall, not semantic search. Pair with a vector store only for the
  queries that genuinely need meaning-matching.
