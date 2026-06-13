# neuron-db API

How the store is shaped, and how you insert, query, update, and delete — from Rust, the
CLI, or HTTP. There are no tables, no schema, and no SQL. You write facts in plain language
and ask for them back by meaning.

## The data model (there are no tables)

Two ideas, that's the whole model:

- A **fact** is a short sentence of plain text: `"the api key is zeta-9931"`. When you store
  it, neuron-db picks the surprising word as the retrievable **value** (`zeta-9931`) and
  indexes the rest as cues.
- A **scope** (also called a *neuron*) is a named bag of facts, addressed by an id string
  like `user:42` or `session:abc`. Scopes are isolated from each other. A `NeuronDB` is a
  file full of scopes.

If you come from SQL, the rough mental map is: a scope is *like* a row keyed by its id, and
its facts are the columns — except you never declare them. You just state things, and ask
questions. Retrieval is associative (cue overlap), not exact-key, so `"what is the api
key?"` finds the fact above without you naming a column.

```
NeuronDB (one .db file)
├── scope "user:42"   ── facts: "the plan is pro", "the region is us-west-2", ...
├── scope "user:43"   ── facts: "the plan is free", ...
└── scope "team:ops"  ── facts: "the on-call is Dana", ...
```

## Operations at a glance

| intent | SQL-ish | neuron-db |
|---|---|---|
| insert | `INSERT` | `observe(scope, "the plan is pro")` |
| read (one value) | `SELECT … LIMIT 1` | `get(scope, "what plan?") -> "pro"` |
| read (with metadata) | `SELECT *` | `recall(scope, "what plan?") -> {value, fact, coverage}` |
| "update" | `UPDATE` | just `observe` again — newest wins; or `reinforce` (plastic tier) |
| delete | `DELETE` | `forget(scope, "plan")` (by substring) or `forget(scope, None)` (all) |
| converse | — | `turn(scope, "my plan is pro")` → stores or answers, one call |
| encrypted put/get | — | `SecureNeuronDB.put/get(scope, secret, …)` |

There is no `UPDATE` because facts aren't rows you mutate. To change an answer, state the new
fact — recall prefers the most recent matching fact. To make a fact "stick" so it wins over
newer ones, use the plastic tier's `reinforce`.

## 1) Rust library (embedded, no server)

Add the crate (path or git) and enable the `sqlite` feature for the durable `NeuronDB`:

```toml
[dependencies]
neuron-core = { path = "rust/neuron-core", features = ["sqlite"] }
```

```rust
use neuron_core::db::NeuronDB;

let db = NeuronDB::open("app.db", 500);        // file path, max facts per scope

// INSERT
db.observe("user:42", "the plan is pro");
db.observe("user:42", "the region is us-west-2");

// READ — one value, or the full hit
let plan = db.get("user:42", "what plan?");                 // Some("pro")
let hit  = db.recall("user:42", "where is the region?");    // Some(Recall{ value, fact, coverage, .. })

// "UPDATE" — newest matching fact wins
db.observe("user:42", "the plan is enterprise");
assert_eq!(db.get("user:42", "what plan?").as_deref(), Some("enterprise"));

// DELETE — by substring, or everything in the scope
let (forgot, remaining) = db.forget("user:42", Some("region"));   // drop region fact
db.forget("user:42", None);                                       // wipe the scope

// CONVERSE — store or answer in one call
let t = db.turn("user:42", "my color is teal");   // t.kind == "ack"
let a = db.turn("user:42", "what is my color?");  // a.kind == "recall", a.reply == "teal."

// INTROSPECT
let s   = db.stats("user:42");     // { facts, max_facts, created, updated, turns }
let ids = db.neurons();            // every scope id, most-recently-updated first
```

### In-memory only (no file): `Neuron`

For a cache or a unit test, skip SQLite entirely (default features, no deps):

```rust
use neuron_core::Neuron;
let mut n = Neuron::new(500);
n.observe("the wifi password is hunter2");
n.recall("what is the wifi password?").unwrap().value;   // "hunter2"
```

### Adaptive memory: `PlasticNeuron`

Facts gain strength on recall, fade when ignored, and wire together when recalled together.

```rust
use neuron_core::plastic::PlasticNeuron;
let mut n = PlasticNeuron::new(10_000_000, Some(200.0), 3);  // max_facts, half_life, link_window
n.observe("the meeting is on monday");
n.observe("the meeting is on friday");
n.recall("when is the meeting?");              // "friday" (recency)
let id = n.base.episodes.iter().find(|e| e.t.contains("monday")).unwrap().id;
for _ in 0..4 { n.reinforce(id, 1.0); }
n.recall("when is the meeting?");              // "monday" (adapted to use)
n.recall_spreading("when is the meeting?", 2, 10, 0.6, 6);  // neurotransmitter-style, multi-hop
```

### Scaling many facts: `NeuronRouter`

One scope recalls best when it's small. To hold a lot, shard:

```rust
use neuron_core::router::NeuronRouter;
let mut r = NeuronRouter::new(128);            // facts per shard
for f in facts { r.observe(&f); }              // auto-spills into new shards
r.get("what is the north gate code?");         // fans out, returns the single best value
```

### Encrypted: `SecureNeuronDB` (feature `secure`)

Values are AES-256-GCM ciphertext; the index is a keyed hash; the per-scope **secret is
supplied per call and never stored**. A stolen `.db` file is opaque.

```rust
use neuron_core::secure::SecureNeuronDB;
let v = SecureNeuronDB::open("vault.db");
v.put("alice", "alice-secret", "wifi password", "hunter2").unwrap();
v.get("alice", "alice-secret", "what is the wifi password?");  // Some("hunter2")
v.get("alice", "WRONG",        "what is the wifi password?");  // None
```

## 2) CLI — query the file from your shell

Build with `./build.sh` (or `cargo build --release --features "sqlite secure server"`), then:

```sh
neuron --db app.db observe user:42 "the plan is pro"      # INSERT
neuron --db app.db get     user:42 "what plan am i on?"   # -> pro
neuron --db app.db recall  user:42 "what plan?"           # value + fact + coverage
neuron --db app.db turn    user:42 "my color is teal"     # store or answer
neuron --db app.db forget  user:42 "plan"                 # DELETE facts containing "plan"
neuron --db app.db stats   user:42
neuron --db app.db list                                   # all scope ids
neuron --db app.db --json get user:42 "what plan?"        # {"value":"pro"}

# encrypted tier (the secret is your "login" for that scope)
neuron --db vault.db --secret s3cr3t secure-put alice "wifi password" hunter2
neuron --db vault.db --secret s3cr3t secure-get alice "what is the wifi password?"
```

The db path comes from `--db` or `$NEURON_DB` (default `neurons.db`). The secret comes from
`--secret` or `$NEURON_SECRET`.

## 3) HTTP — query it over the network

Start the server (`neuron serve 8088`, the `serve` binary, or Docker). Set `NEURON_DB_KEY`
to require `Authorization: Bearer <key>` on every request.

```
GET  /                       -> { service, endpoint }
GET  /v1/{scope}             -> stats                        (auth)
POST /v1/{scope}   {message} -> { reply, kind, wrote, facts } (turn: store or answer)
POST /v1/{scope}/get    {query}  -> { value }
POST /v1/{scope}/forget {match}  -> { forgot, remaining }
```

```sh
curl -d '{"message":"the api key is zeta-9931"}'      localhost:8088/v1/user:42
curl -d '{"query":"what is the api key?"}'             localhost:8088/v1/user:42/get
curl -H "authorization: Bearer $NEURON_DB_KEY" \
     -d '{"match":"api key"}'                          localhost:8088/v1/user:42/forget
```

## Access control (there is no DB login)

SQLite is **embedded** — a file, not a server, so there is nothing to log into. Control
access at the layer that fits:

- **Local file** — OS filesystem permissions on the `.db` file.
- **Over the network** — run the HTTP server and set `NEURON_DB_KEY`; clients send a bearer
  token. Without the env var, auth is off (fine behind your own gateway).
- **Per-secret encryption** — `SecureNeuronDB`: each scope's data is encrypted under a secret
  the caller supplies every time and that is never written to disk. Lose the secret, lose the
  data; that's the point.

## Durability & concurrency

`NeuronDB` is one SQLite file in WAL mode behind a process-wide lock, so concurrent threads
are safe (writes serialize). Each write persists the scope immediately. See `docs/DEPLOY.md`
for running it as a service.
