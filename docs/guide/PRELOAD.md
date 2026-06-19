# Preloading facts — installing a knowledge base into neuron-db memory

A **pack** is a list of pre-built declarative facts you load into a neuron-db instance's memory in
one shot — a curated knowledge base, or a converted grounding dataset like
[google/FACTS-grounding-public](https://huggingface.co/datasets/google/FACTS-grounding-public). The
same pack loads identically on every surface: the **CLI** (`neuron import`), an **MCP server** (a
boot env var), and the **in-browser WASM** (one op call from your script).

## The pack format

neuron-db mints the neuron and its coordinate on ingest, so a pack carries only the **fact text**
and an optional **scope** — no IDs, embeddings, or metadata. Two text forms, one reader:

**`.jsonl`** — one JSON object per line (converter output):
```jsonl
{"scope":"facts-0","fact":"Acme reported revenue of $4.2B in fiscal 2023."}
{"scope":"facts-0","fact":"The filing was signed by the CFO on March 1."}
{"scope":"facts-1","fact":"Aspirin inhibits cyclooxygenase enzymes."}
```

**`.facts`** — one bare fact per line, with `# scope:` directives (awk/sed-friendly):
```
# scope: chemistry
Water is two hydrogen atoms and one oxygen atom.
Table salt is sodium chloride.
```

`fact` is required; `scope` is optional. A blank line or a `#` comment is skipped. **Scope
precedence:** per-fact `scope` → active `# scope:` directive → the caller's default → error (a fact
with no scope anywhere is rejected). One scope per source document is the recommended layout — it
isolates each document's context-relative facts and keeps recall sharp.

## Loading a pack

### CLI — `neuron import` (the durable path; build a db once, serve it anywhere)
`--db` and `--max` are global flags (they come before the subcommand):
```sh
neuron --db app.db import pack.jsonl                  # facts with per-line scopes
neuron --db app.db import kb.facts --scope notes      # default scope for unscoped lines
neuron --db app.db --max 200000 import big.jsonl      # raise the per-scope cap for a large pack
neuron --db app.db import pack.jsonl --replace        # clear each touched scope first (idempotent re-import)
neuron --db app.db import pack.jsonl --dedup          # drop duplicate (scope,fact) lines within the pack
neuron --db app.db export notes -o notes.facts        # round-trip a scope back out to a pack
neuron --db app.db export --all -o everything.facts
```
Multi-scope packs are O(N); a single huge scope is fastest with `--flush 0` (one write).

### MCP — preload at boot (zero code change for the client)
Point any MCP client's `neuron` server at a pack with an env var:
```jsonc
"neuron": {
  "command": "neuron-mcp",
  "env": {
    "NEURON_MCP_DB": "facts.db",
    "NEURON_MCP_PRELOAD": "pack.jsonl",
    "NEURON_MAX_FACTS": "200000"
  }
}
```
The pack loads **once, before the first request**, writing each scope all-or-nothing — so a boot
killed mid-load leaves each scope either fully loaded or empty, and a restart with the env still set
is a near no-op (fully-loaded scopes are skipped). Set `NEURON_MCP_PRELOAD_FORCE=1` to re-seed (this clears each
touched scope first — including any variables, instructions, or stances stored under it, so
FORCE-preload into **dedicated** scopes, not ones a live user also writes to), or
`NEURON_MCP_PRELOAD_SCOPE` to set the default scope. For very large datasets, prefer building the db
offline with `neuron import` and pointing `NEURON_MCP_DB` at it — the preload then costs nothing at boot.

### WASM — pass the datalist from your script
The `loadmany` op takes any number of scopes in **one** boundary crossing:
`loadmany\t<scope1>\t<facts1>\t<scope2>\t<facts2>…` (each scope's facts newline-joined). A chunked
helper keeps peak memory bounded for large lists:
```js
// mem(): the standard alloc → mem → dealloc wrapper for the neuron-db wasm
function mem(EX, ...fields){
  const b = new TextEncoder().encode(fields.join("\t"));
  const p = EX.alloc(b.length); new Uint8Array(EX.memory.buffer, p, b.length).set(b);
  const n = EX.mem(p, b.length);
  const out = new TextDecoder().decode(new Uint8Array(EX.memory.buffer, EX.answer_ptr(), n));
  EX.dealloc(p, b.length); return out;
}

// preload(facts): facts = [{scope, fact}, …]. Groups by scope, strips wire separators, chunks.
function preload(EX, facts, { maxPerCall = 2000 } = {}){
  const clean = f => (f ?? "").replace(/[\t\n]/g, " ").trim();
  let stored = 0;
  for (let i = 0; i < facts.length; i += maxPerCall){
    const byScope = new Map();
    for (const { scope, fact } of facts.slice(i, i + maxPerCall)){
      const f = clean(fact); if (!f) continue;
      const s = clean(scope);   // the SCOPE must be tab/newline-free too — both are loadmany wire separators
      (byScope.get(s) || byScope.set(s, []).get(s)).push(f);
    }
    const args = [];
    for (const [s, fs] of byScope){ args.push(s, fs.join("\n")); }
    if (args.length) stored += parseInt(mem(EX, "loadmany", ...args).split("\t")[0], 10) || 0;
  }
  return stored;   // episodes stored (≥ lines, since one sentence-rich line can fan out)
}
```
Fetching a remote dataset is CORS-gated — pre-pack it and serve the `.jsonl` from your own origin.

## Converting a grounding dataset

`examples/preloads/facts_to_pack.py` turns a grounding CSV into a pack: each row → one scope, the
prose column sentence-split (abbreviation-guarded) into one fact per sentence.
```sh
python examples/preloads/facts_to_pack.py examples.csv > pack.jsonl
neuron --db facts.db --max 200000 import pack.jsonl
```

## The contract (read before loading untrusted or large data)

- **Facts are tab- and newline-free.** Those are wire separators; a fact carrying either is
  **rejected** (counted, never silently mangled) so a pack is portable across all three surfaces.
- **Count is episodes-stored, not lines-submitted.** A line with multiple sentences fans out into
  several facts; a line containing `?` is dropped as a question. Every surface reports both counts.
- **Bulk load is un-deduped** (so it stays O(N)). Re-running a pack double-writes — use `--replace`
  (CLI) / `NEURON_MCP_PRELOAD_FORCE` (MCP) / `forget` then reload (WASM) for idempotency, or
  `--dedup` to drop duplicates within one pack.
- **Size the cap to the largest scope.** A scope past `max_facts` front-drains its oldest facts
  (the loader warns when this happens). Set `--max` (import) **and** `NEURON_MAX_FACTS` (serve) to at
  least the largest single-scope fact count.
