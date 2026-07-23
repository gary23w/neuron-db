# Proposal: document-preserving recall (sub-scopes, stitching, and a full-document path)

Status: **PROPOSAL — not implemented.** Written after the 2026-07-22 "hello"-session
investigation (book absorbed → summary built from fragments). The pipeline bugs found in that
investigation are fixed separately (seed df-gating in `recall_spreading`; nl-veil `--max`
plumbing + eviction-honest absorb). What remains is structural, and per the design gate it
needs sign-off before implementation.

## The gap this closes

Ingestion sentence-atomizes a document into independent episodes. That atomization is core
design (fine-grained weave), and the storage layer *does* preserve document order — episodes
sit in insertion order in the scope, and `scope_facts()` returns them that way. But no recall
surface uses that structure:

- Episodes carry no document identity beyond an in-band `[label] ` text prefix (which recall
  treats as ordinary words — it was the hub-seeding poison), and continuation fragments
  produced by `observe()`'s re-split lose even that.
- Top-k recall (`recall_many`, `recall_spreading`) returns isolated sentences; there is no
  step that expands a hit into its surrounding passage.
- A "summarize the document" request has no path that reads the document; the model is told
  "recall it any time — no re-read needed," which is false for whole-document questions whose
  content words don't discriminate ("plot", "summary", "ending" appear nowhere in the prose).

## Design

Three pieces, ordered by leverage. None changes the SQLite schema; #2 adds one field to two
result structs.

### 1. Per-document sub-scopes (container = document ID)

`absorb`/`veil rag ingest` store a document into `knowledge__doc-<label>` instead of flat
`knowledge`. The `base__child` convention already exists — `recall_many_across(base, …)`
merges `base` and every `base__*` child and deliberately excludes `base::typed` sub-scopes
(db.rs). The scope becomes the durable document ID; insertion order within it *is* document
order.

- Lexical block recall via `recall_many_across("knowledge", …)` works today, unchanged.
- Spreading recall needs an across variant: `recall_assoc_across(base, query, k, hops)` —
  loop the child scopes, assoc each with the same k, merge by activation, truncate. Core-side
  (db.rs + op vocabulary) so CLI/FFI/MCP/wasm all inherit it; nl-veil's `recall_hive` then
  calls the across op instead of flat-scope assoc.
- The `[label] ` prefix can stay for model-facing provenance, but recall no longer depends on
  it — and one absorbed document can never evict the shared hive or its own head (isolation
  is a stronger guarantee than a bigger cap).

### 2. Neighborhood stitching at recall (the missing reassembly step)

Add the episode's index to recall results (`Recall.idx`, `Spread.idx` — additive, no caller
breaks) and one read primitive:

```
neighbors(scope, idx, before, after) -> Vec<fact>   // insertion-order slice around a hit
```

Surfaced as `neuron context <scope> <query> [--before N --after M]`: run recall, then return
the hit *stitched into its surrounding passage* in document order. With per-doc sub-scopes
(#1), the slice is a contiguous passage of the original document — fragment recall starts
returning coherent context instead of disconnected sentences. Cost: O(before+after) Vec
slicing per hit; no index changes; latency unchanged.

Token tradeoff: a stitched hit is ~(before+after+1)× larger, so callers should drop k
accordingly (k=4 stitched beats k=12 isolated for grounding quality at similar token cost).

### 3. A full-document path for summarization (bypass fragment recall entirely)

Whole-document requests must not go through top-k anything. The read already exists
(`scope_facts(scope)` — insertion order); what's missing is paging and the tool surface:

- Core/CLI: `neuron read <scope> --from <i> --limit <n>` (a paged window over scope_facts).
- App: a `read_doc` tool (or absorb ack guidance) that map-reduces over the document scope in
  pages — from *memory*, not the source file. In the repro the model correctly fell back to
  re-reading the raw file, but the file lived in the conversation workdir and the desk client
  disconnected mid-read; the store copy would have survived.
- Optional distiller tweak: keep chapter-heading lines as marker facts (they're currently
  dropped as structural), so paging can be chapter-aligned.

Honest cost: a 3k-fact book is ~360KB of text through the model in pages. That is what a real
summary costs; the old path was cheap only because it wasn't summarizing the document.

## Migration

- Existing flat scopes stay valid: across-merge includes the base scope, so old facts remain
  recallable with zero migration.
- Optional one-shot: group a flat scope's facts by their `[label] ` prefix and re-import into
  sub-scopes (`export` + `import --scope`, both exist). Unprefixed continuation fragments
  stay in the base scope — they're low-value without their neighbors anyway.
- The damaged live scope (the book's surviving 500-fact tail in the client-mem `knowledge`
  scope) is incomplete regardless — re-absorb the book after deploying the fixes.

## Costs and risks

- **Across-recall latency**: N sub-scope probes instead of 1 (linear in document count, each
  sub-linear in facts). Negligible for tens of documents; at hundreds, add a scope-routing
  pass (recall over scope names) before fanning out — future work, not needed now.
- **Per-spawn load cost**: bigger retained scopes make the CLI's per-spawn O(scope) blob
  parse the dominant cost. Per-doc scopes *reduce* the per-op parse (only the probed scopes
  load). The real cure is the linked `neuron-ffi` handle cache (built, unwired) — this design
  doesn't depend on it but gets faster with it.
- **Behavioral drift**: recall_hive results change shape (across-merged, possibly stitched).
  Callers that assumed one flat scope's ranking (swarm prompt builders, grounding gates)
  should be spot-checked — same list as the seed-gating call-site inventory.
