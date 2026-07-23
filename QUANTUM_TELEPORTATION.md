# Quantum Teleportation Tier — neuron-db

An experimental memory tier that maps the quantum teleportation protocol onto
associative recall, creating ephemeral, spooky, and one-shot memory behaviors that
don't exist in any other tier.

> **This is not literal quantum computing.** There is no hardware, no QPU, and no
> superposition of physical states. It is a faithful structural analogy running
> on classical code — entanglement is a link, teleportation is a joint recall,
> collapse is a decrement. The result is a memory that can *move* an association
> without copying it, hold unresolved alternatives, and burn facts after reading
> them. These behaviors are impossible in the base neuron-db store.

**Status:** implemented, experimental. Feature-gated behind `quantum` (in-memory,
std-only) and `quantum-db` (durable state on the NeuronDB SQLite file + the CLI
and HTTP ops). Off by default; a build without the features is byte-identical.

---

## Motivation

The base store writes facts and recalls them by cue overlap. The plastic tier adds
reinforcement and decay. The secure tier encrypts values. But none of them ask:
*what if a fact could be read only once?* or *what if recalling one thing could
teleport the answer to a different question?*

The quantum tier exists for use-cases where the standard "persist until deleted"
model is the wrong abstraction:

| Use-case | Standard store | Quantum tier |
|---|---|---|
| Burn-after-reading secret | Must manually `forget` | `write_once` auto-deletes on the last read |
| Split knowledge (two halves must combine) | Store together, leak together | Entangle across scopes — each half alone is useless |
| Load-balanced recall | Recall hits the same shard | `QuantumRouter` lands the reconstruction on an idle shard |
| Memory ambiguity (don't know yet) | Guess or return nothing | Store a superposition, let recall "measure" it |
| Move an association atomically | Two-step copy+delete leaves a window | `teleport` is one op: measure → channel → reconstruct |

---

## The map

Every quantum concept has a direct analogy in neuron-db's associative store.

| Quantum concept | neuron-db analogy |
|---|---|
| **qubit** | A fact's text — the store's stable identity (base facts carry no numeric id, so identity is `(scope, exact text)`) |
| **entanglement** | An `EntanglementRecord` in a side table: two facts + a classical instruction + an e-bit budget |
| **Bell state measurement** | `teleport`: recall the source by cue, consuming one entanglement unit (e-bit) |
| **classical channel** | The record's plain-text `classical` instruction — written at entangle-time, applied at teleport-time |
| **no-cloning theorem** | A per-fact read budget (`write_once`); each quantum-aware recall decrements it; the read that spends the last one deletes the fact |
| **superposition** | One cue holding `Vec<(value, amplitude)>`; a matching recall "measures" one and decays the rest |
| **collapse** | The e-bit decrement / the alternative decay on measurement |
| **quantum Zeno effect** | The measured candidate is reinforced (×1.1) — repeatedly observing the same outcome pins it |
| **teleportation** | Read the source's association, send the classical instruction to the entangled partner, reconstruct the value there |

---

## Module structure

```
rust/neuron-core/src/
├── lib.rs                     #[cfg(feature = "quantum")] pub mod quantum;
├── quantum.rs                 module root: the storage traits, MemBack, EntangledStore, recall_once
│
├── quantum/
│   ├── entangle.rs            EntanglementRecord, HasEntanglements, entangle/disentangle/entangled_recall
│   ├── teleport.rs            TeleportResult, the 8-step protocol, the classical channel (copy/swap/invert/verbatim)
│   ├── noclone.rs             write_once / reads_remaining (the read budget)
│   ├── superposition.rs       store_super / recall_super / measure (collapse + Zeno + prune + resolve)
│   └── router.rs              QuantumRouter (entangled sharding over NeuronRouter)
│
├── db.rs                      #[cfg(feature = "quantum-db")] the trait impls over NeuronDB (side tables)
└── tests/quantum_tier.rs      integration tests
```

The protocol logic is std-only and generic over three narrow storage traits:

- **`QuantumBack`** — the fact surface (observe / recall_one / has_fact / forget_exact / rewrite_fact)
- **`HasEntanglements`** — the link table (write / read / find / consume_ebit / rebind)
- **`QuantumSide`** — the no-clone budgets and superposition entries

Two backings implement them: **`MemBack`** (feature `quantum`, one Mutex, no
persistence) and **`NeuronDB`** (feature `quantum-db`, impls in db.rs). The same
teleport/burn/collapse code runs over both, so the tiers cannot drift.

---

## Data model

### `EntanglementRecord`

```rust
pub struct EntanglementRecord {
    pub id: u64,                 // primary key
    pub source_scope: String,    // fact identity is (scope, exact text) —
    pub source_text: String,     //   the store's stable identity; there is no FactId
    pub dest_scope: String,
    pub dest_text: String,
    pub classical: String,       // plain-text instruction: "copy" | "swap" | "invert" | anything else
    pub ebits: u32,              // remaining entanglement units
    pub created_at: u64,         // unix millis
}
```

Entanglement units are the resource. Each `teleport` consumes 1 e-bit. When
`ebits` reaches 0 the link is deleted (the pair is "disentangled"). When a
teleport rewrites a fact's text, surviving links are re-pointed at the new text
(text IS identity, so a link left behind would dangle).

### The no-clone budget

`write_once(scope, text, max_reads)` stores the fact normally and arms a read
budget in the tier's side state — **not** on the Episode itself, so the base
store's format and dump()/load() round-trip stay byte-identical. Each
quantum-aware read that returns the fact decrements the budget; the read that
spends the last one deletes the fact (that read still gets the value — the NEXT
reader finds nothing).

### Superpositions

`store_super(scope, text, alternatives)` holds weighted alternatives for a cue,
all at amplitude 1.0, none of them yet a fact. A quantum-aware recall that
matches the cue (stem overlap ≥ 1) measures it:

1. The **highest-amplitude** candidate is returned (ties → first stored).
   Measurement is deliberately deterministic — amplitude decides, not a dice
   roll — so behavior is testable and repeatable.
2. The winner is reinforced ×1.1 (Zeno), the losers decay ×0.5 (collapse).
3. Candidates below amplitude 0.1 are pruned.
4. When a single candidate remains, the superposition **resolves**: the entry is
   deleted and `"<cue text> <winner>"` is stored as an ordinary fact.
   Measurement has produced a classical state.

---

## API (Rust)

Free functions carry the protocol (callable on a borrowed `&NeuronDB` — this is
what the CLI and HTTP arms use); `EntangledStore<S>` is the owned wrapper
mirroring the other tiers' shape:

```rust
use neuron_core::quantum::{EntangledStore, MemBack};

let q = EntangledStore::new(MemBack::new());          // or ::new(NeuronDB::open(..)) with quantum-db

// entangle two facts (observing either side if absent); returns the link id
let id = q.entangle("user:42", "the gate code is 4491",
                    "user:99", "the gate code is ----", "copy", 3);

// the correlated, NON-consuming read: a hit plus everything it is entangled with
let r = q.entangled_recall("user:42", "what is the gate code?");

// the consuming op: measure -> spend an e-bit -> classical channel -> reconstruct on the dest
let t = q.teleport("user:42", "what is the gate code?");
// t.value = "4491", t.dest_scope = "user:99", t.ebits_remaining = 2

q.disentangle(id);

// no-cloning: burns after `max_reads` quantum-aware reads
q.write_once("vault", "the launch code is gamma-7", 1);
q.reads_remaining("vault", "the launch code is gamma-7");   // Some(1)

// superposition: hold alternatives; recall measures one
q.store_super("user:42", "my favorite food is", &["pizza", "sushi", "tacos"]);
q.recall_super("user:42", "what is my favorite food?");     // Some("pizza")

// the combined quantum-aware read (superpositions measure, no-clone facts burn)
q.recall_once("vault", "what is the launch code?");
```

The classical channel at teleport time:

- `"copy"` — the dest fact takes the source's text (and so its association).
- `"swap"` — the two facts exchange texts. A surviving swapped link is re-issued
  with its endpoints exchanged (fresh id).
- `"invert"` — the source's value is negated (numeric) or reversed (string) on
  the dest.
- anything else — stored verbatim as the dest fact's new text.

### `QuantumRouter` (entangled sharding)

Wraps `NeuronRouter`. `entangle` places the dest fact on the **idle** shard
(least-loaded, never the source's own, spawning a fresh one when everything else
is full), so a teleport's reconstruction — the write half of the recall — always
lands off the busy shard. Load-balanced by construction, not by a scheduler.

---

## CLI

New subcommands, compiled with `--features quantum-db`:

```sh
# Entangle two facts (either side is observed if absent)
neuron --db app.db entangle user:42 "the gate code is 4491" \
                            user:99 "the gate code is ----" --classical copy --ebits 3

# Teleport: recall from user:42 moves the association to user:99
neuron --db app.db teleport user:42 "what is the gate code?"
# → teleported "4491" via "copy" to user:99 (2 ebit(s) remaining)

neuron --db app.db entanglements user:42      # list links (ids, endpoints, ebits)
neuron --db app.db disentangle 1

# Burn-after-reading
neuron --db app.db write_once vault "the launch code is gamma-7"
neuron --db app.db get vault "launch code"    # → gamma-7
neuron --db app.db get vault "launch code"    # → (no answer), exit 3 — vanished

# Superposition (alternatives are comma-separated)
neuron --db app.db superposition user:42 mood "happy, tired, curious"
neuron --db app.db get user:42 "mood"         # → happy   (the losers decayed)
```

With `quantum-db` compiled in, the plain `get`/`recall` verbs become
quantum-aware: superpositions measure and write-once facts burn through the
verbs clients already use. A binary built without the feature reads the same
facts without consuming anything — and prints `unknown command` for the quantum
verbs. That boundary is deliberate: the base store's recall stays a pure read.

---

## HTTP API

Same gate (`server` + `quantum-db`); `/get` and `/recall` are quantum-aware too:

```sh
POST /v1/{scope}/entangle      {"fact_a": "...", "fact_b": "...",
                                "scope_b": "user:99", "classical": "copy", "ebits": 3}
POST /v1/{scope}/teleport      {"cue": "what is the gate code?"}
POST /v1/{scope}/write_once    {"text": "the launch code is gamma-7", "reads": 1}
POST /v1/{scope}/superposition {"text": "mood", "alternatives": ["happy","tired","curious"]}
POST /v1/{scope}/disentangle   {"id": 1}
```

---

## Storage

`quantum-db` adds two side tables to the NeuronDB file, created **lazily** on
the first quantum write (the trust-ledger policy): `entanglements` (one row per
link) and `quantum_kv` (the no-clone budgets and superposition entries, keyed
`(kind, scope, k)`). A store that never touches the tier keeps a byte-identical
schema. An `EntanglementRecord` row is ~100 bytes plus its two text endpoints —
this is a niche tier; you entangle a handful of facts for specific one-shot or
split-knowledge use-cases, not the whole store.

---

## Tests

```
tests/quantum_tier.rs            (cargo test --features quantum; durable half needs quantum-db)

  ✓ entangle_then_recall_triggers_side_effect     (symmetric + non-consuming correlated read)
  ✓ teleport_moves_association_not_fact           (source survives; dest answers with its value)
  ✓ teleport_ebit_exhaustion_disentangles         (budget spent → link deleted → next teleport None)
  ✓ classical_channel_copy_swap_invert            (+ the verbatim-payload channel)
  ✓ no_clone_fact_vanishes_after_max_reads
  ✓ no_clone_fact_coexists_with_normal_facts
  ✓ superposition_collapses_on_measurement        (argmax + Zeno boost + loser decay)
  ✓ superposition_removes_decayed_alternatives    (prune → lone survivor resolves to a real fact)
  ✓ quantum_router_fans_out_to_idle_shard
  ✓ entangled_scopes_are_independent_after_disentangle
  ✓ durable_quantum_state_survives_reopen         (quantum-db)
  ✓ durable_noclone_burn_is_durable_across_reopen (quantum-db)
  ✓ quantum_routes_through_http_surface           (server.rs, quantum-db + server)
```

---

## Feature flags

```toml
[features]
quantum     = []                     # base tier: MemBack + EntangledStore + QuantumRouter (in-memory, std-only)
quantum-db  = ["quantum", "sqlite"]  # + durable side tables on NeuronDB, CLI verbs, HTTP endpoints
```

`#[cfg(feature = "quantum")]` guards the module; `quantum-db` guards the db.rs
impls and every CLI/HTTP arm. No extra structs, tables, or match arms unless you
opt in.

---

## Relationship to other tiers

| Tier | Lives alongside? | Notes |
|---|---|---|
| **base** (`Neuron`/`NeuronDB`) | Yes — quantum ops call the same recall engine | Teleport's rewrite carries a fact's learned strength |
| **plastic** (`PlasticNeuron`) | Separate in-memory type | No cross-coupling yet (a reinforce-extends-reads interplay is a future direction) |
| **secure** (`SecureNeuronDB`) | Separate table | The classical instruction is plain text by design (see Limitations) |
| **router** (`NeuronRouter`) | Yes — `QuantumRouter` wraps it | Teleport becomes a shard-to-shard operation |

---

## Limitations

- **No actual quantum hardware** — entirely classical; the name is an analogy.
- **Classical channel is plain text** — anyone who reads the entanglement table
  sees the instruction. If you need secrecy, store an encrypted instruction and
  decrypt before entangling (or keep the halves in a secure tier).
- **Reads are quantum-aware only through the tier's surface** — the CLI/HTTP
  verbs compiled under `quantum-db`, or `recall_once`/`teleport` in Rust. A
  transport built without the feature (or the MCP/wasm surfaces) reads the same
  facts without consuming anything.
- **Measurement is deterministic** (argmax), not amplitude-weighted random — a
  deliberate deviation for testability; amplitude still decides the outcome.
- **No distributed entanglement** — e-bits live in one database file.
- **Superposition decay is global** — all losers decay on any measurement; there
  is no partial measurement.
- **Teleportation moves scalar text** — one fact's text/value, not a compound
  multi-fact state.
- **Same-scope swap** exchanges the two facts correctly but does not re-point
  third-party links on them (the rebinds would collide). Split-knowledge pairs
  are cross-scope, which rebinds fully.

---

## Future directions

| Idea | Requires |
|---|---|
| **Multi-way entanglement** (≥3 facts linked) | N-ary EntanglementRecord |
| **Quantum-net** (sync entanglements across processes) | Consensus layer + network transport |
| **Teleport of compound facts** (multiple fields) | Structured classical instructions |
| **Amplitudes set at write time** | Per-alternative weight on `store_super` |
| **Entanglement swapping** (link two existing links) | New op: `entangle --swap-links` |
| **Density matrix recall** (all alternatives + probabilities, without collapsing) | New op: `recall_density` |
| **Plastic interplay** (reinforcing a no-clone fact extends its reads) | Cross-tier coupling |

---

## Design rationale

**Why feature-gate it?** The tier adds concepts (link records, e-bits,
superposition decay) that are useless if you never call them. Gating keeps the
core binary lean — no extra structs, tables, or match arms unless you opt in.

**Why a "classical channel" at all?** Because real quantum teleportation requires
one (the Bell measurement result must travel by ordinary means). The analogy is
faithful: the classical instruction is the bottleneck that prevents
faster-than-light communication — and here it doubles as the seam where the user
controls *how* the value is reconstructed on the dest.

**Why decay superposition on every recall?** It mirrors measurement: once you
measure, the wavefunction collapses. The Zeno boost is a deliberate second-order
analogy that rewards consistent recall — and the resolve-at-one step means
repeated measurement eventually yields a classical fact, which is what
measurement *does*.

**Why not just store two facts and delete one?** Because `teleport` is atomic —
the full protocol executes (measure → collapse → channel → reconstruct) in one
call. A manual two-step leaves a window where both copies exist un-accounted,
violating the no-cloning spirit. The tier enforces the invariant.
