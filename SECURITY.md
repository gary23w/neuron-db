# Security model

The interesting thing about a neuron is *how data leaves it*. This is an honest
account — the real property and its limits.

## The property: no bulk export

A neuron has no operation that returns all of its values. Facts enter by being stated
and leave only when a cue retrieves a specific one:

- `recall` returns **one value** (or quotes one stored sentence).
- `stats` / `GET` returns **counts and timestamps**, never contents.
- `forget` **removes**; it returns nothing readable.

So a leaked key or a downstream bug yields answers to specific questions, not a dump of
the store. To extract a fact you must already know how to ask for it.

## What this is NOT

- **Not encryption.** Whoever can query the neuron can read facts they know how to ask
  for. The protection is against *mass* exfiltration and accidental over-fetching.
- **Probeable.** Many questions can accumulate answers. Rate-limit and audit recalls;
  those are operational controls, not properties of the store.
- **Fuzzy.** Cue overlap can surface a related fact. A neuron is semantic memory, not an
  access-control boundary for individual secrets.

## Data at rest & in transit

- **At rest: plaintext on disk.** The default `NeuronDB` stores facts as plaintext in the SQLite file
  (and its `-wal`/`-shm` sidecars, and any exported fact packs). It is **not** encrypted at rest —
  protecting the file is the operator's job: use full-disk / volume encryption (LUKS, BitLocker, a
  KMS-backed encrypted volume) on the host. The optional `secure` tier (`SecureNeuronDB`) encrypts
  individual **values** with AES-256-GCM using a per-call key that is never stored, but it is CLI-only,
  does not encrypt the index/cues, and is **not** wired into the HTTP server, MCP server, or WASM build.
- **In transit: no TLS.** The HTTP server speaks plain HTTP. Any bind beyond loopback **MUST** sit behind
  a TLS-terminating reverse proxy (nginx / Caddy / a cloud load balancer) — otherwise the bearer key and
  every fact cross the wire in cleartext. Always set `NEURON_DB_KEY`; an unset key leaves the server open
  to anyone who can reach the port. The key is compared in constant time, and request bodies are capped
  (oversized payloads are rejected before allocation) with read/write socket timeouts.
- **Right to erasure is physical.** `forget` zeroes freed pages (`PRAGMA secure_delete=FAST`) and truncates
  the write-ahead log (`wal_checkpoint(TRUNCATE)`); a full wipe also cascades to a subject's typed
  sub-scopes (`::var`/`::instr`/`::stance`/`::affect`/`::persona`), so a forgotten subject does not survive
  as readable bytes in free pages or WAL frames. (Durability uses WAL `synchronous=NORMAL`: committed
  writes survive a process crash; a power loss can lose the last few unsynced writes — close cleanly for a
  hard durability point.)

## Isolation

Each neuron is a separate row, addressed by id. With the server, set `NEURON_DB_KEY`
and put a **TLS-terminating** reverse proxy / rate limiter in front (see *Data at rest & in transit*).
Namespace neuron ids per tenant (`{tenant}:{name}`) so one tenant can't address another's neuron — note
the server trusts a single global key, so authn is all-or-nothing; per-tenant key scoping is a roadmap item.

## Honest summary

Use a neuron for *recallable knowledge* — preferences, profile facts, context an agent
should remember — where the win is "can't be dumped, only asked." Use a real secrets
vault for credentials and row-level access control for regulated data. A neuron is
memory you can trust not to spill all at once, not a vault.

To report a vulnerability, open an issue or contact the maintainer via GitHub.
