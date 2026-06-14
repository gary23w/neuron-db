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

## Isolation

Each neuron is a separate row, addressed by id. With the server, set `NEURON_DB_KEY`
and put a reverse proxy / rate limiter in front. Namespace neuron ids per tenant
(`{tenant}:{name}`) so one tenant can't address another's neuron.

## Honest summary

Use a neuron for *recallable knowledge* — preferences, profile facts, context an agent
should remember — where the win is "can't be dumped, only asked." Use a real secrets
vault for credentials and row-level access control for regulated data. A neuron is
memory you can trust not to spill all at once, not a vault.

To report a vulnerability, open an issue or contact the maintainer via GitHub.
