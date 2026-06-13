# Threat model

neuron-db has two storage tiers. Choosing the right one is a security decision.

| Tier | Module | At-rest confidentiality | Recall |
|---|---|---|---|
| **Plaintext** | `NeuronDB` | none (facts stored as text) | fuzzy NL, value isolation |
| **Encrypted** | `SecureNeuronDB` | full (AES-GCM, client-held key) | exact key→value |

## What the encrypted tier defends

The database stores only ciphertext and keyed hashes. The per-neuron secret is supplied
by the caller on each request and **never persisted**. The working key is derived as
`HKDF(secret, "neuron-db/" + neuron_id)`, binding it to both the secret and the id.

- **Stolen database file** → opaque blobs. No values, no cues, no keys. (`test_dump_is_ciphertext_only`)
- **Request-key bumping** → changing the neuron id in a request reads nothing: the key is
  bound to the id, so another neuron's secret can't derive this neuron's key, and the
  keyed-hash index won't match either. (`test_id_bump_denied`)
- **Wrong secret** → AEAD authentication fails; recall returns `None`, never the value or
  an error that distinguishes "wrong key" from "no such fact". (`test_wrong_key_denied`)
- **Tampered ciphertext** → detected by the AEAD tag; decryption refuses. (`test_aead_roundtrip_and_tamper`)
- **Loose cue** → a query must cover at least half of a stored key's stems, so a vague
  probe doesn't surface an unrelated secret. (`min_cover`)

## What it does NOT defend

Stated plainly, because overclaiming is the failure mode here.

- **Not protection against a compromised running server.** The caller's secret passes
  through the process during a request. A server that is actively owned at runtime can
  observe secrets in flight. The encryption protects data **at rest** and against
  **id-bumping / mass export**, not against a live attacker inside the process.
- **Probeable with the right secret.** Whoever holds a neuron's secret can ask it many
  questions. Rate-limit and audit. These are operational controls.
- **Metadata.** The number of entries and their approximate stem counts are visible in a
  dump (lengths of the keyed-hash lists). Values, keys, and their text are not.
- **Key management is yours.** If the client loses the secret, the data is unrecoverable
  (that is the point). If the client leaks it, that neuron is readable. Store secrets in
  a real secrets manager; neuron-db does not manage them for you.

## The plaintext tier

`NeuronDB` stores facts as text for fuzzy natural-language recall and value isolation.
Its security property is weaker but still real: there is **no bulk-export operation** —
`recall`/`get` return one value, `stats` returns counts, `forget` returns nothing
readable. A leaked key yields answers to specific questions, not a dump. But the file
itself is plaintext; anyone who reads the disk reads the facts. Use the encrypted tier
for anything sensitive.

## Cipher

AES-256-GCM via the `cryptography` package when installed (`pip install neuron-db[crypto]`).
Without it, a standard-library fallback uses HMAC-SHA256 in counter mode with
encrypt-then-MAC — a sound construction, but for high-assurance deployments install the
vetted AEAD. The stored format records which was used, so a neuron written with one can
be read by the other only if the same backend is present.

## Reporting

Open a GitHub issue or contact the maintainer privately for anything sensitive.
