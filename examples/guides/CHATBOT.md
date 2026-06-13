# Wiring neuron-db into a chatbot (long-term memory)

LLMs forget everything between calls — their only memory is the context window you build for
them. neuron-db gives a bot durable, per-user memory that you can recall *on the fly* and
drop into that context. The store stays deterministic and microsecond-fast; the model only
ever sees a tiny, relevant slice of memory, never the whole database.

## The loop

Every turn does three things:

1. **Recall** — before you call the LLM, pull the few facts relevant to the new message.
2. **Inject** — put those facts into the context (system prompt or a memory block).
3. **Write** — after answering, store any durable facts the user stated.

```
user message ─▶ recall(scope, message) ─▶ build context (system + memory + message)
                      │                              │
                      ▼                              ▼
              neuron-db (µs, local)            your LLM ─▶ reply
                      ▲                              │
                      └────── write new facts ◀──────┘
```

The key idea: **memory lives outside the model.** You decide what enters the prompt. That
keeps token cost flat no matter how much the bot remembers, and the recall is exact and
auditable instead of buried in a giant context dump.

## Minimal example (Node + any chat LLM)

Run the neuron-db server (`serve neurons.db 8088`), then:

```js
const NB = "http://localhost:8088";
const nb = (path, body) =>
  fetch(`${NB}${path}`, { method: "POST", headers: { "content-type": "application/json" },
                          body: JSON.stringify(body) }).then(r => r.json());

// one scope per user keeps memories isolated
const recall = (user, q) => nb(`/v1/${encodeURIComponent(user)}/get`, { query: q }).then(r => r.value);
const remember = (user, fact) => nb(`/v1/${encodeURIComponent(user)}`, { message: fact });

async function chat(user, message) {
  // 1. RECALL relevant memory for this message
  const memory = await recall(user, message);

  // 2. INJECT it into the model context
  const system = memory
    ? `You are a helpful assistant. Known about the user: ${memory}.`
    : `You are a helpful assistant.`;
  const reply = await callYourLLM([             // <- your OpenAI/Anthropic/etc. call
    { role: "system", content: system },
    { role: "user",   content: message },
  ]);

  // 3. WRITE durable facts the user stated (see "what to store" below)
  if (looksLikeAStatement(message)) await remember(user, message);

  return reply;
}
```

That's the whole integration. `recall` is a single HTTP round-trip (microseconds on the
server side); `remember` persists immediately.

## What to store

You have two good options:

- **Store raw statements** (simplest). Pass user messages straight to `remember`. neuron-db
  already extracts the salient value and indexes the rest, so `"my plan is pro"` becomes
  recallable by `"what plan?"`. Skip questions (don't store `"what's my name?"`).
- **Store extracted facts** (cleaner memory). Have your LLM emit short canonical facts
  (`"the user's plan is pro"`, `"the user prefers dark mode"`) and store those. This keeps
  the memory tidy and avoids storing chit-chat.

Either way, keep facts short and declarative — one idea per line.

## When and how much to recall

- Recall against the **incoming user message** (that's the query). For broader context, make
  a few recalls — e.g. on the message plus the user's name/role — and concatenate the hits.
- Gate on **coverage**: only inject a memory when `recall(...).coverage` is high enough
  (start around 0.5) so you don't pollute the prompt with weak matches.
- For a richer memory block, use the library's `recall_related` / `recall_spreading` (the
  plastic tier) to pull a fact plus its associates, then format them as bullet lines:

  ```
  [what you know about this user]
  - plan: pro
  - timezone: PST
  - prefers: dark mode
  ```

## Scopes = users (or sessions, or topics)

The scope id is how you partition memory. Common choices:

- `user:<id>` — durable per-user memory across sessions (most common).
- `session:<id>` — ephemeral, scoped to one conversation.
- `user:<id>:project:<id>` — memory per user *and* workspace.

Scopes are fully isolated, so one user's memory can never leak into another's prompt.

## Persistence, privacy, scale

- **Durable by default** — facts survive restarts; the bot remembers across sessions.
- **Forget on request** — `POST /v1/<user>/forget {match}` deletes matching facts, or wipe
  the whole scope. Wire this to a "forget me" / GDPR delete.
- **Encrypt sensitive memory** — for secrets or PII use the `SecureNeuronDB` tier so values
  are encrypted at rest under a per-user secret (see `examples/` → `encrypted_secrets`).
- **Scale** — keep a hot scope to hundreds–low-thousands of facts; shard with `NeuronRouter`
  beyond that. Recall stays fast because the model only sees the recalled slice.

## Embedded vs server

- **Server (shown above)** — your bot calls neuron-db over HTTP. Language-agnostic; good when
  the bot and the store run as separate services. Set `NEURON_DB_KEY` for bearer auth.
- **Embedded** — if your bot backend is Rust, skip HTTP and call `NeuronDB` directly (see the
  `chatbot_memory` example). Same three steps, zero network hop.

See the runnable `chatbot_memory` example (`cargo run --example chatbot_memory --features
sqlite`) for the embedded version of this loop.
