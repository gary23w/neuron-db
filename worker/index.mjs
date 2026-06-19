// Reference Cloudflare Worker for neuron-db — the canonical "mount the WASM" example.
//
// All neuron-db access goes through the NeuronDB binding (./neuron-db.mjs): no alloc/mem/answer_ptr
// FFI dance, no tab encoding, no per-op parsing in here. This whole file is the wiring an embedder
// writes — a few lines, not a few hundred.
//
//   GET  /             -> health: the build's op surface + whether the cortex is present
//   POST /v1/observe   {scope, facts[]}        -> {stored}
//   POST /v1/recall    {scope, query, k?}      -> {hits:[{fact,coverage,overlap}], coverage}
//   POST /v1/think     {query, facts[]}        -> {type, value}   (cortex dispatch; 501 if no cortex)
//
// The .wasm is imported as a CompiledWasm module (see wrangler.jsonc) and instantiated once per isolate.
import { NeuronDB } from "./neuron-db.mjs";
import wasmModule from "./neuron_core.wasm";

let _db = null;
const db = () => (_db ||= NeuronDB.fromModule(wasmModule)); // one instance per isolate, reused

export default {
  async fetch(req) {
    const url = new URL(req.url);
    const json = (o, s = 200) => new Response(JSON.stringify(o), {
      status: s, headers: { "content-type": "application/json", "access-control-allow-origin": "*" },
    });
    const d = db();
    const body = req.method === "POST" ? await req.json().catch(() => ({})) : {};

    if (url.pathname === "/") {
      return json({ service: "neuron-db (rust/wasm)", ops: d.listOps(), cortex: d.hasCortex,
        endpoints: ["POST /v1/chat {scope,query}  (the cortex loop)", "POST /v1/observe {scope,facts[]}",
                    "POST /v1/recall {scope,query,k?}", "POST /v1/think {query,facts[]}"] });
    }
    if (url.pathname === "/v1/chat" && req.method === "POST") {
      // the headline: the full cortex loop — recall the working set from `scope`, then let gary-neuron
      // route the turn over it. Returns {type, value, facts}; the host acts on type (escalate -> its LLM).
      if (!body.scope || !body.query) return json({ error: "need {scope, query}" }, 400);
      if (!d.hasCortex) return json({ error: "mount a cortex build — the cortex is the dispatcher, never bypass it" }, 501);
      return json(d.route(body.scope, body.query));
    }
    if (url.pathname === "/v1/observe" && req.method === "POST") {
      if (!body.scope || !Array.isArray(body.facts)) return json({ error: "need {scope, facts[]}" }, 400);
      return json({ stored: d.observeMany(body.scope, body.facts) });
    }
    if (url.pathname === "/v1/recall" && req.method === "POST") {
      if (!body.scope || !body.query) return json({ error: "need {scope, query}" }, 400);
      return json({ hits: d.recallScored(body.scope, body.query, body.k || 6), coverage: d.assess(body.scope, body.query).coverage });
    }
    if (url.pathname === "/v1/think" && req.method === "POST") {
      if (!body.query) return json({ error: "need {query, facts[]}" }, 400);
      if (!d.hasCortex) return json({ error: "this build has no cortex (gary-neuron) baked in" }, 501);
      return json(d.dispatch(body.query, body.facts || [])); // {type:"answer"|"escalate"|"fetch"|"store", value}
    }
    return json({ error: "not found" }, 404);
  },
};
