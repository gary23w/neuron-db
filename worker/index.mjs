// Cloudflare Worker that loads neuron-db's Rust core (compiled to WASM, emergence model
// baked in) and serves it. No Python, no external model, no network at request time.
//
//   GET  /            -> health + selftest (store recall + cortex copied a value)
//   POST /v1/think    {"query": "...", "facts": ["...", ...]} -> {"answer": "..."}
//
// The .wasm is imported as a CompiledWasm module (see wrangler.jsonc rules) and
// instantiated once per isolate, then reused across requests.
import wasmModule from "./neuron_core.wasm";

let EX = null;
function inst() {
  if (!EX) EX = new WebAssembly.Instance(wasmModule, {}).exports;
  return EX;
}
const td = new TextDecoder(), te = new TextEncoder();

function think(query, facts) {
  const e = inst();
  const input = [query, ...(facts || [])].join("\n");
  const bytes = te.encode(input);
  const ptr = e.alloc(bytes.length);
  new Uint8Array(e.memory.buffer, ptr, bytes.length).set(bytes);
  const outlen = e.run(ptr, bytes.length);
  return td.decode(new Uint8Array(e.memory.buffer, e.answer_ptr(), outlen));
}

export default {
  async fetch(req) {
    const url = new URL(req.url);
    const json = (o, s = 200) => new Response(JSON.stringify(o), { status: s, headers: { "content-type": "application/json", "access-control-allow-origin": "*" } });

    if (url.pathname === "/") {
      const code = inst().selftest();
      return json({ service: "neuron-db (rust/wasm)", selftest: code,
                    store_recall: !!(code & 1), cortex_copied_value: !!(code & 2),
                    endpoint: "POST /v1/think {query, facts[]}" });
    }
    if (url.pathname === "/v1/think" && req.method === "POST") {
      const body = await req.json().catch(() => ({}));
      if (!body.query) return json({ error: "need {query, facts[]}" }, 400);
      return json({ answer: think(body.query, body.facts || []) });
    }
    return json({ error: "not found" }, 404);
  },
};
