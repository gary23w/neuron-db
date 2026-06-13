// neuron-db HTTP client for Node 18+ (built-in fetch). Run: node node_client.js
const BASE = process.env.NEURON_BASE || "http://localhost:8088";
const KEY  = process.env.NEURON_DB_KEY; // optional bearer token
const headers = { "content-type": "application/json", ...(KEY ? { authorization: `Bearer ${KEY}` } : {}) };

const post = (path, body) => fetch(`${BASE}${path}`, { method: "POST", headers, body: JSON.stringify(body) }).then(r => r.json());

const remember = (scope, message) => post(`/v1/${encodeURIComponent(scope)}`, { message });
const ask      = (scope, query)   => post(`/v1/${encodeURIComponent(scope)}/get`, { query }).then(r => r.value);
const forget   = (scope, match)   => post(`/v1/${encodeURIComponent(scope)}/forget`, { match });

(async () => {
  await remember("user:42", "the plan is pro");
  await remember("user:42", "the region is us-west-2");
  console.log("plan   =", await ask("user:42", "what plan?"));
  console.log("region =", await ask("user:42", "where is the region?"));
  console.log("forget =", await forget("user:42", "region"));
})();
