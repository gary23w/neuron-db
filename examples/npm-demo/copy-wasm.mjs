// Copies the cortex wasm from the repo into public/ before dev/build, so the demo ships the full
// cortex (recall + dispatch) without committing a duplicate 7.6 MB binary. Falls back to the lean
// store-only build that ships inside the installed @gary23w/neuron-db package if the cortex build
// isn't present (e.g. a clean checkout without the worker artifact).
import { copyFileSync, mkdirSync, existsSync } from "node:fs";

mkdirSync("public", { recursive: true });
const candidates = [
  "../../worker/neuron_core.wasm",                          // the cortex build (recall + the ask/route loop)
  "../../docs/demos/neuron_core_http.wasm",
  "node_modules/@gary23w/neuron-db/neuron_core.wasm",       // store-only fallback (memory only; `ask` will explain)
];
const src = candidates.find(existsSync);
if (!src) { console.error("no neuron_core.wasm found to copy"); process.exit(1); }
copyFileSync(src, "public/neuron_core.wasm");
console.log(`copied ${src} -> public/neuron_core.wasm`);
