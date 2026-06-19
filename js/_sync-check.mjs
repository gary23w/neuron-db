// Guard (runs on `npm publish` via prepublishOnly): the published binding (js/index.mjs) must stay
// byte-identical to the in-repo reference (worker/neuron-db.mjs). Not shipped — excluded from `files`.
import { readFileSync } from "node:fs";
const pkg = readFileSync(new URL("./index.mjs", import.meta.url));
const ref = readFileSync(new URL("../worker/neuron-db.mjs", import.meta.url));
if (!pkg.equals(ref)) {
  console.error("DRIFT: js/index.mjs differs from worker/neuron-db.mjs — copy the reference before publishing.");
  process.exit(1);
}
console.log("sync OK: js/index.mjs == worker/neuron-db.mjs");
