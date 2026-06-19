// Guard (runs on `npm publish` via prepublishOnly). Two invariants:
//   1. the published binding (js/index.mjs) is byte-identical to the in-repo reference (worker/neuron-db.mjs);
//   2. the binding's exported VERSION literal matches package.json's version (so the runtime constant can't
//      silently drift behind a release — byte-equality alone wouldn't catch both files being wrong in lockstep).
// Not shipped — excluded from `files`.
import { readFileSync } from "node:fs";

const pkg = readFileSync(new URL("./index.mjs", import.meta.url));
const ref = readFileSync(new URL("../worker/neuron-db.mjs", import.meta.url));
if (!pkg.equals(ref)) {
  console.error("DRIFT: js/index.mjs differs from worker/neuron-db.mjs — copy the reference before publishing.");
  process.exit(1);
}

const pkgVersion = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8")).version;
const m = pkg.toString("utf8").match(/export const VERSION\s*=\s*["']([^"']+)["']/);
if (!m) {
  console.error("DRIFT: could not find the exported VERSION literal in index.mjs.");
  process.exit(1);
}
if (m[1] !== pkgVersion) {
  console.error(`DRIFT: binding VERSION "${m[1]}" != package.json version "${pkgVersion}" — bump the literal in worker/neuron-db.mjs (then re-copy to js/index.mjs).`);
  process.exit(1);
}

console.log(`sync OK: js/index.mjs == worker/neuron-db.mjs, VERSION == ${pkgVersion}`);
