// neuron-db as the demo's database. One instance, loaded from the wasm in public/. Scopes are
// persisted to localStorage with dump()/load() so users and their memories survive a reload.
import { NeuronDB } from "@gary23w/neuron-db";

let db: NeuronDB;

/** Load the wasm and restore any persisted scopes (facts via load(), variables replayed). */
export async function initDB(): Promise<NeuronDB> {
  db = await NeuronDB.forBrowser("/neuron_core.wasm");
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i);
    if (!k) continue;
    if (k.startsWith("ndb:")) {                          // facts: dump()/load() blob
      const blob = localStorage.getItem(k);
      if (blob) db.load(k.slice(4), blob);
    } else if (k.startsWith("ndbvars:")) {               // variables: dump() doesn't capture them, replay
      const scope = k.slice("ndbvars:".length);
      const vars = JSON.parse(localStorage.getItem(k) || "{}") as Record<string, string>;
      for (const [key, val] of Object.entries(vars)) db.setVar(scope, key, val);
    }
  }
  return db;
}

export function ndb(): NeuronDB {
  return db;
}

/** Persist a scope's facts AND variables to localStorage (call after any write). */
export function persist(scope: string): void {
  const blob = db.dump(scope);
  if (blob) localStorage.setItem("ndb:" + scope, blob);
  const vars = db.vars(scope);
  if (Object.keys(vars).length) localStorage.setItem("ndbvars:" + scope, JSON.stringify(vars));
}
