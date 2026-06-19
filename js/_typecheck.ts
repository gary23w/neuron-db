// Type-check guard for index.d.ts (not shipped — excluded from package `files`). Exercises the API
// so `tsc --noEmit --strict` catches any drift between the types and the binding's real surface.
import { NeuronDB, VERSION } from "./index.mjs";
import type { Routed, RecallHit, AssessResult, Capability, DispatchKind } from "./index.mjs";

async function check(): Promise<void> {
  const db: NeuronDB = await NeuronDB.forNode("./neuron_core.wasm");
  const v: string = VERSION;

  const stored: number = db.observeMany("s", ["a fact", "another"]);
  const hits: string[] = db.recall("s", "fact", 5);
  const scored: RecallHit[] = db.recallScored("s", "fact");
  const cov: number = scored[0]?.coverage ?? 0;
  const val: string | null = db.recallValue("s", "fact");
  const a: AssessResult = db.assess("s", "fact");
  const ok: boolean = a.hasValue;
  const assoc: string[] = db.assoc("s", "fact", 8, 2);
  const chain = db.chain("s", "Aurora", ["owner", "manager"]);
  const chainVal: string | null = chain.value;

  db.setVar("s", "k", "v").setVar("s", "k2", "v2");
  const got: string | null = db.getVar("s", "k");
  const vars: Record<string, string> = db.vars("s");
  const deleted: boolean = db.delVar("s", "k");

  const removed: number = db.forget("s", "fact");
  const count: number = db.stats("s");
  const blob: string = db.dump("s");
  const loaded: number = db.load("s2", blob);

  const caps: Capability[] = db.caps();
  const owner: "grounded" | "deferrable" = caps[0].owner;

  if (db.hasCortex) {
    const turn: Routed = db.route("s", "what is the api key?", { k: 6 });
    const kind: DispatchKind = turn.type;
    const facts: string[] = turn.facts;
    const d = db.dispatch("hi", facts);
    void d.value;
  }

  void [stored, hits, cov, val, ok, assoc, chainVal, got, vars, deleted, removed, count, loaded, owner, v];
}

void check;
