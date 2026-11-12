// Validates worker/neuron-db.mjs against a real neuron_core.wasm. Run: node test-binding.mjs <wasm>
import { readFileSync } from "node:fs";
import { NeuronDB } from "./neuron-db.mjs";

const WASM = process.argv[2] || process.env.WASM;
const db = await NeuronDB.fromBytes(readFileSync(WASM));

let pass = 0, fail = 0;
const ok = (name, cond, got) => cond ? pass++ : (fail++, console.log("FAIL:", name, "| got:", JSON.stringify(got)));

// introspection / self-describing surface
ok("listOps nonempty", db.listOps().length > 10, db.listOps().length);
ok("supports recallscored", db.supports("recallscored"));
ok("ops includes obsmany+dump+caps", ["obsmany","dump","caps","ops"].every(o => db.supports(o)));

// write + robust recall (uses recall_many under the hood, not the abstaining single-hit op)
db.observeMany("kb", ["the wifi password is Maple-Cedar-71", "the api key is zeta-9931", "the deploy region is us-west-2"]);
ok("recall finds api key", db.recall("kb", "what is the api key?", 5).some(h => h.includes("zeta-9931")), db.recall("kb","what is the api key?",5));
const scored = db.recallScored("kb", "deploy region", 5);
ok("recallScored has numeric coverage", scored.length > 0 && typeof scored[0].coverage === "number", scored[0]);

// assess (the knowledge-gap signal)
const a = db.assess("kb", "what is the api key?");
ok("assess coverage>0 + hits", a.coverage > 0 && a.hits >= 1, a);
const miss = db.assess("kb", "what is the launch code?");
ok("assess coverage 0 on a miss", miss.coverage === 0, miss);

// value + vars
ok("recallValue", (db.recallValue("kb", "what is the wifi password?") || "").includes("Maple"), db.recallValue("kb","what is the wifi password?"));
db.setVar("kb", "plan", "pro");
ok("getVar", db.getVar("kb", "plan") === "pro", db.getVar("kb","plan"));
ok("vars object", db.vars("kb").plan === "pro", db.vars("kb"));
ok("delVar true", db.delVar("kb", "plan") === true);
ok("getVar null after del", db.getVar("kb", "plan") === null);

// chain
db.observeMany("org", ["Aurora's owner is Marisol", "Marisol's manager is Dana"]);
const ch = db.chain("org", "Aurora", ["owner", "manager"]);
ok("chain resolves Dana", (ch.value || "").includes("Dana"), ch);

// dump/load round-trip (the blob carries tabs — must survive verbatim)
const blob = db.dump("kb");
ok("dump nonempty", blob.length > 0, blob.length);
db.load("kb2", blob);
ok("load restores fact count", db.stats("kb2") === db.stats("kb"), [db.stats("kb"), db.stats("kb2")]);

// scopes + forget
ok("scopes lists kb", db.scopes().some(s => s.scope === "kb"), db.scopes());
ok("forget removes wifi", db.forget("kb", "wifi") >= 1);

// keyed facts: upsert replaces by exact key, forgetKey removes by exact key, keys survive dump/load
db.upsert("crm", "customer:1", "customer Dana Reyes of Oakville phone 905-555-0101 note see rec12 file");
db.upsert("crm", "customer:12", "customer Omar Haddad of Milton phone 905-555-0112");
ok("upsert stores both", db.stats("crm") === 2, db.stats("crm"));
db.upsert("crm", "customer:12", "customer Omar Haddad of Burlington phone 905-555-0112");
ok("upsert replaces, no duplicate", db.stats("crm") === 2, db.stats("crm"));
ok("upsert keeps the other key's fact", db.recall("crm", "Dana Reyes", 5).some(h => h.includes("Oakville")), db.recall("crm", "Dana Reyes", 5));
ok("upsert dropped the old text", !db.recall("crm", "Omar Haddad", 5).some(h => h.includes("Milton")), db.recall("crm", "Omar Haddad", 5));
const keyed = db.recallKeyed("crm", "Omar Haddad", 5);
ok("recallKeyed returns the hit's key", keyed.length > 0 && keyed[0].key === "customer:12" && typeof keyed[0].coverage === "number", keyed);
db.observe("crm", "the office is in Toronto on King Street");
ok("recallKeyed key is null for an unkeyed fact", db.recallKeyed("crm", "office Toronto", 5).some(h => h.fact.includes("King Street") && h.key === null), db.recallKeyed("crm", "office Toronto", 5));
db.forget("crm", "King Street");
db.load("crm2", db.dump("crm"));
ok("forgetKey works after dump/load", db.forgetKey("crm2", "customer:12") === 1 && db.stats("crm2") === 1, db.stats("crm2"));
ok("forgetKey is exact, not a prefix", db.forgetKey("crm", "customer:1") === 1 && db.stats("crm") === 1, db.stats("crm"));
ok("forgetKey on a missing key removes nothing", db.forgetKey("crm", "customer:99") === 0);

// §7 caps
ok("caps recall=grounded", db.caps().some(c => c.name === "recall" && c.owner === "grounded"));
ok("resolveCaps defers summarize", db.resolveCaps(["summarize"]).some(r => r.name === "summarize" && r.disposition === "defer"));

// FAIL LOUD: an unknown op raises (binding validates against the build's op surface), not silent empty
let threw = false, msg = "";
try { db.raw("definitely-not-an-op"); } catch (e) { threw = true; msg = e.message; }
ok("unknown op throws (fail-loud)", threw && /not in this wasm build/.test(msg), msg);
// the wasm-side sentinel backstop: an unchecked raw call to a bogus op returns the  marker
ok("wasm emits sentinel for unknown op", db._raw("bogus-op").charCodeAt(0) === 1, db._raw("bogus-op").slice(0,12));

// cortex path — on a cortex build, dispatch/route return a TYPED decision (never raw model text);
// on a store build both throw a clean error rather than producing garbage.
if (db.hasCortex) {
  const ext = db.dispatch("what is the api key?", ["the api key is zeta-9931"]);
  ok("dispatch returns a typed decision", ["answer", "escalate", "fetch", "store"].includes(ext.type), ext);
  db.observeMany("rt", ["the api key is zeta-9931", "the deploy region is us-west-2"]);
  const r = db.route("rt", "what is the api key?");
  ok("route returns {type, value, facts}", ["answer", "escalate", "fetch", "store"].includes(r.type) && Array.isArray(r.facts), r);
} else {
  let dThrew = false;
  try { db.dispatch("hi", []); } catch (e) { dThrew = /cortex/.test(e.message); }
  ok("dispatch throws on store build (clean error, not garbage)", dThrew);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exitCode = fail ? 1 : 0;
