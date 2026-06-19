// The demo: a login screen, then a memory console — all backed by neuron-db.
import { initDB, ndb, persist } from "./neuron";
import { signup, login, userScope, userCount } from "./auth";

const app = document.getElementById("app") as HTMLElement;
const $ = (sel: string, root: ParentNode = document) => root.querySelector(sel) as HTMLElement;

let user = "";
let scope = "";

// example facts seeded into a new account so recall / ask / chain work right away
const SEED = [
  "the project deadline is Friday",
  "my favorite editor is neovim",
  "the api key is zeta-9931",
  "the deploy region is us-west-2",
  "Aurora's owner is Marisol",
  "Marisol's manager is Dana",
];

boot();

async function boot() {
  app.innerHTML = `<div class="card"><div class="brand"><b>neuron-db</b></div><p class="sub">loading the WebAssembly memory engine…</p></div>`;
  await initDB();
  const resumed = sessionStorage.getItem("demo:user");
  if (resumed && ndb().vars("system:users")[resumed]) enterConsole(resumed);
  else renderLogin();
}

// ───────────────────────── login ─────────────────────────
function renderLogin(msg = "") {
  app.innerHTML = `
    <div class="card">
      <div class="brand"><b>neuron-db</b><span class="v">memory console</span></div>
      <p class="sub">Sign in to your associative memory. Accounts and every memory live in neuron-db itself.</p>
      <div class="field"><label>username</label><input id="u" autocomplete="username" placeholder="ada" /></div>
      <div class="field"><label>password</label><input id="p" type="password" autocomplete="current-password" placeholder="••••" /></div>
      <div class="row"><button id="login">Log in</button><button id="signup" class="ghost">Create account</button></div>
      <div class="status" id="st">${msg}</div>
      <p class="note">Demo only — it runs entirely in your browser (no server). Passwords are PBKDF2-hashed and
      stored in neuron-db, but a client-side store can't be a real auth system. ${userCount()} account(s) on this device.</p>
    </div>`;
  const u = $("#u") as HTMLInputElement, p = $("#p") as HTMLInputElement, st = $("#st");
  const go = async (fn: typeof login, label: string) => {
    st.style.color = "var(--amber)"; st.textContent = `${label}…`;
    const r = await fn(u.value, p.value);
    if (r.ok) { sessionStorage.setItem("demo:user", u.value.trim().toLowerCase()); enterConsole(u.value); }
    else { st.style.color = "var(--pink)"; st.textContent = r.error; }
  };
  $("#login").onclick = () => go(login, "checking");
  $("#signup").onclick = async () => {
    const uname = u.value.trim().toLowerCase();
    const r = await signup(u.value, p.value);
    if (r.ok) {
      ndb().observeMany(userScope(uname), SEED); persist(userScope(uname));   // seed a starter memory
      sessionStorage.setItem("demo:user", uname); enterConsole(u.value);
    } else { st.style.color = "var(--pink)"; st.textContent = r.error; }
  };
  p.onkeydown = (e) => { if (e.key === "Enter") $("#login").click(); };
  u.focus();
}

// ───────────────────────── console ─────────────────────────
function enterConsole(uname: string) {
  user = uname.trim().toLowerCase();
  scope = userScope(user);
  app.innerHTML = `
    <div class="term">
      <div class="bar">
        <span class="dot" style="background:#f472b6"></span><span class="dot" style="background:#f5b454"></span><span class="dot" style="background:#34d399"></span>
        <span class="who">signed in as <b>${user}</b> · <a id="logout" href="#">logout</a></span>
      </div>
      <div class="out scrollbar" id="out"></div>
      <div class="prompt"><span>›</span><input id="cmd" autocomplete="off" spellcheck="false" placeholder="type a command — try: help" /></div>
    </div>`;
  $("#logout").onclick = (e) => { e.preventDefault(); sessionStorage.removeItem("demo:user"); renderLogin("logged out"); };
  const cmd = $("#cmd") as HTMLInputElement;
  cmd.onkeydown = (e) => { if (e.key === "Enter" && cmd.value.trim()) { const v = cmd.value; cmd.value = ""; run(v); } };
  cmd.focus();

  const cortex = ndb().hasCortex;
  print("sys", `neuron-db ready · ${ndb().stats(scope)} facts in your memory · cortex ${cortex ? "on" : "off (store-only build)"}`);
  print("sys", `type ` + b("help") + ` to see commands. your memory was seeded with a few facts — try ` + b("recall deadline") + `.`);
}

function out(): HTMLElement { return $("#out"); }
function print(cls: string, html: string) {
  const p = document.createElement("p");
  p.className = "l " + cls; p.innerHTML = html;
  out().appendChild(p); out().scrollTop = out().scrollHeight;
}
const esc = (s: string) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c] as string));
const b = (s: string) => `<b class="ok">${esc(s)}</b>`;

// ───────────────────────── commands ─────────────────────────
async function run(line: string) {
  print("you", esc(line));
  const sp = line.indexOf(" ");
  const cmd = (sp < 0 ? line : line.slice(0, sp)).trim().toLowerCase();
  const arg = sp < 0 ? "" : line.slice(sp + 1).trim();
  const db = ndb();
  try {
    switch (cmd) {
      case "help":
        print("sys",
          [`commands — your memory is the scope ` + b(scope),
           `  ` + b("remember") + ` <fact>        store a durable fact`,
           `  ` + b("recall") + ` <query>         associative recall, with coverage`,
           `  ` + b("ask") + ` <question>         route through the gary-neuron cortex`,
           `  ` + b("assoc") + ` <topic>          spreading-activation recall (linked facts)`,
           `  ` + b("chain") + ` A | rel | rel    walk a relation chain (e.g. chain Aurora | owner | manager)`,
           `  ` + b("forget") + ` <text>          drop facts containing <text>`,
           `  ` + b("stats") + `                  how much you remember`,
           `  ` + b("clear") + ` · ` + b("logout"),
          ].join("\n"));
        break;

      case "remember": case "note": {
        if (!arg) return print("warn", "usage: remember <fact>");
        const n = db.observe(scope, arg); persist(scope);
        print("ok", `stored — ${n} episode(s). you now hold ${db.stats(scope)} facts.`);
        break;
      }

      case "recall": {
        if (!arg) return print("warn", "usage: recall <query>");
        const t = performance.now();
        const hits = db.recallScored(scope, arg, 6);
        const us = ((performance.now() - t) * 1000).toFixed(0);
        if (!hits.length) { print("sys", `nothing recalled for “${esc(arg)}”.`); break; }
        for (const h of hits) print("ans", `• ${esc(h.fact)}  <span class="meta">[coverage ${(h.coverage * 100).toFixed(0)}% · overlap ${h.overlap}]</span>`);
        print("stat", `recalled ${hits.length} in ${us} µs`);
        break;
      }

      case "ask": {
        if (!arg) return print("warn", "usage: ask <question>");
        if (!db.hasCortex) { print("warn", "this build has no cortex — mount a cortex build for ask. (try recall instead)"); break; }
        const t = performance.now();
        const r = db.route(scope, arg, { k: 6 });
        const ms = (performance.now() - t).toFixed(0);
        print("meta", `cortex → ${r.type.toUpperCase()}${r.value ? " · " + esc(r.value) : ""}  <span class="stat">[${ms} ms over ${r.facts.length} recalled]</span>`);
        if (r.type === "answer") print("ans", esc(r.value));
        else if (r.type === "escalate") print("sys", `memory can't answer this — a real app would hand it to your LLM with the ${r.facts.length} recalled fact(s) as context.`);
        else if (r.type === "fetch") print("sys", `a live-world question — a real app would search the web for: ${esc(r.value)}`);
        else if (r.type === "store") { db.observe(scope, arg); persist(scope); print("ok", `recognized a fact to remember — stored it.`); }
        break;
      }

      case "assoc": {
        if (!arg) return print("warn", "usage: assoc <topic>");
        const linked = db.assoc(scope, arg, 8, 2);
        if (!linked.length) { print("sys", `no associations for “${esc(arg)}”.`); break; }
        for (const f of linked) print("ans", `↦ ${esc(f)}`);
        break;
      }

      case "chain": {
        const parts = arg.split("|").map((s) => s.trim()).filter(Boolean);
        if (parts.length < 2) return print("warn", "usage: chain <start> | <relation> | <relation> …");
        const [start, ...path] = parts;
        const r = db.chain(scope, start, path);
        if (r.value) print("ans", `${esc(r.value)}  <span class="meta">[${esc(r.trail)}]</span>`);
        else print("sys", `the chain didn't resolve — a relation may be missing from memory.`);
        break;
      }

      case "forget": {
        if (!arg) return print("warn", "usage: forget <text>");
        const n = db.forget(scope, arg); persist(scope);
        print(n ? "ok" : "sys", n ? `forgot ${n} fact(s).` : `nothing matched “${esc(arg)}”.`);
        break;
      }

      case "stats": {
        const facts = db.stats(scope);
        const blob = db.dump(scope);
        print("sys", `you hold <b class="ok">${facts}</b> facts · ${(blob.length / 1024).toFixed(1)} KB persisted to this browser · scope ${b(scope)}`);
        break;
      }

      case "clear": out().innerHTML = ""; break;
      case "logout": $("#logout").click(); break;
      default: print("err", `unknown command: ${esc(cmd)} — type ${b("help")}`);
    }
  } catch (e) {
    print("err", `error: ${esc(String((e as Error).message || e))}`);
  }
}
