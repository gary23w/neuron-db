// A demo login flow with neuron-db as the user store. Each account is a variable in the
// `system:users` scope: username -> "salt:iterations:hash" (PBKDF2-SHA256 via Web Crypto). Each
// user's memories live in their own scope, `user:<name>`.
//
// NOTE: this is a DEMO. It is entirely client-side — the "database" lives in the browser, so it is
// not a secure authentication system (no server, no real session, anyone with the page can read the
// stored hashes). It exists to show neuron-db holding records + per-user memory, not to gate secrets.
import { ndb, persist } from "./neuron";

const USERS = "system:users";
const ITER = 100_000;
const enc = new TextEncoder();

function hex(buf: ArrayBuffer): string {
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function randHex(n: number): string {
  const a = new Uint8Array(n);
  crypto.getRandomValues(a);
  return hex(a.buffer);
}
async function derive(pw: string, salt: string, iter: number): Promise<string> {
  const key = await crypto.subtle.importKey("raw", enc.encode(pw), "PBKDF2", false, ["deriveBits"]);
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt: enc.encode(salt), iterations: iter, hash: "SHA-256" },
    key, 256,
  );
  return hex(bits);
}

export type AuthResult = { ok: true } | { ok: false; error: string };

const norm = (u: string) => u.trim().toLowerCase();

export async function signup(username: string, pw: string): Promise<AuthResult> {
  const u = norm(username);
  if (!/^[a-z0-9_.-]{2,32}$/.test(u)) return { ok: false, error: "username: 2-32 chars, [a-z0-9_.-]" };
  if (pw.length < 4) return { ok: false, error: "password must be at least 4 characters" };
  if (ndb().getVar(USERS, u)) return { ok: false, error: "that username is already taken" };
  const salt = randHex(16);
  const h = await derive(pw, salt, ITER);
  ndb().setVar(USERS, u, `${salt}:${ITER}:${h}`);
  persist(USERS);
  return { ok: true };
}

export async function login(username: string, pw: string): Promise<AuthResult> {
  const u = norm(username);
  const rec = ndb().getVar(USERS, u);
  if (!rec) return { ok: false, error: "no such user — sign up first" };
  const [salt, iter, h] = rec.split(":");
  const got = await derive(pw, salt, parseInt(iter, 10) || ITER);
  if (got !== h) return { ok: false, error: "wrong password" };
  return { ok: true };
}

export const userScope = (username: string) => "user:" + norm(username);
export const userCount = () => Object.keys(ndb().vars(USERS)).length;
