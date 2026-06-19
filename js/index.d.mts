// Type definitions for neuron-db — the host binding over the Rust core compiled to WebAssembly.

/** The binding version (matches the npm package). */
export declare const VERSION: string;

/** The cortex's decision for a turn. */
export type DispatchKind = "answer" | "escalate" | "fetch" | "store";

/** A recalled fact with its per-hit confidence. */
export interface RecallHit {
  fact: string;
  /** 0..1 — share of the query's content covered. */
  coverage: number;
  /** count of overlapping cue terms. */
  overlap: number;
}

/** The knowledge-gap signal for a query (drive escalate/fetch decisions off this). */
export interface AssessResult {
  coverage: number;
  overlap: number;
  exact: number;
  /** how many facts matched the query at all. */
  hits: number;
  hasValue: boolean;
  topFact: string;
}

/** One advertised capability and who owns it (§7 grounded-beats-tier). */
export interface Capability {
  name: string;
  owner: "grounded" | "deferrable";
  about: string;
}

/** A resolved keep/defer disposition for a capability given a host's advertised tools. */
export interface CapResolution {
  name: string;
  disposition: "keep" | "defer";
}

/** A multi-hop relation walk result. */
export interface ChainResult {
  value: string | null;
  /** the resolved path, e.g. "Aurora → Marisol → Dana". */
  trail: string;
}

/** A cortex dispatch decision (no recalled facts attached). */
export interface Dispatch {
  type: DispatchKind;
  /** the answer / topic / fact; empty string for "escalate". */
  value: string;
}

/** A full cortex route: the dispatch decision plus the working set it saw. */
export interface Routed extends Dispatch {
  facts: string[];
}

/** A live scope and its fact count. */
export interface ScopeInfo {
  scope: string;
  count: number;
}

/** Options for instantiating the binding. */
export interface LoadOptions {
  /** Wire the wasm's `host_http` (the `http` feature) so a FETCH route / the `fetch` op can reach the
   *  network. Called with a URL; resolve with the response body. */
  httpHandler?: (url: string) => string | Promise<string>;
  /** Extra WebAssembly imports merged in (advanced; the binding already supplies `env.host_http`). */
  imports?: WebAssembly.Imports;
}

/**
 * A typed, self-validating wrapper over a neuron-db `neuron_core.wasm` build. The cortex (gary-neuron)
 * is the headline: mount a cortex build and use {@link NeuronDB.route} — recall + dispatch in one call.
 */
export declare class NeuronDB {
  /** The wire ops this build exposes (read from the build's `ops` surface at load). */
  readonly ops: Set<string>;
  /** Whether a cortex (gary-neuron) is baked into this build — `route`/`dispatch` are available iff true. */
  readonly hasCortex: boolean;

  /** Wrap already-instantiated wasm exports. Prefer the static constructors. */
  constructor(exports: WebAssembly.Exports);

  /** From a `WebAssembly.Module` (Cloudflare's CompiledWasm import gives you one). Synchronous. */
  static fromModule(mod: WebAssembly.Module, opts?: LoadOptions): NeuronDB;
  /** From wasm bytes (browser fetch, Node `fs.readFile`). Compiles then instantiates. */
  static fromBytes(bytes: BufferSource, opts?: LoadOptions): Promise<NeuronDB>;
  /** Node convenience: read a `.wasm` file and instantiate. */
  static forNode(path: string, opts?: LoadOptions): Promise<NeuronDB>;
  /** Browser / Deno convenience: fetch a `.wasm` URL and instantiate. */
  static forBrowser(url: string | URL, opts?: LoadOptions): Promise<NeuronDB>;

  // ── introspection ──
  /** The wire ops this build exposes. */
  listOps(): string[];
  /** Whether an op is present in this build. */
  supports(op: string): boolean;
  /** Raw escape hatch for an op the typed API doesn't cover. Returns the raw tab/newline string. */
  raw(op: string, ...args: string[]): string;

  // ── write ──
  /** Store one fact (the core may split it into sentence-level episodes). Returns episodes stored. */
  observe(scope: string, fact: string): number;
  /** Bulk-store many facts in one crossing. Returns episodes stored. */
  observeMany(scope: string, facts: string[]): number;

  // ── read ──
  /** Top-k recalled facts for a query, best-first (the robust default — uses recall_many). */
  recall(scope: string, query: string, k?: number): string[];
  /** Recall with per-hit confidence, best-first. */
  recallScored(scope: string, query: string, k?: number): RecallHit[];
  /** A single best-matching value for a direct question, or null. */
  recallValue(scope: string, query: string): string | null;
  /** The knowledge-gap signal: how well memory covers a query. */
  assess(scope: string, query: string): AssessResult;
  /** Spreading-activation recall: facts linked to the query through shared entities. */
  assoc(scope: string, query: string, k?: number, hops?: number): string[];
  /** Walk a relation chain server-side: chain("o","Aurora",["owner","manager"]). */
  chain(scope: string, start: string, path: string[]): ChainResult;

  // ── variables ──
  setVar(scope: string, key: string, value: string): this;
  getVar(scope: string, key: string): string | null;
  vars(scope: string): Record<string, string>;
  delVar(scope: string, key: string): boolean;

  // ── lifecycle ──
  /** Drop facts containing `match` (substring), or the whole scope if omitted. Returns count removed. */
  forget(scope: string, match?: string): number;
  /** Fact count held by a scope. */
  stats(scope: string): number;
  /** Every live scope. */
  scopes(): ScopeInfo[];
  /** Serialize a scope to a portable blob (persist it; restore with `load`). */
  dump(scope: string): string;
  /** Restore a scope from a dump blob (replaces it). Returns the fact count loaded. */
  load(scope: string, blob: string): number;

  // ── capability manifest (§7) ──
  /** neuron's capability manifest. */
  caps(): Capability[];
  /** Resolve keep/defer for a host advertising `hostCaps` (grounded caps always stay local). */
  resolveCaps(hostCaps?: string[]): CapResolution[];

  // ── cortex (only when the build bakes gary-neuron) ──
  /** Route a query through the cortex over a given fact set. Throws on a no-cortex build. */
  dispatch(query: string, facts?: string[]): Dispatch;
  /** The full cortex loop: recall the working set from `scope`, then dispatch. Throws on a no-cortex build. */
  route(scope: string, query: string, opts?: { k?: number }): Routed;
}

export default NeuronDB;
