import { defineConfig } from "vite";

export default defineConfig({
  // @gary23w/neuron-db is a normal npm dependency (installed under node_modules), and the wasm lives in
  // public/ (copied there by copy-wasm.mjs), served at /neuron_core.wasm — so no fs.allow escape is needed.
  server: { port: 8076 },
  assetsInclude: ["**/*.wasm"],
  // the binding's Node-only forNode() helper does `import("node:fs")`; the demo uses forBrowser(), so
  // mark node: builtins external to keep the browser build quiet (the dynamic import never runs here).
  build: { rollupOptions: { external: [/^node:/] } },
});
