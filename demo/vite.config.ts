import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 8076,
    // allow Vite to read the linked local package (file:../js) outside the demo root
    fs: { allow: [".."] },
  },
  // the wasm lives in public/ (copied by copy-wasm.mjs) and is served at /neuron_core.wasm
  assetsInclude: ["**/*.wasm"],
  // the binding's Node-only forNode() helper does `import("node:fs")`; the demo uses forBrowser(), so
  // mark node: builtins external to keep the browser build quiet (the dynamic import never runs here).
  build: { rollupOptions: { external: [/^node:/] } },
});
