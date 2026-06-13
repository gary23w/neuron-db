#!/usr/bin/env bash
# Compile neuron-db's Rust core (emergence model baked in) to WASM for the Worker.
set -e
cd "$(dirname "$0")"
rustup target add wasm32-unknown-unknown 2>/dev/null || true
( cd ../rust/neuron-core && cargo build --release --lib --target wasm32-unknown-unknown )
cp ../rust/neuron-core/target/wasm32-unknown-unknown/release/neuron_core.wasm ./neuron_core.wasm
echo "built worker/neuron_core.wasm ($(du -h neuron_core.wasm | cut -f1))"
