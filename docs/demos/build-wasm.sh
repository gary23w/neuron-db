#!/usr/bin/env bash
# Build the WASM the in-browser demos load. Unlike the worker build, this enables the
# `semantic` feature so the semantic-space demo gets sem_* exports (and synapse-3d still
# gets syn_*). Output: docs/demos/neuron_core.wasm (superset, ~1.5 MB).
set -e
cd "$(dirname "$0")"
rustup target add wasm32-unknown-unknown 2>/dev/null || true
( cd ../../rust/neuron-core && cargo build --release --lib --target wasm32-unknown-unknown --features semantic )
cp ../../rust/neuron-core/target/wasm32-unknown-unknown/release/neuron_core.wasm ./neuron_core.wasm
echo "built docs/demos/neuron_core.wasm with syn_* + sem_* exports ($(du -h neuron_core.wasm | cut -f1))"
