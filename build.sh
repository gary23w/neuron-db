#!/usr/bin/env bash
# Build neuron-db (Rust). Friendly one-shot: checks the toolchain, builds, points you at it.
set -euo pipefail
cd "$(dirname "$0")/rust/neuron-core"
if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust/cargo not found. Install it from https://rustup.rs then re-run ./build.sh"; exit 1
fi
FEATURES="${FEATURES:-sqlite secure server trust}"
echo "==> building neuron-core (features: $FEATURES)"
cargo build --release --features "$FEATURES"
BIN="$(pwd)/target/release"
cat <<EOF

  built. binaries:
    $BIN/neuron   CLI over a local SQLite db
    $BIN/serve    HTTP server

  try it:
    $BIN/neuron --db demo.db turn me 'my plan is pro'
    $BIN/neuron --db demo.db get  me 'what plan am i on?'

  install onto your PATH (optional):
    cargo install --path "$(pwd)" --features "$FEATURES"
EOF
