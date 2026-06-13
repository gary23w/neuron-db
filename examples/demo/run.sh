#!/usr/bin/env bash
# Start the real neuron-db server and open the two-pane demo pointed at it.
set -euo pipefail
cd "$(dirname "$0")"
CORE="../../rust/neuron-core"; BIN="$CORE/target/release/serve"
if [ ! -x "$BIN" ]; then
  if command -v cargo >/dev/null 2>&1; then
    echo "building serve (first run)…"; (cd "$CORE" && cargo build --release --features server --bin serve)
  else
    echo "No serve binary and no cargo. Either install Rust (https://rustup.rs) or run the"
    echo "server with Docker from the repo root:  docker compose up -d"; exit 1
  fi
fi
echo "neuron-db -> http://localhost:8088   (db: $PWD/demo.db)"
"$BIN" "$PWD/demo.db" 8088 & SV=$!
sleep 1
(xdg-open index.html 2>/dev/null || open index.html 2>/dev/null || echo "open examples/demo/index.html in your browser")
echo "running (pid $SV). Ctrl-C to stop."; wait $SV
