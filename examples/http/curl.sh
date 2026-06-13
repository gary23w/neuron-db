#!/usr/bin/env bash
# Talk to a running neuron-db server (start it: `serve neurons.db 8088`).
# If NEURON_DB_KEY is set on the server, export the same value here.
set -euo pipefail
BASE="${BASE:-http://localhost:8088}"
AUTH=(); [ -n "${NEURON_DB_KEY:-}" ] && AUTH=(-H "authorization: Bearer $NEURON_DB_KEY")
H=(-H "content-type: application/json")

# store a fact / converse (turn = store-or-answer)
curl -s "${AUTH[@]}" "${H[@]}" -d '{"message":"the api key is zeta-9931"}' "$BASE/v1/user:42"
# ask for a value
curl -s "${AUTH[@]}" "${H[@]}" -d '{"query":"what is the api key?"}'      "$BASE/v1/user:42/get"
# stats for the scope
curl -s "${AUTH[@]}" "$BASE/v1/user:42"
# forget facts matching a substring
curl -s "${AUTH[@]}" "${H[@]}" -d '{"match":"api key"}'                   "$BASE/v1/user:42/forget"
