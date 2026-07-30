#!/usr/bin/env bash
set -euo pipefail

readonly DEMO_URL="${DEMO_URL:-http://127.0.0.1:7860}"
readonly TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

curl -fsS --max-time 30 \
  -H 'Content-Type: application/json' \
  -d '{}' \
  "${DEMO_URL}/api/reset" >"${TMP_DIR}/reset.json"

curl -fsS --max-time 60 \
  -H 'Content-Type: application/json' \
  -d '{"action":7}' \
  "${DEMO_URL}/api/step" >"${TMP_DIR}/step-1.json"

curl -fsS --max-time 60 \
  -H 'Content-Type: application/json' \
  -d '{"action":8}' \
  "${DEMO_URL}/api/step" >"${TMP_DIR}/step-2.json"

jq -e '.step == 1 and .action_id == 7 and
  (.frame | startswith("data:image/png;base64,"))' \
  "${TMP_DIR}/step-1.json" >/dev/null
jq -e '.step == 2 and .action_id == 8 and
  (.frame | startswith("data:image/png;base64,"))' \
  "${TMP_DIR}/step-2.json" >/dev/null

echo "PASS: reset followed by two consecutive model steps"
