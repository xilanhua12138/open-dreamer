#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/mnt/workspace/open-dreamer"
readonly EXPERIMENT="${ROOT}/logs/coinrun-dynamics-fixed20k-20260728"
readonly STATUS="${EXPERIMENT}/STATUS_EXTENSION"
readonly READY="${EXPERIMENT}/LIVE_DEMO_READY.json"
readonly SELECTION="${EXPERIMENT}/best-model.json"

mkdir -p "${EXPERIMENT}"
exec > >(tee -a "${EXPERIMENT}/pipeline-extension.log") 2>&1

stage() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee "${STATUS}"
}

finish() {
  local exit_code=$?
  if [[ "${exit_code}" -ne 0 ]]; then
    stage "FAILED exit_code=${exit_code}"
  fi
}
trap finish EXIT

cd "${ROOT}"
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"

if [[ ! -s "${EXPERIMENT}/eval/large/metrics.json" ]]; then
  stage "LARGE_TRAIN_AND_EVAL"
  bash run_coinrun_large_fixed20k.sh
else
  stage "LARGE_ALREADY_COMPLETE"
fi

stage "SELECT_BEST"
.venv/bin/python select_best_coinrun_checkpoint.py \
  --experiment "${EXPERIMENT}" \
  --output "${SELECTION}"
best_name="$(
  .venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["name"])' "${SELECTION}"
)"
best_checkpoint="$(
  .venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["checkpoint"])' "${SELECTION}"
)"

if [[ ! -s "${EXPERIMENT}/context-ablation-${best_name}/summary.json" ]]; then
  stage "CONTEXT_ABLATION_${best_name^^}"
  bash run_coinrun_context_ablation.sh "${best_checkpoint}" "${best_name}"
else
  stage "CONTEXT_ABLATION_ALREADY_COMPLETE_${best_name^^}"
fi

stage "START_LIVE_DEMO_${best_name^^}"
bash run_coinrun_live_demo.sh

for attempt in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:7860/health > "${EXPERIMENT}/live-demo-health.json"; then
    .venv/bin/python - "${READY}" "${best_name}" "${best_checkpoint}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ready_path, name, checkpoint = sys.argv[1:]
payload = {
    "ready": True,
    "model": name,
    "checkpoint": checkpoint,
    "url_over_ssh_tunnel": "http://127.0.0.1:7860",
    "created_at": datetime.now(timezone.utc).isoformat(),
}
Path(ready_path).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
    stage "DEMO_READY"
    exit 0
  fi
  sleep 5
done

stage "FAILED live_demo_not_ready"
exit 1
