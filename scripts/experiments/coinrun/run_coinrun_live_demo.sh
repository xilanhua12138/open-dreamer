#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/mnt/workspace/open-dreamer"
readonly EXPERIMENT="${ROOT}/logs/coinrun-dynamics-fixed20k-20260728"
readonly EVAL_DATA="/mnt/workspace/datasets/coinrun-minimal-complete-20260728/eval"
readonly SELECTION="${EXPERIMENT}/best-model.json"
readonly PID_FILE="${EXPERIMENT}/live-demo.pid"
readonly LOG_FILE="${EXPERIMENT}/live-demo.log"

cd "${ROOT}"
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"

if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  printf 'Live demo already running as PID %s\n' "$(cat "${PID_FILE}")"
  exit 0
fi

.venv/bin/python select_best_coinrun_checkpoint.py \
  --experiment "${EXPERIMENT}" \
  --output "${SELECTION}"

model_name="$(
  .venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["name"])' "${SELECTION}"
)"
checkpoint="$(
  .venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["checkpoint"])' "${SELECTION}"
)"

nohup .venv/bin/python live_coinrun_demo.py \
  --checkpoint "${checkpoint}" \
  --dataset "${EVAL_DATA}" \
  --model-name "${model_name}" \
  --context 16 \
  --denoise-steps 4 \
  --host 127.0.0.1 \
  --port 7860 \
  >"${LOG_FILE}" 2>&1 &
printf '%s\n' "$!" > "${PID_FILE}"
printf 'Starting model=%s pid=%s log=%s\n' "${model_name}" "$(cat "${PID_FILE}")" "${LOG_FILE}"
