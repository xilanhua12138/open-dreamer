#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-ppo-parity}"
readonly RUN_ROOT="${OPEN_DREAMER_RUN_ROOT:-${ROOT}/logs/coinrun-ppo-official-parity-v1}"
readonly PPO_PYTHON="${OPEN_DREAMER_PPO_PYTHON:-/mnt/workspace/open-dreamer-ppo-collector/.venv-ppo/bin/python}"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
readonly TRAIN_RUN="${RUN_ROOT}/ppo-train"
readonly CHECKPOINT="${TRAIN_RUN}/checkpoints/env-steps-025165824.msgpack"
readonly WANDB_MODE_EFFECTIVE="${WANDB_MODE:-offline}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

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
test -x "${PPO_PYTHON}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"

stage "TRAINING_OFFICIAL_RECIPE_PARITY_CONTROL"
"${PPO_PYTHON}" scripts/experiments/run_recorded.py \
  --experiment-id CR-PPO-0002 \
  --run-name coinrun-ppo-official-parity-seed0 \
  --run-dir "${TRAIN_RUN}" \
  --experiment-dir "${ROOT}/experiments/CR-PPO-0002" \
  -- \
  "${PPO_PYTHON}" scripts/experiments/coinrun/train_coinrun_ppo.py \
    --experiment-id CR-PPO-0002 \
    --run-dir "${TRAIN_RUN}" \
    --run-name coinrun-ppo-official-parity-seed0 \
    --total-env-steps 25165824 \
    --num-envs 64 \
    --rollout-steps 256 \
    --num-minibatches 8 \
    --update-epochs 3 \
    --learning-rate 0.0005 \
    --gamma 0.999 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-clip-epsilon 0.2 \
    --value-coefficient 0.5 \
    --entropy-coefficient 0.01 \
    --max-grad-norm 0.5 \
    --reward-normalization-gamma 0.99 \
    --advantage-normalization minibatch \
    --backbone-kernel-init glorot_uniform \
    --train-start-level 0 \
    --train-num-levels 200 \
    --eval-start-level 0 \
    --eval-num-levels 0 \
    --distribution-mode easy \
    --checkpoint-every-env-steps 1048576 \
    --evaluation-every-env-steps 1048576 \
    --evaluation-episodes 128 \
    --final-evaluation-episodes 512 \
    --evaluation-envs 16 \
    --evaluation-seed 4242 \
    --evaluation-policy stochastic \
    --evaluation-max-vector-steps 40000 \
    --visual-episodes 4 \
    --visual-max-frames 256 \
    --wandb-mode "${WANDB_MODE_EFFECTIVE}" \
    --wandb-group CR-PPO-0002

test -s "${CHECKPOINT}"
test -s "${TRAIN_RUN}/final-validation/env-steps-025165824/metrics.json"
stage "COMPLETE OFFICIAL_RECIPE_PARITY_CONTROL WORLD_MODEL_NOT_STARTED"
