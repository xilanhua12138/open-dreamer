#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-ppo-combined-control}"
readonly RUN_ROOT="${OPEN_DREAMER_RUN_ROOT:-${ROOT}/logs/coinrun-ppo-combined-control-v1}"
readonly PPO_PYTHON="${OPEN_DREAMER_PPO_PYTHON:-/mnt/workspace/open-dreamer-ppo-collector/.venv-ppo/bin/python}"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
readonly WANDB_MODE_EFFECTIVE="${WANDB_MODE:-offline}"
readonly TOTAL_ENV_STEPS=6291456
readonly HISTORICAL_CHECKPOINT="/mnt/workspace/open-dreamer-ppo-collector/logs/coinrun-ppo-collector-v1/ppo-train/checkpoints/env-steps-006291456.msgpack"
readonly REFERENCE_METRICS="/mnt/workspace/open-dreamer-ppo-bug-screen/logs/coinrun-ppo-bug-screen-v1/arms/reference/final-validation/env-steps-006291456/metrics.json"
readonly HISTORICAL_DIR="${RUN_ROOT}/historical-old-checkpoint-eval"
readonly COMBINED_DIR="${RUN_ROOT}/fresh-all-old-combined"

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
test -s "${HISTORICAL_CHECKPOINT}"
test -s "${REFERENCE_METRICS}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"

if [[ ! -s "${HISTORICAL_DIR}/metrics.json" ]]; then
  stage "REEVALUATING_HISTORICAL_OLD_CHECKPOINT"
  "${PPO_PYTHON}" scripts/experiments/run_recorded.py \
    --experiment-id CR-PPO-0004 \
    --run-name coinrun-ppo-historical-checkpoint-reeval \
    --run-dir "${RUN_ROOT}/historical-eval-runtime" \
    --experiment-dir "${ROOT}/experiments/CR-PPO-0004" \
    -- \
    "${PPO_PYTHON}" scripts/experiments/coinrun/evaluate_coinrun_ppo_checkpoint.py \
      --checkpoint "${HISTORICAL_CHECKPOINT}" \
      --output-dir "${HISTORICAL_DIR}" \
      --episodes 256 \
      --envs 16 \
      --seed 4242 \
      --start-level 0 \
      --num-levels 0 \
      --policy stochastic \
      --max-vector-steps 40000 \
      --visual-episodes 2 \
      --visual-max-frames 256
fi
test -s "${HISTORICAL_DIR}/metrics.json"

if [[ ! -s "${COMBINED_DIR}/final-validation/env-steps-006291456/metrics.json" ]]; then
  stage "TRAINING_FRESH_ALL_OLD_COMBINED"
  "${PPO_PYTHON}" scripts/experiments/run_recorded.py \
    --experiment-id CR-PPO-0004 \
    --run-name coinrun-ppo-fresh-all-old-combined \
    --run-dir "${COMBINED_DIR}" \
    --experiment-dir "${ROOT}/experiments/CR-PPO-0004" \
    -- \
    "${PPO_PYTHON}" scripts/experiments/coinrun/train_coinrun_ppo.py \
      --run-dir "${COMBINED_DIR}" \
      --run-name coinrun-ppo-fresh-all-old-combined \
      --experiment-id CR-PPO-0004 \
      --total-env-steps "${TOTAL_ENV_STEPS}" \
      --num-envs 64 \
      --rollout-steps 256 \
      --num-minibatches 8 \
      --update-epochs 3 \
      --train-start-level 0 \
      --train-num-levels 500 \
      --reward-normalization-gamma 0.999 \
      --advantage-normalization batch \
      --backbone-kernel-init orthogonal_sqrt2 \
      --eval-start-level 0 \
      --eval-num-levels 0 \
      --evaluation-policy stochastic \
      --checkpoint-every-env-steps "${TOTAL_ENV_STEPS}" \
      --evaluation-every-env-steps "${TOTAL_ENV_STEPS}" \
      --evaluation-episodes 128 \
      --final-evaluation-episodes 256 \
      --evaluation-envs 16 \
      --evaluation-seed 4242 \
      --evaluation-max-vector-steps 40000 \
      --visual-episodes 2 \
      --visual-max-frames 256 \
      --wandb-mode "${WANDB_MODE_EFFECTIVE}" \
      --wandb-group CR-PPO-0004
fi
test -s "${COMBINED_DIR}/final-validation/env-steps-006291456/metrics.json"

stage "SUMMARIZING_COMBINED_CONTROL"
"${PPO_PYTHON}" scripts/experiments/coinrun/summarize_coinrun_ppo_combined_control.py \
  --reference "${REFERENCE_METRICS}" \
  --historical "${HISTORICAL_DIR}/metrics.json" \
  --combined "${COMBINED_DIR}/final-validation/env-steps-006291456/metrics.json" \
  --output "${RUN_ROOT}/comparison.json" \
  --expected-env-steps "${TOTAL_ENV_STEPS}"

stage "COMPLETE HISTORICAL_REEVALUATED ALL_OLD_COMBINED_TRAINED WORLD_MODEL_NOT_STARTED"
