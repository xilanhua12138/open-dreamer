#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-ppo-bug-screen}"
readonly RUN_ROOT="${OPEN_DREAMER_RUN_ROOT:-${ROOT}/logs/coinrun-ppo-bug-screen-v1}"
readonly PPO_PYTHON="${OPEN_DREAMER_PPO_PYTHON:-/mnt/workspace/open-dreamer-ppo-collector/.venv-ppo/bin/python}"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
readonly WANDB_MODE_EFFECTIVE="${WANDB_MODE:-offline}"
readonly TOTAL_ENV_STEPS=6291456

mkdir -p "${RUN_ROOT}/arms"
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

run_arm() {
  local arm="$1"
  local train_levels="$2"
  local reward_gamma="$3"
  local advantage_normalization="$4"
  local backbone_init="$5"
  local arm_dir="${RUN_ROOT}/arms/${arm}"
  local final_metrics="${arm_dir}/final-validation/env-steps-006291456/metrics.json"

  if [[ -s "${final_metrics}" ]]; then
    stage "ARM_ALREADY_COMPLETE ${arm}"
    return
  fi

  stage "TRAINING_ARM ${arm}"
  "${PPO_PYTHON}" scripts/experiments/run_recorded.py \
    --experiment-id CR-PPO-0003 \
    --run-name "coinrun-ppo-bug-screen-${arm}" \
    --run-dir "${arm_dir}" \
    --experiment-dir "${ROOT}/experiments/CR-PPO-0003" \
    -- \
    "${PPO_PYTHON}" scripts/experiments/coinrun/train_coinrun_ppo.py \
      --run-dir "${arm_dir}" \
      --run-name "coinrun-ppo-bug-screen-${arm}" \
      --experiment-id CR-PPO-0003 \
      --total-env-steps "${TOTAL_ENV_STEPS}" \
      --num-envs 64 \
      --rollout-steps 256 \
      --num-minibatches 8 \
      --update-epochs 3 \
      --train-start-level 0 \
      --train-num-levels "${train_levels}" \
      --reward-normalization-gamma "${reward_gamma}" \
      --advantage-normalization "${advantage_normalization}" \
      --backbone-kernel-init "${backbone_init}" \
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
      --wandb-group CR-PPO-0003
  test -s "${final_metrics}"
}

run_arm reference 200 0.99 minibatch glorot_uniform
run_arm levels500 500 0.99 minibatch glorot_uniform
run_arm reward_gamma0999 200 0.999 minibatch glorot_uniform
run_arm batch_advantage 200 0.99 batch glorot_uniform
run_arm orthogonal_init 200 0.99 minibatch orthogonal_sqrt2

stage "SUMMARIZING_ONE_FACTOR_CONTROLS"
"${PPO_PYTHON}" scripts/experiments/coinrun/summarize_coinrun_ppo_bug_screen.py \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/comparison.json" \
  --expected-env-steps "${TOTAL_ENV_STEPS}"

stage "COMPLETE FIVE_ONE_FACTOR_ARMS WORLD_MODEL_NOT_STARTED"
