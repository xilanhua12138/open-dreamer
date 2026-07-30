#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-ppo-initializer-mitigation}"
readonly RUN_ROOT="${OPEN_DREAMER_RUN_ROOT:-${ROOT}/logs/coinrun-ppo-initializer-mitigation-v1}"
readonly PPO_PYTHON="${OPEN_DREAMER_PPO_PYTHON:-/mnt/workspace/open-dreamer-ppo-collector/.venv-ppo/bin/python}"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
readonly WANDB_MODE_EFFECTIVE="${WANDB_MODE:-offline}"
readonly TOTAL_ENV_STEPS=6291456
readonly DEPTH_SCALE=0.4082482904638631

mkdir -p "${RUN_ROOT}/arms" "${RUN_ROOT}/probe"
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

if [[ ! -s "${RUN_ROOT}/probe/initialization-probe.json" ]]; then
  stage "PROBING_INITIAL_SIGNAL_AND_GRADIENT_PROPAGATION"
  "${PPO_PYTHON}" scripts/experiments/run_recorded.py \
    --experiment-id CR-PPO-0005 \
    --run-name coinrun-ppo-initialization-probe \
    --run-dir "${RUN_ROOT}/probe" \
    --experiment-dir "${ROOT}/experiments/CR-PPO-0005" \
    -- \
    "${PPO_PYTHON}" scripts/experiments/coinrun/probe_coinrun_ppo_initialization.py \
      --output "${RUN_ROOT}/probe/initialization-probe.json" \
      --num-envs 8 \
      --observation-vector-steps 3 \
      --observation-seed 4242 \
      --num-initialization-seeds 16
fi
test -s "${RUN_ROOT}/probe/initialization-probe.json"

run_arm() {
  local arm="$1"
  local backbone_init="$2"
  local residual_scale="$3"
  local last_kernel_init="$4"
  local use_skip_init="$5"
  local arm_dir="${RUN_ROOT}/arms/${arm}"
  local final_metrics="${arm_dir}/final-validation/env-steps-006291456/metrics.json"
  local skip_init_args=()

  if [[ -s "${final_metrics}" ]]; then
    stage "ARM_ALREADY_COMPLETE ${arm}"
    return
  fi
  if [[ "${use_skip_init}" == "true" ]]; then
    skip_init_args+=(--residual-skip-init)
  fi

  stage "TRAINING_ARM ${arm}"
  "${PPO_PYTHON}" scripts/experiments/run_recorded.py \
    --experiment-id CR-PPO-0005 \
    --run-name "coinrun-ppo-initializer-mitigation-${arm}" \
    --run-dir "${arm_dir}" \
    --experiment-dir "${ROOT}/experiments/CR-PPO-0005" \
    -- \
    "${PPO_PYTHON}" scripts/experiments/coinrun/train_coinrun_ppo.py \
      --run-dir "${arm_dir}" \
      --run-name "coinrun-ppo-initializer-mitigation-${arm}" \
      --experiment-id CR-PPO-0005 \
      --total-env-steps "${TOTAL_ENV_STEPS}" \
      --num-envs 64 \
      --rollout-steps 256 \
      --num-minibatches 8 \
      --update-epochs 3 \
      --train-start-level 0 \
      --train-num-levels 200 \
      --reward-normalization-gamma 0.99 \
      --advantage-normalization minibatch \
      --backbone-kernel-init "${backbone_init}" \
      --residual-branch-scale "${residual_scale}" \
      --residual-last-kernel-init "${last_kernel_init}" \
      "${skip_init_args[@]}" \
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
      --wandb-group CR-PPO-0005
  test -s "${final_metrics}"
}

run_arm orthogonal_gain1 orthogonal_gain1 1.0 same false
run_arm orthogonal_sqrt2_depth_scaled orthogonal_sqrt2 "${DEPTH_SCALE}" same false
run_arm orthogonal_sqrt2_zero_last orthogonal_sqrt2 1.0 zeros false
run_arm orthogonal_sqrt2_skipinit orthogonal_sqrt2 1.0 same true

stage "SUMMARIZING_INITIALIZER_MITIGATIONS"
"${PPO_PYTHON}" scripts/experiments/coinrun/summarize_coinrun_ppo_initializer_mitigation.py \
  --run-root "${RUN_ROOT}" \
  --anchor-comparison "${ROOT}/experiments/CR-PPO-0003/raw/comparison.json" \
  --output "${RUN_ROOT}/comparison.json" \
  --plot "${RUN_ROOT}/comparison.png" \
  --expected-env-steps "${TOTAL_ENV_STEPS}"

stage "COMPLETE INITIALIZATION_PROBE FOUR_MITIGATION_ARMS WORLD_MODEL_NOT_STARTED"
