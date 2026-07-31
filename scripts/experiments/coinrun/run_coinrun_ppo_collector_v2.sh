#!/usr/bin/env bash
set -Eeuo pipefail

: "${OPEN_DREAMER_EXPERIMENT_ID:?Set a newly preregistered experiment ID}"

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-ppo-collector}"
readonly RUN_ROOT="${OPEN_DREAMER_RUN_ROOT:-${ROOT}/logs/coinrun-ppo-collector-v2}"
readonly DATA_ROOT="${OPEN_DREAMER_DATA_ROOT:-/mnt/workspace/datasets/coinrun-ppo-v2}"
readonly PPO_PYTHON="${OPEN_DREAMER_PPO_PYTHON:-${ROOT}/.venv-ppo/bin/python}"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
readonly TRAIN_RUN="${RUN_ROOT}/ppo-train"
readonly CHECKPOINT="${TRAIN_RUN}/checkpoints/env-steps-025165824.msgpack"
readonly FINAL_EVALUATION="${TRAIN_RUN}/final-validation/env-steps-025165824/metrics.json"
readonly QUALITY_GATE="${RUN_ROOT}/policy-quality-gate.json"
readonly WANDB_MODE_EFFECTIVE="${WANDB_MODE:-offline}"

mkdir -p "${RUN_ROOT}" "${DATA_ROOT}"
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
test -f "${ROOT}/experiments/${OPEN_DREAMER_EXPERIMENT_ID}/manifest.json"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"

stage "TRAINING_REFERENCE_ALIGNED_COINRUN_PPO"
"${PPO_PYTHON}" scripts/experiments/run_recorded.py \
  --experiment-id "${OPEN_DREAMER_EXPERIMENT_ID}" \
  --run-name coinrun-ppo-collector-v2-seed0 \
  --run-dir "${TRAIN_RUN}" \
  --experiment-dir "${ROOT}/experiments/${OPEN_DREAMER_EXPERIMENT_ID}" \
  -- \
  "${PPO_PYTHON}" scripts/experiments/coinrun/train_coinrun_ppo.py \
    --run-dir "${TRAIN_RUN}" \
    --run-name coinrun-ppo-collector-v2-seed0 \
    --experiment-id "${OPEN_DREAMER_EXPERIMENT_ID}" \
    --total-env-steps 25165824 \
    --num-envs 64 \
    --rollout-steps 256 \
    --num-minibatches 8 \
    --update-epochs 3 \
    --train-start-level 0 \
    --train-num-levels 200 \
    --reward-normalization-gamma 0.99 \
    --advantage-normalization minibatch \
    --backbone-kernel-init glorot_uniform \
    --eval-start-level 0 \
    --eval-num-levels 0 \
    --evaluation-policy stochastic \
    --checkpoint-every-env-steps 1048576 \
    --evaluation-every-env-steps 1048576 \
    --evaluation-episodes 128 \
    --final-evaluation-episodes 512 \
    --evaluation-envs 16 \
    --evaluation-seed 4242 \
    --evaluation-max-vector-steps 40000 \
    --visual-episodes 4 \
    --visual-max-frames 256 \
    --wandb-mode "${WANDB_MODE_EFFECTIVE}" \
    --wandb-group "${OPEN_DREAMER_EXPERIMENT_ID}"

test -s "${CHECKPOINT}"

stage "VALIDATING_FROZEN_POLICY_QUALITY"
"${PPO_PYTHON}" scripts/experiments/coinrun/validate_coinrun_ppo_quality.py \
  --metrics "${FINAL_EVALUATION}" \
  --output "${QUALITY_GATE}" \
  --expected-env-steps 25165824 \
  --expected-episodes 512 \
  --min-mean-return 8.0 \
  --min-success-rate 0.8
test -s "${QUALITY_GATE}"

stage "COLLECTING_DYNAMICS_TRAIN_TRAJECTORIES"
"${PPO_PYTHON}" scripts/experiments/run_recorded.py \
  --experiment-id "${OPEN_DREAMER_EXPERIMENT_ID}" \
  --run-name coinrun-ppo-v2-dynamics-train \
  --run-dir "${RUN_ROOT}/collection-train-runtime" \
  --experiment-dir "${ROOT}/experiments/${OPEN_DREAMER_EXPERIMENT_ID}" \
  -- \
  "${PPO_PYTHON}" scripts/experiments/coinrun/collect_coinrun_ppo_records.py \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${DATA_ROOT}/train" \
    --run-dir "${RUN_ROOT}/collection-train-runtime" \
    --run-name coinrun-ppo-v2-dynamics-train \
    --records 4096 \
    --frames 64 \
    --envs 32 \
    --records-per-shard 256 \
    --seed 20240 \
    --start-level 0 \
    --num-levels 200 \
    --temperature 1.0 \
    --exploration-epsilon 0.05

stage "COLLECTING_DYNAMICS_EVAL_TRAJECTORIES"
"${PPO_PYTHON}" scripts/experiments/run_recorded.py \
  --experiment-id "${OPEN_DREAMER_EXPERIMENT_ID}" \
  --run-name coinrun-ppo-v2-dynamics-eval \
  --run-dir "${RUN_ROOT}/collection-eval-runtime" \
  --experiment-dir "${ROOT}/experiments/${OPEN_DREAMER_EXPERIMENT_ID}" \
  -- \
  "${PPO_PYTHON}" scripts/experiments/coinrun/collect_coinrun_ppo_records.py \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${DATA_ROOT}/eval" \
    --run-dir "${RUN_ROOT}/collection-eval-runtime" \
    --run-name coinrun-ppo-v2-dynamics-eval \
    --records 512 \
    --frames 64 \
    --envs 32 \
    --records-per-shard 256 \
    --seed 30240 \
    --start-level 10000 \
    --num-levels 500 \
    --temperature 1.0 \
    --exploration-epsilon 0.05

stage "AUDITING_ACTION_CONDITIONED_DATASETS"
"${PPO_PYTHON}" scripts/experiments/coinrun/audit_coinrun_records.py \
  --dataset "${DATA_ROOT}/train" \
  --output "${RUN_ROOT}/train-dataset-audit.json" \
  --records-to-check 4096
"${PPO_PYTHON}" scripts/experiments/coinrun/audit_coinrun_records.py \
  --dataset "${DATA_ROOT}/eval" \
  --output "${RUN_ROOT}/eval-dataset-audit.json" \
  --records-to-check 512
"${PPO_PYTHON}" scripts/experiments/coinrun/verify_coinrun_dataset_pair.py \
  --train "${DATA_ROOT}/train" \
  --eval "${DATA_ROOT}/eval" \
  --train-audit "${RUN_ROOT}/train-dataset-audit.json" \
  --eval-audit "${RUN_ROOT}/eval-dataset-audit.json" \
  --output "${RUN_ROOT}/dataset-pair-audit.json"

stage "COMPLETE PPO_QUALITY_GATE_PASSED TRAJECTORIES_AUDITED WORLD_MODEL_NOT_STARTED"
