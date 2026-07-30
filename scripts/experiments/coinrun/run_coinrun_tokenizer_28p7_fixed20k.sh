#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-tokenizer-28p7}"
readonly RUN_ROOT="${OPEN_DREAMER_RUN_ROOT:-${ROOT}/logs/coinrun-tokenizer-28p7-fixed20k-20260729}"
readonly DATA_ROOT="/mnt/workspace/datasets/coinrun-structured-v2-20260729"
readonly TRAIN_DATA="${DATA_ROOT}/train"
readonly EVAL_DATA="${DATA_ROOT}/eval"
readonly BASELINE_RUN_ROOT="${OPEN_DREAMER_BASELINE_RUN_ROOT:-/mnt/workspace/open-dreamer-tokenizer-quality-first/logs/coinrun-tokenizer-quality-first-20k-20260729}"
readonly BASELINE_METRICS="${BASELINE_RUN_ROOT}/runs/n16p6m/milestones/updates-20000/heldout_eval.json"
readonly BASELINE_IMAGE="${BASELINE_RUN_ROOT}/runs/n16p6m/milestones/updates-20000/heldout_reconstruction.png"
readonly RUN_DIR="${RUN_ROOT}/runs/n28p7m"
readonly CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
readonly PLAN="${RUN_ROOT}/plan.json"
readonly PROBE="${RUN_ROOT}/tokenizer-28p7-probe.json"
readonly COMPARISON="${RUN_ROOT}/tokenizer-16p6m-vs-28p7m-comparison.json"
readonly WANDB_MODE_EFFECTIVE="${WANDB_MODE:-offline}"

mkdir -p "${RUN_ROOT}/runs"
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
source .venv/bin/activate
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="${HTTP_PROXY}"
export ALL_PROXY="${HTTP_PROXY}"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"
export all_proxy="${ALL_PROXY}"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="${NO_PROXY}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"
export HYDRA_FULL_ERROR="1"

test -s "${TRAIN_DATA}/metadata.json"
test -s "${EVAL_DATA}/metadata.json"
test -s "${BASELINE_METRICS}"
test -s "${BASELINE_IMAGE}"

stage "WRITING_28P7_FIXED_20K_PLAN"
.venv/bin/python scripts/experiments/coinrun/tokenizer_28p7_protocol.py \
  --output "${PLAN}"

stage "PROBING_TOKENIZER_N28P7M"
.venv/bin/python scripts/experiments/coinrun/probe_coinrun_tokenizer_28p7.py \
  --output "${PROBE}"

stage "TRAINING_TOKENIZER_N28P7M_TO_20000 wandb_mode=${WANDB_MODE_EFFECTIVE}"
.venv/bin/python scripts/experiments/run_recorded.py \
  --experiment-id CR-TOK-0004 \
  --run-name n28p7m-seed0 \
  --run-dir "${RUN_DIR}" \
  --experiment-dir "${ROOT}/experiments/CR-TOK-0004" \
  -- \
  .venv/bin/python scripts/train_tokenizer.py \
    dataset=coinrun \
    dataset.array_record_path="${TRAIN_DATA}" \
    dataset.p_include_reward=0.0 \
    dataset.dataloader_cfg.B=128 \
    dataset.dataloader_cfg.short_T=16 \
    dataset.dataloader_cfg.long_T=16 \
    dataset.dataloader_cfg.num_workers=4 \
    dataset.dataloader_cfg.prefetch_buffer_size=4 \
    tokenizer.encoder.n_latents=16 \
    tokenizer.encoder.d_bottleneck=16 \
    tokenizer.encoder.depth=6 \
    tokenizer.encoder.d_model=384 \
    tokenizer.encoder.n_heads=6 \
    tokenizer.encoder.n_kv_heads=1 \
    tokenizer.encoder.time_every=6 \
    tokenizer.encoder.time_layer_offset=5 \
    tokenizer.encoder.mae_p_max=0.9 \
    tokenizer.decoder.depth=6 \
    tokenizer.decoder.d_model=384 \
    tokenizer.decoder.n_heads=6 \
    tokenizer.decoder.n_kv_heads=1 \
    tokenizer.decoder.time_every=6 \
    tokenizer.decoder.time_layer_offset=5 \
    scaling_flops_budget=0.0 \
    scaling_tokens_per_param=0.0 \
    max_steps=20000 \
    tokenizer_loss_type=mse \
    lpips_weight=0.2 \
    optimizer.optimizer_type=muon \
    optimizer.mup_scaling=true \
    ckpt.max_to_keep=6 \
    ckpt.save_interval_steps=1000000 \
    "ckpt.save_on_steps=[2499,4999,9999,14999,19999]" \
    logger.log_every=50 \
    logger.wandb_group=CR-TOK-0004 \
    "logger.wandb_tags=[coinrun,tokenizer,n28p7m,seed0,fixed20k]" \
    logger.wandb_mode="${WANDB_MODE_EFFECTIVE}" \
    use_wandb=true \
    validation.enabled=true \
    validation.dataset_path="${EVAL_DATA}" \
    validation.every_steps=2500 \
    validation.batch_size=8 \
    validation.batches=2 \
    validation.frames=16 \
    validation.seed=4242 \
    validation.max_visual_samples=4 \
    validation.fps=8 \
    visualize_every=0 \
    run_name=coinrun-tokenizer-28p7-fixed20k \
    hydra.run.dir="${RUN_DIR}"

for completed_updates in 2500 5000 10000 20000; do
  checkpoint_step="$((completed_updates - 1))"
  milestone_dir="${RUN_DIR}/milestones/updates-$(printf '%05d' "${completed_updates}")"
  mkdir -p "${milestone_dir}"
  if [[ ! -s "${milestone_dir}/heldout_eval.json" ]]; then
    stage "EVALUATING_TOKENIZER_N28P7M_UPDATES_${completed_updates}"
    .venv/bin/python scripts/experiments/coinrun/eval_coinrun_tokenizer_psnr.py \
      --checkpoint "${CHECKPOINT_DIR}" \
      --checkpoint-step "${checkpoint_step}" \
      --dataset "${EVAL_DATA}" \
      --output "${milestone_dir}/heldout_eval.json" \
      --visualization "${milestone_dir}/heldout_reconstruction.png" \
      --batch-size 32 \
      --frames 16 \
      --batches 16 \
      --seed 4242
  fi
  test -s "${milestone_dir}/heldout_reconstruction.png"
done

stage "COMPARING_TOKENIZER_N16P6M_TO_N28P7M"
.venv/bin/python scripts/experiments/coinrun/compare_tokenizer_extension.py \
  --baseline-name n16.6m \
  --baseline-metrics "${BASELINE_METRICS}" \
  --candidate-name n28.7m \
  --candidate-metrics "${RUN_DIR}/milestones/updates-20000/heldout_eval.json" \
  --output "${COMPARISON}"

.venv/bin/python scripts/experiments/coinrun/build_tokenizer_scale_grid.py \
  --image "n16.6m=${BASELINE_IMAGE}" \
  --image "n28.7m=${RUN_DIR}/milestones/updates-20000/heldout_reconstruction.png" \
  --output "${RUN_ROOT}/tokenizer-16p6m-vs-28p7m.png" \
  --manifest "${RUN_ROOT}/tokenizer-16p6m-vs-28p7m.json"

stage "COMPLETE 28P7_FIXED_20K_EVALUATED awaiting_visual_review dynamics_blocked wandb_mode=${WANDB_MODE_EFFECTIVE}"
