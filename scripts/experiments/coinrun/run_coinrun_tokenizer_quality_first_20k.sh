#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-tokenizer-quality-first}"
readonly RUN_ROOT="${OPEN_DREAMER_RUN_ROOT:-${ROOT}/logs/coinrun-tokenizer-quality-first-20k-20260729}"
readonly DATA_ROOT="/mnt/workspace/datasets/coinrun-structured-v2-20260729"
readonly TRAIN_DATA="${DATA_ROOT}/train"
readonly EVAL_DATA="${DATA_ROOT}/eval"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
readonly PLAN="${RUN_ROOT}/plan.json"
readonly PROBE="${RUN_ROOT}/tokenizer-quality-first-probe.json"
readonly CURVES="${RUN_ROOT}/learning-curves.json"
readonly SELECTION="${RUN_ROOT}/selection.json"

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

stage "WRITING_FIXED_20K_PLAN"
.venv/bin/python scripts/experiments/coinrun/tokenizer_quality_first_protocol.py \
  --output "${PLAN}"

stage "PROBING_TOKENIZER_SCALES"
.venv/bin/python scripts/experiments/coinrun/probe_coinrun_tokenizer_quality_first.py \
  --output "${PROBE}"

run_candidate() {
  local name="$1"
  local directory_name="$2"
  local depth="$3"
  local d_model="$4"
  local max_steps="$5"
  local run_dir="${RUN_ROOT}/runs/${directory_name}"
  local checkpoint_dir="${run_dir}/checkpoints"

  if [[ ! -d "${checkpoint_dir}/19999" ]]; then
    stage "TRAINING_TOKENIZER_${directory_name^^}_TO_20000"
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
      tokenizer.encoder.depth="${depth}" \
      tokenizer.encoder.d_model="${d_model}" \
      tokenizer.encoder.n_heads="$((d_model / 64))" \
      tokenizer.encoder.n_kv_heads=1 \
      tokenizer.encoder.time_every="${depth}" \
      tokenizer.encoder.time_layer_offset="$((depth - 1))" \
      tokenizer.encoder.mae_p_max=0.9 \
      tokenizer.decoder.depth="${depth}" \
      tokenizer.decoder.d_model="${d_model}" \
      tokenizer.decoder.n_heads="$((d_model / 64))" \
      tokenizer.decoder.n_kv_heads=1 \
      tokenizer.decoder.time_every="${depth}" \
      tokenizer.decoder.time_layer_offset="$((depth - 1))" \
      scaling_flops_budget=0.0 \
      scaling_tokens_per_param=0.0 \
      max_steps="${max_steps}" \
      tokenizer_loss_type=mse \
      lpips_weight=0.2 \
      optimizer.optimizer_type=muon \
      optimizer.mup_scaling=true \
      ckpt.max_to_keep=6 \
      ckpt.save_interval_steps=1000000 \
      "ckpt.save_on_steps=[2499,4999,9999,14999,19999]" \
      logger.log_every=50 \
      visualize_every=0 \
      run_name="coinrun-tokenizer-quality-first-${directory_name}" \
      hydra.run.dir="${run_dir}"
  fi

  local completed_updates
  for completed_updates in 2500 5000 10000 20000; do
    local checkpoint_step="$((completed_updates - 1))"
    local milestone_dir
    milestone_dir="${run_dir}/milestones/updates-$(printf '%05d' "${completed_updates}")"
    mkdir -p "${milestone_dir}"
    if [[ ! -s "${milestone_dir}/heldout_eval.json" ]]; then
      stage "EVALUATING_TOKENIZER_${directory_name^^}_UPDATES_${completed_updates}"
      .venv/bin/python scripts/experiments/coinrun/eval_coinrun_tokenizer_psnr.py \
        --checkpoint "${checkpoint_dir}" \
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
  printf '%s=%s\n' \
    "${name}" \
    "${run_dir}/milestones/updates-20000/heldout_eval.json"
}

while IFS=$'\t' read -r name directory_name depth d_model max_steps; do
  run_candidate "${name}" "${directory_name}" "${depth}" "${d_model}" "${max_steps}"
done < <(
  .venv/bin/python \
    scripts/experiments/coinrun/tokenizer_quality_first_protocol.py \
    --format tsv
)

stage "SUMMARIZING_ALL_MILESTONES"
.venv/bin/python scripts/experiments/coinrun/summarize_tokenizer_quality_first.py \
  --run-root "${RUN_ROOT}" \
  --output "${CURVES}"

.venv/bin/python scripts/experiments/coinrun/select_coinrun_tokenizer_scale.py \
  --probe "${PROBE}" \
  --metrics "n0.17m=${RUN_ROOT}/runs/n0p17m/milestones/updates-20000/heldout_eval.json" \
  --metrics "n1.1m=${RUN_ROOT}/runs/n1p1m/milestones/updates-20000/heldout_eval.json" \
  --metrics "n3.7m=${RUN_ROOT}/runs/n3p7m/milestones/updates-20000/heldout_eval.json" \
  --metrics "n8.6m=${RUN_ROOT}/runs/n8p6m/milestones/updates-20000/heldout_eval.json" \
  --metrics "n16.6m=${RUN_ROOT}/runs/n16p6m/milestones/updates-20000/heldout_eval.json" \
  --output "${SELECTION}"

.venv/bin/python scripts/experiments/coinrun/build_tokenizer_scale_grid.py \
  --image "n0.17m=${RUN_ROOT}/runs/n0p17m/milestones/updates-20000/heldout_reconstruction.png" \
  --image "n1.1m=${RUN_ROOT}/runs/n1p1m/milestones/updates-20000/heldout_reconstruction.png" \
  --image "n3.7m=${RUN_ROOT}/runs/n3p7m/milestones/updates-20000/heldout_reconstruction.png" \
  --image "n8.6m=${RUN_ROOT}/runs/n8p6m/milestones/updates-20000/heldout_reconstruction.png" \
  --image "n16.6m=${RUN_ROOT}/runs/n16p6m/milestones/updates-20000/heldout_reconstruction.png" \
  --output "${RUN_ROOT}/tokenizer-scale-comparison-20k.png" \
  --manifest "${RUN_ROOT}/tokenizer-scale-comparison-20k.json"

selected="$(
  .venv/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["name"])' \
    "${SELECTION}"
)"
stage "COMPLETE ALL_FIVE_FIXED_20K_EVALUATED selected=${selected} awaiting_visual_review dynamics_blocked"
