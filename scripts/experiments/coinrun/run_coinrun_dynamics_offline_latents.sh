#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-dynamics-offline-latents}"
readonly RUN_ROOT="${OPEN_DREAMER_RUN_ROOT:-${ROOT}/logs/coinrun-dynamics-offline-latents-v1}"
readonly RAW_ROOT="${OPEN_DREAMER_RAW_ROOT:-/mnt/workspace/datasets/coinrun-dynamics-repair-reference-v1}"
readonly DATA_ROOT="${OPEN_DREAMER_DATA_ROOT:-/mnt/workspace/datasets/coinrun-dynamics-offline-latents-v1}"
readonly TOKENIZER="${OPEN_DREAMER_TOKENIZER:-/mnt/workspace/open-dreamer-tokenizer-quality-first/logs/coinrun-tokenizer-quality-first-20k-20260729/runs/n16p6m/checkpoints}"
readonly DYNAMICS_PYTHON="${OPEN_DREAMER_DYNAMICS_PYTHON:-/mnt/workspace/open-dreamer-tokenizer-quality-first/.venv/bin/python}"
readonly REPAIR_ROOT="${OPEN_DREAMER_REPAIR_ROOT:-/mnt/workspace/open-dreamer-dynamics-repair-reference/logs/coinrun-dynamics-repair-reference-v1}"
readonly BASELINE_EVAL="${REPAIR_ROOT}/baseline/final-only-medium"
readonly LATENT_STATS="${REPAIR_ROOT}/latent-stats-n16p6m.json"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly RAW_TRAIN="${RAW_ROOT}/train-final-policy-160"
readonly RAW_EVAL="${RAW_ROOT}/eval-final-policy-160"
readonly LATENT_TRAIN="${DATA_ROOT}/train-final-policy-160-n16p6m"
readonly LATENT_EVAL="${DATA_ROOT}/eval-final-policy-160-n16p6m"
readonly TRAIN_AUDIT="${RUN_ROOT}/data/train-latent-audit.json"
readonly EVAL_AUDIT="${RUN_ROOT}/data/eval-latent-audit.json"
readonly TRAIN_RUN="${RUN_ROOT}/training/final-policy-medium-offline-latents"
readonly FINAL_CHECKPOINT="${TRAIN_RUN}/checkpoints/199999/_CHECKPOINT_METADATA"
readonly REPAIR_EVAL="${RUN_ROOT}/repair/final-policy-medium-offline-latents"
readonly USE_WANDB="${OPEN_DREAMER_USE_WANDB:-false}"
readonly MAX_STEPS=200000

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/data" "${DATA_ROOT}"
exec > >(tee -a "${RUN_ROOT}/pipeline.log") 2>&1

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
test -x "${DYNAMICS_PYTHON}"
test -f "${TOKENIZER}/19999/_CHECKPOINT_METADATA"
test -s "${RAW_TRAIN}/metadata.json"
test -s "${RAW_EVAL}/metadata.json"
test -s "${LATENT_STATS}"
test -s "${BASELINE_EVAL}/shortcut-metrics.json"
test -s "${BASELINE_EVAL}/action-conditioning/action-conditioning.json"
test -f "${ROOT}/experiments/CR-DYN-0011/manifest.json"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"
export HYDRA_FULL_ERROR="1"

tokenize_split() {
  local split="$1"
  local raw_dataset="$2"
  local latent_dataset="$3"
  local records="$4"

  if [[ -s "${latent_dataset}/metadata.json" ]]; then
    printf 'Skipping completed %s latent tokenization\n' "${split}"
    return
  fi
  stage "TOKENIZING_${split^^}_COINRUN_RECORDS_OFFLINE"
  "${DYNAMICS_PYTHON}" \
    scripts/experiments/coinrun/tokenize_coinrun_dataset.py \
    --checkpoint "${TOKENIZER}" \
    --input-dir "${raw_dataset}" \
    --output-dir "${latent_dataset}" \
    --split "${split}" \
    --expected-records "${records}" \
    --expected-frames 160 \
    --batch-records 4 \
    --records-per-shard 256
}

audit_split() {
  local raw_dataset="$1"
  local latent_dataset="$2"
  local output="$3"
  if [[ ! -s "${output}" ]]; then
    "${DYNAMICS_PYTHON}" \
      scripts/experiments/coinrun/audit_coinrun_latent_dataset.py \
      --raw-dataset "${raw_dataset}" \
      --latent-dataset "${latent_dataset}" \
      --output "${output}" \
      --records-to-check 0
  fi
  "${DYNAMICS_PYTHON}" -c \
    'import json,sys; assert json.load(open(sys.argv[1]))["valid"] is True' \
    "${output}"
}

generate_rollout_metrics() {
  local checkpoint="$1"
  local rollout_type="$2"
  local num_videos="$3"
  local batch_size="$4"
  local output_dir="$5"
  local metrics="$6"

  if [[ -s "${metrics}" ]]; then
    printf 'Skipping completed rollout metrics %s\n' "${metrics}"
    return
  fi
  if compgen -G "${output_dir}/${rollout_type}/pred_*.mp4" >/dev/null; then
    printf 'Partial rollout exists without metrics: %s\n' "${output_dir}" >&2
    return 1
  fi
  "${DYNAMICS_PYTHON}" scripts/eval_fvd.py \
    dataset=coinrun \
    dataset.array_record_path="${RAW_EVAL}" \
    dataset.p_include_reward=0.0 \
    dataset.dataloader_cfg.B="${batch_size}" \
    dataset.dataloader_cfg.short_T=32 \
    dataset.dataloader_cfg.long_T=32 \
    dataset.dataloader_cfg.long_ratio=0.0 \
    dataset.dataloader_cfg.num_workers=0 \
    dynamics_ckpt="${checkpoint}" \
    mode=generate \
    rollout_type="${rollout_type}" \
    num_videos="${num_videos}" \
    ctx_length=16 \
    horizon=16 \
    seed=4242 \
    video_dir="${output_dir}"
  "${DYNAMICS_PYTHON}" \
    scripts/experiments/coinrun/score_coinrun_rollouts.py \
    "${output_dir}/${rollout_type}" \
    --context 16 \
    --output "${metrics}"
  test -s "${metrics}"
}

tokenize_split train "${RAW_TRAIN}" "${LATENT_TRAIN}" 4096
tokenize_split eval "${RAW_EVAL}" "${LATENT_EVAL}" 512

stage "AUDITING_OFFLINE_LATENT_ALIGNMENT"
audit_split "${RAW_TRAIN}" "${LATENT_TRAIN}" "${TRAIN_AUDIT}"
audit_split "${RAW_EVAL}" "${LATENT_EVAL}" "${EVAL_AUDIT}"

if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
  latent_mean="$(
    "${DYNAMICS_PYTHON}" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["mean"])' \
      "${LATENT_STATS}"
  )"
  latent_std="$(
    "${DYNAMICS_PYTHON}" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["std"])' \
      "${LATENT_STATS}"
  )"
  stage "TRAINING_MEDIUM_200K_FROM_OFFLINE_LATENTS"
  "${DYNAMICS_PYTHON}" scripts/experiments/run_recorded.py \
    --experiment-id CR-DYN-0011 \
    --run-name coinrun-dynamics-medium-offline-latents \
    --run-dir "${TRAIN_RUN}" \
    --experiment-dir "${ROOT}/experiments/CR-DYN-0011" \
    -- \
    "${DYNAMICS_PYTHON}" scripts/train_dynamics.py \
      dataset=coinrun_latent \
      dataset.array_record_path="${LATENT_TRAIN}" \
      dataset.index_max=16 \
      dataset.validation_array_record_path="${LATENT_EVAL}" \
      dataset.validation_index_max=2 \
      dataset.validation_seed=4242 \
      dataset.validation_batch_size=4 \
      dataset.validation_sequence_length=32 \
      dataset.p_include_reward=0.5 \
      dataset.dataloader_cfg.B=16 \
      dataset.dataloader_cfg.short_T=64 \
      dataset.dataloader_cfg.long_T=128 \
      dataset.dataloader_cfg.long_ratio=0.1 \
      dataset.dataloader_cfg.num_workers=4 \
      dataset.dataloader_cfg.prefetch_buffer_size=4 \
      "dataset.latent_mean=${latent_mean}" \
      "dataset.latent_std=${latent_std}" \
      tokenizer_ckpt="${TOKENIZER}" \
      dynamics.d_bottleneck=16 \
      dynamics.depth=4 \
      dynamics.d_model=256 \
      dynamics.n_heads=4 \
      dynamics.n_kv_heads=1 \
      dynamics.packing_factor=2 \
      dynamics.n_register=32 \
      dynamics.qk_norm_type=qknorm \
      dynamics.time_every=2 \
      dynamics.time_layer_offset=1 \
      dynamics.k_max=256 \
      dynamics.context_length=128 \
      periodic_eval_context_frames=16 \
      periodic_eval_include_diffusion=false \
      max_steps="${MAX_STEPS}" \
      bootstrap_start=100000 \
      bootstrap_fraction=0.25 \
      image_fraction=0.0 \
      ot.enabled=true \
      ot.pairing=barycentric \
      loss_weighting=v_space \
      scaling_flops_budget=0 \
      scaling_tokens_per_param=0 \
      ckpt.max_to_keep=9 \
      ckpt.save_interval_steps=25000 \
      logger.log_every=50 \
      use_wandb="${USE_WANDB}" \
      logger.wandb_group=CR-DYN-0011 \
      optimizer.optimizer_type=muon \
      optimizer.mup_scaling=false \
      lr_schedule.schedule_type=wsd \
      lr_schedule.lr=0.0003 \
      lr_schedule.warmup_ratio=0.05 \
      lr_schedule.decay_ratio=0.1 \
      write_video_every=10000 \
      run_name=coinrun-dynamics-medium-offline-latents \
      hydra.run.dir="${TRAIN_RUN}"
fi
test -f "${FINAL_CHECKPOINT}"

stage "EVALUATING_OFFLINE_LATENT_REPAIR_SHORTCUT"
generate_rollout_metrics \
  "${TRAIN_RUN}/checkpoints" \
  ema_shortcut \
  128 \
  4 \
  "${REPAIR_EVAL}/shortcut" \
  "${REPAIR_EVAL}/shortcut-metrics.json"

stage "EVALUATING_OFFLINE_LATENT_REPAIR_FULL_256_STEP_DIFFUSION"
generate_rollout_metrics \
  "${TRAIN_RUN}/checkpoints" \
  ema_diffusion \
  8 \
  2 \
  "${REPAIR_EVAL}/diffusion-256-step" \
  "${REPAIR_EVAL}/diffusion-256-step-metrics.json"

stage "EVALUATING_OFFLINE_LATENT_REPAIR_ACTION_CONDITIONING"
if [[ ! -s "${REPAIR_EVAL}/action-conditioning/action-conditioning.json" ]]; then
  "${DYNAMICS_PYTHON}" \
    scripts/experiments/coinrun/eval_coinrun_action_conditioning.py \
    --checkpoint "${TRAIN_RUN}/checkpoints" \
    --dataset "${RAW_EVAL}" \
    --output "${REPAIR_EVAL}/action-conditioning" \
    --context 16 \
    --horizon 16 \
    --num-videos 64 \
    --visual-videos 4 \
    --batch-size 4 \
    --seed 4242 \
    --denoise-steps 4
fi

stage "WRITING_OFFLINE_LATENT_REPAIR_ASSESSMENT"
"${DYNAMICS_PYTHON}" \
  scripts/experiments/coinrun/summarize_coinrun_dynamics_repair.py \
  --experiment-id CR-DYN-0011 \
  --repair-arm final-policy-medium-offline-latents \
  --baseline-shortcut "${BASELINE_EVAL}/shortcut-metrics.json" \
  --baseline-actions \
    "${BASELINE_EVAL}/action-conditioning/action-conditioning.json" \
  --repair-shortcut "${REPAIR_EVAL}/shortcut-metrics.json" \
  --repair-diffusion "${REPAIR_EVAL}/diffusion-256-step-metrics.json" \
  --repair-actions \
    "${REPAIR_EVAL}/action-conditioning/action-conditioning.json" \
  --output "${RUN_ROOT}/repair-assessment.json"

stage "COMPLETE OFFLINE_LATENT_REPAIR_EVALUATED VISUAL_REVIEW_REQUIRED"
