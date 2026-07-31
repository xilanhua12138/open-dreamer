#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-dynamics-repair-reference}"
readonly RUN_ROOT="${OPEN_DREAMER_RUN_ROOT:-${ROOT}/logs/coinrun-dynamics-repair-reference-v1}"
readonly DATA_ROOT="${OPEN_DREAMER_DATA_ROOT:-/mnt/workspace/datasets/coinrun-dynamics-repair-reference-v1}"
readonly PPO_RUN="${OPEN_DREAMER_PPO_RUN:-/mnt/workspace/open-dreamer-ppo-parity/logs/coinrun-ppo-official-parity-v1/ppo-train}"
readonly PPO_CHECKPOINT="${PPO_RUN}/checkpoints/env-steps-025165824.msgpack"
readonly TOKENIZER="${OPEN_DREAMER_TOKENIZER:-/mnt/workspace/open-dreamer-tokenizer-quality-first/logs/coinrun-tokenizer-quality-first-20k-20260729/runs/n16p6m/checkpoints}"
readonly PPO_PYTHON="${OPEN_DREAMER_PPO_PYTHON:-/mnt/workspace/open-dreamer-ppo-collector/.venv-ppo/bin/python}"
readonly DYNAMICS_PYTHON="${OPEN_DREAMER_DYNAMICS_PYTHON:-/mnt/workspace/open-dreamer-tokenizer-quality-first/.venv/bin/python}"
readonly BASELINE_ROOT="${OPEN_DREAMER_BASELINE_ROOT:-/mnt/workspace/open-dreamer-dynamics-checkpoint-mixtures/logs/coinrun-dynamics-checkpoint-mixtures-v1}"
readonly BASELINE_CHECKPOINT="${BASELINE_ROOT}/scale-ablation/runs/final_only-medium/checkpoints"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly TRAIN_DATA="${DATA_ROOT}/train-final-policy-160"
readonly EVAL_DATA="${DATA_ROOT}/eval-final-policy-160"
readonly TRAIN_AUDIT="${RUN_ROOT}/data/train-audit.json"
readonly EVAL_AUDIT="${RUN_ROOT}/data/eval-audit.json"
readonly PAIR_AUDIT="${RUN_ROOT}/data/pair-audit.json"
readonly WINDOW_AUDIT="${RUN_ROOT}/data/reward-windowing-audit.json"
readonly LATENT_STATS="${RUN_ROOT}/latent-stats-n16p6m.json"
readonly TRAIN_RUN="${RUN_ROOT}/training/final-policy-medium-reference-repair"
readonly FINAL_CHECKPOINT="${TRAIN_RUN}/checkpoints/199999/_CHECKPOINT_METADATA"
readonly BASELINE_EVAL="${RUN_ROOT}/baseline/final-only-medium"
readonly REPAIR_EVAL="${RUN_ROOT}/repair/final-policy-medium-reference-repair"
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
test -x "${PPO_PYTHON}"
test -x "${DYNAMICS_PYTHON}"
test -s "${PPO_CHECKPOINT}"
test -f "${TOKENIZER}/19999/_CHECKPOINT_METADATA"
test -f "${BASELINE_CHECKPOINT}/19999/_CHECKPOINT_METADATA"
test -f "${ROOT}/experiments/CR-DYN-0010/manifest.json"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"
export HYDRA_FULL_ERROR="1"

collect_records() {
  local split="$1"
  local output_dir="$2"
  local records="$3"
  local seed="$4"
  local start_level="$5"
  local runtime_dir="${RUN_ROOT}/collection/${split}"

  if [[ -s "${output_dir}/metadata.json" ]]; then
    printf 'Skipping completed %s collection\n' "${split}"
    return
  fi
  if compgen -G "${output_dir}/shard-*.array_record" >/dev/null; then
    printf 'Partial dataset exists without metadata: %s\n' "${output_dir}" >&2
    return 1
  fi

  stage "COLLECTING_${split^^}_FINAL_POLICY_160_FRAME_RECORDS"
  "${PPO_PYTHON}" scripts/experiments/run_recorded.py \
    --experiment-id CR-DYN-0010 \
    --run-name "coinrun-dynamics-repair-${split}-collection" \
    --run-dir "${runtime_dir}" \
    --experiment-dir "${ROOT}/experiments/CR-DYN-0010" \
    -- \
    "${PPO_PYTHON}" scripts/experiments/coinrun/collect_coinrun_ppo_records.py \
      --experiment-id CR-DYN-0010 \
      --checkpoint "${PPO_CHECKPOINT}" \
      --output-dir "${output_dir}" \
      --run-dir "${runtime_dir}" \
      --run-name "coinrun-dynamics-repair-${split}-collection" \
      --records "${records}" \
      --frames 160 \
      --envs 32 \
      --records-per-shard 256 \
      --seed "${seed}" \
      --start-level "${start_level}" \
      --num-levels 500 \
      --temperature 1.0 \
      --exploration-epsilon 0.05
}

audit_records() {
  local dataset="$1"
  local output="$2"
  local records="$3"
  if [[ ! -s "${output}" ]]; then
    "${PPO_PYTHON}" scripts/experiments/coinrun/audit_coinrun_records.py \
      --dataset "${dataset}" \
      --output "${output}" \
      --records-to-check "${records}"
  fi
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
    dataset.array_record_path="${EVAL_DATA}" \
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
  "${DYNAMICS_PYTHON}" scripts/experiments/coinrun/score_coinrun_rollouts.py \
    "${output_dir}/${rollout_type}" \
    --context 16 \
    --output "${metrics}"
  test -s "${metrics}"
}

stage "COLLECTING_FRESH_EPISODE_SAFE_CORPUS"
collect_records train "${TRAIN_DATA}" 4096 50240 0
collect_records eval "${EVAL_DATA}" 512 60240 10000

stage "AUDITING_FRESH_CORPUS_AND_WINDOWING"
audit_records "${TRAIN_DATA}" "${TRAIN_AUDIT}" 4096
audit_records "${EVAL_DATA}" "${EVAL_AUDIT}" 512
if [[ ! -s "${PAIR_AUDIT}" ]]; then
  "${PPO_PYTHON}" scripts/experiments/coinrun/verify_coinrun_dataset_pair.py \
    --train "${TRAIN_DATA}" \
    --eval "${EVAL_DATA}" \
    --train-audit "${TRAIN_AUDIT}" \
    --eval-audit "${EVAL_AUDIT}" \
    --output "${PAIR_AUDIT}"
fi
if [[ ! -s "${WINDOW_AUDIT}" ]]; then
  "${PPO_PYTHON}" scripts/experiments/coinrun/audit_coinrun_reward_windowing.py \
    --dataset "${TRAIN_DATA}" \
    --output "${WINDOW_AUDIT}" \
    --short-window 64 \
    --long-window 128 \
    --p-include-reward 0.5
fi

stage "EVALUATING_REJECTED_BASELINE_ON_FIXED_REPAIR_FUTURES"
generate_rollout_metrics \
  "${BASELINE_CHECKPOINT}" \
  ema_shortcut \
  128 \
  4 \
  "${BASELINE_EVAL}/shortcut" \
  "${BASELINE_EVAL}/shortcut-metrics.json"
if [[ ! -s "${BASELINE_EVAL}/action-conditioning/action-conditioning.json" ]]; then
  "${DYNAMICS_PYTHON}" \
    scripts/experiments/coinrun/eval_coinrun_action_conditioning.py \
    --checkpoint "${BASELINE_CHECKPOINT}" \
    --dataset "${EVAL_DATA}" \
    --output "${BASELINE_EVAL}/action-conditioning" \
    --context 16 \
    --horizon 16 \
    --num-videos 64 \
    --visual-videos 4 \
    --batch-size 4 \
    --seed 4242 \
    --denoise-steps 4
fi

if [[ ! -s "${LATENT_STATS}" ]]; then
  stage "COMPUTING_N16P6M_LATENT_STATS_ON_REPAIR_CORPUS"
  "${DYNAMICS_PYTHON}" \
    scripts/experiments/coinrun/compute_coinrun_latent_stats.py \
    --checkpoint "${TOKENIZER}" \
    --dataset "${TRAIN_DATA}" \
    --output "${LATENT_STATS}" \
    --batch-size 16 \
    --frames 128 \
    --batches 16
fi

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
  stage "TRAINING_REFERENCE_REPAIR_MEDIUM_200K"
  "${DYNAMICS_PYTHON}" scripts/experiments/run_recorded.py \
    --experiment-id CR-DYN-0010 \
    --run-name coinrun-dynamics-reference-repair-medium \
    --run-dir "${TRAIN_RUN}" \
    --experiment-dir "${ROOT}/experiments/CR-DYN-0010" \
    -- \
    "${DYNAMICS_PYTHON}" scripts/train_dynamics.py \
      dataset=coinrun \
      dataset.array_record_path="${TRAIN_DATA}" \
      dataset.validation_array_record_path="${EVAL_DATA}" \
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
      "+dataset.latent_mean=${latent_mean}" \
      "+dataset.latent_std=${latent_std}" \
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
      logger.wandb_group=CR-DYN-0010 \
      optimizer.optimizer_type=muon \
      optimizer.mup_scaling=false \
      lr_schedule.schedule_type=wsd \
      lr_schedule.lr=0.0003 \
      lr_schedule.warmup_ratio=0.05 \
      lr_schedule.decay_ratio=0.1 \
      write_video_every=10000 \
      run_name=coinrun-dynamics-reference-repair-medium \
      hydra.run.dir="${TRAIN_RUN}"
fi
test -f "${FINAL_CHECKPOINT}"

stage "EVALUATING_REFERENCE_REPAIR_SHORTCUT"
generate_rollout_metrics \
  "${TRAIN_RUN}/checkpoints" \
  ema_shortcut \
  128 \
  4 \
  "${REPAIR_EVAL}/shortcut" \
  "${REPAIR_EVAL}/shortcut-metrics.json"

stage "EVALUATING_REFERENCE_REPAIR_FULL_256_STEP_DIFFUSION"
generate_rollout_metrics \
  "${TRAIN_RUN}/checkpoints" \
  ema_diffusion \
  8 \
  2 \
  "${REPAIR_EVAL}/diffusion-256-step" \
  "${REPAIR_EVAL}/diffusion-256-step-metrics.json"

stage "EVALUATING_REFERENCE_REPAIR_ACTION_CONDITIONING"
if [[ ! -s "${REPAIR_EVAL}/action-conditioning/action-conditioning.json" ]]; then
  "${DYNAMICS_PYTHON}" \
    scripts/experiments/coinrun/eval_coinrun_action_conditioning.py \
    --checkpoint "${TRAIN_RUN}/checkpoints" \
    --dataset "${EVAL_DATA}" \
    --output "${REPAIR_EVAL}/action-conditioning" \
    --context 16 \
    --horizon 16 \
    --num-videos 64 \
    --visual-videos 4 \
    --batch-size 4 \
    --seed 4242 \
    --denoise-steps 4
fi

stage "WRITING_PREREGISTERED_REPAIR_ASSESSMENT"
"${DYNAMICS_PYTHON}" \
  scripts/experiments/coinrun/summarize_coinrun_dynamics_repair.py \
  --baseline-shortcut "${BASELINE_EVAL}/shortcut-metrics.json" \
  --baseline-actions \
    "${BASELINE_EVAL}/action-conditioning/action-conditioning.json" \
  --repair-shortcut "${REPAIR_EVAL}/shortcut-metrics.json" \
  --repair-diffusion "${REPAIR_EVAL}/diffusion-256-step-metrics.json" \
  --repair-actions \
    "${REPAIR_EVAL}/action-conditioning/action-conditioning.json" \
  --output "${RUN_ROOT}/repair-assessment.json"

stage "COMPLETE REFERENCE_REPAIR_EVALUATED VISUAL_REVIEW_REQUIRED"
