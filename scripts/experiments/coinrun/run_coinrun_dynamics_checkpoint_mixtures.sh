#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-dynamics-checkpoint-mixtures}"
readonly RUN_ROOT="${OPEN_DREAMER_RUN_ROOT:-${ROOT}/logs/coinrun-dynamics-checkpoint-mixtures-v1}"
readonly DATA_ROOT="${OPEN_DREAMER_DATA_ROOT:-/mnt/workspace/datasets/coinrun-ppo-checkpoint-mixtures-v1}"
readonly PPO_RUN="${OPEN_DREAMER_PPO_RUN:-/mnt/workspace/open-dreamer-ppo-parity/logs/coinrun-ppo-official-parity-v1/ppo-train}"
readonly TOKENIZER="${OPEN_DREAMER_TOKENIZER:-/mnt/workspace/open-dreamer-tokenizer-quality-first/logs/coinrun-tokenizer-quality-first-20k-20260729/runs/n16p6m/checkpoints}"
readonly PPO_PYTHON="${OPEN_DREAMER_PPO_PYTHON:-/mnt/workspace/open-dreamer-ppo-collector/.venv-ppo/bin/python}"
readonly DYNAMICS_PYTHON="${OPEN_DREAMER_DYNAMICS_PYTHON:-/mnt/workspace/open-dreamer-tokenizer-quality-first/.venv/bin/python}"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly LATENT_STATS="${RUN_ROOT}/latent-stats-n16p6m.json"
readonly CORPUS_AUDIT="${RUN_ROOT}/checkpoint-mixture-corpus-audit.json"
readonly SOURCE_ROOT="${DATA_ROOT}/source-pools/train"
readonly MIXTURE_ROOT="${DATA_ROOT}/mixtures"
readonly EVAL_DATA="${DATA_ROOT}/eval-final-policy"
readonly USE_WANDB="${OPEN_DREAMER_USE_WANDB:-true}"
readonly MAX_STEPS=20000

mkdir -p \
  "${RUN_ROOT}" \
  "${SOURCE_ROOT}" \
  "${MIXTURE_ROOT}" \
  "${DATA_ROOT}/audits/source-pools" \
  "${DATA_ROOT}/audits/mixtures"
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
test -f "${ROOT}/experiments/CR-DYN-0008/manifest.json"
test -f "${ROOT}/experiments/CR-DYN-0009/manifest.json"
test -f "${TOKENIZER}/19999/_CHECKPOINT_METADATA"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"
export HYDRA_FULL_ERROR="1"

checkpoint_path() {
  printf '%s/checkpoints/env-steps-%09d.msgpack\n' "${PPO_RUN}" "$1"
}

collect_training_source() {
  local stage_name="$1"
  local env_steps="$2"
  local checkpoint
  checkpoint="$(checkpoint_path "${env_steps}")"
  local output_dir="${SOURCE_ROOT}/${stage_name}"
  local runtime_dir="${RUN_ROOT}/collection/${stage_name}"
  local audit_path="${DATA_ROOT}/audits/source-pools/${stage_name}.json"

  test -s "${checkpoint}"
  if [[ ! -s "${output_dir}/metadata.json" ]]; then
    if compgen -G "${output_dir}/shard-*.array_record" >/dev/null; then
      printf 'Partial dataset exists without metadata: %s\n' "${output_dir}" >&2
      return 1
    fi
    stage "COLLECTING_SOURCE_${stage_name^^}"
    "${PPO_PYTHON}" scripts/experiments/run_recorded.py \
      --experiment-id CR-DYN-0008 \
      --run-name "coinrun-mixture-source-${stage_name}" \
      --run-dir "${runtime_dir}" \
      --experiment-dir "${ROOT}/experiments/CR-DYN-0008" \
      -- \
      "${PPO_PYTHON}" scripts/experiments/coinrun/collect_coinrun_ppo_records.py \
        --experiment-id CR-DYN-0008 \
        --checkpoint "${checkpoint}" \
        --output-dir "${output_dir}" \
        --run-dir "${runtime_dir}" \
        --run-name "coinrun-mixture-source-${stage_name}" \
        --records 2048 \
        --frames 64 \
        --envs 32 \
        --records-per-shard 256 \
        --seed 20240 \
        --start-level 0 \
        --num-levels 200 \
        --temperature 1.0 \
        --exploration-epsilon 0.05
  fi

  if [[ ! -s "${audit_path}" ]]; then
    stage "AUDITING_SOURCE_${stage_name^^}"
    "${PPO_PYTHON}" scripts/experiments/coinrun/audit_coinrun_records.py \
      --dataset "${output_dir}" \
      --output "${audit_path}" \
      --records-to-check 2048
  fi
}

collect_evaluation_data() {
  local checkpoint
  checkpoint="$(checkpoint_path 25165824)"
  local runtime_dir="${RUN_ROOT}/collection/eval-final-policy"
  local audit_path="${DATA_ROOT}/audits/eval-final-policy.json"

  test -s "${checkpoint}"
  if [[ ! -s "${EVAL_DATA}/metadata.json" ]]; then
    if compgen -G "${EVAL_DATA}/shard-*.array_record" >/dev/null; then
      printf 'Partial dataset exists without metadata: %s\n' "${EVAL_DATA}" >&2
      return 1
    fi
    stage "COLLECTING_FIXED_FINAL_POLICY_EVAL"
    "${PPO_PYTHON}" scripts/experiments/run_recorded.py \
      --experiment-id CR-DYN-0008 \
      --run-name coinrun-mixture-eval-final-policy \
      --run-dir "${runtime_dir}" \
      --experiment-dir "${ROOT}/experiments/CR-DYN-0008" \
      -- \
      "${PPO_PYTHON}" scripts/experiments/coinrun/collect_coinrun_ppo_records.py \
        --experiment-id CR-DYN-0008 \
        --checkpoint "${checkpoint}" \
        --output-dir "${EVAL_DATA}" \
        --run-dir "${runtime_dir}" \
        --run-name coinrun-mixture-eval-final-policy \
        --records 512 \
        --frames 64 \
        --envs 32 \
        --records-per-shard 256 \
        --seed 30240 \
        --start-level 10000 \
        --num-levels 500 \
        --temperature 1.0 \
        --exploration-epsilon 0.05
  fi

  if [[ ! -s "${audit_path}" ]]; then
    stage "AUDITING_FIXED_FINAL_POLICY_EVAL"
    "${PPO_PYTHON}" scripts/experiments/coinrun/audit_coinrun_records.py \
      --dataset "${EVAL_DATA}" \
      --output "${audit_path}" \
      --records-to-check 512
  fi
}

audit_mixture() {
  local mixture_name="$1"
  local audit_path="${DATA_ROOT}/audits/mixtures/${mixture_name}.json"
  if [[ ! -s "${audit_path}" ]]; then
    stage "AUDITING_MIXTURE_${mixture_name^^}"
    "${PPO_PYTHON}" scripts/experiments/coinrun/audit_coinrun_records.py \
      --dataset "${MIXTURE_ROOT}/${mixture_name}" \
      --output "${audit_path}" \
      --records-to-check 2048
  fi
}

train_dynamics_arm() {
  local experiment_id="$1"
  local group="$2"
  local train_data="$3"
  local arm_name="$4"
  local depth="$5"
  local d_model="$6"
  local n_heads="$7"
  local n_register="$8"
  local run_dir="${RUN_ROOT}/${group}/runs/${arm_name}"
  local final_checkpoint="${run_dir}/checkpoints/19999/_CHECKPOINT_METADATA"

  if [[ -f "${final_checkpoint}" ]]; then
    printf 'Skipping completed training arm %s\n' "${arm_name}"
    return
  fi

  local latent_mean
  local latent_std
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

  stage "TRAINING_${group^^}_${arm_name^^}"
  "${DYNAMICS_PYTHON}" scripts/experiments/run_recorded.py \
    --experiment-id "${experiment_id}" \
    --run-name "coinrun-dynamics-${arm_name}" \
    --run-dir "${run_dir}" \
    --experiment-dir "${ROOT}/experiments/${experiment_id}" \
    -- \
    "${DYNAMICS_PYTHON}" scripts/train_dynamics.py \
      dataset=coinrun \
      dataset.array_record_path="${train_data}" \
      dataset.dataloader_cfg.B=16 \
      dataset.dataloader_cfg.short_T=64 \
      dataset.dataloader_cfg.long_T=64 \
      dataset.dataloader_cfg.long_ratio=0.0 \
      dataset.dataloader_cfg.num_workers=4 \
      dataset.dataloader_cfg.prefetch_buffer_size=4 \
      "+dataset.latent_mean=${latent_mean}" \
      "+dataset.latent_std=${latent_std}" \
      tokenizer_ckpt="${TOKENIZER}" \
      dynamics.d_bottleneck=16 \
      dynamics.depth="${depth}" \
      dynamics.d_model="${d_model}" \
      dynamics.n_heads="${n_heads}" \
      dynamics.n_kv_heads=1 \
      dynamics.packing_factor=2 \
      dynamics.n_register="${n_register}" \
      dynamics.qk_norm_type=qknorm \
      dynamics.time_every=2 \
      dynamics.time_layer_offset=1 \
      dynamics.k_max=8 \
      dynamics.context_length=64 \
      max_steps="${MAX_STEPS}" \
      bootstrap_start=10000 \
      bootstrap_fraction=0.25 \
      image_fraction=0.0 \
      ot.enabled=true \
      ot.pairing=barycentric \
      loss_weighting=v_space \
      scaling_flops_budget=0 \
      scaling_tokens_per_param=0 \
      ckpt.max_to_keep=5 \
      ckpt.save_interval_steps=5000 \
      logger.log_every=50 \
      use_wandb="${USE_WANDB}" \
      logger.wandb_group="${experiment_id}" \
      optimizer.optimizer_type=muon \
      optimizer.mup_scaling=false \
      lr_schedule.schedule_type=wsd \
      lr_schedule.lr=0.0003 \
      lr_schedule.warmup_ratio=0.05 \
      lr_schedule.decay_ratio=0.1 \
      write_video_every=2500 \
      run_name="coinrun-dynamics-${arm_name}" \
      hydra.run.dir="${run_dir}"
  test -f "${final_checkpoint}"
}

evaluate_dynamics_arm() {
  local group="$1"
  local arm_name="$2"
  local run_dir="${RUN_ROOT}/${group}/runs/${arm_name}"
  local eval_dir="${RUN_ROOT}/${group}/eval/${arm_name}"
  local metrics="${eval_dir}/metrics.json"

  if [[ -s "${metrics}" ]]; then
    printf 'Skipping completed evaluation arm %s\n' "${arm_name}"
    return
  fi
  if compgen -G "${eval_dir}/ema_shortcut/pred_*.mp4" >/dev/null; then
    printf 'Partial evaluation exists without metrics: %s\n' "${eval_dir}" >&2
    return 1
  fi

  stage "EVALUATING_${group^^}_${arm_name^^}"
  "${DYNAMICS_PYTHON}" scripts/eval_fvd.py \
    dataset=coinrun \
    dataset.array_record_path="${EVAL_DATA}" \
    dataset.dataloader_cfg.B=2 \
    dataset.dataloader_cfg.short_T=32 \
    dataset.dataloader_cfg.long_T=32 \
    dataset.dataloader_cfg.num_workers=0 \
    dynamics_ckpt="${run_dir}/checkpoints" \
    mode=generate \
    rollout_type=ema_shortcut \
    num_videos=32 \
    ctx_length=16 \
    horizon=16 \
    seed=4242 \
    video_dir="${eval_dir}"

  "${DYNAMICS_PYTHON}" scripts/experiments/coinrun/score_coinrun_rollouts.py \
    "${eval_dir}/ema_shortcut" \
    --context 16 \
    --output "${metrics}"
  test -s "${metrics}"
}

stage "COLLECTING_PPO_CHECKPOINT_SOURCE_POOLS"
collect_training_source ppo01p05m 1048576
collect_training_source ppo06p29m 6291456
collect_training_source ppo12p58m 12582912
collect_training_source ppo25p17m 25165824
collect_evaluation_data

stage "MATERIALIZING_CHECKPOINT_MIXTURES"
"${PPO_PYTHON}" \
  scripts/experiments/coinrun/build_coinrun_checkpoint_mixtures.py \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${MIXTURE_ROOT}" \
  --experiment-id CR-DYN-0008 \
  --selection-seed 40240

audit_mixture final_only
audit_mixture uniform
audit_mixture recency_weighted

stage "VERIFYING_CHECKPOINT_MIXTURE_CORPUS"
"${PPO_PYTHON}" \
  scripts/experiments/coinrun/verify_coinrun_checkpoint_mixtures.py \
  --data-root "${DATA_ROOT}" \
  --output "${CORPUS_AUDIT}"
test -s "${CORPUS_AUDIT}"

if [[ ! -s "${LATENT_STATS}" ]]; then
  stage "COMPUTING_N16P6M_LATENT_STATS"
  "${DYNAMICS_PYTHON}" \
    scripts/experiments/coinrun/compute_coinrun_latent_stats.py \
    --checkpoint "${TOKENIZER}" \
    --dataset "${MIXTURE_ROOT}/uniform" \
    --output "${LATENT_STATS}" \
    --batch-size 16 \
    --frames 64 \
    --batches 16
fi

stage "RUNNING_CHECKPOINT_MIXTURE_ABLATION"
for mixture_name in final_only uniform recency_weighted; do
  train_dynamics_arm \
    CR-DYN-0008 \
    mixture-ablation \
    "${MIXTURE_ROOT}/${mixture_name}" \
    "${mixture_name}-medium" \
    4 256 4 32
  evaluate_dynamics_arm \
    mixture-ablation \
    "${mixture_name}-medium"
done

stage "SELECTING_CHECKPOINT_MIXTURE"
"${DYNAMICS_PYTHON}" \
  scripts/experiments/coinrun/summarize_coinrun_dynamics_checkpoint_mixtures.py \
  --run-root "${RUN_ROOT}" \
  --mixture-output "${RUN_ROOT}/checkpoint-mixture-comparison.json" \
  --selection-output "${RUN_ROOT}/checkpoint-mixture-selection.json"
selected_mixture="$(
  "${DYNAMICS_PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected_mixture"])' \
    "${RUN_ROOT}/checkpoint-mixture-selection.json"
)"
case "${selected_mixture}" in
  final_only|uniform|recency_weighted) ;;
  *)
    printf 'Invalid selected mixture: %s\n' "${selected_mixture}" >&2
    exit 1
    ;;
esac

stage "RUNNING_SELECTED_MIXTURE_DYNAMICS_SCALE_ABLATION"
train_dynamics_arm \
  CR-DYN-0009 \
  scale-ablation \
  "${MIXTURE_ROOT}/${selected_mixture}" \
  "${selected_mixture}-tiny" \
  2 64 1 8
evaluate_dynamics_arm \
  scale-ablation \
  "${selected_mixture}-tiny"

train_dynamics_arm \
  CR-DYN-0009 \
  scale-ablation \
  "${MIXTURE_ROOT}/${selected_mixture}" \
  "${selected_mixture}-small" \
  2 128 2 16
evaluate_dynamics_arm \
  scale-ablation \
  "${selected_mixture}-small"

mkdir -p \
  "${RUN_ROOT}/scale-ablation/runs" \
  "${RUN_ROOT}/scale-ablation/eval"
if [[ ! -e "${RUN_ROOT}/scale-ablation/runs/${selected_mixture}-medium" ]]; then
  ln -s \
    "../../mixture-ablation/runs/${selected_mixture}-medium" \
    "${RUN_ROOT}/scale-ablation/runs/${selected_mixture}-medium"
fi
if [[ ! -e "${RUN_ROOT}/scale-ablation/eval/${selected_mixture}-medium" ]]; then
  ln -s \
    "../../mixture-ablation/eval/${selected_mixture}-medium" \
    "${RUN_ROOT}/scale-ablation/eval/${selected_mixture}-medium"
fi

train_dynamics_arm \
  CR-DYN-0009 \
  scale-ablation \
  "${MIXTURE_ROOT}/${selected_mixture}" \
  "${selected_mixture}-large" \
  6 384 6 32
evaluate_dynamics_arm \
  scale-ablation \
  "${selected_mixture}-large"

stage "WRITING_FINAL_COMPARISON_TABLES"
"${DYNAMICS_PYTHON}" \
  scripts/experiments/coinrun/summarize_coinrun_dynamics_checkpoint_mixtures.py \
  --run-root "${RUN_ROOT}" \
  --mixture-output "${RUN_ROOT}/checkpoint-mixture-comparison.json" \
  --selection-output "${RUN_ROOT}/checkpoint-mixture-selection.json" \
  --scale-output "${RUN_ROOT}/selected-mixture-dynamics-scale-comparison.json"

stage "COMPLETE CHECKPOINT_MIXTURE_AND_DYNAMICS_SCALE_ABLATIONS"
