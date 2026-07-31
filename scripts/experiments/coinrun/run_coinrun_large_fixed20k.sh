#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/mnt/workspace/open-dreamer"
readonly EXPERIMENT="${ROOT}/logs/coinrun-dynamics-fixed20k-20260728"
readonly RUN_DIR="${EXPERIMENT}/runs/large"
readonly EVAL_DIR="${EXPERIMENT}/eval/large"
readonly TRAIN_DATA="/mnt/workspace/datasets/coinrun-minimal-complete-20260728/train"
readonly EVAL_DATA="/mnt/workspace/datasets/coinrun-minimal-complete-20260728/eval"
readonly TOKENIZER="${ROOT}/logs/coinrun-official-min-n0.17m-c1e16-20260728/checkpoints"
readonly STATS="${ROOT}/logs/coinrun-dynamics-scaling-20260728/latent_stats.json"
readonly STATUS="${EXPERIMENT}/STATUS_LARGE"
readonly MAX_STEPS=20000
readonly BOOTSTRAP_START=10000

mkdir -p "${RUN_DIR}" "${EVAL_DIR}"
exec > >(tee -a "${EXPERIMENT}/pipeline-large.log") 2>&1

stage() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee "${STATUS}"
}

finish() {
  local exit_code=$?
  if [[ "${exit_code}" -eq 0 ]]; then
    stage "COMPLETE"
  else
    stage "FAILED exit_code=${exit_code}"
  fi
}
trap finish EXIT

cd "${ROOT}"
source .venv/bin/activate
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
test -s "${STATS}"

latent_mean="$(
  .venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["mean"])' "${STATS}"
)"
latent_std="$(
  .venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["std"])' "${STATS}"
)"

stage "TRAINING_LARGE"
.venv/bin/python scripts/train_dynamics.py \
  dataset=coinrun \
  dataset.array_record_path="${TRAIN_DATA}" \
  dataset.dataloader_cfg.B=32 \
  dataset.dataloader_cfg.short_T=64 \
  dataset.dataloader_cfg.long_T=64 \
  dataset.dataloader_cfg.long_ratio=0.0 \
  dataset.dataloader_cfg.num_workers=4 \
  dataset.dataloader_cfg.prefetch_buffer_size=4 \
  "+dataset.latent_mean=${latent_mean}" \
  "+dataset.latent_std=${latent_std}" \
  tokenizer_ckpt="${TOKENIZER}" \
  dynamics.d_bottleneck=16 \
  dynamics.depth=6 \
  dynamics.d_model=384 \
  dynamics.n_heads=6 \
  dynamics.n_kv_heads=1 \
  dynamics.packing_factor=2 \
  dynamics.n_register=32 \
  dynamics.qk_norm_type=qknorm \
  dynamics.time_every=2 \
  dynamics.time_layer_offset=1 \
  dynamics.k_max=8 \
  dynamics.context_length=64 \
  max_steps="${MAX_STEPS}" \
  bootstrap_start="${BOOTSTRAP_START}" \
  bootstrap_fraction=0.25 \
  image_fraction=0.0 \
  ot.enabled=true \
  ot.pairing=barycentric \
  loss_weighting=v_space \
  ckpt.max_to_keep=2 \
  ckpt.save_interval_steps=10000 \
  logger.log_every=50 \
  optimizer.optimizer_type=muon \
  optimizer.mup_scaling=false \
  lr_schedule.schedule_type=wsd \
  lr_schedule.lr=0.0003 \
  lr_schedule.warmup_ratio=0.05 \
  lr_schedule.decay_ratio=0.1 \
  write_video_every=0 \
  run_name="coinrun-dynamics-large-fixed20k" \
  hydra.run.dir="${RUN_DIR}"

stage "EVALUATING_LARGE"
.venv/bin/python scripts/eval_fvd.py \
  dataset=coinrun \
  dataset.array_record_path="${EVAL_DATA}" \
  dataset.dataloader_cfg.B=2 \
  dataset.dataloader_cfg.long_T=20 \
  dataset.dataloader_cfg.num_workers=0 \
  dynamics_ckpt="${RUN_DIR}/checkpoints" \
  mode=generate \
  rollout_type=ema_shortcut \
  num_videos=8 \
  ctx_length=4 \
  horizon=16 \
  seed=4242 \
  video_dir="${EVAL_DIR}"

stage "SCORING_LARGE"
.venv/bin/python "${ROOT}/score_coinrun_rollouts.py" \
  "${EVAL_DIR}/ema_shortcut" \
  --context 4 \
  --output "${EVAL_DIR}/metrics.json"

stage "COMPLETE"
