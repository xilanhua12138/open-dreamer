#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/mnt/workspace/open-dreamer"
readonly EXPERIMENT_DIR="${ROOT}/logs/coinrun-minimal-complete-20260728"
readonly DATASET_DIR="/mnt/workspace/datasets/coinrun-minimal-complete-20260728"
readonly TRAIN_DATA="${DATASET_DIR}/train"
readonly EVAL_DATA="${DATASET_DIR}/eval"
readonly TOKENIZER_DIR="${EXPERIMENT_DIR}/tokenizer"
readonly DYNAMICS_DIR="${EXPERIMENT_DIR}/dynamics"
readonly STATS_FILE="${EXPERIMENT_DIR}/latent_stats.json"
readonly STATUS_FILE="${EXPERIMENT_DIR}/STATUS"

mkdir -p "${EXPERIMENT_DIR}"
exec > >(tee -a "${EXPERIMENT_DIR}/pipeline.log") 2>&1

stage() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee "${STATUS_FILE}"
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

stage "GENERATING_DATA"
if [[ ! -s "${TRAIN_DATA}/metadata.json" ]]; then
  .venv-procgen/bin/python logs/generate_coinrun_records.py \
    --output-dir "${TRAIN_DATA}" \
    --records 2048 \
    --frames 64 \
    --envs 32 \
    --records-per-shard 256 \
    --seed 42
fi
if [[ ! -s "${EVAL_DATA}/metadata.json" ]]; then
  .venv-procgen/bin/python logs/generate_coinrun_records.py \
    --output-dir "${EVAL_DATA}" \
    --records 256 \
    --frames 64 \
    --envs 16 \
    --records-per-shard 128 \
    --seed 4242
fi

stage "TRAINING_TOKENIZER"
python scripts/train_tokenizer.py \
  dataset=coinrun \
  dataset.array_record_path="${TRAIN_DATA}" \
  dataset.dataloader_cfg.B=128 \
  dataset.dataloader_cfg.short_T=16 \
  dataset.dataloader_cfg.long_T=16 \
  dataset.dataloader_cfg.num_workers=4 \
  dataset.dataloader_cfg.prefetch_buffer_size=4 \
  tokenizer.encoder.n_latents=16 \
  tokenizer.encoder.d_bottleneck=16 \
  tokenizer.encoder.depth=2 \
  tokenizer.encoder.d_model=128 \
  tokenizer.encoder.n_heads=2 \
  tokenizer.encoder.n_kv_heads=1 \
  tokenizer.encoder.time_every=2 \
  tokenizer.encoder.time_layer_offset=1 \
  tokenizer.encoder.mae_p_max=0.9 \
  tokenizer.decoder.depth=2 \
  tokenizer.decoder.d_model=128 \
  tokenizer.decoder.n_heads=2 \
  tokenizer.decoder.n_kv_heads=1 \
  max_steps=25000 \
  ckpt.max_to_keep=1 \
  ckpt.save_interval_steps=5000 \
  logger.log_every=100 \
  optimizer.optimizer_type=muon \
  optimizer.mup_scaling=false \
  lr_schedule.schedule_type=wsd \
  lr_schedule.lr=0.003 \
  lr_schedule.warmup_ratio=0.05 \
  lr_schedule.decay_ratio=0.4 \
  lpips_weight=0 \
  visualize_every=2500 \
  run_name=coinrun-minimal-tokenizer-1.1m \
  hydra.run.dir="${TOKENIZER_DIR}"

stage "COMPUTING_LATENT_STATS"
python logs/compute_coinrun_latent_stats.py \
  --checkpoint "${TOKENIZER_DIR}/checkpoints" \
  --dataset "${EVAL_DATA}" \
  --output "${STATS_FILE}" \
  --batch-size 32 \
  --frames 64 \
  --batches 8

latent_mean="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["mean"])' "${STATS_FILE}"
)"
latent_std="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["std"])' "${STATS_FILE}"
)"

stage "TRAINING_DYNAMICS"
python scripts/train_dynamics.py \
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
  tokenizer_ckpt="${TOKENIZER_DIR}/checkpoints" \
  dynamics.d_bottleneck=16 \
  dynamics.depth=2 \
  dynamics.d_model=128 \
  dynamics.n_heads=2 \
  dynamics.n_kv_heads=1 \
  dynamics.packing_factor=2 \
  dynamics.n_register=16 \
  dynamics.time_every=2 \
  dynamics.time_layer_offset=1 \
  dynamics.k_max=8 \
  dynamics.context_length=64 \
  max_steps=15000 \
  bootstrap_start=5000 \
  bootstrap_fraction=0.25 \
  image_fraction=0.0 \
  ot.enabled=false \
  ckpt.max_to_keep=1 \
  ckpt.save_interval_steps=5000 \
  logger.log_every=100 \
  optimizer.optimizer_type=muon \
  optimizer.mup_scaling=false \
  lr_schedule.schedule_type=wsd \
  lr_schedule.lr=0.0003 \
  lr_schedule.warmup_ratio=0.05 \
  lr_schedule.decay_ratio=0.1 \
  write_video_every=0 \
  run_name=coinrun-minimal-dynamics \
  hydra.run.dir="${DYNAMICS_DIR}"

stage "GENERATING_TEST_VIDEOS"
python scripts/eval_fvd.py \
  dataset=coinrun \
  dataset.array_record_path="${EVAL_DATA}" \
  dataset.dataloader_cfg.B=2 \
  dataset.dataloader_cfg.long_T=20 \
  dataset.dataloader_cfg.num_workers=0 \
  dynamics_ckpt="${DYNAMICS_DIR}/checkpoints" \
  mode=generate \
  rollout_type=ema_shortcut \
  num_videos=4 \
  ctx_length=4 \
  horizon=16 \
  video_dir="${EXPERIMENT_DIR}/test_videos"

python - <<'PY'
from pathlib import Path

root = Path("/mnt/workspace/open-dreamer/logs/coinrun-minimal-complete-20260728")
video_dir = root / "test_videos" / "ema_shortcut"
summary = root / "RESULTS.txt"
lines = [
    "OpenDreamer minimal complete CoinRun experiment",
    "",
    f"Tokenizer checkpoint: {root / 'tokenizer' / 'checkpoints'}",
    f"Dynamics checkpoint: {root / 'dynamics' / 'checkpoints'}",
    f"Training rollout grid: {root / 'dynamics' / 'viz' / 'step_014999' / 'rollouts_grid.mp4'}",
    f"Test videos: {video_dir}",
    "",
    "Files:",
]
lines.extend(f"- {path.name}" for path in sorted(video_dir.glob("*.mp4")))
summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary.read_text(encoding="utf-8"))
PY
