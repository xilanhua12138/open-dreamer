#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/mnt/workspace/open-dreamer"
readonly EXPERIMENT_DIR="${ROOT}/logs/coinrun-minimal-complete-20260728"
readonly EVAL_DATA="/mnt/workspace/datasets/coinrun-minimal-complete-20260728/eval"
readonly OUTPUT_DIR="${EXPERIMENT_DIR}/manual_test_$(date +%Y%m%d_%H%M%S)"

cd "${ROOT}"
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"

python scripts/eval_fvd.py \
  dataset=coinrun \
  dataset.array_record_path="${EVAL_DATA}" \
  dataset.dataloader_cfg.B=2 \
  dataset.dataloader_cfg.long_T=20 \
  dataset.dataloader_cfg.num_workers=0 \
  dynamics_ckpt="${EXPERIMENT_DIR}/dynamics/checkpoints" \
  mode=generate \
  rollout_type=ema_shortcut \
  num_videos=4 \
  ctx_length=4 \
  horizon=16 \
  video_dir="${OUTPUT_DIR}"

printf '测试视频已生成到: %s/ema_shortcut\n' "${OUTPUT_DIR}"
