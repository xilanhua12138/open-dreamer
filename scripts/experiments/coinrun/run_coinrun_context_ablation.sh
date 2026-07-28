#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/mnt/workspace/open-dreamer"
readonly EXPERIMENT="${ROOT}/logs/coinrun-dynamics-fixed20k-20260728"
readonly CHECKPOINT="${1:?usage: run_coinrun_context_ablation.sh CHECKPOINT_DIR [MODEL_NAME]}"
readonly MODEL_NAME="${2:-best}"
readonly OUTPUT="${EXPERIMENT}/context-ablation-${MODEL_NAME}"
readonly EVAL_DATA="/mnt/workspace/datasets/coinrun-minimal-complete-20260728/eval"

cd "${ROOT}"
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED="1"

.venv/bin/python eval_coinrun_context_ablation.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${EVAL_DATA}" \
  --output "${OUTPUT}" \
  --contexts 4 16 32 \
  --horizon 16 \
  --num-videos 8 \
  --batch-size 2 \
  --seed 4242 \
  --denoise-steps 4

.venv/bin/python - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
rows = []
for context in (4, 16, 32):
    metrics = json.loads((output / f"ctx_{context}" / "metrics.json").read_text())
    rows.append(
        {
            "context_frames": context,
            "mean_frame_psnr_db": metrics["mean_frame_psnr_db"],
            "mean_ssim": metrics["mean_ssim"],
            "psnr_by_horizon_db": metrics["psnr_by_horizon_db"],
            "ssim_by_horizon": metrics["ssim_by_horizon"],
        }
    )
(output / "summary.json").write_text(json.dumps({"results": rows}, indent=2) + "\n")
print(json.dumps({"results": rows}, indent=2))
PY
