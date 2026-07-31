#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/mnt/workspace/open-dreamer"
readonly RUN_DIR="${ROOT}/logs/coinrun-official-min-n0.17m-c1e16-20260728"
readonly TRAIN_DATA="/mnt/workspace/datasets/coinrun-minimal-complete-20260728/train"
readonly EVAL_DATA="/mnt/workspace/datasets/coinrun-minimal-complete-20260728/eval"
readonly STATUS_FILE="${RUN_DIR}/STATUS"
readonly EVAL_JSON="${RUN_DIR}/heldout_eval.json"
readonly EVAL_IMAGE="${RUN_DIR}/heldout_reconstruction.png"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/pipeline.log") 2>&1

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

test -s "${TRAIN_DATA}/metadata.json"
test -s "${EVAL_DATA}/metadata.json"

stage "TRAINING_TOKENIZER_N0.17M_C1E16"
.venv/bin/python scripts/train_tokenizer.py \
  dataset=coinrun \
  dataset.array_record_path="${TRAIN_DATA}" \
  dataset.dataloader_cfg.B=128 \
  dataset.dataloader_cfg.short_T=16 \
  dataset.dataloader_cfg.long_T=16 \
  dataset.dataloader_cfg.num_workers=4 \
  dataset.dataloader_cfg.prefetch_buffer_size=4 \
  tokenizer.encoder.n_latents=16 \
  tokenizer.encoder.d_bottleneck=16 \
  tokenizer.encoder.depth=1 \
  tokenizer.encoder.d_model=64 \
  tokenizer.encoder.n_heads=1 \
  tokenizer.encoder.n_kv_heads=1 \
  tokenizer.encoder.time_every=4 \
  tokenizer.encoder.time_layer_offset=3 \
  tokenizer.decoder.depth=1 \
  tokenizer.decoder.d_model=64 \
  tokenizer.decoder.n_heads=1 \
  tokenizer.decoder.n_kv_heads=1 \
  tokenizer.decoder.time_every=4 \
  tokenizer.decoder.time_layer_offset=3 \
  scaling_flops_budget=1e16 \
  tokenizer_loss_type=mse \
  lpips_weight=0.2 \
  optimizer.optimizer_type=muon \
  optimizer.mup_scaling=true \
  ckpt.max_to_keep=1 \
  logger.log_every=100 \
  run_name=coinrun-official-min-n0.17m-c1e16 \
  hydra.run.dir="${RUN_DIR}"

stage "EVALUATING_HELDOUT_TOKENIZER_PSNR"
.venv/bin/python logs/eval_coinrun_tokenizer_psnr.py \
  --checkpoint "${RUN_DIR}/checkpoints" \
  --dataset "${EVAL_DATA}" \
  --output "${EVAL_JSON}" \
  --visualization "${EVAL_IMAGE}" \
  --batch-size 32 \
  --frames 16 \
  --batches 8 \
  --seed 4242

stage "WRITING_RESULTS"
.venv/bin/python - "${RUN_DIR}" "${EVAL_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
eval_json = Path(sys.argv[2])
evaluation = json.loads(eval_json.read_text(encoding="utf-8"))
results_csv = run_dir.parent / "results.csv"
matching_rows = []
if results_csv.exists():
    with results_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if row and row[0] == "coinrun-official-min-n0.17m-c1e16":
                matching_rows.append(row)

metrics = evaluation["metrics"]
lines = [
    "OpenDreamer CoinRun smallest published scaling point",
    "",
    "Published target:",
    "- Parameter point: approximately 0.17M",
    "- Compute budget: 1e16 model FLOPs",
    "- Plot estimate: approximately 21.24 dB training PSNR",
    "",
    "This run:",
    "- Exact model parameters: 163,392",
    "- Native scaling_flops_budget: 1e16",
    f"- Held-out online masked PSNR: {metrics['online_masked_psnr']:.4f} dB",
    f"- Held-out EMA masked PSNR: {metrics['ema_masked_psnr']:.4f} dB",
    f"- Held-out online clean PSNR: {metrics['online_clean_psnr']:.4f} dB",
    f"- Held-out EMA clean PSNR: {metrics['ema_clean_psnr']:.4f} dB",
    f"- Reconstruction grid: {run_dir / 'heldout_reconstruction.png'}",
    f"- Checkpoint: {run_dir / 'checkpoints'}",
]
if matching_rows:
    lines.extend(("", "Native results.csv row:", ",".join(matching_rows[-1])))
(run_dir / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((run_dir / "RESULTS.txt").read_text(encoding="utf-8"))
PY
