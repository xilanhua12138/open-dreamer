#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/mnt/workspace/open-dreamer"
readonly EXPERIMENT="${ROOT}/logs/coinrun-dynamics-scaling-20260728"
readonly TRAIN_DATA="/mnt/workspace/datasets/coinrun-minimal-complete-20260728/train"
readonly EVAL_DATA="/mnt/workspace/datasets/coinrun-minimal-complete-20260728/eval"
readonly TOKENIZER="${ROOT}/logs/coinrun-official-min-n0.17m-c1e16-20260728/checkpoints"
readonly STATS="${EXPERIMENT}/latent_stats.json"
readonly STATUS="${EXPERIMENT}/STATUS"
readonly BUDGET="1e15"

mkdir -p "${EXPERIMENT}/runs" "${EXPERIMENT}/eval"
exec > >(tee -a "${EXPERIMENT}/pipeline.log") 2>&1

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

run_one() {
  local name="$1"
  local depth="$2"
  local d_model="$3"
  local n_heads="$4"
  local n_register="$5"
  local bootstrap_start="$6"
  local run_dir="${EXPERIMENT}/runs/${name}"

  stage "TRAINING_${name^^}"
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
    scaling_flops_budget="${BUDGET}" \
    bootstrap_start="${bootstrap_start}" \
    bootstrap_fraction=0.25 \
    image_fraction=0.0 \
    ot.enabled=true \
    ot.pairing=barycentric \
    loss_weighting=v_space \
    ckpt.max_to_keep=1 \
    ckpt.save_interval_steps=1000000 \
    logger.log_every=50 \
    optimizer.optimizer_type=muon \
    optimizer.mup_scaling=false \
    lr_schedule.schedule_type=wsd \
    lr_schedule.lr=0.0003 \
    lr_schedule.warmup_ratio=0.05 \
    lr_schedule.decay_ratio=0.1 \
    write_video_every=0 \
    run_name="coinrun-dynamics-${name}-c1e15" \
    hydra.run.dir="${run_dir}"

  stage "EVALUATING_${name^^}"
  .venv/bin/python scripts/eval_fvd.py \
    dataset=coinrun \
    dataset.array_record_path="${EVAL_DATA}" \
    dataset.dataloader_cfg.B=2 \
    dataset.dataloader_cfg.long_T=20 \
    dataset.dataloader_cfg.num_workers=0 \
    dynamics_ckpt="${run_dir}/checkpoints" \
    mode=generate \
    rollout_type=ema_shortcut \
    num_videos=8 \
    ctx_length=4 \
    horizon=16 \
    seed=4242 \
    video_dir="${EXPERIMENT}/eval/${name}"
}

# Largest first: its first compiled step is also the peak-memory smoke test.
run_one medium 4 256 4 32 237
run_one small 2 128 2 16 2384
run_one tiny 2 64 1 8 7608

stage "WRITING_RESULTS"
.venv/bin/python - "${EXPERIMENT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
probe = json.loads((root / "probe.json").read_text(encoding="utf-8"))
csv_path = root / "runs" / "results.csv"
rows = []
if csv_path.exists():
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

payload = {
    "protocol": {
        "dataset": "CoinRun random actions",
        "train_records": 2048,
        "frames_per_record": 64,
        "tokenizer_params": 163392,
        "flops_budget": 1e15,
        "batch_size": 32,
        "sequence_length": 64,
        "heldout_videos": 8,
        "context_frames": 4,
        "predicted_frames": 16,
    },
    "probe": probe["candidates"],
    "native_results_csv": rows,
}
(root / "scaling_results.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
PY
