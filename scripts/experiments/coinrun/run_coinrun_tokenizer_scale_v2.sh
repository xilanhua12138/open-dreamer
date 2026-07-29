#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-v2}"
readonly RUN_ROOT="${ROOT}/logs/coinrun-tokenizer-scale-v2-20260729"
readonly DATA_ROOT="/mnt/workspace/datasets/coinrun-structured-v2-20260729"
readonly TRAIN_DATA="${DATA_ROOT}/train"
readonly EVAL_DATA="${DATA_ROOT}/eval"
readonly STATUS="${RUN_ROOT}/STATUS"
readonly PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
readonly PROBE="${RUN_ROOT}/tokenizer-scale-probe.json"
readonly SELECTION="${RUN_ROOT}/selection.json"

mkdir -p "${RUN_ROOT}/runs" "${RUN_ROOT}/audit"
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

ensure_empty_or_complete_dataset() {
  local directory="$1"
  if [[ -s "${directory}/metadata.json" ]]; then
    return
  fi
  if find "${directory}" -maxdepth 1 -name '*.array_record' -print -quit 2>/dev/null | grep -q .; then
    echo "partial dataset exists without metadata: ${directory}" >&2
    return 1
  fi
}

stage "PREPARING_STRUCTURED_DATASET"
ensure_empty_or_complete_dataset "${TRAIN_DATA}"
ensure_empty_or_complete_dataset "${EVAL_DATA}"
if [[ ! -s "${TRAIN_DATA}/metadata.json" ]]; then
  .venv-procgen/bin/python scripts/experiments/coinrun/generate_coinrun_records.py \
    --output-dir "${TRAIN_DATA}" \
    --records 4096 \
    --frames 64 \
    --envs 64 \
    --records-per-shard 256 \
    --seed 7000 \
    --policy structured \
    --goal-directed-fraction 0.65
fi
if [[ ! -s "${EVAL_DATA}/metadata.json" ]]; then
  .venv-procgen/bin/python scripts/experiments/coinrun/generate_coinrun_records.py \
    --output-dir "${EVAL_DATA}" \
    --records 512 \
    --frames 64 \
    --envs 32 \
    --records-per-shard 128 \
    --seed 20000 \
    --policy structured \
    --goal-directed-fraction 0.65
fi

stage "AUDITING_DATASET"
.venv/bin/python scripts/experiments/coinrun/audit_coinrun_records.py \
  --dataset "${TRAIN_DATA}" \
  --output "${RUN_ROOT}/audit/train.json" \
  --records-to-check 256
.venv/bin/python scripts/experiments/coinrun/audit_coinrun_records.py \
  --dataset "${EVAL_DATA}" \
  --output "${RUN_ROOT}/audit/eval.json" \
  --records-to-check 256
.venv/bin/python - "${TRAIN_DATA}/metadata.json" "${EVAL_DATA}/metadata.json" <<'PY'
import json
import sys

train = json.load(open(sys.argv[1], encoding="utf-8"))
eval_ = json.load(open(sys.argv[2], encoding="utf-8"))
train_levels = set(range(train["seed"], train["seed"] + train["num_levels"]))
eval_levels = set(range(eval_["seed"], eval_["seed"] + eval_["num_levels"]))
overlap = train_levels & eval_levels
if overlap:
    raise SystemExit(f"train/eval level overlap: {sorted(overlap)[:10]}")
print(
    f"split isolation verified: train={min(train_levels)}..{max(train_levels)} "
    f"eval={min(eval_levels)}..{max(eval_levels)}"
)
PY

stage "PROBING_TOKENIZER_SCALES"
.venv/bin/python scripts/experiments/coinrun/probe_coinrun_tokenizer_scale_v2.py \
  --output "${PROBE}"

run_candidate() {
  local name="$1"
  local directory_name="$2"
  local depth="$3"
  local d_model="$4"
  local flops_budget="$5"
  local run_dir="${RUN_ROOT}/runs/${directory_name}"
  local evaluation="${run_dir}/heldout_eval.json"

  if [[ ! -s "${evaluation}" ]]; then
    stage "TRAINING_TOKENIZER_${directory_name^^}"
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
      scaling_flops_budget="${flops_budget}" \
      tokenizer_loss_type=mse \
      lpips_weight=0.2 \
      optimizer.optimizer_type=muon \
      optimizer.mup_scaling=true \
      ckpt.max_to_keep=1 \
      ckpt.save_interval_steps=1000 \
      logger.log_every=50 \
      visualize_every=0 \
      run_name="coinrun-tokenizer-v2-${directory_name}" \
      hydra.run.dir="${run_dir}"

    stage "EVALUATING_TOKENIZER_${directory_name^^}"
    .venv/bin/python scripts/experiments/coinrun/eval_coinrun_tokenizer_psnr.py \
      --checkpoint "${run_dir}/checkpoints" \
      --dataset "${EVAL_DATA}" \
      --output "${evaluation}" \
      --visualization "${run_dir}/heldout_reconstruction.png" \
      --batch-size 32 \
      --frames 16 \
      --batches 16 \
      --seed 4242
  fi
  test -s "${run_dir}/heldout_reconstruction.png"
  printf '%s=%s\n' "${name}" "${evaluation}"
}

write_selection() {
  local -a candidates=("$@")
  local -a arguments=()
  local candidate
  for candidate in "${candidates[@]}"; do
    case "${candidate}" in
      n0.17m) arguments+=(--metrics "n0.17m=${RUN_ROOT}/runs/n0p17m/heldout_eval.json") ;;
      n1.1m) arguments+=(--metrics "n1.1m=${RUN_ROOT}/runs/n1p1m/heldout_eval.json") ;;
      n3.7m) arguments+=(--metrics "n3.7m=${RUN_ROOT}/runs/n3p7m/heldout_eval.json") ;;
      n8.6m) arguments+=(--metrics "n8.6m=${RUN_ROOT}/runs/n8p6m/heldout_eval.json") ;;
      n16.6m) arguments+=(--metrics "n16.6m=${RUN_ROOT}/runs/n16p6m/heldout_eval.json") ;;
      *) echo "unknown candidate ${candidate}" >&2; return 1 ;;
    esac
  done
  .venv/bin/python scripts/experiments/coinrun/select_coinrun_tokenizer_scale.py \
    --probe "${PROBE}" \
    "${arguments[@]}" \
    --output "${SELECTION}"
}

quality_gate_passed() {
  .venv/bin/python -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["quality_gate_passed"] else 1)' \
    "${SELECTION}"
}

run_candidate "n0.17m" "n0p17m" 1 64 1e16
run_candidate "n1.1m" "n1p1m" 2 128 1e16
run_candidate "n3.7m" "n3p7m" 3 192 1e16
evaluated=("n0.17m" "n1.1m" "n3.7m")
write_selection "${evaluated[@]}"

if ! quality_gate_passed; then
  stage "PRIMARY_GATE_MISSED_EXTENDING_TO_N8P6M"
  run_candidate "n8.6m" "n8p6m" 4 256 2e16
  evaluated+=("n8.6m")
  write_selection "${evaluated[@]}"
fi

if ! quality_gate_passed; then
  stage "N8P6M_GATE_MISSED_EXTENDING_TO_N16P6M"
  run_candidate "n16.6m" "n16p6m" 5 320 4e16
  evaluated+=("n16.6m")
  write_selection "${evaluated[@]}"
fi

grid_arguments=()
for candidate in "${evaluated[@]}"; do
  case "${candidate}" in
    n0.17m) grid_arguments+=(--image "n0.17m=${RUN_ROOT}/runs/n0p17m/heldout_reconstruction.png") ;;
    n1.1m) grid_arguments+=(--image "n1.1m=${RUN_ROOT}/runs/n1p1m/heldout_reconstruction.png") ;;
    n3.7m) grid_arguments+=(--image "n3.7m=${RUN_ROOT}/runs/n3p7m/heldout_reconstruction.png") ;;
    n8.6m) grid_arguments+=(--image "n8.6m=${RUN_ROOT}/runs/n8p6m/heldout_reconstruction.png") ;;
    n16.6m) grid_arguments+=(--image "n16.6m=${RUN_ROOT}/runs/n16p6m/heldout_reconstruction.png") ;;
  esac
done
.venv/bin/python scripts/experiments/coinrun/build_tokenizer_scale_grid.py \
  "${grid_arguments[@]}" \
  --output "${RUN_ROOT}/tokenizer-scale-comparison.png" \
  --manifest "${RUN_ROOT}/tokenizer-scale-comparison.json"

if quality_gate_passed; then
  selected="$(
    .venv/bin/python -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["name"])' \
      "${SELECTION}"
  )"
  stage "COMPLETE QUALITY_GATE_PASSED selected=${selected} awaiting_visual_review"
else
  stage "COMPLETE QUALITY_GATE_FAILED awaiting_visual_review"
fi
