#!/usr/bin/env bash
set -Eeuo pipefail

readonly EXPERIMENT_ID="CR-DYN-0012"
readonly XLARGE_DEPTH=9
readonly XLARGE_D_MODEL=640
readonly XLARGE_N_HEADS=10
readonly XLARGE_N_KV_HEADS=1
readonly XLARGE_N_REGISTER=32

export OPEN_DREAMER_EXPERIMENT_ID="${EXPERIMENT_ID}"
export OPEN_DREAMER_MODEL_LABEL=xlarge
export OPEN_DREAMER_TRAIN_ARM=final-policy-xlarge-offline-latents
export OPEN_DREAMER_DYNAMICS_DEPTH="${XLARGE_DEPTH}"
export OPEN_DREAMER_DYNAMICS_D_MODEL="${XLARGE_D_MODEL}"
export OPEN_DREAMER_DYNAMICS_N_HEADS="${XLARGE_N_HEADS}"
export OPEN_DREAMER_DYNAMICS_N_KV_HEADS="${XLARGE_N_KV_HEADS}"
export OPEN_DREAMER_DYNAMICS_N_REGISTER="${XLARGE_N_REGISTER}"

exec bash "${OPEN_DREAMER_ROOT}/scripts/experiments/coinrun/run_coinrun_dynamics_offline_latents.sh"
