#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="${OPEN_DREAMER_ROOT:-/mnt/workspace/open-dreamer-ppo-collector}"
readonly UV="${OPEN_DREAMER_UV:-/root/.local/bin/uv}"
readonly PYTHON="${OPEN_DREAMER_PYTHON310:-/usr/bin/python3.10}"
readonly VENV="${OPEN_DREAMER_PPO_VENV:-${ROOT}/.venv-ppo}"

test -x "${UV}"
test -x "${PYTHON}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${UV}" venv --python "${PYTHON}" "${VENV}"
fi

"${UV}" pip install --python "${VENV}/bin/python" \
  "numpy==2.2.6" \
  "scipy==1.14.1" \
  "jax[cuda12]==0.4.35" \
  "flax==0.10.2" \
  "optax==0.2.4" \
  "procgen==0.10.7" \
  "gym3==0.3.3" \
  "array-record==0.8.1" \
  "grain==0.2.11" \
  "imageio==2.36.1" \
  "pillow==11.1.0" \
  "msgpack==1.1.0" \
  "wandb==0.19.7"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu
export OPEN_DREAMER_PPO_RUNTIME_JSON="${VENV}/runtime-versions.json"

"${VENV}/bin/python" - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path

import jax
import numpy as np
from procgen import ProcgenGym3Env

from dreamer.coinrun import COINRUN_ACTION_DIM
from dreamer.coinrun_ppo import CoinRunActorCritic

env = ProcgenGym3Env(
    num=1,
    env_name="coinrun",
    start_level=0,
    num_levels=1,
    distribution_mode="easy",
    rand_seed=0,
)
action_dim = int(env.ac_space.eltype.n)
if action_dim != COINRUN_ACTION_DIM:
    raise RuntimeError(
        f"CoinRun action dimension {action_dim} != {COINRUN_ACTION_DIM}"
    )
_, observations, _ = env.observe()
model = CoinRunActorCritic(action_dim=COINRUN_ACTION_DIM)
variables = model.init(jax.random.PRNGKey(0), observations["rgb"])
logits, values = model.apply(variables, observations["rgb"])
if logits.shape != (1, COINRUN_ACTION_DIM) or values.shape != (1,):
    raise RuntimeError(
        f"actor-critic smoke shapes are {logits.shape} and {values.shape}"
    )
payload = {
    "schema_version": "1.0",
    "python": os.sys.version,
    "packages": {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "scipy",
            "jax",
            "jaxlib",
            "flax",
            "optax",
            "procgen",
            "gym3",
            "array-record",
            "grain",
            "imageio",
            "wandb",
        )
    },
    "jax_devices_under_cpu_smoke": [str(device) for device in jax.devices()],
    "coinrun_action_dim": action_dim,
    "coinrun_rgb_shape": list(
        np.asarray(observations["rgb"]).shape
    ),
    "actor_logits_shape": list(logits.shape),
    "actor_values_shape": list(values.shape),
}
path = Path(os.environ["OPEN_DREAMER_PPO_RUNTIME_JSON"])
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

unset JAX_PLATFORMS
