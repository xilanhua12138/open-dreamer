#!/usr/bin/env python3
"""Measure signal and gradient propagation before PPO training starts."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from dreamer.coinrun import COINRUN_ACTION_DIM
from dreamer.coinrun_ppo import (
    CoinRunActorCritic,
    categorical_entropy,
    categorical_log_prob,
)
from dreamer.experiment_runtime import atomic_write_json


RESIDUAL_DEPTH = 6
DEPTH_SCALE = 1.0 / math.sqrt(RESIDUAL_DEPTH)
ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "glorot_reference": {
        "backbone_kernel_init": "glorot_uniform",
        "residual_branch_scale": 1.0,
        "residual_last_kernel_init": "same",
        "residual_skip_init": False,
    },
    "orthogonal_sqrt2_anchor": {
        "backbone_kernel_init": "orthogonal_sqrt2",
        "residual_branch_scale": 1.0,
        "residual_last_kernel_init": "same",
        "residual_skip_init": False,
    },
    "orthogonal_gain1": {
        "backbone_kernel_init": "orthogonal_gain1",
        "residual_branch_scale": 1.0,
        "residual_last_kernel_init": "same",
        "residual_skip_init": False,
    },
    "orthogonal_sqrt2_depth_scaled": {
        "backbone_kernel_init": "orthogonal_sqrt2",
        "residual_branch_scale": DEPTH_SCALE,
        "residual_last_kernel_init": "same",
        "residual_skip_init": False,
    },
    "orthogonal_sqrt2_zero_last": {
        "backbone_kernel_init": "orthogonal_sqrt2",
        "residual_branch_scale": 1.0,
        "residual_last_kernel_init": "zeros",
        "residual_skip_init": False,
    },
    "orthogonal_sqrt2_skipinit": {
        "backbone_kernel_init": "orthogonal_sqrt2",
        "residual_branch_scale": 1.0,
        "residual_last_kernel_init": "same",
        "residual_skip_init": True,
    },
}


def _rms(value: jax.Array) -> float:
    value = jnp.asarray(value, dtype=jnp.float32)
    return float(jax.device_get(jnp.sqrt(jnp.mean(jnp.square(value)))))


def _single_sown_value(value: Any) -> jax.Array:
    if not isinstance(value, tuple) or len(value) != 1:
        raise ValueError(
            "diagnostic sow must contain exactly one value from one apply call"
        )
    return value[0]


def _tree_l2_norm(
    tree: Mapping[str, Any],
    *,
    include: Callable[[tuple[str, ...]], bool] | None = None,
) -> float:
    squared = 0.0
    flattened = traverse_util.flatten_dict(tree)
    selected = 0
    for path, value in flattened.items():
        if include is not None and not include(path):
            continue
        squared += float(
            jax.device_get(jnp.sum(jnp.square(jnp.asarray(value))))
        )
        selected += 1
    if selected == 0:
        raise ValueError("gradient selector matched no parameter leaves")
    return math.sqrt(squared)


def summarize_scalar_rows(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("at least one diagnostic row is required")
    keys = tuple(sorted(rows[0]))
    if any(tuple(sorted(row)) != keys for row in rows):
        raise ValueError("diagnostic rows must have identical scalar fields")
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"diagnostic metric {key} contains non-finite values")
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def _collect_observations(
    *,
    num_envs: int,
    vector_steps: int,
    seed: int,
) -> np.ndarray:
    try:
        from procgen import ProcgenGym3Env
    except ImportError as error:
        raise RuntimeError("Procgen is required for initialization probing") from error

    env = ProcgenGym3Env(
        num=num_envs,
        env_name="coinrun",
        start_level=0,
        num_levels=200,
        distribution_mode="easy",
        rand_seed=seed,
        num_threads=0,
    )
    generator = np.random.default_rng(seed)
    observations = []
    for step in range(vector_steps + 1):
        _, observation, _ = env.observe()
        observations.append(np.asarray(observation["rgb"], dtype=np.uint8))
        if step < vector_steps:
            env.act(
                generator.integers(
                    0,
                    COINRUN_ACTION_DIM,
                    size=(num_envs,),
                    dtype=np.int32,
                )
            )
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return np.concatenate(observations, axis=0)


def _synthetic_loss(
    model: CoinRunActorCritic,
    params: Mapping[str, Any],
    observations: jax.Array,
) -> jax.Array:
    logits, values = model.apply({"params": params}, observations)
    batch_size = observations.shape[0]
    actions = jnp.arange(batch_size, dtype=jnp.int32) % COINRUN_ACTION_DIM
    advantages = jnp.linspace(-1.0, 1.0, batch_size, dtype=jnp.float32)
    advantages = (advantages - jnp.mean(advantages)) / (
        jnp.std(advantages) + 1e-8
    )
    returns = jnp.sin(
        jnp.arange(batch_size, dtype=jnp.float32) * jnp.asarray(0.37)
    )
    policy_loss = -jnp.mean(categorical_log_prob(logits, actions) * advantages)
    value_loss = 0.5 * jnp.mean(jnp.square(values - returns))
    entropy = jnp.mean(categorical_entropy(logits))
    return policy_loss + 0.5 * value_loss - 0.01 * entropy


def _diagnose_seed(
    *,
    config: Mapping[str, Any],
    observations: jax.Array,
    seed: int,
) -> dict[str, float]:
    plain_model = CoinRunActorCritic(
        action_dim=COINRUN_ACTION_DIM,
        **config,
    )
    diagnostic_model = CoinRunActorCritic(
        action_dim=COINRUN_ACTION_DIM,
        record_diagnostics=True,
        **config,
    )
    parameter_key, tangent_key = jax.random.split(jax.random.PRNGKey(seed))
    params = plain_model.init(parameter_key, observations)["params"]
    (logits, values), diagnostic_state = diagnostic_model.apply(
        {"params": params},
        observations,
        mutable=["diagnostics"],
    )
    diagnostics = diagnostic_state["diagnostics"]

    metrics: dict[str, float] = {
        "encoder_rms": _rms(
            _single_sown_value(diagnostics["encoder_features"])
        ),
        "logits_rms": _rms(logits),
        "logits_std": float(jax.device_get(jnp.std(logits))),
        "values_rms": _rms(values),
        "values_std": float(jax.device_get(jnp.std(values))),
        "policy_entropy": float(
            jax.device_get(jnp.mean(categorical_entropy(logits)))
        ),
    }
    for sequence_index in range(3):
        sequence = diagnostics[f"ImpalaConvSequence_{sequence_index}"]
        for block_index in range(2):
            block = sequence[f"ResidualBlock_{block_index}"]
            prefix = f"sequence{sequence_index}.block{block_index}"
            skip_rms = _rms(_single_sown_value(block["skip"]))
            unscaled_rms = _rms(
                _single_sown_value(block["branch_unscaled"])
            )
            scaled_rms = _rms(_single_sown_value(block["branch_scaled"]))
            output_rms = _rms(_single_sown_value(block["output"]))
            metrics.update(
                {
                    f"{prefix}.skip_rms": skip_rms,
                    f"{prefix}.branch_unscaled_rms": unscaled_rms,
                    f"{prefix}.branch_scaled_rms": scaled_rms,
                    f"{prefix}.output_rms": output_rms,
                    f"{prefix}.branch_to_skip_rms": (
                        scaled_rms / max(skip_rms, 1e-12)
                    ),
                    f"{prefix}.output_to_skip_rms": (
                        output_rms / max(skip_rms, 1e-12)
                    ),
                }
            )

    loss, gradients = jax.value_and_grad(
        lambda current_params: _synthetic_loss(
            plain_model,
            current_params,
            observations,
        )
    )(params)
    metrics.update(
        {
            "synthetic_ppo_loss": float(jax.device_get(loss)),
            "gradient_global_l2": _tree_l2_norm(gradients),
            "gradient_backbone_l2": _tree_l2_norm(
                gradients,
                include=lambda path: (
                    "policy_head" not in path and "value_head" not in path
                ),
            ),
            "gradient_policy_head_l2": _tree_l2_norm(
                gradients,
                include=lambda path: "policy_head" in path,
            ),
            "gradient_value_head_l2": _tree_l2_norm(
                gradients,
                include=lambda path: "value_head" in path,
            ),
        }
    )

    tangent = jax.random.rademacher(
        tangent_key,
        observations.shape,
        dtype=jnp.float32,
    )

    def encoder_features(current_observations: jax.Array) -> jax.Array:
        _, current_diagnostics = diagnostic_model.apply(
            {"params": params},
            current_observations,
            mutable=["diagnostics"],
        )
        return _single_sown_value(
            current_diagnostics["diagnostics"]["encoder_features"]
        )

    _, feature_tangent = jax.jvp(
        encoder_features,
        (observations.astype(jnp.float32),),
        (tangent,),
    )
    normalized_input_tangent_rms = _rms(tangent / 255.0)
    metrics["encoder_jvp_rms_gain"] = (
        _rms(feature_tangent) / max(normalized_input_tangent_rms, 1e-12)
    )
    return metrics


def run_probe(
    *,
    observations: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, Any]:
    if observations.ndim != 4 or observations.shape[1:] != (64, 64, 3):
        raise ValueError(
            "observations must have shape (batch, 64, 64, 3), got "
            f"{observations.shape}"
        )
    if not seeds:
        raise ValueError("at least one initialization seed is required")
    observation_array = jnp.asarray(observations)
    arm_rows = []
    for arm, config in ARM_CONFIGS.items():
        per_seed = []
        scalar_rows = []
        for seed in seeds:
            metrics = _diagnose_seed(
                config=config,
                observations=observation_array,
                seed=seed,
            )
            per_seed.append({"seed": seed, "metrics": metrics})
            scalar_rows.append(metrics)
        arm_rows.append(
            {
                "arm": arm,
                "config": config,
                "per_seed": per_seed,
                "aggregate": summarize_scalar_rows(scalar_rows),
            }
        )
    return {
        "schema_version": "1.0",
        "experiment_id": "CR-PPO-0005",
        "residual_depth": RESIDUAL_DEPTH,
        "depth_scale": DEPTH_SCALE,
        "observation_identity": {
            "shape": list(observations.shape),
            "dtype": str(observations.dtype),
            "sha256": hashlib.sha256(observations.tobytes()).hexdigest(),
        },
        "initialization_seeds": list(seeds),
        "arms": arm_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--observation-vector-steps", type=int, default=3)
    parser.add_argument("--observation-seed", type=int, default=4_242)
    parser.add_argument("--num-initialization-seeds", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.num_envs,
        args.num_initialization_seeds,
    ) <= 0:
        raise ValueError("probe counts must be positive")
    if args.observation_vector_steps < 0:
        raise ValueError("observation_vector_steps must be non-negative")
    observations = _collect_observations(
        num_envs=args.num_envs,
        vector_steps=args.observation_vector_steps,
        seed=args.observation_seed,
    )
    atomic_write_json(
        args.output,
        run_probe(
            observations=observations,
            seeds=tuple(range(args.num_initialization_seeds)),
        ),
    )


if __name__ == "__main__":
    main()
