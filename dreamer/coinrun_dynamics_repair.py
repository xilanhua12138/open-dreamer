"""Protocol checks and action controls for repaired CoinRun dynamics runs."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dreamer.actions import Actions


def assess_dynamics_repair(
    *,
    shortcut_metrics: dict[str, Any],
    action_metrics: dict[str, Any],
    min_mean_frame_psnr_db: float,
    min_horizon_16_psnr_db: float,
    min_shuffled_advantage_db: float,
    min_noop_advantage_db: float,
) -> dict[str, Any]:
    """Apply the preregistered absolute-quality and action-use checks."""

    mean_frame_psnr_db = float(shortcut_metrics["mean_frame_psnr_db"])
    horizon_16_psnr_db = float(
        shortcut_metrics["psnr_by_horizon_db"]["16"]
    )
    aligned_vs_shuffled_db = float(
        action_metrics["aligned_psnr_advantage_db"]["batch_shuffled"]["16"]
    )
    aligned_vs_noop_db = float(
        action_metrics["aligned_psnr_advantage_db"]["all_noop"]["16"]
    )

    quality_pass = (
        mean_frame_psnr_db >= min_mean_frame_psnr_db
        and horizon_16_psnr_db >= min_horizon_16_psnr_db
    )
    action_use_pass = (
        aligned_vs_shuffled_db >= min_shuffled_advantage_db
        and aligned_vs_noop_db >= min_noop_advantage_db
    )

    return {
        "observed": {
            "mean_frame_psnr_db": mean_frame_psnr_db,
            "horizon_16_psnr_db": horizon_16_psnr_db,
            "aligned_vs_batch_shuffled_horizon_16_db": aligned_vs_shuffled_db,
            "aligned_vs_all_noop_horizon_16_db": aligned_vs_noop_db,
        },
        "thresholds": {
            "min_mean_frame_psnr_db": min_mean_frame_psnr_db,
            "min_horizon_16_psnr_db": min_horizon_16_psnr_db,
            "min_shuffled_advantage_db": min_shuffled_advantage_db,
            "min_noop_advantage_db": min_noop_advantage_db,
        },
        "quality_pass": quality_pass,
        "action_use_pass": action_use_pass,
        "overall_pass": quality_pass and action_use_pass,
    }


def validate_reward_windowing(
    *,
    record_frames: int,
    short_window: int,
    long_window: int,
    p_include_reward: float,
) -> dict[str, Any]:
    """Validate that reward-biased slicing can change the sampled window."""

    if min(record_frames, short_window, long_window) <= 0:
        raise ValueError("record_frames and window lengths must be positive")
    if not 0.0 <= p_include_reward <= 1.0:
        raise ValueError("p_include_reward must be in [0, 1]")
    if short_window > long_window:
        raise ValueError("short_window must be <= long_window")
    if long_window > record_frames:
        raise ValueError(
            f"long_window={long_window} exceeds record_frames={record_frames}"
        )

    short_start_positions = record_frames - short_window + 1
    long_start_positions = record_frames - long_window + 1
    reward_bias_operational = (
        p_include_reward > 0.0 and short_start_positions > 1
    )
    if p_include_reward > 0.0 and not reward_bias_operational:
        raise ValueError(
            "reward-biased slicing requires more than one start position; "
            f"record_frames={record_frames}, short_window={short_window}"
        )

    return {
        "record_frames": record_frames,
        "short_window": short_window,
        "long_window": long_window,
        "p_include_reward": p_include_reward,
        "short_start_positions": short_start_positions,
        "long_start_positions": long_start_positions,
        "reward_bias_operational": reward_bias_operational,
    }


def build_future_action_conditions(
    actions: Actions,
    *,
    context_frames: int,
    categorical_noop_action: int,
) -> dict[str, Actions]:
    """Build deterministic future-action corruption controls.

    Context actions stay byte-identical. Only actions used for predicted future
    frames are changed, so every condition shares the same observed history.
    """

    categorical = actions.categorical
    if categorical is None:
        raise ValueError("CoinRun action controls require categorical actions")
    if categorical.ndim != 2:
        raise ValueError(
            "categorical actions must have shape (batch, time), got "
            f"{categorical.shape}"
        )
    batch_size, sequence_length = categorical.shape
    if batch_size < 2:
        raise ValueError("batch_shuffled control requires at least two samples")
    if not 0 < context_frames < sequence_length:
        raise ValueError(
            "context_frames must be in "
            f"[1, {sequence_length - 1}], got {context_frames}"
        )

    def replace_future(
        source: jax.Array | None,
        future: jax.Array | None,
    ) -> jax.Array | None:
        if source is None:
            return None
        if future is None:
            raise ValueError("future action payload cannot be None")
        return jnp.concatenate([source[:, :context_frames], future], axis=1)

    categorical_future = categorical[:, context_frames:]
    shuffled_categorical = jnp.roll(categorical_future, shift=1, axis=0)
    shifted_categorical = jnp.concatenate(
        [
            jnp.full_like(
                categorical_future[:, :1],
                categorical_noop_action,
            ),
            categorical_future[:, :-1],
        ],
        axis=1,
    )
    noop_categorical = jnp.full_like(
        categorical_future,
        categorical_noop_action,
    )

    def roll_future(source: jax.Array | None) -> jax.Array | None:
        if source is None:
            return None
        return replace_future(
            source,
            jnp.roll(source[:, context_frames:], shift=1, axis=0),
        )

    def shift_future(source: jax.Array | None, fill: float) -> jax.Array | None:
        if source is None:
            return None
        future = source[:, context_frames:]
        shifted = jnp.concatenate(
            [jnp.full_like(future[:, :1], fill), future[:, :-1]],
            axis=1,
        )
        return replace_future(source, shifted)

    def zero_future(source: jax.Array | None) -> jax.Array | None:
        if source is None:
            return None
        return replace_future(source, jnp.zeros_like(source[:, context_frames:]))

    return {
        "aligned": actions,
        "batch_shuffled": Actions(
            binary=roll_future(actions.binary),
            categorical=replace_future(categorical, shuffled_categorical),
            continuous=roll_future(actions.continuous),
        ),
        "one_step_shifted": Actions(
            binary=shift_future(actions.binary, 0),
            categorical=replace_future(categorical, shifted_categorical),
            continuous=shift_future(actions.continuous, 0.0),
        ),
        "all_noop": Actions(
            binary=zero_future(actions.binary),
            categorical=replace_future(categorical, noop_categorical),
            continuous=zero_future(actions.continuous),
        ),
    }
