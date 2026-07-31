"""Training utilities for the real-environment CoinRun PPO collector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.coinrun_ppo import (
    PPOBatch,
    categorical_log_prob,
    compute_gae,
)


@dataclass(frozen=True)
class PPOTrainConfig:
    total_env_steps: int = 25_165_824
    num_envs: int = 64
    rollout_steps: int = 256
    num_minibatches: int = 8
    update_epochs: int = 3
    learning_rate: float = 5e-4
    adam_epsilon: float = 1e-5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    reward_clip: float = 10.0
    reward_normalization_gamma: float = 0.99
    advantage_normalization: str = "minibatch"
    backbone_kernel_init: str = "glorot_uniform"
    seed: int = 0
    start_level: int = 0
    num_levels: int = 200
    distribution_mode: str = "easy"

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_env_steps // self.batch_size

    @property
    def optimizer_minibatches_per_update(self) -> int:
        return self.num_minibatches * self.update_epochs

    def validate(self) -> None:
        positive_integer_fields = {
            "total_env_steps": self.total_env_steps,
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "num_minibatches": self.num_minibatches,
            "update_epochs": self.update_epochs,
            "num_levels": self.num_levels,
        }
        invalid = {
            name: value
            for name, value in positive_integer_fields.items()
            if value <= 0
        }
        if invalid:
            raise ValueError(f"PPO integer fields must be positive: {invalid}")
        if self.total_env_steps % self.batch_size:
            raise ValueError(
                f"total_env_steps {self.total_env_steps} must be divisible by "
                f"num_envs*rollout_steps={self.batch_size}"
            )
        if self.batch_size % self.num_minibatches:
            raise ValueError(
                f"batch_size {self.batch_size} must be divisible by "
                f"num_minibatches {self.num_minibatches}"
            )
        unit_interval_fields = {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "reward_normalization_gamma": self.reward_normalization_gamma,
        }
        invalid_fractions = {
            name: value
            for name, value in unit_interval_fields.items()
            if not 0.0 <= value <= 1.0
        }
        if invalid_fractions:
            raise ValueError(
                f"PPO discount fields must be in [0, 1]: {invalid_fractions}"
            )
        positive_float_fields = {
            "learning_rate": self.learning_rate,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "reward_clip": self.reward_clip,
        }
        invalid_positive = {
            name: value
            for name, value in positive_float_fields.items()
            if value <= 0.0
        }
        if invalid_positive:
            raise ValueError(f"PPO float fields must be positive: {invalid_positive}")
        if self.clip_epsilon < 0.0 or self.value_clip_epsilon < 0.0:
            raise ValueError("PPO clipping radii must be non-negative")
        if self.advantage_normalization not in {"batch", "minibatch"}:
            raise ValueError(
                "advantage_normalization must be 'batch' or 'minibatch', got "
                f"{self.advantage_normalization!r}"
            )
        if self.backbone_kernel_init not in {
            "orthogonal_sqrt2",
            "glorot_uniform",
        }:
            raise ValueError(
                "backbone_kernel_init must be 'orthogonal_sqrt2' or "
                f"'glorot_uniform', got {self.backbone_kernel_init!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "batch_size": self.batch_size,
                "minibatch_size": self.minibatch_size,
                "num_updates": self.num_updates,
                "optimizer_minibatches_per_update": (
                    self.optimizer_minibatches_per_update
                ),
            }
        )
        return payload


class RunningMeanVariance:
    def __init__(self, *, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.variance = 1.0
        self.count = float(epsilon)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            raise ValueError("cannot update running statistics with no values")
        batch_mean = float(np.mean(values))
        batch_variance = float(np.var(values))
        batch_count = values.size
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        combined_mean = self.mean + delta * batch_count / total_count
        left_moment = self.variance * self.count
        right_moment = batch_variance * batch_count
        correction = delta * delta * self.count * batch_count / total_count
        self.mean = combined_mean
        self.variance = (left_moment + right_moment + correction) / total_count
        self.count = total_count

    def state_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "count": self.count,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        required = {"mean", "variance", "count"}
        if set(payload) != required:
            raise ValueError(
                f"running-stat state must contain exactly {sorted(required)}"
            )
        self.mean = float(payload["mean"])
        self.variance = float(payload["variance"])
        self.count = float(payload["count"])
        if self.variance < 0.0 or self.count <= 0.0:
            raise ValueError("invalid running-stat variance/count")


class RewardNormalizer:
    """Normalize rewards by discounted-return variance, then clip."""

    def __init__(
        self,
        *,
        num_envs: int,
        gamma: float,
        clip: float,
        epsilon: float = 1e-8,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if clip <= 0.0 or epsilon <= 0.0:
            raise ValueError("clip and epsilon must be positive")
        self.num_envs = num_envs
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self.discounted_returns = np.zeros(num_envs, dtype=np.float64)
        self.running = RunningMeanVariance()

    def normalize(
        self,
        rewards: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float32)
        terminals = np.asarray(terminals, dtype=bool)
        expected_shape = (self.num_envs,)
        if rewards.shape != expected_shape or terminals.shape != expected_shape:
            raise ValueError(
                f"rewards and terminals must have shape {expected_shape}"
            )
        self.discounted_returns = (
            self.gamma * self.discounted_returns + rewards
        )
        self.running.update(self.discounted_returns)
        normalized = rewards / np.sqrt(self.running.variance + self.epsilon)
        normalized = np.clip(normalized, -self.clip, self.clip).astype(np.float32)
        self.discounted_returns = np.where(
            terminals,
            0.0,
            self.discounted_returns,
        )
        return normalized

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "gamma": self.gamma,
            "clip": self.clip,
            "epsilon": self.epsilon,
            "discounted_returns": self.discounted_returns.tolist(),
            "running": self.running.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if int(payload.get("num_envs", -1)) != self.num_envs:
            raise ValueError("reward-normalizer num_envs mismatch")
        if float(payload.get("gamma", -1.0)) != self.gamma:
            raise ValueError("reward-normalizer gamma mismatch")
        if float(payload.get("clip", -1.0)) != self.clip:
            raise ValueError("reward-normalizer clip mismatch")
        restored_returns = np.asarray(
            payload.get("discounted_returns"),
            dtype=np.float64,
        )
        if restored_returns.shape != (self.num_envs,):
            raise ValueError("reward-normalizer discounted_returns shape mismatch")
        self.discounted_returns = restored_returns
        self.running.load_state_dict(payload["running"])


def sample_actions_and_values(
    *,
    apply_fn: Any,
    params: Any,
    observations: jax.Array,
    key: jax.Array,
    temperature: float = 1.0,
    deterministic: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    next_key, action_key = jax.random.split(key)
    logits, values = apply_fn({"params": params}, observations)
    if deterministic:
        actions = jnp.argmax(logits, axis=-1)
    else:
        actions = jax.random.categorical(
            action_key,
            logits / temperature,
            axis=-1,
        )
    log_probs = categorical_log_prob(logits, actions)
    return actions.astype(jnp.int32), log_probs, values, next_key


def make_action_sampler(
    *,
    apply_fn: Callable[..., tuple[jax.Array, jax.Array]],
    temperature: float = 1.0,
    deterministic: bool = False,
) -> Callable[
    [Any, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
]:
    """Compile one reusable visual-policy sampler for rollout collection."""

    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    def sample(
        params: Any,
        observations: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return sample_actions_and_values(
            apply_fn=apply_fn,
            params=params,
            observations=observations,
            key=key,
            temperature=temperature,
            deterministic=deterministic,
        )

    return jax.jit(sample)


def build_training_batch(
    *,
    observations: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    values: jax.Array,
    rewards: jax.Array,
    terminals: jax.Array,
    next_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> PPOBatch:
    observations = jnp.asarray(observations)
    actions = jnp.asarray(actions)
    old_log_probs = jnp.asarray(old_log_probs)
    values = jnp.asarray(values)
    rewards = jnp.asarray(rewards)
    terminals = jnp.asarray(terminals)
    rollout_shape = actions.shape
    if actions.ndim != 2:
        raise ValueError(f"rollout actions must have shape (time, env), got {rollout_shape}")
    for name, value in (
        ("old_log_probs", old_log_probs),
        ("values", values),
        ("rewards", rewards),
        ("terminals", terminals),
    ):
        if value.shape != rollout_shape:
            raise ValueError(
                f"{name} must have shape {rollout_shape}, got {value.shape}"
            )
    if observations.shape[:2] != rollout_shape:
        raise ValueError(
            f"observations must begin with {rollout_shape}, got {observations.shape}"
        )
    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        next_value=next_value,
        terminals=terminals,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    flat_size = rollout_shape[0] * rollout_shape[1]
    return PPOBatch(
        observations=observations.reshape((flat_size, *observations.shape[2:])),
        actions=actions.reshape((flat_size,)),
        old_log_probs=old_log_probs.reshape((flat_size,)),
        old_values=values.reshape((flat_size,)),
        advantages=advantages.reshape((flat_size,)),
        returns=returns.reshape((flat_size,)),
    )
