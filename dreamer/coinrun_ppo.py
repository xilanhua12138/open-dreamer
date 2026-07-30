"""PPO primitives and Gym3 semantics for a real-environment CoinRun policy.

This module intentionally does not depend on Procgen at import time.  The core
losses, model, checkpoint format, and environment adapter can therefore be
tested on CPU without installing the native Procgen package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import flax.linen as nn
from flax import serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax


class PPOLosses(NamedTuple):
    total_loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array


class PPOBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    old_log_probs: jax.Array
    old_values: jax.Array
    advantages: jax.Array
    returns: jax.Array


@dataclass(frozen=True)
class PPOHyperparameters:
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    minibatch_size: int = 2_048
    update_epochs: int = 3
    advantage_normalization: str = "minibatch"


def compute_gae(
    *,
    rewards: jax.Array,
    values: jax.Array,
    next_value: jax.Array,
    terminals: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute generalized advantages without crossing episode boundaries.

    ``terminals[t]`` describes whether the action at timestep ``t`` ended its
    episode.  This convention matches :class:`Gym3VectorEnvAdapter`, which
    associates the reward and next ``first`` flag with the action that caused
    them.
    """

    rewards = jnp.asarray(rewards)
    values = jnp.asarray(values)
    terminals = jnp.asarray(terminals, dtype=bool)
    next_value = jnp.asarray(next_value)
    if rewards.shape != values.shape or rewards.shape != terminals.shape:
        raise ValueError(
            "rewards, values, and terminals must have the same shape, got "
            f"{rewards.shape}, {values.shape}, and {terminals.shape}"
        )
    if rewards.ndim != 2:
        raise ValueError(
            f"rollouts must have shape (time, env), got {rewards.shape}"
        )
    if next_value.shape != rewards.shape[1:]:
        raise ValueError(
            f"next_value must have shape {rewards.shape[1:]}, got {next_value.shape}"
        )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

    following_values = jnp.concatenate(
        [values[1:], next_value[None]],
        axis=0,
    )

    def reverse_step(
        following_advantage: jax.Array,
        timestep: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        reward, value, following_value, terminal = timestep
        nonterminal = 1.0 - terminal.astype(jnp.float32)
        delta = reward + gamma * nonterminal * following_value - value
        advantage = (
            delta
            + gamma * gae_lambda * nonterminal * following_advantage
        )
        return advantage, advantage

    _, reversed_advantages = jax.lax.scan(
        reverse_step,
        jnp.zeros_like(next_value, dtype=jnp.float32),
        (
            rewards[::-1],
            values[::-1],
            following_values[::-1],
            terminals[::-1],
        ),
    )
    advantages = reversed_advantages[::-1]
    return advantages, advantages + values


def categorical_log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    logits = jnp.asarray(logits)
    actions = jnp.asarray(actions)
    if logits.ndim < 2:
        raise ValueError(f"logits must have an action axis, got {logits.shape}")
    if actions.shape != logits.shape[:-1]:
        raise ValueError(
            f"actions must have shape {logits.shape[:-1]}, got {actions.shape}"
        )
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]


def categorical_entropy(logits: jax.Array) -> jax.Array:
    logits = jnp.asarray(logits)
    if logits.ndim < 2:
        raise ValueError(f"logits must have an action axis, got {logits.shape}")
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probs)
    return -jnp.sum(probabilities * log_probs, axis=-1)


def ppo_loss(
    *,
    new_log_probs: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    new_values: jax.Array,
    old_values: jax.Array,
    returns: jax.Array,
    entropy: jax.Array,
    clip_epsilon: float,
    value_clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> PPOLosses:
    """Return the standard clipped PPO policy and value objectives."""

    arrays = tuple(
        jnp.asarray(value)
        for value in (
            new_log_probs,
            old_log_probs,
            advantages,
            new_values,
            old_values,
            returns,
            entropy,
        )
    )
    expected_shape = arrays[0].shape
    if any(value.shape != expected_shape for value in arrays[1:]):
        raise ValueError("all PPO loss inputs must have the same shape")
    if clip_epsilon < 0.0 or value_clip_epsilon < 0.0:
        raise ValueError("PPO clipping radii must be non-negative")

    (
        new_log_probs,
        old_log_probs,
        advantages,
        new_values,
        old_values,
        returns,
        entropy,
    ) = arrays
    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    unclipped_policy = ratio * advantages
    clipped_policy = (
        jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(unclipped_policy, clipped_policy))

    clipped_values = old_values + jnp.clip(
        new_values - old_values,
        -value_clip_epsilon,
        value_clip_epsilon,
    )
    value_loss_unclipped = jnp.square(new_values - returns)
    value_loss_clipped = jnp.square(clipped_values - returns)
    value_loss = 0.5 * jnp.mean(
        jnp.maximum(value_loss_unclipped, value_loss_clipped)
    )
    mean_entropy = jnp.mean(entropy)
    total_loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * mean_entropy
    )
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean(
        (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32)
    )
    return PPOLosses(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=mean_entropy,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )


def make_minibatch_indices(
    *,
    key: jax.Array,
    batch_size: int,
    minibatch_size: int,
    update_epochs: int,
) -> jax.Array:
    if min(batch_size, minibatch_size, update_epochs) <= 0:
        raise ValueError("batch_size, minibatch_size, and update_epochs must be positive")
    if batch_size % minibatch_size:
        raise ValueError(
            f"batch_size {batch_size} must be divisible by minibatch_size "
            f"{minibatch_size}"
        )
    keys = jax.random.split(key, update_epochs)
    permutations = [
        jax.random.permutation(epoch_key, batch_size) for epoch_key in keys
    ]
    return jnp.stack(permutations).reshape(
        update_epochs,
        batch_size // minibatch_size,
        minibatch_size,
    )


def make_ppo_minibatch_step(
    *,
    apply_fn: Callable[..., tuple[jax.Array, jax.Array]],
    hyperparameters: PPOHyperparameters,
) -> Callable[
    [train_state.TrainState, PPOBatch],
    tuple[train_state.TrainState, PPOLosses],
]:
    """Compile one reusable optimizer step for all PPO updates."""

    def step(
        state: train_state.TrainState,
        minibatch: PPOBatch,
    ) -> tuple[train_state.TrainState, PPOLosses]:
        def minibatch_loss(params: Any):
            logits, new_values = apply_fn(
                {"params": params},
                minibatch.observations,
            )
            losses = ppo_loss(
                new_log_probs=categorical_log_prob(
                    logits,
                    minibatch.actions,
                ),
                old_log_probs=minibatch.old_log_probs,
                advantages=minibatch.advantages,
                new_values=new_values,
                old_values=minibatch.old_values,
                returns=minibatch.returns,
                entropy=categorical_entropy(logits),
                clip_epsilon=hyperparameters.clip_epsilon,
                value_clip_epsilon=hyperparameters.value_clip_epsilon,
                value_coefficient=hyperparameters.value_coefficient,
                entropy_coefficient=hyperparameters.entropy_coefficient,
            )
            return losses.total_loss, losses

        (_, losses), gradients = jax.value_and_grad(
            minibatch_loss,
            has_aux=True,
        )(state.params)
        return state.apply_gradients(grads=gradients), losses

    return jax.jit(step)


def update_ppo(
    *,
    state: train_state.TrainState,
    batch: PPOBatch,
    key: jax.Array,
    hyperparameters: PPOHyperparameters,
    minibatch_step: Callable[
        [train_state.TrainState, PPOBatch],
        tuple[train_state.TrainState, PPOLosses],
    ]
    | None = None,
) -> tuple[train_state.TrainState, dict[str, float | int]]:
    """Apply shuffled PPO minibatch epochs and return aggregate diagnostics."""

    batch_size = int(batch.actions.shape[0])
    scalar_fields = (
        batch.actions,
        batch.old_log_probs,
        batch.old_values,
        batch.advantages,
        batch.returns,
    )
    if batch.observations.shape[0] != batch_size or any(
        field.shape != (batch_size,) for field in scalar_fields
    ):
        raise ValueError("PPOBatch fields must share one flat batch dimension")
    indices = make_minibatch_indices(
        key=key,
        batch_size=batch_size,
        minibatch_size=hyperparameters.minibatch_size,
        update_epochs=hyperparameters.update_epochs,
    )
    if hyperparameters.advantage_normalization not in {"batch", "minibatch"}:
        raise ValueError(
            "advantage_normalization must be 'batch' or 'minibatch', got "
            f"{hyperparameters.advantage_normalization!r}"
        )
    if hyperparameters.advantage_normalization == "batch":
        normalized_advantages = (
            batch.advantages - jnp.mean(batch.advantages)
        ) / (jnp.std(batch.advantages) + 1e-8)
        batch = batch._replace(advantages=normalized_advantages)

    if minibatch_step is None:
        minibatch_step = make_ppo_minibatch_step(
            apply_fn=state.apply_fn,
            hyperparameters=hyperparameters,
        )
    metric_rows: list[PPOLosses] = []
    for minibatch_indices in np.asarray(indices).reshape(
        -1,
        hyperparameters.minibatch_size,
    ):
        minibatch = jax.tree.map(
            lambda value: value[minibatch_indices],
            batch,
        )
        if hyperparameters.advantage_normalization == "minibatch":
            normalized_advantages = (
                minibatch.advantages - jnp.mean(minibatch.advantages)
            ) / (jnp.std(minibatch.advantages) + 1e-8)
            minibatch = minibatch._replace(advantages=normalized_advantages)
        state, losses = minibatch_step(state, minibatch)
        metric_rows.append(losses)

    stacked = jax.tree.map(lambda *values: jnp.stack(values), *metric_rows)
    metrics = {
        name: float(jax.device_get(jnp.mean(getattr(stacked, name))))
        for name in PPOLosses._fields
    }
    metrics["optimizer_minibatches"] = len(metric_rows)
    return state, metrics


def backbone_kernel_initializer(name: str) -> Callable[..., jax.Array]:
    if name == "orthogonal_sqrt2":
        return nn.initializers.orthogonal(np.sqrt(2.0))
    if name == "orthogonal_gain1":
        return nn.initializers.orthogonal(1.0)
    if name == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    raise ValueError(
        "backbone_kernel_init must be 'orthogonal_sqrt2', "
        f"'orthogonal_gain1' or 'glorot_uniform', got {name!r}"
    )


class ResidualBlock(nn.Module):
    channels: int
    backbone_kernel_init: str = "glorot_uniform"
    residual_branch_scale: float = 1.0
    residual_last_kernel_init: str = "same"
    residual_skip_init: bool = False
    record_diagnostics: bool = False

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        if not np.isfinite(self.residual_branch_scale):
            raise ValueError("residual_branch_scale must be finite")
        if self.residual_branch_scale < 0.0:
            raise ValueError("residual_branch_scale must be non-negative")
        if self.residual_last_kernel_init not in {"same", "zeros"}:
            raise ValueError(
                "residual_last_kernel_init must be 'same' or 'zeros', got "
                f"{self.residual_last_kernel_init!r}"
            )
        residual = inputs
        hidden = nn.relu(inputs)
        hidden = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=backbone_kernel_initializer(
                self.backbone_kernel_init
            ),
        )(hidden)
        hidden = nn.relu(hidden)
        hidden = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=(
                nn.initializers.zeros_init()
                if self.residual_last_kernel_init == "zeros"
                else backbone_kernel_initializer(self.backbone_kernel_init)
            ),
        )(hidden)
        branch_gain: jax.Array | float = self.residual_branch_scale
        if self.residual_skip_init:
            branch_gain = branch_gain * self.param(
                "skip_init_gain",
                nn.initializers.zeros_init(),
                (),
            )
        scaled_branch = hidden * branch_gain
        output = residual + scaled_branch
        if self.record_diagnostics:
            self.sow("diagnostics", "skip", residual)
            self.sow("diagnostics", "branch_unscaled", hidden)
            self.sow("diagnostics", "branch_scaled", scaled_branch)
            self.sow("diagnostics", "output", output)
        return output


class ImpalaConvSequence(nn.Module):
    channels: int
    backbone_kernel_init: str = "glorot_uniform"
    residual_branch_scale: float = 1.0
    residual_last_kernel_init: str = "same"
    residual_skip_init: bool = False
    record_diagnostics: bool = False

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        hidden = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=backbone_kernel_initializer(
                self.backbone_kernel_init
            ),
        )(inputs)
        hidden = nn.max_pool(
            hidden,
            window_shape=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )
        hidden = ResidualBlock(
            self.channels,
            backbone_kernel_init=self.backbone_kernel_init,
            residual_branch_scale=self.residual_branch_scale,
            residual_last_kernel_init=self.residual_last_kernel_init,
            residual_skip_init=self.residual_skip_init,
            record_diagnostics=self.record_diagnostics,
        )(hidden)
        return ResidualBlock(
            self.channels,
            backbone_kernel_init=self.backbone_kernel_init,
            residual_branch_scale=self.residual_branch_scale,
            residual_last_kernel_init=self.residual_last_kernel_init,
            residual_skip_init=self.residual_skip_init,
            record_diagnostics=self.record_diagnostics,
        )(hidden)


class CoinRunActorCritic(nn.Module):
    """IMPALA-style visual encoder with categorical policy and value heads."""

    action_dim: int
    backbone_kernel_init: str = "glorot_uniform"
    residual_branch_scale: float = 1.0
    residual_last_kernel_init: str = "same"
    residual_skip_init: bool = False
    record_diagnostics: bool = False

    @nn.compact
    def __call__(self, observations: jax.Array) -> tuple[jax.Array, jax.Array]:
        observations = jnp.asarray(observations)
        if observations.ndim != 4 or observations.shape[1:] != (64, 64, 3):
            raise ValueError(
                "CoinRun observations must have shape (batch, 64, 64, 3), "
                f"got {observations.shape}"
            )
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        hidden = observations.astype(jnp.float32) / 255.0
        for channels in (16, 32, 32):
            hidden = ImpalaConvSequence(
                channels,
                backbone_kernel_init=self.backbone_kernel_init,
                residual_branch_scale=self.residual_branch_scale,
                residual_last_kernel_init=self.residual_last_kernel_init,
                residual_skip_init=self.residual_skip_init,
                record_diagnostics=self.record_diagnostics,
            )(hidden)
        hidden = nn.relu(hidden)
        hidden = hidden.reshape((hidden.shape[0], -1))
        hidden = nn.Dense(
            256,
            kernel_init=backbone_kernel_initializer(
                self.backbone_kernel_init
            ),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)
        hidden = nn.relu(hidden)
        if self.record_diagnostics:
            self.sow("diagnostics", "encoder_features", hidden)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros_init(),
            name="policy_head",
        )(hidden)
        values = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros_init(),
            name="value_head",
        )(hidden)
        return logits, values[..., 0]


def save_train_state(
    path: Path,
    state: Any,
    *,
    metadata: dict[str, Any],
) -> None:
    """Atomically persist a Flax train state and JSON-compatible metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        json.dumps(metadata)
    except TypeError as error:
        raise ValueError("checkpoint metadata must be JSON serializable") from error
    payload = {
        "schema_version": "1.0",
        "train_state": serialization.to_state_dict(state),
        "metadata": metadata,
    }
    encoded = serialization.msgpack_serialize(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def load_train_state(path: Path, target: Any) -> tuple[Any, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = serialization.msgpack_restore(path.read_bytes())
    if payload.get("schema_version") != "1.0":
        raise ValueError(
            f"unsupported PPO checkpoint schema: {payload.get('schema_version')}"
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("PPO checkpoint metadata must be an object")
    restored = serialization.from_state_dict(target, payload["train_state"])
    return restored, metadata


class PPOCoinRunPolicy:
    """Frozen stochastic CoinRun policy used to collect dynamics trajectories."""

    def __init__(
        self,
        *,
        model: CoinRunActorCritic,
        params: Any,
        metadata: dict[str, Any],
        num_envs: int,
        seed: int,
        temperature: float = 1.0,
        exploration_epsilon: float = 0.0,
        deterministic: bool = False,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not 0.0 <= exploration_epsilon <= 1.0:
            raise ValueError(
                "exploration_epsilon must be in [0, 1], "
                f"got {exploration_epsilon}"
            )
        self.model = model
        self.params = params
        self.metadata = dict(metadata)
        self.num_envs = num_envs
        self.temperature = temperature
        self.exploration_epsilon = exploration_epsilon
        self.deterministic = deterministic
        self.key = jax.random.PRNGKey(seed)
        self._apply = jax.jit(
            lambda params, observations: model.apply(
                {"params": params},
                observations,
            )
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        num_envs: int,
        seed: int,
        temperature: float = 1.0,
        exploration_epsilon: float = 0.0,
        deterministic: bool = False,
    ) -> "PPOCoinRunPolicy":
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = serialization.msgpack_restore(checkpoint.read_bytes())
        if payload.get("schema_version") != "1.0":
            raise ValueError(
                "unsupported PPO checkpoint schema: "
                f"{payload.get('schema_version')}"
            )
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("PPO checkpoint metadata must be an object")
        action_dim = int(metadata.get("action_dim", -1))
        from dreamer.coinrun import COINRUN_ACTION_DIM

        if action_dim != COINRUN_ACTION_DIM:
            raise ValueError(
                f"checkpoint action_dim={action_dim}, expected {COINRUN_ACTION_DIM}"
            )
        architecture = metadata.get("architecture")
        if architecture != "impala_cnn":
            raise ValueError(
                f"checkpoint architecture={architecture!r}, expected 'impala_cnn'"
            )
        train_state_payload = payload.get("train_state")
        if not isinstance(train_state_payload, dict) or "params" not in train_state_payload:
            raise ValueError("PPO checkpoint is missing train_state.params")
        train_config = metadata.get("train_config")
        backbone_kernel_init = (
            train_config.get(
                "backbone_kernel_init",
                "orthogonal_sqrt2",
            )
            if isinstance(train_config, dict)
            else "orthogonal_sqrt2"
        )
        residual_branch_scale = (
            float(train_config.get("residual_branch_scale", 1.0))
            if isinstance(train_config, dict)
            else 1.0
        )
        residual_last_kernel_init = (
            str(train_config.get("residual_last_kernel_init", "same"))
            if isinstance(train_config, dict)
            else "same"
        )
        residual_skip_init = (
            bool(train_config.get("residual_skip_init", False))
            if isinstance(train_config, dict)
            else False
        )
        model = CoinRunActorCritic(
            action_dim=action_dim,
            backbone_kernel_init=backbone_kernel_init,
            residual_branch_scale=residual_branch_scale,
            residual_last_kernel_init=residual_last_kernel_init,
            residual_skip_init=residual_skip_init,
        )
        return cls(
            model=model,
            params=train_state_payload["params"],
            metadata=metadata,
            num_envs=num_envs,
            seed=seed,
            temperature=temperature,
            exploration_epsilon=exploration_epsilon,
            deterministic=deterministic,
        )

    def sample(
        self,
        observations: np.ndarray,
        first: np.ndarray,
    ) -> np.ndarray:
        observations = np.asarray(observations)
        first = np.asarray(first, dtype=bool)
        expected_observations = (self.num_envs, 64, 64, 3)
        if observations.shape != expected_observations:
            raise ValueError(
                f"observations must have shape {expected_observations}, "
                f"got {observations.shape}"
            )
        if first.shape != (self.num_envs,):
            raise ValueError(
                f"first must have shape {(self.num_envs,)}, got {first.shape}"
            )
        del first  # Feed-forward policy has no recurrent state to reset.
        self.key, action_key, explore_key, random_key = jax.random.split(
            self.key,
            4,
        )
        logits, _ = self._apply(self.params, jnp.asarray(observations))
        if self.deterministic:
            actions = jnp.argmax(logits, axis=-1)
        else:
            actions = jax.random.categorical(
                action_key,
                logits / self.temperature,
                axis=-1,
            )
        if self.exploration_epsilon:
            explore = (
                jax.random.uniform(explore_key, shape=actions.shape)
                < self.exploration_epsilon
            )
            random_actions = jax.random.randint(
                random_key,
                shape=actions.shape,
                minval=0,
                maxval=self.model.action_dim,
            )
            actions = jnp.where(explore, random_actions, actions)
        return np.asarray(jax.device_get(actions), dtype=np.int32)


@dataclass(frozen=True)
class Gym3Observation:
    observation: np.ndarray
    incoming_rewards: np.ndarray
    first: np.ndarray


@dataclass(frozen=True)
class Gym3Transition:
    observation: np.ndarray
    first: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    next_observation: np.ndarray
    next_first: np.ndarray


class Gym3VectorEnvAdapter:
    """Expose action-aligned transitions from Gym3's observe/act protocol."""

    def __init__(self, env: Any, *, action_dim: int) -> None:
        self.env = env
        self.num_envs = int(env.num)
        self.action_dim = int(action_dim)
        if self.num_envs <= 0:
            raise ValueError(f"env.num must be positive, got {self.num_envs}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        self._current = self._observe()

    @property
    def current(self) -> Gym3Observation:
        return self._current

    def _observe(self) -> Gym3Observation:
        rewards, observations, first = self.env.observe()
        if not isinstance(observations, dict) or "rgb" not in observations:
            raise ValueError("Gym3 CoinRun observations must contain an rgb field")
        rgb = np.asarray(observations["rgb"])
        expected_shape = (self.num_envs, 64, 64, 3)
        if rgb.shape != expected_shape:
            raise ValueError(
                f"CoinRun rgb observations must have shape {expected_shape}, "
                f"got {rgb.shape}"
            )
        rewards = np.asarray(rewards, dtype=np.float32)
        first = np.asarray(first, dtype=bool)
        expected_vector = (self.num_envs,)
        if rewards.shape != expected_vector:
            raise ValueError(
                f"Gym3 rewards must have shape {expected_vector}, got {rewards.shape}"
            )
        if first.shape != expected_vector:
            raise ValueError(
                f"Gym3 first flags must have shape {expected_vector}, got {first.shape}"
            )
        return Gym3Observation(
            observation=rgb.copy(),
            incoming_rewards=rewards.copy(),
            first=first.copy(),
        )

    def step(self, actions: np.ndarray) -> Gym3Transition:
        actions = np.asarray(actions)
        expected_shape = (self.num_envs,)
        if actions.shape != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}, got {actions.shape}"
            )
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(f"actions must be integers, got {actions.dtype}")
        actions = actions.astype(np.int32, copy=False)
        if np.any(actions < 0) or np.any(actions >= self.action_dim):
            raise ValueError(
                f"actions outside [0, {self.action_dim}): {actions.tolist()}"
            )

        previous = self._current
        self.env.act(actions)
        following = self._observe()
        transition = Gym3Transition(
            observation=previous.observation,
            first=previous.first,
            actions=actions.copy(),
            rewards=following.incoming_rewards,
            terminals=following.first,
            next_observation=following.observation,
            next_first=following.first,
        )
        self._current = following
        return transition
