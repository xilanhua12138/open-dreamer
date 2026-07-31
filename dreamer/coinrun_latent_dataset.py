"""CoinRun raw-video to latent-record protocol helpers.

The offline tokenizer pipeline must preserve the transition convention used by
the collector:

``observation_t, action_t, reward_after_action_t``.

Keeping these helpers independent from JAX, Grain and ArrayRecord makes the
alignment contract cheap to test before a GPU run is launched.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


TRANSITION_ALIGNMENT = (
    "latents[t]=tokenizer(raw_video[t]), "
    "actions[t]=action_t, rewards[t]=reward_after_action_t"
)


class WindowRng(Protocol):
    def random(self) -> float: ...

    def choice(self, values: np.ndarray) -> Any: ...

    def integers(self, low: int, high: int) -> Any: ...


def _aligned_vector(
    raw_record: dict[str, Any],
    name: str,
    *,
    sequence_length: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    if name not in raw_record:
        raise ValueError(f"raw CoinRun record is missing {name}")
    value = np.asarray(raw_record[name], dtype=dtype)
    if value.shape != (sequence_length,):
        raise ValueError(
            f"raw {name} must have shape ({sequence_length},), got {value.shape}"
        )
    return value


def build_coinrun_latent_record(
    raw_record: dict[str, Any],
    latents: np.ndarray,
    *,
    record_index: int,
    raw_tree_sha256: str,
) -> dict[str, Any]:
    """Build one msgpack-ready latent record without changing time alignment."""
    sequence_length = int(raw_record["sequence_length"])
    latents = np.asarray(latents)
    if latents.ndim != 3:
        raise ValueError(
            "CoinRun latents must have shape "
            f"(time, n_latents, d_bottleneck), got {latents.shape}"
        )
    if latents.shape[0] != sequence_length:
        raise ValueError(
            f"latent time axis {latents.shape[0]} does not match "
            f"raw sequence_length {sequence_length}"
        )

    actions = _aligned_vector(
        raw_record,
        "actions",
        sequence_length=sequence_length,
        dtype=np.dtype(np.int32),
    )
    rewards = _aligned_vector(
        raw_record,
        "rewards",
        sequence_length=sequence_length,
        dtype=np.dtype(np.float32),
    )
    terminals = _aligned_vector(
        raw_record,
        "terminals",
        sequence_length=sequence_length,
        dtype=np.dtype(bool),
    )

    return {
        "latents": latents,
        "actions": {
            "binary": None,
            "categorical": actions.copy(),
            "continuous": None,
        },
        "rewards": rewards.copy(),
        "terminals": terminals.copy(),
        "source": {
            "record_index": int(record_index),
            "raw_tree_sha256": raw_tree_sha256,
            "transition_alignment": TRANSITION_ALIGNMENT,
        },
    }


def validate_coinrun_latent_pair(
    raw_record: dict[str, Any],
    latent_record: dict[str, Any],
) -> dict[str, Any]:
    """Validate that a latent record preserves every non-pixel time series."""
    sequence_length = int(raw_record["sequence_length"])
    latents = np.asarray(latent_record["latents"])
    if latents.ndim != 3 or latents.shape[0] != sequence_length:
        raise ValueError(
            "latent shape does not preserve raw sequence length: "
            f"raw={sequence_length}, latent={latents.shape}"
        )

    raw_actions = _aligned_vector(
        raw_record,
        "actions",
        sequence_length=sequence_length,
        dtype=np.dtype(np.int32),
    )
    latent_actions = np.asarray(
        latent_record["actions"]["categorical"],
        dtype=np.int32,
    )
    if not np.array_equal(raw_actions, latent_actions):
        raise ValueError("latent categorical actions differ from raw record")

    raw_rewards = _aligned_vector(
        raw_record,
        "rewards",
        sequence_length=sequence_length,
        dtype=np.dtype(np.float32),
    )
    latent_rewards = np.asarray(latent_record["rewards"], dtype=np.float32)
    if not np.array_equal(raw_rewards, latent_rewards):
        raise ValueError("latent rewards differ from raw record")

    raw_terminals = _aligned_vector(
        raw_record,
        "terminals",
        sequence_length=sequence_length,
        dtype=np.dtype(bool),
    )
    latent_terminals = np.asarray(latent_record["terminals"], dtype=bool)
    if not np.array_equal(raw_terminals, latent_terminals):
        raise ValueError("latent terminals differ from raw record")

    return {
        "sequence_length": sequence_length,
        "latent_shape": list(latents.shape),
        "actions_aligned": True,
        "rewards_aligned": True,
        "terminals_aligned": True,
    }


def choose_coinrun_window_start(
    *,
    episode_len: int,
    seq_len: int,
    rewards: np.ndarray | None,
    p_include_reward: float,
    rng: WindowRng,
) -> int:
    """Choose a legal CoinRun crop with the raw-video reward-bias semantics."""
    if episode_len < seq_len:
        raise ValueError(
            f"episode length {episode_len} is shorter than requested "
            f"sequence length {seq_len}"
        )
    if not 0.0 <= p_include_reward <= 1.0:
        raise ValueError(
            f"p_include_reward must be in [0, 1], got {p_include_reward}"
        )

    max_start_idx = episode_len - seq_len
    start_idx: int | None = None
    if (
        rewards is not None
        and p_include_reward > 0.0
        and rng.random() < p_include_reward
    ):
        rewards = np.asarray(rewards)
        if rewards.shape != (episode_len,):
            raise ValueError(
                f"rewards must have shape ({episode_len},), got {rewards.shape}"
            )
        reward_ts = np.flatnonzero(rewards > 0)
        if reward_ts.size:
            reward_t = int(rng.choice(reward_ts))
            start_min = max(0, reward_t - (seq_len - 1))
            start_max = min(reward_t, max_start_idx)
            start_idx = int(rng.integers(start_min, start_max + 1))

    if start_idx is None:
        start_idx = int(rng.integers(0, max_start_idx + 1))
    return start_idx
