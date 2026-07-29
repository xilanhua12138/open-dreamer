"""Episode-safe CoinRun record assembly for dynamics data collection."""

from __future__ import annotations

from typing import Any

import numpy as np

from dreamer.coinrun_ppo import Gym3Transition


class CoinRunRecordAccumulator:
    """Build fixed-length records without crossing auto-reset boundaries.

    Each stored tuple is ``(observation_t, action_t, reward_after_action_t)``.
    The action causes the transition to the following observation.  This makes
    reward accounting auditable while keeping the action convention expected
    by dynamics training explicit.  The world-model pipeline may ignore reward
    fields; they are retained to measure collector quality and success rate.
    """

    def __init__(self, *, num_envs: int, frames_per_record: int) -> None:
        if min(num_envs, frames_per_record) <= 0:
            raise ValueError("num_envs and frames_per_record must be positive")
        self.num_envs = num_envs
        self.frames_per_record = frames_per_record
        self._videos: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._rewards: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self._terminals: list[list[np.ndarray]] = [[] for _ in range(num_envs)]

    @property
    def buffer_lengths(self) -> tuple[int, ...]:
        return tuple(len(buffer) for buffer in self._videos)

    def _validate(self, transition: Gym3Transition) -> None:
        observation_shape = (self.num_envs, 64, 64, 3)
        vector_shape = (self.num_envs,)
        if transition.observation.shape != observation_shape:
            raise ValueError(
                "transition observation num_envs/shape mismatch: "
                f"expected {observation_shape}, got {transition.observation.shape}"
            )
        for name in ("first", "actions", "rewards", "terminals", "next_first"):
            value = np.asarray(getattr(transition, name))
            if value.shape != vector_shape:
                raise ValueError(
                    f"transition {name} num_envs/shape mismatch: "
                    f"expected {vector_shape}, got {value.shape}"
                )
        if transition.next_observation.shape != observation_shape:
            raise ValueError(
                "transition next_observation num_envs/shape mismatch: "
                f"expected {observation_shape}, "
                f"got {transition.next_observation.shape}"
            )

    def _clear(self, env_index: int) -> None:
        self._videos[env_index].clear()
        self._actions[env_index].clear()
        self._rewards[env_index].clear()
        self._terminals[env_index].clear()

    def append(
        self,
        transition: Gym3Transition,
    ) -> list[tuple[int, dict[str, Any]]]:
        self._validate(transition)
        records: list[tuple[int, dict[str, Any]]] = []
        for env_index in range(self.num_envs):
            self._videos[env_index].append(
                np.asarray(transition.observation[env_index], dtype=np.uint8).copy()
            )
            self._actions[env_index].append(
                np.asarray(transition.actions[env_index], dtype=np.int32).copy()
            )
            self._rewards[env_index].append(
                np.asarray(transition.rewards[env_index], dtype=np.float32).copy()
            )
            self._terminals[env_index].append(
                np.asarray(transition.terminals[env_index], dtype=bool).copy()
            )
            if len(self._videos[env_index]) == self.frames_per_record:
                record = {
                    "raw_video": np.stack(self._videos[env_index])
                    .astype(np.uint8)
                    .tobytes(),
                    "sequence_length": self.frames_per_record,
                    "actions": np.stack(self._actions[env_index]).astype(np.int32),
                    "rewards": np.stack(self._rewards[env_index]).astype(np.float32),
                    "terminals": np.stack(self._terminals[env_index]).astype(bool),
                }
                records.append((env_index, record))
                self._clear(env_index)
            if bool(transition.terminals[env_index]):
                self._clear(env_index)
        return records
