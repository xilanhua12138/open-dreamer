"""CoinRun-specific action contract and structured data-collection policy."""

from __future__ import annotations

import numpy as np


COINRUN_ACTION_DIM = 15
COINRUN_NOOP_ACTION = 4
COINRUN_CONTROL_ACTIONS = (1, 2, 4, 5, 7, 8)


class StructuredCoinRunPolicy:
    """Emit persistent, diverse control macros with goal-directed coverage.

    Most episodes favor rightward running and jumping so the dataset reaches
    useful platform configurations. The remaining episodes use balanced
    persistent macros so left, jump and no-op effects remain identifiable.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        goal_directed_fraction: float = 0.65,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if not 0.0 <= goal_directed_fraction <= 1.0:
            raise ValueError(
                "goal_directed_fraction must be in [0, 1], "
                f"got {goal_directed_fraction}"
            )

        self.num_envs = num_envs
        self.goal_directed_fraction = goal_directed_fraction
        self.rng = np.random.RandomState(seed)
        self.goal_directed = np.zeros(num_envs, dtype=bool)
        self.remaining = np.zeros(num_envs, dtype=np.int32)
        self.current = np.full(
            num_envs,
            COINRUN_NOOP_ACTION,
            dtype=np.int32,
        )

    def sample(self, first: np.ndarray) -> np.ndarray:
        first = np.asarray(first, dtype=bool)
        if first.shape != (self.num_envs,):
            raise ValueError(
                f"first must have shape {(self.num_envs,)}, got {first.shape}"
            )

        reset_indices = np.flatnonzero(first)
        if reset_indices.size:
            self.goal_directed[reset_indices] = (
                self.rng.rand(reset_indices.size) < self.goal_directed_fraction
            )
            self.remaining[reset_indices] = 0

        for index in np.flatnonzero(self.remaining <= 0):
            if self.goal_directed[index]:
                self.current[index] = self.rng.choice(
                    (4, 7, 8),
                    p=(0.08, 0.17, 0.75),
                )
                self.remaining[index] = self.rng.randint(3, 11)
            else:
                self.current[index] = self.rng.choice(
                    COINRUN_CONTROL_ACTIONS,
                    p=(0.13, 0.15, 0.14, 0.14, 0.20, 0.24),
                )
                self.remaining[index] = self.rng.randint(4, 13)

        actions = self.current.copy()
        self.remaining -= 1
        return actions
