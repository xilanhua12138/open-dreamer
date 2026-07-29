from __future__ import annotations

import unittest

import numpy as np

from dreamer.coinrun import (
    COINRUN_CONTROL_ACTIONS,
    StructuredCoinRunPolicy,
)


class StructuredCoinRunPolicyTests(unittest.TestCase):
    def test_policy_is_deterministic_for_seed_and_reset_sequence(self) -> None:
        left = StructuredCoinRunPolicy(num_envs=8, seed=123)
        right = StructuredCoinRunPolicy(num_envs=8, seed=123)
        reset_sequence = [
            np.ones(8, dtype=bool),
            np.zeros(8, dtype=bool),
            np.asarray([True, False, False, True, False, False, False, False]),
        ]

        for first in reset_sequence:
            self.assertEqual(left.sample(first).tolist(), right.sample(first).tolist())

    def test_policy_emits_only_supported_control_actions_with_persistence(self) -> None:
        policy = StructuredCoinRunPolicy(num_envs=64, seed=20260729)
        previous = None
        repeated = 0
        total_transitions = 0
        seen: set[int] = set()

        for step in range(400):
            first = np.ones(64, dtype=bool) if step == 0 else np.zeros(64, dtype=bool)
            actions = policy.sample(first)
            seen.update(int(action) for action in actions)
            if previous is not None:
                repeated += int(np.sum(actions == previous))
                total_transitions += actions.size
            previous = actions

        self.assertEqual(seen, set(COINRUN_CONTROL_ACTIONS))
        self.assertGreater(repeated / total_transitions, 0.70)

    def test_policy_rejects_invalid_environment_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_envs"):
            StructuredCoinRunPolicy(num_envs=0, seed=1)


if __name__ == "__main__":
    unittest.main()
