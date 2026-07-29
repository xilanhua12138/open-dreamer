from __future__ import annotations

import unittest

import numpy as np

from dreamer.coinrun import COINRUN_ACTION_DIM
from dreamer.coinrun_ppo import Gym3VectorEnvAdapter


class FakeGym3Env:
    def __init__(self) -> None:
        self.num = 2
        self._step = 0
        self.actions: list[np.ndarray] = []

    def observe(self):
        observations = {
            "rgb": np.full((2, 64, 64, 3), self._step, dtype=np.uint8)
        }
        if self._step == 0:
            rewards = np.asarray([0.0, 0.0], dtype=np.float32)
            first = np.asarray([True, True])
        elif self._step == 1:
            rewards = np.asarray([1.5, -2.0], dtype=np.float32)
            first = np.asarray([False, True])
        else:
            rewards = np.asarray([3.0, 4.0], dtype=np.float32)
            first = np.asarray([False, False])
        return rewards, observations, first

    def act(self, actions) -> None:
        self.actions.append(np.asarray(actions, dtype=np.int32).copy())
        self._step += 1


class Gym3VectorEnvAdapterTests(unittest.TestCase):
    def test_step_aligns_reward_and_terminal_with_action_that_caused_them(self) -> None:
        env = FakeGym3Env()
        adapter = Gym3VectorEnvAdapter(env, action_dim=COINRUN_ACTION_DIM)

        initial = adapter.current
        transition = adapter.step(np.asarray([7, 8], dtype=np.int32))

        self.assertEqual(initial.observation[:, 0, 0, 0].tolist(), [0, 0])
        self.assertEqual(initial.first.tolist(), [True, True])
        self.assertEqual(
            transition.observation[:, 0, 0, 0].tolist(),
            [0, 0],
        )
        self.assertEqual(
            transition.next_observation[:, 0, 0, 0].tolist(),
            [1, 1],
        )
        self.assertEqual(transition.actions.tolist(), [7, 8])
        self.assertEqual(transition.rewards.tolist(), [1.5, -2.0])
        self.assertEqual(transition.terminals.tolist(), [False, True])
        self.assertEqual(env.actions[0].tolist(), [7, 8])

    def test_step_rejects_wrong_action_shape(self) -> None:
        adapter = Gym3VectorEnvAdapter(
            FakeGym3Env(),
            action_dim=COINRUN_ACTION_DIM,
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter.step(np.asarray([1], dtype=np.int32))

    def test_step_rejects_out_of_range_actions_before_touching_env(self) -> None:
        env = FakeGym3Env()
        adapter = Gym3VectorEnvAdapter(env, action_dim=COINRUN_ACTION_DIM)

        with self.assertRaisesRegex(ValueError, "outside"):
            adapter.step(np.asarray([0, COINRUN_ACTION_DIM], dtype=np.int32))

        self.assertEqual(env.actions, [])

    def test_constructor_rejects_non_rgb_observation_shape(self) -> None:
        class BrokenEnv(FakeGym3Env):
            def observe(self):
                rewards, observations, first = super().observe()
                observations["rgb"] = np.zeros((2, 32, 32, 3), dtype=np.uint8)
                return rewards, observations, first

        with self.assertRaisesRegex(ValueError, "64, 64, 3"):
            Gym3VectorEnvAdapter(BrokenEnv(), action_dim=COINRUN_ACTION_DIM)


if __name__ == "__main__":
    unittest.main()
