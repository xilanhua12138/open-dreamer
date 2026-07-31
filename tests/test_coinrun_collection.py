from __future__ import annotations

import unittest

import numpy as np

from dreamer.coinrun_collection import CoinRunRecordAccumulator
from dreamer.coinrun_ppo import Gym3Transition


def transition(
    *,
    frame_value: int,
    actions: tuple[int, int],
    rewards: tuple[float, float],
    terminals: tuple[bool, bool],
) -> Gym3Transition:
    observation = np.full((2, 64, 64, 3), frame_value, dtype=np.uint8)
    return Gym3Transition(
        observation=observation,
        first=np.asarray([frame_value == 0, frame_value == 0]),
        actions=np.asarray(actions, dtype=np.int32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminals=np.asarray(terminals, dtype=bool),
        next_observation=np.full(
            (2, 64, 64, 3),
            frame_value + 1,
            dtype=np.uint8,
        ),
        next_first=np.asarray(terminals, dtype=bool),
    )


class CoinRunRecordAccumulatorTests(unittest.TestCase):
    def test_records_keep_observation_action_and_resulting_reward_aligned(
        self,
    ) -> None:
        accumulator = CoinRunRecordAccumulator(num_envs=2, frames_per_record=2)
        first = accumulator.append(
            transition(
                frame_value=0,
                actions=(7, 8),
                rewards=(0.25, 1.0),
                terminals=(False, True),
            )
        )
        second = accumulator.append(
            transition(
                frame_value=1,
                actions=(8, 4),
                rewards=(2.0, 3.0),
                terminals=(False, False),
            )
        )

        self.assertEqual(first, [])
        self.assertEqual(len(second), 1)
        env_index, record = second[0]
        self.assertEqual(env_index, 0)
        video = np.frombuffer(record["raw_video"], dtype=np.uint8).reshape(
            2,
            64,
            64,
            3,
        )
        self.assertEqual(video[:, 0, 0, 0].tolist(), [0, 1])
        self.assertEqual(record["actions"].tolist(), [7, 8])
        self.assertEqual(record["rewards"].tolist(), [0.25, 2.0])
        self.assertEqual(record["terminals"].tolist(), [False, False])
        self.assertEqual(record["sequence_length"], 2)

    def test_terminal_discards_partial_chunk_instead_of_crossing_episode(
        self,
    ) -> None:
        accumulator = CoinRunRecordAccumulator(num_envs=2, frames_per_record=3)
        accumulator.append(
            transition(
                frame_value=0,
                actions=(7, 8),
                rewards=(0.0, 1.0),
                terminals=(False, True),
            )
        )
        accumulator.append(
            transition(
                frame_value=1,
                actions=(8, 4),
                rewards=(0.0, 0.0),
                terminals=(False, False),
            )
        )
        records = accumulator.append(
            transition(
                frame_value=2,
                actions=(8, 7),
                rewards=(1.0, 0.0),
                terminals=(False, False),
            )
        )

        self.assertEqual([env_index for env_index, _ in records], [0])
        self.assertEqual(accumulator.buffer_lengths, (0, 2))

    def test_rejects_transition_with_wrong_environment_count(self) -> None:
        accumulator = CoinRunRecordAccumulator(num_envs=2, frames_per_record=2)
        broken = transition(
            frame_value=0,
            actions=(7, 8),
            rewards=(0.0, 0.0),
            terminals=(False, False),
        )
        broken = Gym3Transition(
            observation=broken.observation[:1],
            first=broken.first[:1],
            actions=broken.actions[:1],
            rewards=broken.rewards[:1],
            terminals=broken.terminals[:1],
            next_observation=broken.next_observation[:1],
            next_first=broken.next_first[:1],
        )

        with self.assertRaisesRegex(ValueError, "num_envs"):
            accumulator.append(broken)


if __name__ == "__main__":
    unittest.main()
