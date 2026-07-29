from __future__ import annotations

import unittest

import numpy as np

from dreamer.coinrun_ppo_evaluation import (
    CompletedEpisode,
    EpisodeStatisticsTracker,
    sample_video_frames,
    validate_level_split,
)


class CoinRunPPOEvaluationTests(unittest.TestCase):
    def test_episode_tracker_attributes_terminal_reward_to_completed_episode(
        self,
    ) -> None:
        tracker = EpisodeStatisticsTracker(num_envs=2)

        first = tracker.observe(
            rewards=np.asarray([1.0, 2.0], dtype=np.float32),
            terminals=np.asarray([False, True]),
        )
        second = tracker.observe(
            rewards=np.asarray([3.0, -1.0], dtype=np.float32),
            terminals=np.asarray([True, False]),
        )

        self.assertEqual(
            first,
            [CompletedEpisode(env_index=1, episode_return=2.0, length=1)],
        )
        self.assertEqual(
            second,
            [CompletedEpisode(env_index=0, episode_return=4.0, length=2)],
        )
        self.assertEqual(tracker.active_returns.tolist(), [0.0, -1.0])
        self.assertEqual(tracker.active_lengths.tolist(), [0, 1])

    def test_episode_tracker_rejects_wrong_vector_shape(self) -> None:
        tracker = EpisodeStatisticsTracker(num_envs=2)

        with self.assertRaisesRegex(ValueError, "shape"):
            tracker.observe(
                rewards=np.asarray([1.0], dtype=np.float32),
                terminals=np.asarray([False]),
            )

    def test_level_split_rejects_any_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_level_split(
                train_start_level=0,
                train_num_levels=500,
                eval_start_level=499,
                eval_num_levels=500,
            )

    def test_level_split_accepts_adjacent_ranges(self) -> None:
        validate_level_split(
            train_start_level=0,
            train_num_levels=500,
            eval_start_level=500,
            eval_num_levels=500,
        )

    def test_visual_sampling_keeps_episode_endpoints_with_bounded_frames(
        self,
    ) -> None:
        video = np.arange(300, dtype=np.uint16).reshape(300, 1, 1, 1)

        sampled = sample_video_frames(video, max_frames=64)

        self.assertEqual(sampled.shape, (64, 1, 1, 1))
        self.assertEqual(int(sampled[0, 0, 0, 0]), 0)
        self.assertEqual(int(sampled[-1, 0, 0, 0]), 299)


if __name__ == "__main__":
    unittest.main()
