from __future__ import annotations

import unittest

from scripts.experiments.coinrun.evaluate_coinrun_ppo_checkpoint import (
    build_evaluation_record,
)


class CoinRunPPOCheckpointEvaluationTests(unittest.TestCase):
    def test_record_preserves_checkpoint_and_full_distribution_identity(self) -> None:
        result = build_evaluation_record(
            summary={
                "episodes": 256,
                "mean_return": 7.5,
                "success_rate": 0.75,
            },
            metadata={
                "completed_env_steps": 6_291_456,
                "train_config": {"start_level": 0, "num_levels": 500},
            },
            checkpoint_sha256="a" * 64,
            episodes=256,
            seed=4242,
            start_level=0,
            num_levels=0,
            policy="stochastic",
            distribution_mode="easy",
        )

        self.assertEqual(result["completed_env_steps"], 6_291_456)
        self.assertEqual(result["checkpoint_sha256"], "a" * 64)
        self.assertEqual(result["evaluation_distribution"], "full_distribution")
        self.assertIsNone(result["evaluation_level_range"])
        self.assertEqual(result["train_level_range"], [0, 500])
        self.assertEqual(result["mean_return"], 7.5)

    def test_record_rejects_wrong_episode_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "summary episodes"):
            build_evaluation_record(
                summary={"episodes": 64},
                metadata={
                    "completed_env_steps": 6_291_456,
                    "train_config": {"start_level": 0, "num_levels": 500},
                },
                checkpoint_sha256="b" * 64,
                episodes=256,
                seed=4242,
                start_level=0,
                num_levels=0,
                policy="stochastic",
                distribution_mode="easy",
            )


if __name__ == "__main__":
    unittest.main()
