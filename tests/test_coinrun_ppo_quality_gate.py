from __future__ import annotations

import unittest

from scripts.experiments.coinrun.validate_coinrun_ppo_quality import (
    validate_policy_quality,
)


class CoinRunPPOQualityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_metrics = {
            "completed_env_steps": 25_165_824,
            "episodes": 512,
            "evaluation_distribution": "full_distribution",
            "num_levels": 0,
            "policy": "stochastic",
            "mean_return": 8.671875,
            "success_rate": 0.8671875,
        }

    def test_accepts_exact_final_evaluation_above_both_thresholds(self) -> None:
        result = validate_policy_quality(
            self.valid_metrics,
            expected_env_steps=25_165_824,
            expected_episodes=512,
            min_mean_return=8.0,
            min_success_rate=0.8,
        )

        self.assertEqual(result["decision"], "accepted_for_collection")
        self.assertEqual(result["mean_return"], 8.671875)
        self.assertEqual(result["success_rate"], 0.8671875)

    def test_rejects_policy_when_either_quality_threshold_fails(self) -> None:
        metrics = dict(self.valid_metrics)
        metrics["mean_return"] = 7.99

        with self.assertRaisesRegex(ValueError, "mean_return"):
            validate_policy_quality(
                metrics,
                expected_env_steps=25_165_824,
                expected_episodes=512,
                min_mean_return=8.0,
                min_success_rate=0.8,
            )

        metrics = dict(self.valid_metrics)
        metrics["success_rate"] = 0.79
        with self.assertRaisesRegex(ValueError, "success_rate"):
            validate_policy_quality(
                metrics,
                expected_env_steps=25_165_824,
                expected_episodes=512,
                min_mean_return=8.0,
                min_success_rate=0.8,
            )

    def test_rejects_wrong_evaluation_identity_even_with_high_reward(self) -> None:
        for field, value in (
            ("completed_env_steps", 25_000_000),
            ("episodes", 64),
            ("evaluation_distribution", "fixed_level_range"),
            ("num_levels", 500),
            ("policy", "deterministic_argmax"),
        ):
            metrics = dict(self.valid_metrics)
            metrics[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    validate_policy_quality(
                        metrics,
                        expected_env_steps=25_165_824,
                        expected_episodes=512,
                        min_mean_return=8.0,
                        min_success_rate=0.8,
                    )


if __name__ == "__main__":
    unittest.main()
