from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.experiments.coinrun.summarize_coinrun_ppo_bug_screen import (
    ARMS,
    summarize_arms,
)


class CoinRunPPOBugScreenTests(unittest.TestCase):
    def _write_arm(
        self,
        root: Path,
        arm: str,
        *,
        mean_return: float,
        success_rate: float,
        final_episodes: int = 256,
    ) -> None:
        for milestone, episodes in (
            ("validation", 128),
            ("final-validation", final_episodes),
        ):
            path = (
                root
                / "arms"
                / arm
                / milestone
                / "env-steps-006291456"
                / "metrics.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "completed_env_steps": 6_291_456,
                        "episodes": episodes,
                        "evaluation_distribution": "full_distribution",
                        "num_levels": 0,
                        "policy": "stochastic",
                        "seed": 4242,
                        "mean_return": mean_return,
                        "success_rate": success_rate,
                    }
                ),
                encoding="utf-8",
            )

    def test_summary_preserves_all_arms_and_computes_signed_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, arm in enumerate(ARMS):
                self._write_arm(
                    root,
                    arm,
                    mean_return=8.0 - index,
                    success_rate=0.8 - index / 10,
                )

            result = summarize_arms(
                run_root=root,
                expected_env_steps=6_291_456,
            )

        self.assertEqual(
            [row["arm"] for row in result["arms"]],
            list(ARMS),
        )
        self.assertEqual(
            result["arms"][1]["delta_vs_reference_final_256"],
            {"mean_return": -1.0, "success_rate": -0.09999999999999998},
        )

    def test_summary_rejects_wrong_final_episode_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in ARMS:
                self._write_arm(
                    root,
                    arm,
                    mean_return=8.0,
                    success_rate=0.8,
                    final_episodes=128,
                )

            with self.assertRaisesRegex(ValueError, "episodes"):
                summarize_arms(
                    run_root=root,
                    expected_env_steps=6_291_456,
                )


if __name__ == "__main__":
    unittest.main()
