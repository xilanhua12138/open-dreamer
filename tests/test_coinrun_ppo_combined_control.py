from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.experiments.coinrun.summarize_coinrun_ppo_combined_control import (
    summarize_combined_control,
)


class CoinRunPPOCombinedControlTests(unittest.TestCase):
    def _write(self, path: Path, mean_return: float, success_rate: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "completed_env_steps": 6_291_456,
                    "episodes": 256,
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

    def test_summary_distinguishes_historical_and_fresh_combined_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.json"
            historical = root / "historical.json"
            combined = root / "combined.json"
            self._write(reference, 7.75, 0.78)
            self._write(historical, 4.5, 0.45)
            self._write(combined, 7.5, 0.75)

            result = summarize_combined_control(
                reference_path=reference,
                historical_path=historical,
                combined_path=combined,
                expected_env_steps=6_291_456,
            )

        self.assertTrue(result["arms"][1]["independent_collapse"])
        self.assertFalse(result["arms"][2]["independent_collapse"])
        self.assertEqual(
            result["arms"][2]["delta_vs_reference"]["mean_return"],
            -0.25,
        )

    def test_summary_rejects_mismatched_evaluation_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / f"{index}.json" for index in range(3)]
            for path in paths:
                self._write(path, 7.5, 0.75)
            payload = json.loads(paths[2].read_text(encoding="utf-8"))
            payload["policy"] = "deterministic_argmax"
            paths[2].write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "policy"):
                summarize_combined_control(
                    reference_path=paths[0],
                    historical_path=paths[1],
                    combined_path=paths[2],
                    expected_env_steps=6_291_456,
                )


if __name__ == "__main__":
    unittest.main()
