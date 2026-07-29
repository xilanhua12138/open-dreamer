from __future__ import annotations

import unittest

from scripts.experiments.coinrun.select_coinrun_tokenizer_scale import (
    select_candidate,
)


def candidate(
    name: str,
    parameters: int,
    clean: float,
    edge: float,
    temporal: float,
) -> dict:
    return {
        "name": name,
        "parameters": parameters,
        "metrics": {
            "ema_clean_psnr": clean,
            "ema_clean_edge_psnr": edge,
            "ema_clean_temporal_change_psnr": temporal,
        },
    }


class TokenizerSelectionTests(unittest.TestCase):
    def test_selects_smallest_candidate_that_clears_all_quality_gates(self) -> None:
        candidates = [
            candidate("n0.17m", 163_392, 24.2, 18.0, 17.0),
            candidate("n1.1m", 1_048_192, 25.8, 18.9, 17.8),
            candidate("n3.7m", 3_342_528, 27.0, 20.0, 19.0),
        ]

        result = select_candidate(candidates)

        self.assertTrue(result["quality_gate_passed"])
        self.assertEqual(result["selected"]["name"], "n3.7m")
        self.assertEqual(result["best_quality"]["name"], "n3.7m")

    def test_rejects_scale_when_any_region_metric_misses_gate(self) -> None:
        candidates = [
            candidate("n0.17m", 163_392, 24.2, 18.0, 17.0),
            candidate("n1.1m", 1_048_192, 25.8, 18.9, 17.8),
            candidate("n3.7m", 3_342_528, 26.5, 18.6, 18.0),
        ]

        result = select_candidate(candidates)

        self.assertFalse(result["quality_gate_passed"])
        self.assertIsNone(result["selected"])
        self.assertEqual(result["best_quality"]["name"], "n3.7m")

    def test_requires_smallest_scale_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            select_candidate([candidate("n0.17m", 163_392, 24.2, 18.0, 17.0)])


if __name__ == "__main__":
    unittest.main()
