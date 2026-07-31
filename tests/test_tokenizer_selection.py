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
    def test_selects_highest_detail_aware_quality_after_full_sweep(self) -> None:
        candidates = [
            candidate("n0.17m", 163_392, 24.2, 18.0, 17.0),
            candidate("n1.1m", 1_048_192, 25.8, 18.9, 17.8),
            candidate("n3.7m", 3_342_528, 27.0, 20.0, 19.0),
        ]

        result = select_candidate(candidates)

        self.assertEqual(result["selected"]["name"], "n3.7m")
        self.assertEqual(
            result["selection_basis"],
            "highest_unweighted_sum_of_clean_edge_and_temporal_change_psnr",
        )
        self.assertEqual(
            [row["name"] for row in result["ranking"]],
            ["n3.7m", "n1.1m", "n0.17m"],
        )

    def test_does_not_reject_candidates_with_an_absolute_gate(self) -> None:
        candidates = [
            candidate("n0.17m", 163_392, 24.2, 18.0, 17.0),
            candidate("n1.1m", 1_048_192, 25.8, 18.9, 17.8),
            candidate("n3.7m", 3_342_528, 26.5, 18.6, 18.0),
        ]

        result = select_candidate(candidates)

        self.assertEqual(result["selected"]["name"], "n3.7m")
        self.assertNotIn("quality_gate_passed", result)
        self.assertNotIn("thresholds", result)
        self.assertTrue(all("eligible" not in row for row in result["candidates"]))

    def test_requires_smallest_scale_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            select_candidate([candidate("n0.17m", 163_392, 24.2, 18.0, 17.0)])


if __name__ == "__main__":
    unittest.main()
