from __future__ import annotations

import unittest

from scripts.experiments.coinrun.compare_tokenizer_extension import compare_metrics


def heldout(
    *,
    clean: float,
    masked: float,
    edge: float,
    temporal_change: float,
    seed: int = 4_242,
) -> dict:
    return {
        "dataset": "/eval",
        "seed": seed,
        "batch_size": 32,
        "frames": 16,
        "batches": 16,
        "num_clips": 512,
        "completed_updates": 20_000,
        "metrics": {
            "ema_clean_psnr": clean,
            "ema_masked_psnr": masked,
            "ema_clean_edge_psnr": edge,
            "ema_clean_temporal_change_psnr": temporal_change,
        },
    }


class TokenizerExtensionComparisonTests(unittest.TestCase):
    def test_comparison_reports_exact_deltas_and_descriptive_score(self) -> None:
        comparison = compare_metrics(
            baseline_name="n16.6m",
            baseline=heldout(
                clean=35.0,
                masked=32.0,
                edge=26.0,
                temporal_change=27.0,
            ),
            candidate_name="n28.7m",
            candidate=heldout(
                clean=36.5,
                masked=33.0,
                edge=27.25,
                temporal_change=28.75,
            ),
        )

        self.assertEqual(comparison["evaluation_identity"]["num_clips"], 512)
        self.assertEqual(
            comparison["candidate_minus_baseline_psnr_db"],
            {
                "ema_clean": 1.5,
                "ema_masked": 1.0,
                "ema_clean_edge": 1.25,
                "ema_clean_temporal_change": 1.75,
            },
        )
        self.assertEqual(comparison["descriptive_score_delta"], 4.5)
        self.assertNotIn("eligible", comparison)
        self.assertNotIn("quality_gate", comparison)

    def test_mismatched_heldout_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluation identity differs"):
            compare_metrics(
                baseline_name="n16.6m",
                baseline=heldout(
                    clean=35.0,
                    masked=32.0,
                    edge=26.0,
                    temporal_change=27.0,
                ),
                candidate_name="n28.7m",
                candidate=heldout(
                    clean=36.5,
                    masked=33.0,
                    edge=27.25,
                    temporal_change=28.75,
                    seed=9_999,
                ),
            )


if __name__ == "__main__":
    unittest.main()
