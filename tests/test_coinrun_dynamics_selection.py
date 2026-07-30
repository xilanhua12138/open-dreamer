from __future__ import annotations

import unittest

from dreamer.coinrun_dynamics_selection import select_checkpoint_mixture


def row(
    name: str,
    *,
    frame_psnr: float,
    ssim: float,
    horizon_16: float,
) -> dict:
    return {
        "mixture": name,
        "mean_frame_psnr_db": frame_psnr,
        "mean_ssim": ssim,
        "psnr_by_horizon_db": {"16": horizon_16},
    }


class CoinRunDynamicsSelectionTests(unittest.TestCase):
    def test_selects_primary_metric_before_tie_breakers(self) -> None:
        selection = select_checkpoint_mixture(
            [
                row(
                    "final_only",
                    frame_psnr=20.0,
                    ssim=0.9,
                    horizon_16=15.0,
                ),
                row(
                    "uniform",
                    frame_psnr=20.1,
                    ssim=0.8,
                    horizon_16=14.0,
                ),
                row(
                    "recency_weighted",
                    frame_psnr=20.05,
                    ssim=0.95,
                    horizon_16=16.0,
                ),
            ]
        )

        self.assertEqual(selection["selected_mixture"], "uniform")
        self.assertEqual(
            selection["ranked_mixtures"],
            ["uniform", "recency_weighted", "final_only"],
        )

    def test_uses_ssim_then_horizon_for_exact_psnr_ties(self) -> None:
        selection = select_checkpoint_mixture(
            [
                row(
                    "final_only",
                    frame_psnr=20.0,
                    ssim=0.9,
                    horizon_16=15.0,
                ),
                row(
                    "uniform",
                    frame_psnr=20.0,
                    ssim=0.91,
                    horizon_16=14.0,
                ),
                row(
                    "recency_weighted",
                    frame_psnr=20.0,
                    ssim=0.91,
                    horizon_16=16.0,
                ),
            ]
        )

        self.assertEqual(
            selection["ranked_mixtures"],
            ["recency_weighted", "uniform", "final_only"],
        )

    def test_rejects_missing_arm(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "mixture rows must be exactly",
        ):
            select_checkpoint_mixture(
                [
                    row(
                        "final_only",
                        frame_psnr=20.0,
                        ssim=0.9,
                        horizon_16=15.0,
                    )
                ]
            )


if __name__ == "__main__":
    unittest.main()
