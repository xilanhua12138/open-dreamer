from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class CoinRunBackfillConsistencyTests(unittest.TestCase):
    def test_tokenizer_observations_match_raw_heldout_metrics(self) -> None:
        directory = EXPERIMENTS / "CR-TOK-0001"
        results = load(directory / "results.json")
        raw_metrics = load(directory / "raw" / "heldout_eval.json")["metrics"]
        observations = {
            observation["name"]: observation["value"]
            for observation in results["observations"]
        }

        self.assertEqual(
            observations["heldout_online_clean_psnr"],
            raw_metrics["online_clean_psnr"],
        )
        self.assertEqual(
            observations["heldout_online_masked_psnr"],
            raw_metrics["online_masked_psnr"],
        )
        self.assertEqual(
            observations["heldout_ema_clean_psnr"],
            raw_metrics["ema_clean_psnr"],
        )
        self.assertEqual(
            observations["heldout_ema_masked_psnr"],
            raw_metrics["ema_masked_psnr"],
        )

    def test_fixed_flops_completed_arms_match_raw_and_tiny_is_missing(self) -> None:
        directory = EXPERIMENTS / "CR-DYN-0002"
        results = load(directory / "results.json")
        observations = {
            observation["candidate"]: observation
            for observation in results["observations"]
        }

        for candidate in ("small", "medium"):
            raw = load(directory / "raw" / f"{candidate}-metrics.json")
            recorded = observations[candidate]
            self.assertEqual(recorded["mean_video_psnr_db"], raw["mean_video_psnr_db"])
            self.assertEqual(recorded["mean_frame_psnr_db"], raw["mean_frame_psnr_db"])
            self.assertEqual(recorded["mean_ssim"], raw["mean_ssim"])
            self.assertEqual(
                recorded["psnr_by_horizon_db"], raw["psnr_by_horizon_db"]
            )

        self.assertEqual(results["execution_status"], "aborted")
        self.assertEqual(observations["tiny"]["last_observed_progress_step"], 4452)
        self.assertEqual(observations["tiny"]["evaluation"], "missing")
        self.assertFalse((directory / "raw" / "tiny-metrics.json").exists())

    def test_fixed_20k_arms_match_raw_and_primary_metrics_are_monotonic(self) -> None:
        directory = EXPERIMENTS / "CR-DYN-0003"
        results = load(directory / "results.json")
        observations = {
            observation["candidate"]: observation
            for observation in results["observations"]
        }

        psnr: list[float] = []
        ssim: list[float] = []
        for candidate in ("tiny", "small", "medium"):
            raw = load(directory / "raw" / f"{candidate}-metrics.json")
            recorded = observations[candidate]
            self.assertEqual(recorded["mean_video_psnr_db"], raw["mean_video_psnr_db"])
            self.assertEqual(recorded["mean_frame_psnr_db"], raw["mean_frame_psnr_db"])
            self.assertEqual(recorded["mean_ssim"], raw["mean_ssim"])
            self.assertEqual(
                recorded["psnr_by_horizon_db"], raw["psnr_by_horizon_db"]
            )
            self.assertEqual(recorded["ssim_by_horizon"], raw["ssim_by_horizon"])
            psnr.append(recorded["mean_frame_psnr_db"])
            ssim.append(recorded["mean_ssim"])

        self.assertLess(psnr[0], psnr[1])
        self.assertLess(psnr[1], psnr[2])
        self.assertLess(ssim[0], ssim[1])
        self.assertLess(ssim[1], ssim[2])


if __name__ == "__main__":
    unittest.main()
