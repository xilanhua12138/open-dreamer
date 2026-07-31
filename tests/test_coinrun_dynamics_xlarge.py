from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.experiments.coinrun.dynamics_xlarge_protocol import (
    XLARGE_ARCHITECTURE,
    XLARGE_PROTOCOL,
)


ROOT = Path(__file__).resolve().parents[1]


class DynamicsXlargeProtocolTests(unittest.TestCase):
    def test_xlarge_changes_only_dynamics_capacity(self) -> None:
        self.assertEqual(
            XLARGE_ARCHITECTURE,
            {
                "depth": 9,
                "d_model": 640,
                "n_heads": 10,
                "n_kv_heads": 1,
                "n_register": 32,
            },
        )
        self.assertEqual(XLARGE_PROTOCOL["max_steps"], 200_000)
        self.assertEqual(XLARGE_PROTOCOL["batch_size"], 16)
        self.assertEqual(XLARGE_PROTOCOL["short_T"], 64)
        self.assertEqual(XLARGE_PROTOCOL["long_T"], 128)
        self.assertEqual(XLARGE_PROTOCOL["long_ratio"], 0.1)
        self.assertEqual(XLARGE_PROTOCOL["k_max"], 256)
        self.assertEqual(XLARGE_PROTOCOL["bootstrap_start"], 100_000)
        self.assertEqual(XLARGE_PROTOCOL["parameters"], 52_801_152)

    def test_manifest_and_runner_are_preregistered_for_xlarge_only(self) -> None:
        manifest = json.loads(
            (ROOT / "experiments/CR-DYN-0012/manifest.json").read_text()
        )
        self.assertEqual(manifest["experiment_id"], "CR-DYN-0012")
        self.assertEqual(manifest["baseline_id"], "CR-DYN-0011")
        self.assertEqual(
            manifest["controlled_variable"]["name"],
            "dynamics_model_capacity",
        )
        self.assertEqual(
            manifest["protocol"]["training"]["architecture"],
            XLARGE_ARCHITECTURE,
        )
        self.assertEqual(
            manifest["protocol"]["training"]["max_steps"],
            XLARGE_PROTOCOL["max_steps"],
        )
        runner = (
            ROOT
            / "scripts/experiments/coinrun/run_coinrun_dynamics_xlarge_offline_latents.sh"
        ).read_text()
        self.assertIn('readonly EXPERIMENT_ID="CR-DYN-0012"', runner)
        self.assertIn(
            'export OPEN_DREAMER_DYNAMICS_DEPTH="${XLARGE_DEPTH}"', runner
        )
        self.assertIn(
            'export OPEN_DREAMER_DYNAMICS_D_MODEL="${XLARGE_D_MODEL}"', runner
        )
        self.assertIn("run_coinrun_dynamics_offline_latents.sh", runner)


if __name__ == "__main__":
    unittest.main()
