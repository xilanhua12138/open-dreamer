from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CoinRunPPOProtocolTests(unittest.TestCase):
    def test_planned_experiment_ends_at_world_model_data_collection(self) -> None:
        manifest = json.loads(
            (ROOT / "experiments/CR-PPO-0001/manifest.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            manifest["protocol"]["downstream_scope"],
            {
                "collect_action_conditioned_trajectories": True,
                "train_dynamics_after_separate_authorization": True,
                "behavior_cloning": False,
                "policy_training_inside_world_model": False,
            },
        )

    def test_runner_records_runtime_and_never_starts_dynamics_or_bc(self) -> None:
        runner = (
            ROOT / "scripts/experiments/coinrun/run_coinrun_ppo_collector.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("scripts/experiments/run_recorded.py", runner)
        self.assertIn("train_coinrun_ppo.py", runner)
        self.assertNotIn("train_dynamics.py", runner)
        self.assertNotIn("behavior", runner.lower())
        self.assertNotIn("clone", runner.lower())

    def test_train_and_collection_scripts_have_distinct_contracts(self) -> None:
        training = (
            ROOT / "scripts/experiments/coinrun/train_coinrun_ppo.py"
        ).read_text(encoding="utf-8")
        collection = (
            ROOT / "scripts/experiments/coinrun/collect_coinrun_ppo_records.py"
        ).read_text(encoding="utf-8")

        self.assertIn("build_logger", training)
        self.assertIn("evaluation_every_env_steps", training)
        self.assertIn("PPOCoinRunPolicy.from_checkpoint", collection)
        self.assertIn("CoinRunRecordAccumulator", collection)
        self.assertIn("checkpoint_sha256", collection)

    def test_runtime_bootstrap_builds_one_procgen_and_jax_environment(self) -> None:
        bootstrap = (
            ROOT / "scripts/experiments/coinrun/prepare_coinrun_ppo_runtime.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("python3.10", bootstrap)
        self.assertIn("procgen==0.10.7", bootstrap)
        self.assertIn("jax[cuda12]==0.4.35", bootstrap)
        self.assertIn("flax==0.10.2", bootstrap)
        self.assertIn("JAX_PLATFORMS=cpu", bootstrap)


if __name__ == "__main__":
    unittest.main()
