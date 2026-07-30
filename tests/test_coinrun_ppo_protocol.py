from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class CoinRunPPOProtocolTests(unittest.TestCase):
    @staticmethod
    def _load_training_entrypoint():
        path = ROOT / "scripts/experiments/coinrun/train_coinrun_ppo.py"
        spec = importlib.util.spec_from_file_location(
            "coinrun_ppo_training_entrypoint",
            path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

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

    def test_v2_collector_cannot_run_before_machine_checked_policy_quality(
        self,
    ) -> None:
        runner = (
            ROOT / "scripts/experiments/coinrun/run_coinrun_ppo_collector_v2.sh"
        ).read_text(encoding="utf-8")

        gate_position = runner.index("validate_coinrun_ppo_quality.py")
        collection_position = runner.index("collect_coinrun_ppo_records.py")
        self.assertLess(gate_position, collection_position)
        self.assertIn("--min-mean-return 8.0", runner)
        self.assertIn("--min-success-rate 0.8", runner)

    def test_training_cli_defaults_to_validated_recipe_and_requires_id(
        self,
    ) -> None:
        entrypoint = self._load_training_entrypoint()

        args = entrypoint.parse_args(
            [
                "--run-dir",
                "/tmp/coinrun-ppo-test",
                "--experiment-id",
                "CR-PPO-TEST",
            ]
        )

        self.assertEqual(args.train_num_levels, 200)
        self.assertEqual(args.eval_start_level, 0)
        self.assertEqual(args.eval_num_levels, 0)
        self.assertEqual(args.evaluation_policy, "stochastic")
        self.assertEqual(args.evaluation_episodes, 128)
        self.assertEqual(args.final_evaluation_episodes, 512)
        self.assertEqual(args.reward_normalization_gamma, 0.99)
        self.assertEqual(args.advantage_normalization, "minibatch")
        self.assertEqual(args.backbone_kernel_init, "glorot_uniform")

        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                entrypoint.parse_args(
                    [
                        "--run-dir",
                        "/tmp/coinrun-ppo-missing-experiment-id",
                    ]
                )

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

    def test_runtime_bootstrap_pins_nvcc_layout_for_jax_0_4_35(self) -> None:
        bootstrap = (
            ROOT / "scripts/experiments/coinrun/prepare_coinrun_ppo_runtime.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            '"nvidia-cuda-nvcc-cu12==12.4.131"',
            bootstrap,
            (
                "JAX 0.4.35 imports nvidia.cuda_nvcc.__file__, but the "
                "unbounded CUDA 12.9 wheel exposes it as a namespace package"
            ),
        )

    def test_official_parity_runner_freezes_the_public_easy_200_recipe(
        self,
    ) -> None:
        runner = (
            ROOT
            / "scripts/experiments/coinrun/run_coinrun_ppo_official_parity.sh"
        ).read_text(encoding="utf-8")

        for expected in (
            "--train-num-levels 200",
            "--eval-num-levels 0",
            "--evaluation-policy stochastic",
            "--reward-normalization-gamma 0.99",
            "--advantage-normalization minibatch",
            "--backbone-kernel-init glorot_uniform",
            "--total-env-steps 25165824",
            "--final-evaluation-episodes 512",
            "--evaluation-max-vector-steps 40000",
            "CR-PPO-0002",
        ):
            self.assertIn(expected, runner)
        self.assertNotIn("collect_coinrun_ppo_records.py", runner)
        self.assertNotIn("train_dynamics.py", runner)


if __name__ == "__main__":
    unittest.main()
