from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class CoinRunPPOProtocolTests(unittest.TestCase):
    @staticmethod
    def _load_coinrun_script(filename: str, module_name: str):
        path = ROOT / "scripts/experiments/coinrun" / filename
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _load_training_entrypoint(cls):
        return cls._load_coinrun_script(
            "train_coinrun_ppo.py",
            "coinrun_ppo_training_entrypoint",
        )

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
        self.assertEqual(args.residual_branch_scale, 1.0)
        self.assertEqual(args.residual_last_kernel_init, "same")
        self.assertFalse(args.residual_skip_init)

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

    def test_bug_screen_changes_exactly_one_recipe_factor_per_arm(
        self,
    ) -> None:
        runner = (
            ROOT / "scripts/experiments/coinrun/run_coinrun_ppo_bug_screen.sh"
        ).read_text(encoding="utf-8")

        for expected in (
            "run_arm reference 200 0.99 minibatch glorot_uniform",
            "run_arm levels500 500 0.99 minibatch glorot_uniform",
            "run_arm reward_gamma0999 200 0.999 minibatch glorot_uniform",
            "run_arm batch_advantage 200 0.99 batch glorot_uniform",
            "run_arm orthogonal_init 200 0.99 minibatch orthogonal_sqrt2",
            "--total-env-steps",
            "--final-evaluation-episodes 256",
            "CR-PPO-0003",
        ):
            self.assertIn(expected, runner)
        self.assertNotIn("collect_coinrun_ppo_records.py", runner)
        self.assertNotIn("train_dynamics.py", runner)

    def test_initializer_mitigation_runner_probes_before_four_targeted_arms(
        self,
    ) -> None:
        runner = (
            ROOT
            / "scripts/experiments/coinrun/run_coinrun_ppo_initializer_mitigation.sh"
        ).read_text(encoding="utf-8")

        self.assertLess(
            runner.index("probe_coinrun_ppo_initialization.py"),
            runner.index("run_arm orthogonal_gain1"),
        )
        self.assertIn("probe_is_complete", runner)
        self.assertIn("probe/runtime-identity.json", runner)
        self.assertIn("probe/run-state.json", runner)
        for expected in (
            "run_arm orthogonal_gain1 orthogonal_gain1 1.0 same false",
            'run_arm orthogonal_sqrt2_depth_scaled orthogonal_sqrt2 "${DEPTH_SCALE}" same false',
            "run_arm orthogonal_sqrt2_zero_last orthogonal_sqrt2 1.0 zeros false",
            "run_arm orthogonal_sqrt2_skipinit orthogonal_sqrt2 1.0 same true",
            "--total-env-steps",
            "--final-evaluation-episodes 256",
            "CR-PPO-0005",
        ):
            self.assertIn(expected, runner)
        self.assertNotIn("collect_coinrun_ppo_records.py", runner)
        self.assertNotIn("train_dynamics.py", runner)

    def test_initializer_probe_freezes_six_mechanistically_distinct_configs(
        self,
    ) -> None:
        probe = self._load_coinrun_script(
            "probe_coinrun_ppo_initialization.py",
            "coinrun_ppo_initialization_probe",
        )

        self.assertEqual(set(probe.ARM_CONFIGS), {
            "glorot_reference",
            "orthogonal_sqrt2_anchor",
            "orthogonal_gain1",
            "orthogonal_sqrt2_depth_scaled",
            "orthogonal_sqrt2_zero_last",
            "orthogonal_sqrt2_skipinit",
        })
        self.assertAlmostEqual(
            probe.ARM_CONFIGS[
                "orthogonal_sqrt2_depth_scaled"
            ]["residual_branch_scale"],
            1.0 / (6.0 ** 0.5),
            places=15,
        )
        summary = probe.summarize_scalar_rows(
            [
                {"gradient": 2.0, "ratio": 1.0},
                {"gradient": 4.0, "ratio": 3.0},
            ]
        )
        self.assertEqual(
            summary,
            {
                "gradient": {
                    "mean": 3.0,
                    "std": 1.0,
                    "min": 2.0,
                    "max": 4.0,
                },
                "ratio": {
                    "mean": 2.0,
                    "std": 1.0,
                    "min": 1.0,
                    "max": 3.0,
                },
            },
        )

    def test_initializer_probe_materializes_runtime_identity_and_terminal_state(
        self,
    ) -> None:
        probe = self._load_coinrun_script(
            "probe_coinrun_ppo_initialization.py",
            "coinrun_ppo_initialization_probe_recording",
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            output = run_dir / "initialization-probe.json"
            arguments = SimpleNamespace(
                output=output,
                num_envs=2,
                observation_vector_steps=0,
                observation_seed=4242,
                num_initialization_seeds=1,
            )
            payload = {
                "schema_version": "1.0",
                "experiment_id": "CR-PPO-0005",
                "arms": [{"arm": "glorot_reference"}],
            }
            with (
                mock.patch.object(
                    probe,
                    "_collect_observations",
                    return_value=b"observations",
                ),
                mock.patch.object(probe, "run_probe", return_value=payload),
                mock.patch(
                    "dreamer.experiment_runtime.collect_runtime_identity",
                    return_value={
                        "schema_version": "1.0",
                        "run_id": "probe-run",
                        "attempt_id": "probe-attempt",
                    },
                ),
            ):
                probe.execute_probe(arguments)

            runtime = json.loads(
                (run_dir / "runtime-identity.json").read_text(encoding="utf-8")
            )
            state = json.loads(
                (run_dir / "run-state.json").read_text(encoding="utf-8")
            )
            artifacts = [
                json.loads(line)
                for line in (run_dir / "artifacts.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(runtime["attempt_id"], "probe-attempt")
            self.assertEqual(state["state"], "COMPLETED")
            self.assertEqual(state["last_completed_updates"], 1)
            self.assertEqual(artifacts[0]["key"], "probe/initialization")
            self.assertEqual(artifacts[0]["uri"], "initialization-probe.json")

    def test_initializer_mitigation_summary_uses_identical_final_evaluator(
        self,
    ) -> None:
        summarizer = self._load_coinrun_script(
            "summarize_coinrun_ppo_initializer_mitigation.py",
            "coinrun_ppo_initializer_mitigation_summary",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = root / "anchor.json"
            anchor.write_text(
                json.dumps(
                    {
                        "experiment_id": "CR-PPO-0003",
                        "expected_env_steps": 6_291_456,
                        "arms": [
                            {
                                "arm": "reference",
                                "final_256": {
                                    "mean_return": 8.0,
                                    "success_rate": 0.8,
                                },
                            },
                            {
                                "arm": "orthogonal_init",
                                "final_256": {
                                    "mean_return": 5.0,
                                    "success_rate": 0.5,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            for index, arm in enumerate(summarizer.NEW_ARMS, start=1):
                metrics_path = (
                    root
                    / "arms"
                    / arm
                    / "final-validation"
                    / "env-steps-006291456"
                    / "metrics.json"
                )
                metrics_path.parent.mkdir(parents=True)
                metrics_path.write_text(
                    json.dumps(
                        {
                            "completed_env_steps": 6_291_456,
                            "episodes": 256,
                            "evaluation_distribution": "full_distribution",
                            "num_levels": 0,
                            "policy": "stochastic",
                            "seed": 4242,
                            "mean_return": 5.0 + index,
                            "success_rate": 0.5 + index / 20.0,
                        }
                    ),
                    encoding="utf-8",
                )

            payload = summarizer.summarize_mitigations(
                run_root=root,
                anchor_comparison_path=anchor,
                expected_env_steps=6_291_456,
            )
            plot = root / "comparison.png"
            summarizer.render_comparison(payload, plot)

            self.assertEqual(len(payload["arms"]), 6)
            self.assertAlmostEqual(
                payload["arms"][2]["recovery_vs_orthogonal_sqrt2"][
                    "mean_return_fraction_of_gap"
                ],
                1.0 / 3.0,
            )
            self.assertEqual(plot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
