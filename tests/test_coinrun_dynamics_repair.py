from __future__ import annotations

import json
import subprocess
import sys
import types
import unittest
from pathlib import Path

import jax.numpy as jnp

from dreamer.actions import Actions
from dreamer.coinrun_dynamics_repair import (
    assess_dynamics_repair,
    build_future_action_conditions,
    validate_reward_windowing,
)
from dreamer.dynamics_validation import (
    build_fixed_validation_batch,
    periodic_rollout_names,
)


class CoinRunDynamicsRepairTests(unittest.TestCase):
    @property
    def repository_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def test_reward_biased_windowing_rejects_an_inert_equal_length_record(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "reward-biased slicing requires more than one start position",
        ):
            validate_reward_windowing(
                record_frames=64,
                short_window=64,
                long_window=64,
                p_include_reward=0.5,
            )

    def test_reward_biased_windowing_reports_short_and_long_start_counts(
        self,
    ) -> None:
        audit = validate_reward_windowing(
            record_frames=160,
            short_window=64,
            long_window=128,
            p_include_reward=0.5,
        )

        self.assertEqual(
            audit,
            {
                "record_frames": 160,
                "short_window": 64,
                "long_window": 128,
                "p_include_reward": 0.5,
                "short_start_positions": 97,
                "long_start_positions": 33,
                "reward_bias_operational": True,
            },
        )

    def test_action_corruptions_preserve_context_and_change_only_future(
        self,
    ) -> None:
        source = Actions(
            categorical=jnp.asarray(
                [
                    [4, 7, 7, 8, 8, 3],
                    [4, 6, 6, 5, 5, 2],
                ],
                dtype=jnp.int32,
            )
        )

        conditions = build_future_action_conditions(
            source,
            context_frames=2,
            categorical_noop_action=4,
        )

        self.assertEqual(
            set(conditions),
            {"aligned", "batch_shuffled", "one_step_shifted", "all_noop"},
        )
        for condition in conditions.values():
            self.assertEqual(
                condition.categorical[:, :2].tolist(),
                source.categorical[:, :2].tolist(),
            )
        self.assertEqual(
            conditions["aligned"].categorical.tolist(),
            source.categorical.tolist(),
        )
        self.assertEqual(
            conditions["batch_shuffled"].categorical[:, 2:].tolist(),
            source.categorical[::-1, 2:].tolist(),
        )
        self.assertEqual(
            conditions["one_step_shifted"].categorical[:, 2:].tolist(),
            [[4, 7, 8, 8], [4, 6, 5, 5]],
        )
        self.assertEqual(
            conditions["all_noop"].categorical[:, 2:].tolist(),
            [[4, 4, 4, 4], [4, 4, 4, 4]],
        )

    def test_fixed_validation_batch_uses_a_copy_and_one_frozen_batch(
        self,
    ) -> None:
        train_cfg = types.SimpleNamespace(
            array_record_path="/train",
            p_include_reward=0.5,
            dataloader_cfg=types.SimpleNamespace(
                B=16,
                num_workers=4,
                prefetch_buffer_size=4,
                device_prefetch_buffer_size=1,
            ),
        )
        validation_batch = {
            "videos": jnp.zeros((4, 32, 64, 64, 3), dtype=jnp.uint8),
            "actions": Actions(
                categorical=jnp.full((4, 32), 4, dtype=jnp.int32)
            ),
        }
        calls: list[tuple[object, dict[str, object]]] = []

        def fake_builder(cfg, **kwargs):
            calls.append((cfg, kwargs))
            return iter([validation_batch])

        observed = build_fixed_validation_batch(
            train_cfg,
            validation_array_record_path="/eval",
            validation_seed=4242,
            validation_batch_size=4,
            validation_sequence_length=32,
            device="device-sharding",
            dtype="bfloat16",
            iterator_builder=fake_builder,
        )

        self.assertIs(observed, validation_batch)
        self.assertEqual(train_cfg.array_record_path, "/train")
        self.assertEqual(train_cfg.dataloader_cfg.B, 16)
        validation_cfg, kwargs = calls[0]
        self.assertEqual(validation_cfg.array_record_path, "/eval")
        self.assertEqual(validation_cfg.p_include_reward, 0.0)
        self.assertEqual(validation_cfg.dataloader_cfg.B, 4)
        self.assertEqual(validation_cfg.dataloader_cfg.num_workers, 0)
        self.assertEqual(
            kwargs,
            {
                "seed": 4242,
                "device": "device-sharding",
                "dtype": "bfloat16",
                "seq_len": 32,
                "return_actions": True,
            },
        )

    def test_validation_helpers_import_without_optional_data_dependencies(
        self,
    ) -> None:
        code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "dreamer.data" or name.startswith("dreamer.data."):
        raise ModuleNotFoundError("blocked optional data dependency")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import dreamer.dynamics_validation
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=self.repository_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )

    def test_periodic_validation_can_avoid_full_256_step_diffusion(
        self,
    ) -> None:
        self.assertEqual(
            periodic_rollout_names(include_diffusion=False),
            ("online_shortcut", "ema_shortcut"),
        )
        self.assertEqual(
            periodic_rollout_names(include_diffusion=True),
            (
                "online_diffusion",
                "ema_diffusion",
                "online_shortcut",
                "ema_shortcut",
            ),
        )

    def test_repair_assessment_requires_absolute_quality_and_action_use(
        self,
    ) -> None:
        passing = assess_dynamics_repair(
            shortcut_metrics={
                "mean_frame_psnr_db": 22.5,
                "psnr_by_horizon_db": {"16": 20.25},
            },
            action_metrics={
                "aligned_psnr_advantage_db": {
                    "batch_shuffled": {"16": 0.30},
                    "all_noop": {"16": 0.55},
                }
            },
            min_mean_frame_psnr_db=22.0,
            min_horizon_16_psnr_db=20.0,
            min_shuffled_advantage_db=0.25,
            min_noop_advantage_db=0.50,
        )
        failing = assess_dynamics_repair(
            shortcut_metrics={
                "mean_frame_psnr_db": 23.0,
                "psnr_by_horizon_db": {"16": 20.5},
            },
            action_metrics={
                "aligned_psnr_advantage_db": {
                    "batch_shuffled": {"16": 0.05},
                    "all_noop": {"16": -0.10},
                }
            },
            min_mean_frame_psnr_db=22.0,
            min_horizon_16_psnr_db=20.0,
            min_shuffled_advantage_db=0.25,
            min_noop_advantage_db=0.50,
        )

        self.assertTrue(passing["quality_pass"])
        self.assertTrue(passing["action_use_pass"])
        self.assertTrue(passing["overall_pass"])
        self.assertTrue(failing["quality_pass"])
        self.assertFalse(failing["action_use_pass"])
        self.assertFalse(failing["overall_pass"])

    def test_repair_runner_freezes_long_records_and_reference_schedule(
        self,
    ) -> None:
        runner = (
            self.repository_root
            / "scripts/experiments/coinrun"
            / "run_coinrun_dynamics_repair_reference.sh"
        ).read_text(encoding="utf-8")

        required_fragments = (
            "readonly MAX_STEPS=200000",
            "--frames 160",
            "dataset.dataloader_cfg.short_T=64",
            "dataset.dataloader_cfg.long_T=128",
            "dataset.dataloader_cfg.long_ratio=0.1",
            "dynamics.k_max=256",
            "dynamics.context_length=128",
            'bootstrap_start=100000',
            'dataset.validation_array_record_path="${EVAL_DATA}"',
            "periodic_eval_context_frames=16",
            "periodic_eval_include_diffusion=false",
            "eval_coinrun_action_conditioning.py",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, runner)
        self.assertNotIn("dynamics.k_max=8", runner)

    def test_repair_manifest_preregisters_quality_and_action_thresholds(
        self,
    ) -> None:
        manifest = json.loads(
            (
                self.repository_root
                / "experiments/CR-DYN-0010/manifest.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(manifest["experiment_id"], "CR-DYN-0010")
        self.assertEqual(manifest["protocol"]["data"]["record_frames"], 160)
        self.assertEqual(manifest["protocol"]["training"]["max_steps"], 200000)
        self.assertEqual(manifest["protocol"]["training"]["k_max"], 256)
        self.assertEqual(
            manifest["protocol"]["success_criteria"],
            {
                "min_mean_frame_psnr_db": 22.0,
                "min_horizon_16_psnr_db": 20.0,
                "min_aligned_vs_batch_shuffled_horizon_16_db": 0.25,
                "min_aligned_vs_all_noop_horizon_16_db": 0.5,
                "visual_review_required": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
