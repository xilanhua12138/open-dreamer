from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from dreamer.coinrun_latent_dataset import (
    build_coinrun_latent_record,
    choose_coinrun_window_start,
    validate_coinrun_latent_pair,
)


class _RewardChoosingRng:
    def random(self) -> float:
        return 0.0

    def choice(self, values: np.ndarray) -> np.int64:
        return np.int64(values[-1])

    def integers(self, low: int, high: int) -> np.int64:
        self.bounds = (low, high)
        return np.int64(high - 1)


class CoinRunLatentDatasetTests(unittest.TestCase):
    @property
    def repository_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _raw_record(self) -> dict[str, object]:
        frames = np.arange(6 * 2 * 2 * 3, dtype=np.uint8).reshape(6, 2, 2, 3)
        return {
            "raw_video": frames.tobytes(),
            "sequence_length": 6,
            "actions": np.asarray([4, 7, 7, 8, 8, 3], dtype=np.int32),
            "rewards": np.asarray([0, 0, 0, 0, 1, 0], dtype=np.float32),
            "terminals": np.asarray(
                [False, False, False, False, False, True],
                dtype=bool,
            ),
        }

    def test_latent_record_preserves_action_reward_terminal_alignment(self) -> None:
        raw = self._raw_record()
        latents = np.arange(6 * 16 * 16, dtype=np.float32).reshape(6, 16, 16)

        latent_record = build_coinrun_latent_record(
            raw,
            latents,
            record_index=17,
            raw_tree_sha256="a" * 64,
        )

        np.testing.assert_array_equal(latent_record["latents"], latents)
        np.testing.assert_array_equal(
            latent_record["actions"]["categorical"],
            raw["actions"],
        )
        self.assertIsNone(latent_record["actions"]["binary"])
        self.assertIsNone(latent_record["actions"]["continuous"])
        np.testing.assert_array_equal(latent_record["rewards"], raw["rewards"])
        np.testing.assert_array_equal(latent_record["terminals"], raw["terminals"])
        self.assertEqual(
            latent_record["source"],
            {
                "record_index": 17,
                "raw_tree_sha256": "a" * 64,
                "transition_alignment": (
                    "latents[t]=tokenizer(raw_video[t]), "
                    "actions[t]=action_t, rewards[t]=reward_after_action_t"
                ),
            },
        )
        audit = validate_coinrun_latent_pair(raw, latent_record)
        self.assertEqual(
            audit,
            {
                "sequence_length": 6,
                "latent_shape": [6, 16, 16],
                "actions_aligned": True,
                "rewards_aligned": True,
                "terminals_aligned": True,
            },
        )

    def test_latent_record_rejects_a_time_axis_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "latent time axis 5 does not match raw sequence_length 6",
        ):
            build_coinrun_latent_record(
                self._raw_record(),
                np.zeros((5, 16, 16), dtype=np.float32),
                record_index=0,
                raw_tree_sha256="b" * 64,
            )

    def test_reward_biased_latent_window_uses_same_legal_bounds_as_video(
        self,
    ) -> None:
        rng = _RewardChoosingRng()
        start = choose_coinrun_window_start(
            episode_len=6,
            seq_len=3,
            rewards=np.asarray([0, 0, 0, 0, 1, 0], dtype=np.float32),
            p_include_reward=1.0,
            rng=rng,
        )

        self.assertEqual(rng.bounds, (2, 4))
        self.assertEqual(start, 3)
        self.assertLessEqual(start, 4)
        self.assertGreaterEqual(start + 2, 4)

    def test_pair_audit_rejects_action_drift(self) -> None:
        raw = self._raw_record()
        latent_record = build_coinrun_latent_record(
            raw,
            np.zeros((6, 16, 16), dtype=np.float32),
            record_index=0,
            raw_tree_sha256="c" * 64,
        )
        latent_record["actions"]["categorical"][2] = 14

        with self.assertRaisesRegex(
            ValueError,
            "latent categorical actions differ from raw record",
        ):
            validate_coinrun_latent_pair(raw, latent_record)

    def test_offline_latent_runner_freezes_only_the_input_representation(
        self,
    ) -> None:
        runner = (
            self.repository_root
            / "scripts/experiments/coinrun"
            / "run_coinrun_dynamics_offline_latents.sh"
        ).read_text(encoding="utf-8")

        for fragment in (
            "readonly MAX_STEPS=200000",
            "tokenize_coinrun_dataset.py",
            "audit_coinrun_latent_dataset.py",
            "dataset=coinrun_latent",
            "dataset.dataloader_cfg.B=16",
            "dataset.dataloader_cfg.short_T=64",
            "dataset.dataloader_cfg.long_T=128",
            "dataset.dataloader_cfg.long_ratio=0.1",
            "dynamics.k_max=256",
            "dynamics.context_length=128",
            "bootstrap_start=100000",
            "write_video_every=10000",
        ):
            self.assertIn(fragment, runner)
        self.assertIn(
            '"${DYNAMICS_PYTHON}" scripts/train_dynamics.py \\\n'
            "      dataset=coinrun_latent",
            runner,
        )

    def test_offline_latent_experiment_is_preregistered_as_a_new_id(
        self,
    ) -> None:
        manifest = json.loads(
            (
                self.repository_root
                / "experiments/CR-DYN-0011/manifest.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(manifest["experiment_id"], "CR-DYN-0011")
        self.assertEqual(manifest["baseline_id"], "CR-DYN-0010")
        self.assertEqual(
            manifest["controlled_variable"]["name"],
            "dynamics_training_input_representation",
        )
        self.assertEqual(
            manifest["protocol"]["training"]["max_steps"],
            200000,
        )
        self.assertEqual(
            manifest["protocol"]["training"]["data_type"],
            "latent",
        )

    def test_latent_audit_cli_can_import_repository_modules(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/experiments/coinrun/audit_coinrun_latent_dataset.py",
                "--help",
            ],
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


if __name__ == "__main__":
    unittest.main()
