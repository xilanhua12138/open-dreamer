from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.configs import (
    DataloaderConfig,
    DatasetConfig,
    TokenizerValidationConfig,
)
from dreamer.tokenizer_validation import (
    build_comparison_video,
    build_validation_dataset_config,
    evaluate_fixed_validation,
    should_run_validation,
    write_validation_artifacts,
)


class _ShiftTokenizer(nnx.Module):
    def __init__(self, shift: float):
        self.shift = shift

    def __call__(self, videos, *, deterministic: bool):
        del deterministic
        reconstruction = videos.astype(jnp.float32) + self.shift
        return reconstruction, (None, None), None


class TokenizerValidationRuntimeTests(unittest.TestCase):
    def test_validation_schedule_uses_completed_updates(self) -> None:
        self.assertFalse(should_run_validation(step=0, max_steps=20_000, every_steps=2_500))
        self.assertTrue(
            should_run_validation(step=2_499, max_steps=20_000, every_steps=2_500)
        )
        self.assertTrue(
            should_run_validation(step=19_999, max_steps=20_000, every_steps=2_500)
        )
        self.assertFalse(
            should_run_validation(step=2_499, max_steps=20_000, every_steps=0)
        )

    def test_comparison_video_preserves_time_and_labels_three_columns(self) -> None:
        target = np.zeros((2, 4, 8, 8, 3), dtype=np.uint8)
        online = np.full_like(target, 64)
        ema = np.full_like(target, 128)

        video = build_comparison_video(target, online, ema, max_samples=2)

        self.assertEqual(video.shape, (4, 16, 24, 3))
        self.assertEqual(int(video[:, :, :8].max()), 0)
        self.assertEqual(int(video[:, :, 8:16].min()), 64)
        self.assertEqual(int(video[:, :, 16:].min()), 128)

    def test_validation_artifacts_are_machine_readable(self) -> None:
        target = np.zeros((1, 2, 16, 16, 3), dtype=np.uint8)
        online = np.full_like(target, 64)
        ema = np.full_like(target, 128)
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = write_validation_artifacts(
                output_dir=Path(temporary),
                completed_updates=2_500,
                metrics={"online_clean_psnr": 10.0, "ema_clean_psnr": 11.0},
                target=target,
                online=online,
                ema=ema,
                validation_set_sha256="b" * 64,
                fps=4,
            )

            payload = json_load(artifacts["metrics"])
            self.assertEqual(payload["completed_updates"], 2_500)
            self.assertEqual(payload["validation_set_sha256"], "b" * 64)
            self.assertTrue(artifacts["image"].is_file())
            self.assertGreater(artifacts["video"].stat().st_size, 0)

    def test_validation_config_requires_disjoint_dataset_and_positive_shape(self) -> None:
        training = DatasetConfig(
            array_record_path="/train",
            dataloader_cfg=DataloaderConfig(B=128, short_T=16, long_T=16),
        )
        with self.assertRaisesRegex(ValueError, "dataset_path"):
            build_validation_dataset_config(
                training,
                TokenizerValidationConfig(enabled=True, dataset_path=""),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            build_validation_dataset_config(
                training,
                TokenizerValidationConfig(
                    enabled=True,
                    dataset_path="/eval",
                    batches=0,
                ),
            )

    def test_validation_aggregates_exact_sse_for_online_and_ema_models(self) -> None:
        batches = [{"videos": jnp.zeros((1, 2, 4, 4, 3), dtype=jnp.uint8)}]

        metrics, target, online, ema = evaluate_fixed_validation(
            _ShiftTokenizer(255.0),
            _ShiftTokenizer(0.0),
            batches,
        )

        self.assertEqual(metrics["online_clean_mse"], 1.0)
        self.assertEqual(metrics["online_clean_psnr"], 0.0)
        self.assertEqual(metrics["ema_clean_mse"], 0.0)
        self.assertEqual(metrics["ema_clean_psnr"], 120.0)
        self.assertTrue(np.array_equal(target, ema))
        self.assertEqual(int(online.min()), 255)


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
