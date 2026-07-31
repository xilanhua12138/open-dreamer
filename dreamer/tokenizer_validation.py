"""Fixed held-out tokenizer validation, visualization, and durable artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Sequence

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import DictConfig, OmegaConf

from dreamer.configs import (
    DataloaderConfig,
    DatasetConfig,
    TokenizerValidationConfig,
)
from dreamer.experiment_runtime import atomic_write_json


def should_run_validation(*, step: int, max_steps: int, every_steps: int) -> bool:
    if every_steps <= 0:
        return False
    completed_updates = step + 1
    return completed_updates % every_steps == 0 or completed_updates == max_steps


def build_validation_dataset_config(
    training_dataset: DatasetConfig | DictConfig,
    validation: TokenizerValidationConfig,
) -> DatasetConfig:
    if isinstance(training_dataset, DictConfig):
        merged_dataset = OmegaConf.to_container(
            OmegaConf.merge(
                OmegaConf.create(asdict(DatasetConfig())),
                training_dataset,
            ),
            resolve=True,
        )
        if not isinstance(merged_dataset, dict):
            raise TypeError(
                "DatasetConfig merge must produce a mapping, got "
                f"{type(merged_dataset).__name__}"
            )
        loader_values = merged_dataset.pop("dataloader_cfg")
        if not isinstance(loader_values, dict):
            raise TypeError(
                "dataloader_cfg must materialize as a mapping, got "
                f"{type(loader_values).__name__}"
            )
        training_dataset = DatasetConfig(
            **merged_dataset,
            dataloader_cfg=DataloaderConfig(**loader_values),
        )
    elif not isinstance(training_dataset, DatasetConfig):
        raise TypeError(
            "training_dataset must be DatasetConfig or DictConfig, got "
            f"{type(training_dataset).__name__}"
        )
    if not validation.dataset_path:
        raise ValueError("validation.dataset_path is required when validation is enabled")
    positive_fields = {
        "every_steps": validation.every_steps,
        "batch_size": validation.batch_size,
        "batches": validation.batches,
        "frames": validation.frames,
        "max_visual_samples": validation.max_visual_samples,
        "fps": validation.fps,
    }
    invalid = {
        name: value for name, value in positive_fields.items() if int(value) <= 0
    }
    if invalid:
        raise ValueError(f"validation values must be positive, got {invalid}")
    if validation.dataset_path == training_dataset.array_record_path:
        raise ValueError(
            "validation.dataset_path must be disjoint from the training dataset path"
        )
    training_loader = training_dataset.dataloader_cfg
    loader_values = {
        field.name: getattr(training_loader, field.name)
        for field in fields(DataloaderConfig)
    }
    loader_values.update(
        {
            "B": validation.batch_size,
            "num_workers": validation.num_workers,
            "prefetch_buffer_size": validation.prefetch_buffer_size,
            "device_prefetch_buffer_size": validation.device_prefetch_buffer_size,
            "short_T": validation.frames,
            "long_T": validation.frames,
            "long_ratio": 0.0,
        }
    )
    dataset_values = {
        field.name: getattr(training_dataset, field.name)
        for field in fields(DatasetConfig)
        if field.name != "dataloader_cfg"
    }
    dataset_values.update(
        {
            "array_record_path": validation.dataset_path,
            "p_include_reward": 0.0,
            "dataloader_cfg": DataloaderConfig(**loader_values),
        }
    )
    return DatasetConfig(**dataset_values)


def validation_set_sha256(batches: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        videos = np.asarray(jax.device_get(batch["videos"]))
        digest.update(str(videos.shape).encode("utf-8"))
        digest.update(str(videos.dtype).encode("utf-8"))
        digest.update(videos.tobytes(order="C"))
    return digest.hexdigest()


@nnx.jit
def _validation_step(model, model_ema, videos):
    online, _, _ = model(videos, deterministic=True)
    ema, _, _ = model_ema(videos, deterministic=True)
    target_01 = videos.astype(jnp.float32) / 255.0
    online_01 = online.astype(jnp.float32) / 255.0
    ema_01 = ema.astype(jnp.float32) / 255.0
    online_squared_error = jnp.sum((online_01 - target_01) ** 2)
    ema_squared_error = jnp.sum((ema_01 - target_01) ** 2)
    value_count = jnp.asarray(target_01.size, dtype=jnp.int32)
    return online_squared_error, ema_squared_error, value_count, online, ema


def evaluate_fixed_validation(
    model,
    model_ema,
    batches: Sequence[Any],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    if not batches:
        raise ValueError("fixed validation requires at least one batch")
    online_squared_error = 0.0
    ema_squared_error = 0.0
    value_count = 0
    first_visual = None
    for batch in batches:
        online_sse, ema_sse, count, online, ema = _validation_step(
            model, model_ema, batch["videos"]
        )
        online_squared_error += float(jax.device_get(online_sse))
        ema_squared_error += float(jax.device_get(ema_sse))
        value_count += int(jax.device_get(count))
        if first_visual is None:
            first_visual = (
                np.asarray(jax.device_get(batch["videos"])),
                np.asarray(jax.device_get(online)),
                np.asarray(jax.device_get(ema)),
            )
    online_mse = online_squared_error / value_count
    ema_mse = ema_squared_error / value_count
    metrics = {
        "online_clean_mse": online_mse,
        "online_clean_psnr": 10.0 * np.log10(1.0 / max(online_mse, 1e-12)),
        "ema_clean_mse": ema_mse,
        "ema_clean_psnr": 10.0 * np.log10(1.0 / max(ema_mse, 1e-12)),
        "num_clips": sum(int(batch["videos"].shape[0]) for batch in batches),
    }
    assert first_visual is not None
    target, online, ema = first_visual
    return metrics, to_uint8(target), to_uint8(online), to_uint8(ema)


def to_uint8(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=np.float32), 0, 255).astype(np.uint8)


def build_comparison_video(
    target: np.ndarray,
    online: np.ndarray,
    ema: np.ndarray,
    *,
    max_samples: int,
) -> np.ndarray:
    arrays = [to_uint8(value) for value in (target, online, ema)]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(
            f"target/online/ema shapes differ: {[array.shape for array in arrays]}"
        )
    if arrays[0].ndim != 5 or arrays[0].shape[-1] != 3:
        raise ValueError(f"expected (B,T,H,W,3), got {arrays[0].shape}")
    sample_count = min(max_samples, arrays[0].shape[0])
    columns = np.concatenate(
        [array[:sample_count] for array in arrays],
        axis=3,
    )
    return np.transpose(columns, (1, 0, 2, 3, 4)).reshape(
        columns.shape[1],
        sample_count * columns.shape[2],
        columns.shape[3],
        columns.shape[4],
    )


def write_validation_artifacts(
    *,
    output_dir: Path,
    completed_updates: int,
    metrics: dict[str, Any],
    target: np.ndarray,
    online: np.ndarray,
    ema: np.ndarray,
    validation_set_sha256: str,
    fps: int,
    max_samples: int = 4,
    write_video: bool = True,
) -> dict[str, Path]:
    milestone = Path(output_dir) / f"updates-{completed_updates:08d}"
    milestone.mkdir(parents=True, exist_ok=True)
    video = build_comparison_video(
        target,
        online,
        ema,
        max_samples=max_samples,
    )
    metrics_path = milestone / "metrics.json"
    image_path = milestone / "reconstruction.png"
    video_path = milestone / "reconstruction.gif"
    payload = {
        "schema_version": "1.0",
        "completed_updates": completed_updates,
        "validation_set_sha256": validation_set_sha256,
        "layout": {
            "columns": ["target", "online", "ema"],
            "rows": min(max_samples, target.shape[0]),
        },
        "metrics": metrics,
    }
    atomic_write_json(metrics_path, payload)
    iio.imwrite(image_path, video[0])
    artifacts = {"metrics": metrics_path, "image": image_path}
    if write_video:
        iio.imwrite(
            video_path,
            video,
            plugin="pillow",
            duration=1_000 / fps,
            loop=0,
        )
        artifacts["video"] = video_path
    return artifacts
