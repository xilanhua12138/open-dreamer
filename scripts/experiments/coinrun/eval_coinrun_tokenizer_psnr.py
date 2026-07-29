#!/usr/bin/env python3
"""Evaluate a CoinRun tokenizer checkpoint on held-out ArrayRecords."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.configs import DataloaderConfig, DatasetConfig
from dreamer.data import build_iterator
from dreamer.image_metrics import (
    edge_mask,
    psnr_from_squared_error,
    region_squared_error,
    temporal_change_mask,
)
from dreamer.parallel import build_parallel
from dreamer.training import compute_psnr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        help="Exact zero-based checkpoint step. Defaults to the latest checkpoint.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4242)
    return parser.parse_args()


def pixel_metrics(pred: jax.Array, target: jax.Array) -> tuple[jax.Array, jax.Array]:
    pred_01 = jnp.clip(pred.astype(jnp.float32) / 255.0, 0.0, 1.0)
    target_01 = jnp.clip(target.astype(jnp.float32) / 255.0, 0.0, 1.0)
    mse = jnp.mean((pred_01 - target_01) ** 2)
    return mse, compute_psnr(pred_01, target_01)


@nnx.jit
def evaluate_clean(model, videos):
    pred, _, _ = model(videos, deterministic=True)
    mse, psnr = pixel_metrics(pred, videos)
    return mse, psnr, pred


@nnx.jit
def evaluate_masked(model, videos, mae_key, dropout_key):
    pred, (frame_mask, _), _ = model(
        videos,
        deterministic=False,
        rngs=nnx.Rngs(mae=mae_key, dropout=dropout_key),
    )
    mse, psnr = pixel_metrics(pred, videos)
    return mse, psnr, pred, frame_mask


def to_uint8(value: jax.Array | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(jax.device_get(value), dtype=np.float32), 0, 255).astype(
        np.uint8
    )


def main() -> None:
    args = parse_args()
    mesh, data_sharding, mesh_rules = build_parallel("data")
    dataset_cfg = DatasetConfig(
        name="coinrun",
        data_type="video",
        array_record_path=str(args.dataset),
        H=64,
        W=64,
        C=3,
        patch_size=8,
        categorical_action_dim=15,
        categorical_noop_action=4,
        p_include_reward=0.5,
        dataloader_cfg=DataloaderConfig(
            B=args.batch_size,
            num_workers=2,
            prefetch_buffer_size=2,
            device_prefetch_buffer_size=1,
            short_T=args.frames,
            long_T=args.frames,
            long_ratio=0.0,
            dtype="bfloat16",
        ),
    )

    metric_names = (
        "online_clean_mse",
        "online_clean_psnr",
        "online_masked_mse",
        "online_masked_psnr",
        "ema_clean_mse",
        "ema_clean_psnr",
        "ema_masked_mse",
        "ema_masked_psnr",
    )
    totals = {name: 0.0 for name in metric_names}
    regional_totals = {
        name: {"squared_error": 0.0, "count": 0}
        for name in (
            "online_clean_edge",
            "online_clean_temporal_change",
            "ema_clean_edge",
            "ema_clean_temporal_change",
        )
    }
    first_visualization = None

    with jax.set_mesh(mesh):
        bundle = TokenizerCheckpointBundle.from_pretrained(
            str(args.checkpoint),
            mesh_rules=mesh_rules,
            step=args.checkpoint_step,
        )
        iterator = iter(
            build_iterator(
                dataset_cfg,
                seed=args.seed,
                device=data_sharding,
                dtype="bfloat16",
                seq_len=args.frames,
            )
        )

        for batch_index in range(args.batches):
            videos = next(iterator)["videos"]
            batch_key = jax.random.fold_in(jax.random.key(args.seed), batch_index)
            mae_key, dropout_key = jax.random.split(batch_key)

            online_clean_mse, online_clean_psnr, online_clean_pred = evaluate_clean(
                bundle.tokenizer, videos
            )
            (
                online_masked_mse,
                online_masked_psnr,
                _,
                _,
            ) = evaluate_masked(bundle.tokenizer, videos, mae_key, dropout_key)
            ema_clean_mse, ema_clean_psnr, ema_clean_pred = evaluate_clean(
                bundle.tokenizer_ema, videos
            )
            (
                ema_masked_mse,
                ema_masked_psnr,
                ema_masked_pred,
                frame_mask,
            ) = evaluate_masked(bundle.tokenizer_ema, videos, mae_key, dropout_key)

            values = {
                "online_clean_mse": online_clean_mse,
                "online_clean_psnr": online_clean_psnr,
                "online_masked_mse": online_masked_mse,
                "online_masked_psnr": online_masked_psnr,
                "ema_clean_mse": ema_clean_mse,
                "ema_clean_psnr": ema_clean_psnr,
                "ema_masked_mse": ema_masked_mse,
                "ema_masked_psnr": ema_masked_psnr,
            }
            for name, value in values.items():
                totals[name] += float(jax.device_get(value))

            target_u8 = to_uint8(videos)
            edge = edge_mask(target_u8, threshold=16 / 255)
            temporal_change = temporal_change_mask(
                target_u8,
                threshold=16 / 255,
            )
            for prefix, prediction in (
                ("online_clean", online_clean_pred),
                ("ema_clean", ema_clean_pred),
            ):
                prediction_u8 = to_uint8(prediction)
                for region_name, region_mask in (
                    ("edge", edge),
                    ("temporal_change", temporal_change),
                ):
                    squared_error, count = region_squared_error(
                        prediction_u8,
                        target_u8,
                        region_mask,
                    )
                    aggregate = regional_totals[f"{prefix}_{region_name}"]
                    aggregate["squared_error"] += squared_error
                    aggregate["count"] += count

            if first_visualization is None:
                first_visualization = (
                    target_u8,
                    to_uint8(ema_clean_pred),
                    to_uint8(ema_masked_pred),
                    np.asarray(jax.device_get(frame_mask), dtype=bool),
                )
            print(
                f"eval batch {batch_index + 1}/{args.batches}: "
                f"online_masked_psnr={float(online_masked_psnr):.4f}, "
                f"ema_masked_psnr={float(ema_masked_psnr):.4f}",
                flush=True,
            )

    metrics = {name: totals[name] / args.batches for name in metric_names}
    regions = {}
    for name, aggregate in regional_totals.items():
        metrics[f"{name}_psnr"] = psnr_from_squared_error(
            aggregate["squared_error"],
            aggregate["count"],
        )
        regions[name] = {
            "selected_rgb_values": aggregate["count"],
            "squared_error": aggregate["squared_error"],
        }

    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": args.checkpoint_step,
        "completed_updates": (
            None if args.checkpoint_step is None else args.checkpoint_step + 1
        ),
        "dataset": str(args.dataset),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "frames": args.frames,
        "batches": args.batches,
        "num_clips": args.batch_size * args.batches,
        "metrics": metrics,
        "regions": regions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    assert first_visualization is not None
    target, clean_pred, masked_pred, frame_mask = first_visualization
    sample_count = min(8, target.shape[0])
    time_index = 0
    target_frames = target[:sample_count, time_index]
    masked_input = target_frames.copy()
    mask = frame_mask[:sample_count, time_index]
    masked_input[np.broadcast_to(mask, masked_input.shape)] = 0
    rows = (
        target_frames,
        masked_input,
        masked_pred[:sample_count, time_index],
        clean_pred[:sample_count, time_index],
    )
    grid = np.concatenate(
        [np.concatenate(list(row), axis=1) for row in rows],
        axis=0,
    )
    args.visualization.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.visualization, grid)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
