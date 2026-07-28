#!/usr/bin/env python3
"""Compute tokenizer latent statistics from CoinRun ArrayRecords."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.configs import DataloaderConfig, DatasetConfig
from dreamer.data import build_iterator
from dreamer.parallel import build_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--batches", type=int, default=8)
    return parser.parse_args()


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
        categorical_action_dim=16,
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

    with jax.set_mesh(mesh):
        bundle = TokenizerCheckpointBundle.from_pretrained(
            str(args.checkpoint),
            mesh_rules=mesh_rules,
        )
        tokenizer = bundle.tokenizer

        @nnx.jit
        def encode(videos):
            latents, _, _ = tokenizer.encode(videos, deterministic=True)
            return latents

        iterator = iter(
            build_iterator(
                dataset_cfg,
                device=data_sharding,
                dtype="bfloat16",
                seq_len=args.frames,
            )
        )
        count = 0
        total = None
        total_sq = None

        for batch_index in range(args.batches):
            batch = next(iterator)
            latents = np.asarray(jax.device_get(encode(batch["videos"]))).astype(
                np.float64
            )
            flat = latents.reshape(-1, latents.shape[-1])
            batch_sum = flat.sum(axis=0)
            batch_sum_sq = np.square(flat).sum(axis=0)
            total = batch_sum if total is None else total + batch_sum
            total_sq = batch_sum_sq if total_sq is None else total_sq + batch_sum_sq
            count += flat.shape[0]
            print(f"stats batch {batch_index + 1}/{args.batches}", flush=True)

    assert total is not None and total_sq is not None and count > 0
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("non-finite latent statistics")

    payload = {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "num_samples": count,
        "num_videos": args.batch_size * args.batches,
        "frames": args.frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
