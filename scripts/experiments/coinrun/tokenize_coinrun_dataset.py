#!/usr/bin/env python3
"""Encode full CoinRun records once and write action-aligned latent shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import grain
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.coinrun_latent_dataset import (
    build_coinrun_latent_record,
    validate_coinrun_latent_pair,
)
from dreamer.data.shard_writer import ShardWriter
from dreamer.experiment_runtime import atomic_write_json, sha256_file
from dreamer.parallel import build_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, default=160)
    parser.add_argument("--batch-records", type=int, default=4)
    parser.add_argument("--records-per-shard", type=int, default=256)
    return parser.parse_args()


def _tree_sha256(paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    for path in sorted(paths):
        digest = sha256_file(path)
        size = path.stat().st_size
        entries.append({"name": path.name, "bytes": size, "sha256": digest})
        lines.append(f"{path.name} {size} {digest}")
    tree = hashlib.sha256(
        ("\n".join(lines) + "\n").encode("utf-8")
    ).hexdigest()
    return tree, entries


def _directory_tree_sha256(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {path}")
    lines = [
        f"{item.relative_to(path)} {item.stat().st_size} {sha256_file(item)}"
        for item in files
    ]
    return hashlib.sha256(
        ("\n".join(lines) + "\n").encode("utf-8")
    ).hexdigest()


def _resolve_checkpoint_step(checkpoint: Path) -> Path:
    numeric = sorted(
        (
            child
            for child in checkpoint.iterdir()
            if child.is_dir() and child.name.isdigit()
        ),
        key=lambda child: int(child.name),
    )
    return numeric[-1] if numeric else checkpoint


def _load_raw_record(
    source: grain.sources.ArrayRecordDataSource,
    index: int,
    *,
    expected_frames: int,
) -> tuple[dict[str, Any], np.ndarray]:
    raw = pickle.loads(source[index])
    sequence_length = int(raw["sequence_length"])
    if sequence_length != expected_frames:
        raise ValueError(
            f"record {index} sequence_length={sequence_length}, "
            f"expected {expected_frames}"
        )
    expected_bytes = expected_frames * 64 * 64 * 3
    if len(raw["raw_video"]) != expected_bytes:
        raise ValueError(
            f"record {index} raw_video bytes={len(raw['raw_video'])}, "
            f"expected {expected_bytes}"
        )
    video = np.frombuffer(raw["raw_video"], dtype=np.uint8).reshape(
        expected_frames,
        64,
        64,
        3,
    )
    return raw, video


def main() -> None:
    args = parse_args()
    if min(
        args.expected_records,
        args.expected_frames,
        args.batch_records,
        args.records_per_shard,
    ) <= 0:
        raise ValueError("record, frame, batch and shard sizes must be positive")
    if args.output_dir.exists():
        raise ValueError(
            f"output directory already exists: {args.output_dir}; "
            "offline tokenization requires a fresh immutable target"
        )

    raw_metadata_path = args.input_dir / "metadata.json"
    raw_metadata = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
    raw_shards = sorted(args.input_dir.glob("shard-*.array_record"))
    if not raw_shards:
        raise ValueError(f"no raw ArrayRecord shards in {args.input_dir}")
    raw_tree_sha256, raw_shard_entries = _tree_sha256(raw_shards)
    if raw_tree_sha256 != raw_metadata["tree_sha256"]:
        raise ValueError(
            "raw shard tree differs from collector metadata: "
            f"{raw_tree_sha256} != {raw_metadata['tree_sha256']}"
        )

    source = grain.sources.ArrayRecordDataSource(
        [str(path) for path in raw_shards]
    )
    if len(source) != args.expected_records:
        raise ValueError(
            f"raw record count {len(source)} != expected {args.expected_records}"
        )
    if int(raw_metadata["frames_per_record"]) != args.expected_frames:
        raise ValueError(
            "raw metadata frame count differs from tokenization protocol"
        )

    resolved_checkpoint = _resolve_checkpoint_step(args.checkpoint)
    checkpoint_tree_sha256 = _directory_tree_sha256(resolved_checkpoint)
    staging = args.output_dir.parent / (
        f".{args.output_dir.name}.incomplete-{uuid.uuid4()}"
    )
    staging.mkdir(parents=True, exist_ok=False)

    mesh, _, mesh_rules = build_parallel("data")
    sum_channels: np.ndarray | None = None
    sum_squares: np.ndarray | None = None
    latent_samples = 0
    written = 0

    try:
        with jax.set_mesh(mesh):
            bundle = TokenizerCheckpointBundle.from_pretrained(
                str(args.checkpoint),
                mesh_rules=mesh_rules,
            )
            tokenizer = bundle.tokenizer
            del tokenizer.decoder

            @nnx.jit
            def encode(videos: jax.Array) -> jax.Array:
                latents, _, _ = tokenizer.encode(
                    videos,
                    deterministic=True,
                )
                return latents

            with ShardWriter(
                staging,
                records_per_shard=args.records_per_shard,
            ) as writer:
                for batch_start in range(
                    0,
                    args.expected_records,
                    args.batch_records,
                ):
                    batch_end = min(
                        batch_start + args.batch_records,
                        args.expected_records,
                    )
                    raw_batch: list[dict[str, Any]] = []
                    video_batch: list[np.ndarray] = []
                    for index in range(batch_start, batch_end):
                        raw, video = _load_raw_record(
                            source,
                            index,
                            expected_frames=args.expected_frames,
                        )
                        raw_batch.append(raw)
                        video_batch.append(video)

                    videos = jnp.asarray(np.stack(video_batch)).astype(
                        jnp.bfloat16
                    )
                    latents = np.asarray(
                        jax.device_get(encode(videos)),
                        dtype=np.float32,
                    )
                    flat = latents.reshape(-1, latents.shape[-1]).astype(
                        np.float64
                    )
                    batch_sum = flat.sum(axis=0)
                    batch_sum_squares = np.square(flat).sum(axis=0)
                    sum_channels = (
                        batch_sum
                        if sum_channels is None
                        else sum_channels + batch_sum
                    )
                    sum_squares = (
                        batch_sum_squares
                        if sum_squares is None
                        else sum_squares + batch_sum_squares
                    )
                    latent_samples += flat.shape[0]

                    for offset, raw in enumerate(raw_batch):
                        index = batch_start + offset
                        latent_record = build_coinrun_latent_record(
                            raw,
                            latents[offset],
                            record_index=index,
                            raw_tree_sha256=raw_tree_sha256,
                        )
                        validate_coinrun_latent_pair(raw, latent_record)
                        writer.write(latent_record)
                        written += 1
                    print(
                        f"[{args.split}] tokenized "
                        f"{written}/{args.expected_records} records",
                        flush=True,
                    )

        if written != args.expected_records:
            raise RuntimeError(
                f"wrote {written}/{args.expected_records} latent records"
            )
        if (
            sum_channels is None
            or sum_squares is None
            or latent_samples <= 0
        ):
            raise RuntimeError("no latent statistics were accumulated")

        mean = sum_channels / latent_samples
        variance = np.maximum(
            sum_squares / latent_samples - np.square(mean),
            1e-12,
        )
        std = np.sqrt(variance)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("non-finite latent statistics")

        latent_shards = sorted(staging.glob("shard-*.array_record"))
        latent_tree_sha256, latent_shard_entries = _tree_sha256(latent_shards)
        metadata = {
            "schema_version": "1.0",
            "experiment_id": "CR-DYN-0011",
            "split": args.split,
            "records": written,
            "frames_per_record": args.expected_frames,
            "shards": len(latent_shards),
            "tree_sha256": latent_tree_sha256,
            "shard_entries": latent_shard_entries,
            "latent": {
                "shape_per_record": [
                    args.expected_frames,
                    int(latents.shape[2]),
                    int(latents.shape[3]),
                ],
                "storage_dtype": "float32",
                "mean": mean.astype(np.float32).tolist(),
                "std": std.astype(np.float32).tolist(),
                "num_channel_samples": latent_samples,
            },
            "tokenizer": {
                "checkpoint": str(args.checkpoint),
                "resolved_checkpoint": str(resolved_checkpoint),
                "resolved_checkpoint_tree_sha256": checkpoint_tree_sha256,
                "deterministic": True,
            },
            "source": {
                "raw_dataset": str(args.input_dir),
                "raw_metadata_sha256": sha256_file(raw_metadata_path),
                "raw_tree_sha256": raw_tree_sha256,
                "raw_shard_entries": raw_shard_entries,
            },
            "action_space": raw_metadata["action_space"],
            "action_policy": raw_metadata["action_policy"],
            "policy": raw_metadata["policy"],
            "start_level": raw_metadata["start_level"],
            "num_levels": raw_metadata["num_levels"],
            "transition_alignment": (
                "latents[t]=tokenizer(raw_video[t]), actions[t]=action_t, "
                "rewards[t]=reward observed after action_t"
            ),
            "rewards_preserved": True,
            "terminals_preserved": True,
        }
        atomic_write_json(staging / "metadata.json", metadata)
        os.replace(staging, args.output_dir)
        print(json.dumps(metadata, indent=2), flush=True)
    except BaseException:
        failure = {
            "schema_version": "1.0",
            "experiment_id": "CR-DYN-0011",
            "split": args.split,
            "records_written_before_failure": written,
            "source_raw_tree_sha256": raw_tree_sha256,
            "tokenizer_checkpoint_tree_sha256": checkpoint_tree_sha256,
        }
        atomic_write_json(staging / "failure.json", failure)
        raise
    finally:
        if args.output_dir.exists() and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
