#!/usr/bin/env python3
"""Measure whether CoinRun dynamics uses future actions.

Every condition reuses the same context, future target and PRNG key. Only
future actions change, so aligned-action advantage is attributable to action
conditioning rather than a different sampled future or noise draw.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import operator
import os
from pathlib import Path

import hydra
import imageio.v3 as iio
import jax
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from dreamer.actions import shift_actions
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.coinrun_dynamics_repair import build_future_action_conditions
from dreamer.data import build_iterator
from dreamer.generation import DenoiseSchedule
from dreamer.parallel import build_parallel
from dreamer.sampler import sample_video
from eval_coinrun_context_ablation import score_raw_arrays


os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

for name, resolver in (
    ("mul", lambda *args: functools.reduce(operator.mul, args)),
    ("sum", lambda *args: sum(args)),
    ("floordiv", lambda x, y: x // y),
    ("max", lambda *args: max(args)),
):
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, resolver)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num-videos", type=int, default=64)
    parser.add_argument("--visual-videos", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--denoise-steps", type=int, default=4)
    return parser.parse_args()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def save_video(frames: np.ndarray, path: Path, fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), frames, plugin="pyav", fps=fps, codec="libx264")


def main() -> None:
    args = parse_args()
    if min(
        args.context,
        args.horizon,
        args.num_videos,
        args.batch_size,
        args.denoise_steps,
    ) <= 0:
        raise ValueError("context, horizon, counts and denoise steps must be positive")
    if args.batch_size < 2:
        raise ValueError("batch-size must be >= 2 for batch-shuffled controls")
    if not 0 <= args.visual_videos <= args.num_videos:
        raise ValueError("visual-videos must be in [0, num-videos]")

    sequence_length = args.context + args.horizon
    checkpoint = args.checkpoint
    if checkpoint.name != "checkpoints" and (checkpoint / "checkpoints").is_dir():
        checkpoint = checkpoint / "checkpoints"

    root = Path(__file__).resolve().parents[3]
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(root / "configs"),
    ):
        cfg = hydra.compose(
            config_name="eval_fvd",
            overrides=[
                "dataset=coinrun",
                f"dataset.array_record_path={args.dataset}",
                f"dataset.dataloader_cfg.B={args.batch_size}",
                f"dataset.dataloader_cfg.short_T={sequence_length}",
                f"dataset.dataloader_cfg.long_T={sequence_length}",
                "dataset.dataloader_cfg.num_workers=0",
                f"seed={args.seed}",
                "parallel_strategy=data",
            ],
        )

    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "aligned",
            "batch_shuffled",
            "one_step_shifted",
            "all_noop",
        )
    }
    targets: list[np.ndarray] = []
    target_hashes: list[dict[str, object]] = []

    with jax.set_mesh(mesh):
        bundle = DynamicsCheckpointBundle.from_pretrained(
            str(checkpoint),
            mesh_rules=mesh_rules,
            model_names={"dynamics_ema", "tokenizer"},
        )
        tokenizer = bundle.tokenizer
        dynamics = bundle.dynamics_ema
        schedule = DenoiseSchedule.init(
            args.denoise_steps,
            dynamics.cfg.k_max,
        )
        dataloader = build_iterator(
            cfg.dataset,
            seed=args.seed,
            device=data_sharding,
            return_actions=True,
        )
        rng = jax.random.PRNGKey(args.seed)
        collected = 0
        pbar = tqdm(total=args.num_videos, desc="Action-conditioning controls")

        for batch in dataloader:
            if collected >= args.num_videos:
                break
            shifted_actions = shift_actions(
                batch["actions"],
                cfg.dataset.categorical_action_dim,
                cfg.dataset.categorical_noop_action,
            )
            conditions = build_future_action_conditions(
                shifted_actions,
                context_frames=args.context,
                categorical_noop_action=(
                    cfg.dataset.categorical_noop_action
                ),
            )
            rng, eval_rng = jax.random.split(rng)
            condition_frames = {}
            gt_decoded = None
            for name, condition_actions in conditions.items():
                pred_frames, decoded_frames, _ = sample_video(
                    tokenizer=tokenizer,
                    dynamics=dynamics,
                    frames=batch["videos"],
                    actions=condition_actions,
                    horizon=args.horizon,
                    schedule_config=schedule,
                    rng=eval_rng,
                )
                condition_frames[name] = np.asarray(
                    jax.device_get(pred_frames)
                )
                if gt_decoded is None:
                    gt_decoded = np.asarray(jax.device_get(decoded_frames))

            if gt_decoded is None:
                raise RuntimeError("no target frames were decoded")
            actions_host = np.asarray(
                jax.device_get(shifted_actions.categorical)
            )
            batch_count = min(
                int(batch["videos"].shape[0]),
                args.num_videos - collected,
            )
            future_slice = slice(args.context, sequence_length)
            for batch_index in range(batch_count):
                video_index = collected + batch_index
                target_future = gt_decoded[batch_index, future_slice]
                targets.append(target_future)
                target_hashes.append(
                    {
                        "video": video_index,
                        "future_decoded_sha256": array_sha256(target_future),
                        "aligned_future_actions_sha256": array_sha256(
                            actions_host[batch_index, future_slice]
                        ),
                    }
                )
                for name, frames in condition_frames.items():
                    future = frames[batch_index, future_slice]
                    predictions[name].append(future)
                    if video_index < args.visual_videos:
                        save_video(
                            frames[batch_index],
                            args.output
                            / "visuals"
                            / name
                            / f"pred_{video_index:06d}.mp4",
                        )
                        save_video(
                            gt_decoded[batch_index],
                            args.output
                            / "visuals"
                            / name
                            / f"gt_decoded_{video_index:06d}.mp4",
                        )
            collected += batch_count
            pbar.update(batch_count)
        pbar.close()

    target_array = np.stack(targets)
    metrics = {
        name: score_raw_arrays(
            np.stack(values),
            target_array,
            args.context,
        )
        for name, values in predictions.items()
    }
    aligned_predictions = np.stack(predictions["aligned"]).astype(np.float32) / 255.0
    advantages = {}
    divergence = {}
    for name in ("batch_shuffled", "one_step_shifted", "all_noop"):
        advantages[name] = {
            horizon: (
                metrics["aligned"]["psnr_by_horizon_db"][horizon]
                - metrics[name]["psnr_by_horizon_db"][horizon]
            )
            for horizon in metrics["aligned"]["psnr_by_horizon_db"]
        }
        corrupted = np.stack(predictions[name]).astype(np.float32) / 255.0
        divergence[name] = float(
            np.mean((aligned_predictions - corrupted) ** 2)
        )

    payload = {
        "schema_version": "1.0",
        "checkpoint": str(checkpoint),
        "dataset": str(args.dataset),
        "context_frames": args.context,
        "predicted_frames": args.horizon,
        "num_videos": args.num_videos,
        "seed": args.seed,
        "denoise_steps": args.denoise_steps,
        "fixed_noise": True,
        "condition_protocol": (
            "Same context, future target and PRNG key. Only future actions "
            "change for batch-shuffled, one-step-shifted and all-noop controls."
        ),
        "conditions": metrics,
        "aligned_psnr_advantage_db": advantages,
        "prediction_divergence_mse_from_aligned": divergence,
        "targets": target_hashes,
    }
    output_path = args.output / "action-conditioning.json"
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
