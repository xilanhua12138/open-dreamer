#!/usr/bin/env python3
"""Evaluate CoinRun history length while holding every future target fixed.

Each held-out record is loaded once at ``max(contexts) + horizon`` frames. The
tokenizer encodes that full sequence once, then each context condition slices
the same latent/video/action tensors. Therefore every context length predicts
the exact same future frames with the exact same future actions and RNG seed.
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

from score_coinrun_rollouts import frame_ssim
from dreamer.actions import shift_actions
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import build_iterator
from dreamer.generation import DenoiseSchedule
from dreamer.parallel import build_parallel
from dreamer.sampler import decode_jit, encode_jit, sample_video


os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

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
    parser.add_argument("--contexts", type=int, nargs="+", default=(4, 16, 32))
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num-videos", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--denoise-steps", type=int, default=4)
    return parser.parse_args()


def save_video(frames: np.ndarray, path: Path, fps: int = 20) -> None:
    iio.imwrite(str(path), frames, plugin="pyav", fps=fps, codec="libx264")


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def score_raw_arrays(pred: np.ndarray, gt: np.ndarray, context: int) -> dict:
    """Score exact uint8 tensors, avoiding context-dependent MP4 artifacts."""
    pred_float = pred.astype(np.float64) / 255.0
    gt_float = gt.astype(np.float64) / 255.0
    mse = np.mean((pred_float - gt_float) ** 2, axis=(2, 3, 4))
    psnr = 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))
    ssim = np.asarray(
        [
            [
                frame_ssim(pred_frame, gt_frame)
                for pred_frame, gt_frame in zip(pred_video, gt_video, strict=True)
            ]
            for pred_video, gt_video in zip(pred_float, gt_float, strict=True)
        ]
    )
    windows = [n for n in (1, 3, 8, 16) if n <= psnr.shape[1]]
    per_video_mse = mse.mean(axis=1)
    return {
        "metric_source": "raw uint8 arrays before MP4 encoding",
        "num_videos": int(pred.shape[0]),
        "context_frames": context,
        "predicted_frames": int(pred.shape[1]),
        "mean_video_psnr_db": float(
            np.mean(10.0 * np.log10(1.0 / np.maximum(per_video_mse, 1e-12)))
        ),
        "mean_frame_psnr_db": float(psnr.mean()),
        "mean_ssim": float(ssim.mean()),
        "psnr_by_horizon_db": {
            str(n): float(psnr[:, :n].mean()) for n in windows
        },
        "ssim_by_horizon": {
            str(n): float(ssim[:, :n].mean()) for n in windows
        },
    }


def main() -> None:
    args = parse_args()
    contexts = tuple(sorted(set(args.contexts)))
    if not contexts or contexts[0] < 1:
        raise ValueError(f"All context lengths must be positive, got {contexts}")

    max_context = max(contexts)
    sequence_length = max_context + args.horizon
    checkpoint = args.checkpoint
    if checkpoint.name != "checkpoints" and (checkpoint / "checkpoints").is_dir():
        checkpoint = checkpoint / "checkpoints"

    config_dir = Path(__file__).resolve().parent / "configs"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(
            config_name="eval_fvd",
            overrides=[
                "dataset=coinrun",
                f"dataset.array_record_path={args.dataset}",
                f"dataset.dataloader_cfg.B={args.batch_size}",
                f"dataset.dataloader_cfg.long_T={sequence_length}",
                "dataset.dataloader_cfg.num_workers=0",
                f"seed={args.seed}",
                "parallel_strategy=data",
            ],
        )

    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)
    args.output.mkdir(parents=True, exist_ok=True)
    out_dirs = {}
    for context in contexts:
        out_dir = args.output / f"ctx_{context}" / "ema_shortcut"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[context] = out_dir
    raw_predictions = {context: [] for context in contexts}
    raw_targets = {context: [] for context in contexts}

    manifest = {
        "checkpoint": str(checkpoint),
        "dataset": str(args.dataset),
        "contexts": list(contexts),
        "max_context": max_context,
        "horizon": args.horizon,
        "num_videos": args.num_videos,
        "seed": args.seed,
        "denoise_steps": args.denoise_steps,
        "fixed_target_protocol": (
            "Load and encode max_context+horizon once; slice context prefixes "
            "while reusing the same future latents, raw frames, actions, and RNG."
        ),
        "targets": [],
    }

    with jax.set_mesh(mesh):
        bundle = DynamicsCheckpointBundle.from_pretrained(
            str(checkpoint),
            mesh_rules=mesh_rules,
            model_names={"dynamics_ema", "tokenizer"},
        )
        tokenizer = bundle.tokenizer
        dynamics = bundle.dynamics_ema
        schedule = DenoiseSchedule.init(args.denoise_steps, dynamics.cfg.k_max)
        dataloader = build_iterator(
            cfg.dataset,
            seed=args.seed,
            device=data_sharding,
            return_actions=True,
        )
        rng = jax.random.PRNGKey(args.seed)
        collected = 0
        pbar = tqdm(total=args.num_videos, desc="Fixed-target context ablation")

        for batch in dataloader:
            if collected >= args.num_videos:
                break

            videos = batch["videos"]
            shifted_actions = shift_actions(
                batch["actions"], cfg.dataset.categorical_action_dim
            )
            full_latents = encode_jit(tokenizer, videos)
            full_gt_decoded = jnp_to_u8(decode_jit(tokenizer, full_latents))
            original = jnp_to_u8(videos)
            rng, eval_rng = jax.random.split(rng)
            batch_size = min(videos.shape[0], args.num_videos - collected)

            per_context = {}
            for context in contexts:
                start = max_context - context
                stop = sequence_length
                sliced_latents = full_latents[:, start:stop]
                sliced_actions = shifted_actions[:, start:stop]
                pred_frames, _, _ = sample_video(
                    tokenizer=tokenizer,
                    dynamics=dynamics,
                    frames=None,
                    actions=sliced_actions,
                    horizon=args.horizon,
                    schedule_config=schedule,
                    rng=eval_rng,
                    latents=sliced_latents,
                )
                per_context[context] = jnp_to_u8(pred_frames)

            original_host = np.asarray(jax.device_get(original))
            decoded_host = np.asarray(jax.device_get(full_gt_decoded))
            actions_host = np.asarray(
                jax.device_get(shifted_actions.categorical)
            )
            pred_host = {
                context: np.asarray(jax.device_get(value))
                for context, value in per_context.items()
            }

            for batch_index in range(batch_size):
                idx = collected + batch_index
                future_slice = slice(max_context, sequence_length)
                manifest["targets"].append(
                    {
                        "video": idx,
                        "future_original_sha256": array_sha256(
                            original_host[batch_index, future_slice]
                        ),
                        "future_decoded_sha256": array_sha256(
                            decoded_host[batch_index, future_slice]
                        ),
                        "future_actions_sha256": array_sha256(
                            actions_host[batch_index, future_slice]
                        ),
                    }
                )

                for context in contexts:
                    start = max_context - context
                    stop = sequence_length
                    out_dir = out_dirs[context]
                    raw_predictions[context].append(
                        pred_host[context][batch_index, context:]
                    )
                    raw_targets[context].append(
                        decoded_host[batch_index, max_context:sequence_length]
                    )
                    save_video(
                        pred_host[context][batch_index],
                        out_dir / f"pred_{idx:06d}.mp4",
                    )
                    save_video(
                        decoded_host[batch_index, start:stop],
                        out_dir / f"gt_decoded_{idx:06d}.mp4",
                    )
                    save_video(
                        original_host[batch_index, start:stop],
                        out_dir / f"original_{idx:06d}.mp4",
                    )

            collected += batch_size
            pbar.update(batch_size)

        pbar.close()

    for context in contexts:
        pred = np.stack(raw_predictions[context])
        gt = np.stack(raw_targets[context])
        context_dir = args.output / f"ctx_{context}"
        np.savez_compressed(
            context_dir / "raw_future_arrays.npz",
            pred=pred,
            gt_decoded=gt,
        )
        metrics = score_raw_arrays(pred, gt, context)
        (context_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )

    (args.output / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def jnp_to_u8(array: jax.Array) -> jax.Array:
    return jax.numpy.clip(array, 0, 255).astype(jax.numpy.uint8)


if __name__ == "__main__":
    main()
