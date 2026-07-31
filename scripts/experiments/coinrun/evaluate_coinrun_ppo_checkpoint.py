#!/usr/bin/env python3
"""Evaluate an existing PPO checkpoint under one explicit CoinRun protocol."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from dreamer.coinrun import COINRUN_ACTION_DIM
from dreamer.coinrun_ppo import PPOCoinRunPolicy
from dreamer.coinrun_ppo_evaluation import evaluate_policy, write_evaluation_media
from dreamer.experiment_runtime import atomic_write_json


def build_evaluation_record(
    *,
    summary: dict[str, Any],
    metadata: dict[str, Any],
    checkpoint_sha256: str,
    episodes: int,
    seed: int,
    start_level: int,
    num_levels: int,
    policy: str,
    distribution_mode: str,
) -> dict[str, Any]:
    """Attach checkpoint and evaluation identity to raw episode statistics."""

    completed_env_steps = metadata.get("completed_env_steps")
    train_config = metadata.get("train_config")
    if not isinstance(completed_env_steps, int) or completed_env_steps <= 0:
        raise ValueError("checkpoint metadata has invalid completed_env_steps")
    if not isinstance(train_config, dict):
        raise ValueError("checkpoint metadata is missing train_config")
    if summary.get("episodes") != episodes:
        raise ValueError(
            f"summary episodes: expected {episodes}, got {summary.get('episodes')}"
        )
    if len(checkpoint_sha256) != 64:
        raise ValueError("checkpoint_sha256 must contain 64 hexadecimal characters")
    int(checkpoint_sha256, 16)
    return {
        "schema_version": "1.0",
        **summary,
        "completed_env_steps": completed_env_steps,
        "checkpoint_sha256": checkpoint_sha256,
        "policy": policy,
        "seed": seed,
        "start_level": start_level,
        "num_levels": num_levels,
        "distribution_mode": distribution_mode,
        "evaluation_distribution": (
            "full_distribution" if num_levels == 0 else "fixed_level_range"
        ),
        "evaluation_level_range": (
            None if num_levels == 0 else [start_level, start_level + num_levels]
        ),
        "train_level_range": [
            int(train_config["start_level"]),
            int(train_config["start_level"]) + int(train_config["num_levels"]),
        ],
        "checkpoint_train_config": train_config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--start-level", type=int, default=0)
    parser.add_argument("--num-levels", type=int, default=0)
    parser.add_argument("--distribution-mode", default="easy")
    parser.add_argument(
        "--policy",
        choices=("stochastic", "deterministic_argmax"),
        default="stochastic",
    )
    parser.add_argument("--max-vector-steps", type=int, default=40000)
    parser.add_argument("--visual-episodes", type=int, default=2)
    parser.add_argument("--visual-max-frames", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.episodes, args.envs, args.max_vector_steps) <= 0:
        raise ValueError("episodes, envs and max_vector_steps must be positive")
    if args.visual_episodes < 0 or args.visual_max_frames <= 0:
        raise ValueError(
            "visual_episodes must be non-negative and visual_max_frames positive"
        )
    try:
        from procgen import ProcgenGym3Env
    except ImportError as error:
        raise RuntimeError("Procgen is required for checkpoint evaluation") from error

    checkpoint = args.checkpoint.resolve()
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    env = ProcgenGym3Env(
        num=args.envs,
        env_name="coinrun",
        start_level=args.start_level,
        num_levels=args.num_levels,
        distribution_mode=args.distribution_mode,
        rand_seed=args.seed,
    )
    if int(env.ac_space.eltype.n) != COINRUN_ACTION_DIM:
        raise ValueError("CoinRun action-space identity mismatch")
    policy = PPOCoinRunPolicy.from_checkpoint(
        checkpoint,
        num_envs=args.envs,
        seed=args.seed,
        deterministic=args.policy == "deterministic_argmax",
    )
    summary, videos = evaluate_policy(
        env=env,
        policy=policy,
        num_episodes=args.episodes,
        max_vector_steps=args.max_vector_steps,
        max_visual_episodes=args.visual_episodes,
    )
    record = build_evaluation_record(
        summary=summary,
        metadata=policy.metadata,
        checkpoint_sha256=checkpoint_sha256,
        episodes=args.episodes,
        seed=args.seed,
        start_level=args.start_level,
        num_levels=args.num_levels,
        policy=args.policy,
        distribution_mode=args.distribution_mode,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "metrics.json", record)
    write_evaluation_media(
        videos=videos,
        output_dir=output_dir,
        completed_env_steps=int(record["completed_env_steps"]),
        max_frames_per_video=args.visual_max_frames,
    )


if __name__ == "__main__":
    main()
