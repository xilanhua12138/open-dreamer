#!/usr/bin/env python3
"""Collect episode-safe CoinRun dynamics records with a frozen PPO policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

from dreamer.coinrun import COINRUN_ACTION_DIM, COINRUN_NOOP_ACTION
from dreamer.coinrun_collection import CoinRunRecordAccumulator
from dreamer.coinrun_ppo import Gym3VectorEnvAdapter, PPOCoinRunPolicy
from dreamer.coinrun_ppo_evaluation import EpisodeStatisticsTracker
from dreamer.configs import LoggerConfig
from dreamer.experiment_runtime import atomic_write_json, sha256_file
from dreamer.logging import build_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--records-per-shard", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20_240)
    parser.add_argument("--start-level", type=int, required=True)
    parser.add_argument("--num-levels", type=int, required=True)
    parser.add_argument("--distribution-mode", default="easy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--exploration-epsilon", type=float, default=0.05)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-vector-steps", type=int)
    return parser.parse_args()


def _make_env(args: argparse.Namespace) -> Any:
    try:
        from procgen import ProcgenGym3Env
    except ImportError as error:
        raise RuntimeError(
            "Procgen is required to collect real CoinRun trajectories."
        ) from error
    env = ProcgenGym3Env(
        num=args.envs,
        env_name="coinrun",
        start_level=args.start_level,
        num_levels=args.num_levels,
        distribution_mode=args.distribution_mode,
        rand_seed=args.seed,
    )
    action_dim = int(env.ac_space.eltype.n)
    if action_dim != COINRUN_ACTION_DIM:
        raise ValueError(
            f"expected CoinRun action dimension {COINRUN_ACTION_DIM}, "
            f"got {action_dim}"
        )
    return env


def _tree_sha256(shards: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    entries = []
    lines = []
    for shard in shards:
        digest = sha256_file(shard)
        size = shard.stat().st_size
        entries.append(
            {
                "name": shard.name,
                "bytes": size,
                "sha256": digest,
            }
        )
        lines.append(f"{shard.name} {size} {digest}")
    tree_digest = hashlib.sha256(
        ("\n".join(lines) + "\n").encode("utf-8")
    ).hexdigest()
    return tree_digest, entries


def main() -> None:
    args = parse_args()
    positive = {
        "records": args.records,
        "frames": args.frames,
        "envs": args.envs,
        "records_per_shard": args.records_per_shard,
        "num_levels": args.num_levels,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"collection counts must be positive: {invalid}")
    if args.start_level < 0:
        raise ValueError("start_level must be non-negative")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= args.exploration_epsilon <= 1.0:
        raise ValueError("exploration_epsilon must be in [0, 1]")
    max_vector_steps = (
        args.max_vector_steps
        if args.max_vector_steps is not None
        else max(10_000, (args.records * args.frames * 20) // args.envs)
    )
    if max_vector_steps <= 0:
        raise ValueError("max_vector_steps must be positive")

    checkpoint = args.checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    output_dir = args.output_dir.resolve()
    run_dir = args.run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(output_dir.glob("shard-*.array_record")):
        raise ValueError(
            f"{output_dir} already contains ArrayRecord shards; use a fresh "
            "dataset directory to preserve provenance"
        )
    env = _make_env(args)
    adapter = Gym3VectorEnvAdapter(env, action_dim=COINRUN_ACTION_DIM)
    policy = PPOCoinRunPolicy.from_checkpoint(
        checkpoint,
        num_envs=args.envs,
        seed=args.seed,
        temperature=args.temperature,
        exploration_epsilon=args.exploration_epsilon,
        deterministic=args.deterministic,
    )
    accumulator = CoinRunRecordAccumulator(
        num_envs=args.envs,
        frames_per_record=args.frames,
    )
    episode_tracker = EpisodeStatisticsTracker(num_envs=args.envs)
    action_counts = np.zeros(COINRUN_ACTION_DIM, dtype=np.int64)
    completed_episode_returns: list[float] = []
    written = 0
    vector_steps = 0
    shard_index = -1
    records_in_shard = 0
    writer: ArrayRecordWriter | None = None

    runtime_config = {
        "schema_version": "1.0",
        "experiment_id": "CR-PPO-0001",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "records": args.records,
        "frames": args.frames,
        "envs": args.envs,
        "records_per_shard": args.records_per_shard,
        "seed": args.seed,
        "start_level": args.start_level,
        "num_levels": args.num_levels,
        "distribution_mode": args.distribution_mode,
        "temperature": args.temperature,
        "exploration_epsilon": args.exploration_epsilon,
        "deterministic": args.deterministic,
        "max_vector_steps": max_vector_steps,
        "transition_alignment": "observation_t_action_t_resulting_reward_t",
        "episode_boundary_policy": "discard_partial_chunk_on_terminal",
        "consumer": "action_conditioned_dynamics_only",
    }
    logger_config = LoggerConfig(
        run_name=args.run_name,
        use_wandb=False,
        log_every=1,
        max_steps=args.records,
        telemetry_progress_every_seconds=30.0,
        telemetry_system_every_seconds=60.0,
    )

    def open_next_shard() -> ArrayRecordWriter:
        nonlocal shard_index, records_in_shard
        shard_index += 1
        records_in_shard = 0
        path = output_dir / f"shard-{shard_index:05d}.array_record"
        return ArrayRecordWriter(str(path), "group_size:1")

    with build_logger(
        logger_config,
        config=runtime_config,
        dir=str(run_dir),
    ) as logger:
        try:
            while written < args.records and vector_steps < max_vector_steps:
                current = adapter.current
                actions = policy.sample(
                    current.observation,
                    current.first,
                )
                action_counts += np.bincount(
                    actions,
                    minlength=COINRUN_ACTION_DIM,
                )
                transition = adapter.step(actions)
                vector_steps += 1
                completed_episode_returns.extend(
                    episode.episode_return
                    for episode in episode_tracker.observe(
                        rewards=transition.rewards,
                        terminals=transition.terminals,
                    )
                )
                for _, record in accumulator.append(transition):
                    if writer is None or (
                        records_in_shard >= args.records_per_shard
                    ):
                        if writer is not None:
                            writer.close()
                        writer = open_next_shard()
                    writer.write(
                        pickle.dumps(
                            record,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    )
                    written += 1
                    records_in_shard += 1
                    if written % 16 == 0 or written == args.records:
                        returns = np.asarray(
                            completed_episode_returns,
                            dtype=np.float64,
                        )
                        logger.log_metrics(
                            step=written - 1,
                            prefix="collection/",
                            metrics={
                                "records_written": written,
                                "vector_steps": vector_steps,
                                "completed_episodes": returns.size,
                                **(
                                    {
                                        "episode_return_mean": float(
                                            np.mean(returns)
                                        ),
                                        "episode_success_rate": float(
                                            np.mean(returns > 0.0)
                                        ),
                                    }
                                    if returns.size
                                    else {}
                                ),
                            },
                        )
                    if written == args.records:
                        break
        finally:
            if writer is not None:
                writer.close()
        if written != args.records:
            raise RuntimeError(
                f"collected only {written}/{args.records} complete records "
                f"within {max_vector_steps} vector steps"
            )

        shards = sorted(output_dir.glob("shard-*.array_record"))
        tree_digest, shard_entries = _tree_sha256(shards)
        episode_returns = np.asarray(
            completed_episode_returns,
            dtype=np.float64,
        )
        metadata = {
            "schema_version": "2.0",
            "env": "coinrun",
            "records": written,
            "frames_per_record": args.frames,
            "seed": args.seed,
            "start_level": args.start_level,
            "num_levels": args.num_levels,
            "level_range": [
                args.start_level,
                args.start_level + args.num_levels,
            ],
            "distribution_mode": args.distribution_mode,
            "shards": len(shards),
            "tree_sha256": tree_digest,
            "shard_entries": shard_entries,
            "action_space": {
                "categorical_action_dim": COINRUN_ACTION_DIM,
                "categorical_noop_action": COINRUN_NOOP_ACTION,
            },
            "action_policy": "ppo",
            "policy": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_completed_env_steps": int(
                    policy.metadata["completed_env_steps"]
                ),
                "architecture": policy.metadata["architecture"],
                "temperature": args.temperature,
                "exploration_epsilon": args.exploration_epsilon,
                "deterministic": args.deterministic,
            },
            "transition_alignment": (
                "raw_video[t]=observation_t, actions[t]=action_t, "
                "rewards[t]=reward observed after action_t"
            ),
            "episode_boundary_policy": (
                "records never cross auto-reset boundaries; partial chunks "
                "are discarded when terminals[t] is true"
            ),
            "terminals_in_records": True,
            "action_counts": {
                str(action_id): int(count)
                for action_id, count in enumerate(action_counts)
                if count
            },
            "completed_episodes": int(episode_returns.size),
            "successful_episodes": int(np.sum(episode_returns > 0.0)),
            "success_rate": (
                float(np.mean(episode_returns > 0.0))
                if episode_returns.size
                else None
            ),
            "mean_episode_return": (
                float(np.mean(episode_returns))
                if episode_returns.size
                else None
            ),
            "vector_steps": vector_steps,
            "environment_transitions": vector_steps * args.envs,
            "consumer": "action_conditioned_dynamics_only",
            "excluded_downstream_stages": [
                "behavior_cloning",
                "policy_training_inside_world_model",
            ],
        }
        metadata_path = output_dir / "metadata.json"
        atomic_write_json(metadata_path, metadata)
        logger.recorder.register_artifact(
            step=written - 1,
            key="dataset/metadata",
            path=metadata_path,
        )
        print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
