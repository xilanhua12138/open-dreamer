#!/usr/bin/env python3
"""Generate small CoinRun ArrayRecord shards for tokenizer experiments."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter
from gym3 import types_np
from procgen import ProcgenGym3Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--records-per-shard", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.records, args.frames, args.envs, args.records_per_shard) <= 0:
        raise ValueError("all numeric arguments must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    env = ProcgenGym3Env(
        num=args.envs,
        env_name="coinrun",
        start_level=args.seed,
        num_levels=max(args.records, args.envs),
    )

    video_buffers: list[list[np.ndarray]] = [[] for _ in range(args.envs)]
    action_buffers: list[list[np.ndarray]] = [[] for _ in range(args.envs)]
    reward_buffers: list[list[np.ndarray]] = [[] for _ in range(args.envs)]
    writer: ArrayRecordWriter | None = None
    shard_index = -1
    written = 0

    def open_shard(index: int) -> ArrayRecordWriter:
        path = args.output_dir / f"shard-{index:05d}.array_record"
        return ArrayRecordWriter(str(path), "group_size:1")

    try:
        while written < args.records:
            rewards, observations, first = env.observe()
            actions = types_np.sample(env.ac_space, bshape=(env.num,), rng=rng)

            for index in range(args.envs):
                if bool(first[index]) and video_buffers[index]:
                    video_buffers[index].clear()
                    action_buffers[index].clear()
                    reward_buffers[index].clear()

                video_buffers[index].append(observations["rgb"][index].copy())
                action_buffers[index].append(np.asarray(actions[index]).copy())
                reward_buffers[index].append(np.asarray(rewards[index]).copy())

                if len(video_buffers[index]) != args.frames:
                    continue

                if written % args.records_per_shard == 0:
                    if writer is not None:
                        writer.close()
                    shard_index += 1
                    writer = open_shard(shard_index)

                record = {
                    "raw_video": np.stack(video_buffers[index]).astype(np.uint8).tobytes(),
                    "sequence_length": args.frames,
                    "actions": np.stack(action_buffers[index]),
                    "rewards": np.stack(reward_buffers[index]),
                }
                writer.write(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
                written += 1
                video_buffers[index].clear()
                action_buffers[index].clear()
                reward_buffers[index].clear()

                if written % 16 == 0 or written == args.records:
                    print(f"wrote {written}/{args.records} records", flush=True)
                if written == args.records:
                    break

            env.act(actions)
    finally:
        if writer is not None:
            writer.close()

    metadata = {
        "env": "coinrun",
        "records": written,
        "frames_per_record": args.frames,
        "seed": args.seed,
        "shards": shard_index + 1,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
