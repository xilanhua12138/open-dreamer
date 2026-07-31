#!/usr/bin/env python3
"""Verify that a CoinRun corpus can support repaired reward-biased windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreamer.coinrun_dynamics_repair import validate_reward_windowing
from dreamer.experiment_runtime import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-window", type=int, default=64)
    parser.add_argument("--long-window", type=int, default=128)
    parser.add_argument("--p-include-reward", type=float, default=0.5)
    args = parser.parse_args()

    metadata_path = args.dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit = validate_reward_windowing(
        record_frames=int(metadata["frames_per_record"]),
        short_window=args.short_window,
        long_window=args.long_window,
        p_include_reward=args.p_include_reward,
    )
    payload = {
        "schema_version": "1.0",
        "dataset": str(args.dataset.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        **audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
