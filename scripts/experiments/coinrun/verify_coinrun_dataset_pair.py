#!/usr/bin/env python3
"""Verify that CoinRun dynamics train/eval datasets are comparable and disjoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreamer.coinrun_dataset_pair import validate_dataset_pair
from dreamer.experiment_runtime import atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--eval-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_metadata_path = args.train / "metadata.json"
    eval_metadata_path = args.eval / "metadata.json"
    train_metadata = json.loads(
        train_metadata_path.read_text(encoding="utf-8")
    )
    eval_metadata = json.loads(
        eval_metadata_path.read_text(encoding="utf-8")
    )
    train_audit = json.loads(args.train_audit.read_text(encoding="utf-8"))
    eval_audit = json.loads(args.eval_audit.read_text(encoding="utf-8"))
    errors = validate_dataset_pair(
        train=train_metadata,
        evaluation=eval_metadata,
    )
    if train_audit.get("valid") is not True:
        errors.append("train dataset individual audit is not valid")
    if eval_audit.get("valid") is not True:
        errors.append("eval dataset individual audit is not valid")
    payload = {
        "schema_version": "1.0",
        "valid": not errors,
        "errors": errors,
        "train": {
            "metadata": str(train_metadata_path),
            "metadata_sha256": sha256_file(train_metadata_path),
            "audit": str(args.train_audit),
            "audit_sha256": sha256_file(args.train_audit),
            "level_range": [
                int(train_metadata["start_level"]),
                int(train_metadata["start_level"])
                + int(train_metadata["num_levels"]),
            ],
            "tree_sha256": train_metadata["tree_sha256"],
        },
        "eval": {
            "metadata": str(eval_metadata_path),
            "metadata_sha256": sha256_file(eval_metadata_path),
            "audit": str(args.eval_audit),
            "audit_sha256": sha256_file(args.eval_audit),
            "level_range": [
                int(eval_metadata["start_level"]),
                int(eval_metadata["start_level"])
                + int(eval_metadata["num_levels"]),
            ],
            "tree_sha256": eval_metadata["tree_sha256"],
        },
        "shared_policy_checkpoint_sha256": train_metadata["policy"][
            "checkpoint_sha256"
        ],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
