#!/usr/bin/env python3
"""Validate the four PPO-stage training corpora and fixed evaluation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreamer.coinrun_stage_corpus import (
    PPO_STAGE_ENV_STEPS,
    validate_stage_corpus,
)
from dreamer.experiment_runtime import atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_paths = {
        stage: args.data_root / "train" / stage / "metadata.json"
        for stage in PPO_STAGE_ENV_STEPS
    }
    eval_path = args.data_root / "eval-final-policy" / "metadata.json"
    training = {
        stage: json.loads(path.read_text(encoding="utf-8"))
        for stage, path in metadata_paths.items()
    }
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    errors = validate_stage_corpus(
        training=training,
        evaluation=evaluation,
    )
    payload = {
        "schema_version": "1.0",
        "valid": not errors,
        "errors": errors,
        "training": {
            stage: {
                "metadata": str(path),
                "metadata_sha256": sha256_file(path),
                "tree_sha256": training[stage]["tree_sha256"],
                "checkpoint_completed_env_steps": training[stage]["policy"][
                    "checkpoint_completed_env_steps"
                ],
            }
            for stage, path in metadata_paths.items()
        },
        "evaluation": {
            "metadata": str(eval_path),
            "metadata_sha256": sha256_file(eval_path),
            "tree_sha256": evaluation["tree_sha256"],
            "checkpoint_completed_env_steps": evaluation["policy"][
                "checkpoint_completed_env_steps"
            ],
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
