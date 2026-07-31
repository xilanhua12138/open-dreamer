#!/usr/bin/env python3
"""Verify PPO source pools, fixed eval data and materialized mixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreamer.coinrun_checkpoint_mixture import (
    MIXTURE_SHARDS_BY_STAGE,
    PPO_STAGE_ENV_STEPS,
    validate_mixture_metadata,
)
from dreamer.coinrun_stage_corpus import validate_stage_corpus
from dreamer.experiment_runtime import atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    source_paths = {
        stage: args.data_root
        / "source-pools"
        / "train"
        / stage
        / "metadata.json"
        for stage in PPO_STAGE_ENV_STEPS
    }
    eval_path = args.data_root / "eval-final-policy" / "metadata.json"
    mixture_paths = {
        name: args.data_root / "mixtures" / name / "metadata.json"
        for name in MIXTURE_SHARDS_BY_STAGE
    }
    source_training = {
        stage: _load(path) for stage, path in source_paths.items()
    }
    evaluation = _load(eval_path)
    mixtures = {
        name: _load(path) for name, path in mixture_paths.items()
    }

    errors = validate_stage_corpus(
        training=source_training,
        evaluation=evaluation,
        expected_experiment_id="CR-DYN-0008",
    )
    errors.extend(
        validate_mixture_metadata(
            mixtures=mixtures,
            source_training=source_training,
        )
    )
    audit_paths = {
        name: args.data_root / "audits" / "mixtures" / f"{name}.json"
        for name in MIXTURE_SHARDS_BY_STAGE
    }
    for name, path in audit_paths.items():
        audit = _load(path)
        if audit.get("valid") is not True:
            errors.append(f"{name}: individual ArrayRecord audit is not valid")
        if audit.get("tree_sha256") != mixtures[name].get("tree_sha256"):
            errors.append(f"{name}: audit tree SHA256 differs from metadata")

    payload = {
        "schema_version": "1.0",
        "experiment_id": "CR-DYN-0008",
        "valid": not errors,
        "errors": errors,
        "source_training": {
            stage: {
                "metadata": str(path),
                "metadata_sha256": sha256_file(path),
                "tree_sha256": source_training[stage]["tree_sha256"],
                "checkpoint_completed_env_steps": source_training[stage][
                    "policy"
                ]["checkpoint_completed_env_steps"],
            }
            for stage, path in source_paths.items()
        },
        "evaluation": {
            "metadata": str(eval_path),
            "metadata_sha256": sha256_file(eval_path),
            "tree_sha256": evaluation["tree_sha256"],
        },
        "mixtures": {
            name: {
                "metadata": str(path),
                "metadata_sha256": sha256_file(path),
                "tree_sha256": mixtures[name]["tree_sha256"],
                "records_by_stage": mixtures[name]["mixture"][
                    "records_by_stage"
                ],
                "audit": str(audit_paths[name]),
                "audit_sha256": sha256_file(audit_paths[name]),
            }
            for name, path in mixture_paths.items()
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
