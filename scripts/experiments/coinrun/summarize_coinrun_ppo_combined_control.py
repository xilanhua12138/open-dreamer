#!/usr/bin/env python3
"""Summarize the historical-checkpoint and all-old combined controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dreamer.experiment_runtime import atomic_write_json


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def summarize_combined_control(
    *,
    reference_path: Path,
    historical_path: Path,
    combined_path: Path,
    expected_env_steps: int,
) -> dict[str, Any]:
    rows = []
    for name, path in (
        ("reference", reference_path),
        ("historical_old_checkpoint", historical_path),
        ("fresh_all_old_combined", combined_path),
    ):
        metrics = _load(path)
        expected = {
            "completed_env_steps": expected_env_steps,
            "episodes": 256,
            "evaluation_distribution": "full_distribution",
            "num_levels": 0,
            "policy": "stochastic",
            "seed": 4242,
        }
        for field, value in expected.items():
            if metrics.get(field) != value:
                raise ValueError(
                    f"{name} {field}: expected {value!r}, "
                    f"got {metrics.get(field)!r}"
                )
        rows.append(
            {
                "arm": name,
                "mean_return": float(metrics["mean_return"]),
                "success_rate": float(metrics["success_rate"]),
                "checkpoint_sha256": metrics.get("checkpoint_sha256"),
            }
        )

    reference = rows[0]
    for row in rows:
        row["delta_vs_reference"] = {
            "mean_return": row["mean_return"] - reference["mean_return"],
            "success_rate": row["success_rate"] - reference["success_rate"],
        }
        row["independent_collapse"] = (
            row["delta_vs_reference"]["mean_return"] <= -1.0
            or row["delta_vs_reference"]["success_rate"] <= -0.10
        )
    return {
        "schema_version": "1.0",
        "experiment_id": "CR-PPO-0004",
        "expected_env_steps": expected_env_steps,
        "arms": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-env-steps", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atomic_write_json(
        args.output,
        summarize_combined_control(
            reference_path=args.reference,
            historical_path=args.historical,
            combined_path=args.combined,
            expected_env_steps=args.expected_env_steps,
        ),
    )


if __name__ == "__main__":
    main()
