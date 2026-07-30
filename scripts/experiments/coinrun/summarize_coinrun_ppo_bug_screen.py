#!/usr/bin/env python3
"""Validate and summarize the CR-PPO-0003 one-factor screening arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dreamer.experiment_runtime import atomic_write_json


ARMS = (
    "reference",
    "levels500",
    "reward_gamma0999",
    "batch_advantage",
    "orthogonal_init",
)


def _load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def summarize_arms(
    *,
    run_root: Path,
    expected_env_steps: int,
) -> dict[str, Any]:
    rows = []
    for arm in ARMS:
        arm_root = run_root / "arms" / arm
        periodic_path = (
            arm_root
            / "validation"
            / f"env-steps-{expected_env_steps:09d}"
            / "metrics.json"
        )
        final_path = (
            arm_root
            / "final-validation"
            / f"env-steps-{expected_env_steps:09d}"
            / "metrics.json"
        )
        periodic = _load_metrics(periodic_path)
        final = _load_metrics(final_path)
        for name, metrics, episodes in (
            ("periodic", periodic, 128),
            ("final", final, 256),
        ):
            expected = {
                "completed_env_steps": expected_env_steps,
                "episodes": episodes,
                "evaluation_distribution": "full_distribution",
                "num_levels": 0,
                "policy": "stochastic",
                "seed": 4242,
            }
            for field, value in expected.items():
                if metrics.get(field) != value:
                    raise ValueError(
                        f"{arm}/{name} {field}: expected {value!r}, "
                        f"got {metrics.get(field)!r}"
                    )
        rows.append(
            {
                "arm": arm,
                "periodic_128": {
                    "mean_return": periodic["mean_return"],
                    "success_rate": periodic["success_rate"],
                },
                "final_256": {
                    "mean_return": final["mean_return"],
                    "success_rate": final["success_rate"],
                },
            }
        )

    reference = rows[0]
    reference_return = float(reference["final_256"]["mean_return"])
    reference_success = float(reference["final_256"]["success_rate"])
    for row in rows:
        row["delta_vs_reference_final_256"] = {
            "mean_return": (
                float(row["final_256"]["mean_return"]) - reference_return
            ),
            "success_rate": (
                float(row["final_256"]["success_rate"]) - reference_success
            ),
        }
    return {
        "schema_version": "1.0",
        "experiment_id": "CR-PPO-0003",
        "expected_env_steps": expected_env_steps,
        "arms": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-env-steps", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atomic_write_json(
        args.output,
        summarize_arms(
            run_root=args.run_root,
            expected_env_steps=args.expected_env_steps,
        ),
    )


if __name__ == "__main__":
    main()
