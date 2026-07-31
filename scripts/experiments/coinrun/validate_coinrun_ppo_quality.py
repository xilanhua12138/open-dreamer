#!/usr/bin/env python3
"""Fail closed unless a frozen CoinRun PPO policy passes its declared gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dreamer.experiment_runtime import atomic_write_json


def _require_number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    return float(value)


def validate_policy_quality(
    metrics: dict[str, Any],
    *,
    expected_env_steps: int,
    expected_episodes: int,
    min_mean_return: float,
    min_success_rate: float,
) -> dict[str, Any]:
    """Validate evaluation identity and both conjunctive quality thresholds."""

    expected_identity = {
        "completed_env_steps": expected_env_steps,
        "episodes": expected_episodes,
        "evaluation_distribution": "full_distribution",
        "num_levels": 0,
        "policy": "stochastic",
    }
    for field, expected in expected_identity.items():
        observed = metrics.get(field)
        if observed != expected:
            raise ValueError(
                f"{field} mismatch: expected {expected!r}, got {observed!r}"
            )

    mean_return = _require_number(metrics, "mean_return")
    success_rate = _require_number(metrics, "success_rate")
    if mean_return < min_mean_return:
        raise ValueError(
            f"mean_return {mean_return} is below required {min_mean_return}"
        )
    if success_rate < min_success_rate:
        raise ValueError(
            f"success_rate {success_rate} is below required {min_success_rate}"
        )

    return {
        "schema_version": "1.0",
        "decision": "accepted_for_collection",
        "completed_env_steps": expected_env_steps,
        "episodes": expected_episodes,
        "evaluation_distribution": "full_distribution",
        "num_levels": 0,
        "policy": "stochastic",
        "mean_return": mean_return,
        "success_rate": success_rate,
        "thresholds": {
            "min_mean_return": min_mean_return,
            "min_success_rate": min_success_rate,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-env-steps", type=int, default=25_165_824)
    parser.add_argument("--expected-episodes", type=int, default=512)
    parser.add_argument("--min-mean-return", type=float, default=8.0)
    parser.add_argument("--min-success-rate", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError("metrics must contain one JSON object")
    result = validate_policy_quality(
        metrics,
        expected_env_steps=args.expected_env_steps,
        expected_episodes=args.expected_episodes,
        min_mean_return=args.min_mean_return,
        min_success_rate=args.min_success_rate,
    )
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
