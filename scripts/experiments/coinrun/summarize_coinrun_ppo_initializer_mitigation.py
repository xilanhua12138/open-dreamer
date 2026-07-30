#!/usr/bin/env python3
"""Validate and summarize targeted PPO initializer mitigation arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from dreamer.experiment_runtime import atomic_write_json


NEW_ARMS = (
    "orthogonal_gain1",
    "orthogonal_sqrt2_depth_scaled",
    "orthogonal_sqrt2_zero_last",
    "orthogonal_sqrt2_skipinit",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _find_anchor(
    comparison: dict[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    matches = [row for row in comparison.get("arms", []) if row.get("arm") == arm]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one CR-PPO-0003 anchor arm {arm!r}")
    return matches[0]


def summarize_mitigations(
    *,
    run_root: Path,
    anchor_comparison_path: Path,
    expected_env_steps: int,
) -> dict[str, Any]:
    anchors = _load_object(anchor_comparison_path)
    if anchors.get("experiment_id") != "CR-PPO-0003":
        raise ValueError("anchor comparison must come from CR-PPO-0003")
    if anchors.get("expected_env_steps") != expected_env_steps:
        raise ValueError("anchor and mitigation environment-step budgets differ")

    reference_anchor = _find_anchor(anchors, arm="reference")
    orthogonal_anchor = _find_anchor(anchors, arm="orthogonal_init")
    rows = [
        {
            "arm": "glorot_reference",
            "source_experiment": "CR-PPO-0003",
            "final_256": reference_anchor["final_256"],
        },
        {
            "arm": "orthogonal_sqrt2_anchor",
            "source_experiment": "CR-PPO-0003",
            "final_256": orthogonal_anchor["final_256"],
        },
    ]

    for arm in NEW_ARMS:
        final_path = (
            run_root
            / "arms"
            / arm
            / "final-validation"
            / f"env-steps-{expected_env_steps:09d}"
            / "metrics.json"
        )
        metrics = _load_object(final_path)
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
                    f"{arm} {field}: expected {value!r}, "
                    f"got {metrics.get(field)!r}"
                )
        rows.append(
            {
                "arm": arm,
                "source_experiment": "CR-PPO-0005",
                "final_256": {
                    "mean_return": metrics["mean_return"],
                    "success_rate": metrics["success_rate"],
                },
            }
        )

    reference_return = float(rows[0]["final_256"]["mean_return"])
    reference_success = float(rows[0]["final_256"]["success_rate"])
    orthogonal_return = float(rows[1]["final_256"]["mean_return"])
    orthogonal_success = float(rows[1]["final_256"]["success_rate"])
    return_gap = reference_return - orthogonal_return
    success_gap = reference_success - orthogonal_success
    if return_gap <= 0.0 or success_gap <= 0.0:
        raise ValueError("orthogonal anchor must be below the Glorot reference")

    for row in rows:
        mean_return = float(row["final_256"]["mean_return"])
        success_rate = float(row["final_256"]["success_rate"])
        row["delta_vs_glorot_reference"] = {
            "mean_return": mean_return - reference_return,
            "success_rate": success_rate - reference_success,
        }
        row["recovery_vs_orthogonal_sqrt2"] = {
            "mean_return": mean_return - orthogonal_return,
            "success_rate": success_rate - orthogonal_success,
            "mean_return_fraction_of_gap": (
                (mean_return - orthogonal_return) / return_gap
            ),
            "success_rate_fraction_of_gap": (
                (success_rate - orthogonal_success) / success_gap
            ),
        }
    return {
        "schema_version": "1.0",
        "experiment_id": "CR-PPO-0005",
        "expected_env_steps": expected_env_steps,
        "evaluation_identity": {
            "episodes": 256,
            "evaluation_distribution": "full_distribution",
            "num_levels": 0,
            "policy": "stochastic",
            "seed": 4242,
        },
        "arms": rows,
    }


def render_comparison(payload: dict[str, Any], path: Path) -> None:
    width = 1000
    row_height = 58
    top = 90
    left = 340
    right = 70
    rows = payload["arms"]
    height = top + row_height * len(rows) + 70
    image = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (32, 24),
        "CoinRun PPO initializer mitigation - final 256 episodes",
        fill="#f9fafb",
        font=font,
    )
    draw.text(
        (32, 48),
        "6,291,456 transitions; stochastic full distribution; seed 4242",
        fill="#9ca3af",
        font=font,
    )
    maximum = max(float(row["final_256"]["mean_return"]) for row in rows)
    plot_width = width - left - right
    colors = {
        "glorot_reference": "#22c55e",
        "orthogonal_sqrt2_anchor": "#ef4444",
        "orthogonal_gain1": "#60a5fa",
        "orthogonal_sqrt2_depth_scaled": "#a78bfa",
        "orthogonal_sqrt2_zero_last": "#f59e0b",
        "orthogonal_sqrt2_skipinit": "#14b8a6",
    }
    for index, row in enumerate(rows):
        y = top + index * row_height
        arm = str(row["arm"])
        mean_return = float(row["final_256"]["mean_return"])
        success = float(row["final_256"]["success_rate"])
        draw.text((32, y + 8), arm, fill="#e5e7eb", font=font)
        bar_width = int(plot_width * mean_return / max(maximum, 1e-12))
        draw.rounded_rectangle(
            (left, y, left + bar_width, y + 28),
            radius=5,
            fill=colors[arm],
        )
        draw.text(
            (left + 8, y + 8),
            f"return {mean_return:.3f}",
            fill="#111827",
            font=font,
        )
        draw.text(
            (left, y + 34),
            f"success {success:.2%}",
            fill="#d1d5db",
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--anchor-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--expected-env-steps", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize_mitigations(
        run_root=args.run_root,
        anchor_comparison_path=args.anchor_comparison,
        expected_env_steps=args.expected_env_steps,
    )
    atomic_write_json(args.output, payload)
    render_comparison(payload, args.plot)


if __name__ == "__main__":
    main()
