#!/usr/bin/env python3
"""Collect every fixed-step tokenizer milestone into one learning-curve manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.experiments.coinrun.tokenizer_quality_first_protocol import (
    CANDIDATES,
    EVALUATION_COMPLETED_UPDATES,
)


def summarize(run_root: Path) -> dict:
    rows = []
    for candidate in CANDIDATES:
        run_dir = run_root / "runs" / candidate["directory"]
        for completed_updates in EVALUATION_COMPLETED_UPDATES:
            metrics_path = (
                run_dir
                / "milestones"
                / f"updates-{completed_updates:05d}"
                / "heldout_eval.json"
            )
            if not metrics_path.is_file():
                raise FileNotFoundError(f"missing milestone metrics: {metrics_path}")
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if payload.get("completed_updates") != completed_updates:
                raise ValueError(
                    f"{metrics_path}: expected completed_updates={completed_updates}, "
                    f"got {payload.get('completed_updates')}"
                )
            rows.append(
                {
                    "name": candidate["name"],
                    "directory": candidate["directory"],
                    "completed_updates": completed_updates,
                    "checkpoint_step": payload["checkpoint_step"],
                    "metrics_path": str(metrics_path),
                    "metrics": payload["metrics"],
                }
            )
    return {
        "schema_version": "1.0",
        "training_mode": "fixed_optimizer_updates",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = summarize(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
