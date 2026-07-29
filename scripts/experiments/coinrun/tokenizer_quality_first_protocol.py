#!/usr/bin/env python3
"""Single source of truth for the CoinRun tokenizer fixed-20k sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MAX_STEPS = 20_000
EVALUATION_COMPLETED_UPDATES = (2_500, 5_000, 10_000, 20_000)
CHECKPOINT_STEPS = tuple(updates - 1 for updates in EVALUATION_COMPLETED_UPDATES)
RECOVERY_CHECKPOINT_STEPS = (14_999,)
CANDIDATES = (
    {"name": "n0.17m", "directory": "n0p17m", "depth": 1, "d_model": 64},
    {"name": "n1.1m", "directory": "n1p1m", "depth": 2, "d_model": 128},
    {"name": "n3.7m", "directory": "n3p7m", "depth": 3, "d_model": 192},
    {"name": "n8.6m", "directory": "n8p6m", "depth": 4, "d_model": 256},
    {"name": "n16.6m", "directory": "n16p6m", "depth": 5, "d_model": 320},
)


def build_plan() -> dict:
    candidates = []
    for candidate in CANDIDATES:
        candidates.append(
            {
                **candidate,
                "max_steps": MAX_STEPS,
                "scaling_flops_budget": 0.0,
                "scaling_tokens_per_param": 0.0,
            }
        )
    return {
        "schema_version": "1.0",
        "training_mode": "fixed_optimizer_updates",
        "candidates": candidates,
        "evaluation_completed_updates": list(EVALUATION_COMPLETED_UPDATES),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "recovery_checkpoint_steps": list(RECOVERY_CHECKPOINT_STEPS),
        "dynamics_authorized": False,
        "terminal_state": "awaiting_visual_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    args = parser.parse_args()

    plan = build_plan()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.format == "tsv":
        for candidate in plan["candidates"]:
            print(
                "\t".join(
                    str(candidate[field])
                    for field in ("name", "directory", "depth", "d_model", "max_steps")
                )
            )
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
