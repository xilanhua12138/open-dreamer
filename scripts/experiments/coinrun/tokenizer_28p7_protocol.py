#!/usr/bin/env python3
"""Frozen protocol for the CoinRun tokenizer 28.7M-label extension arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPERIMENT_ID = "CR-TOK-0004"
BASELINE_EXPERIMENT_ID = "CR-TOK-0003"
MAX_STEPS = 20_000
HELDOUT_EVALUATION_COMPLETED_UPDATES = (2_500, 5_000, 10_000, 20_000)
CHECKPOINT_STEPS = tuple(
    updates - 1 for updates in HELDOUT_EVALUATION_COMPLETED_UPDATES
)
RECOVERY_CHECKPOINT_STEPS = (14_999,)
CANDIDATE = {
    "name": "n28.7m",
    "directory": "n28p7m",
    "depth": 6,
    "d_model": 384,
    "expected_parameters": 25_564_032,
}
PERIODIC_VALIDATION = {
    "every_steps": 2_500,
    "seed": 4_242,
    "frames": 16,
    "batch_size": 8,
    "batches": 2,
}


def build_plan() -> dict:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "baseline_experiment_id": BASELINE_EXPERIMENT_ID,
        "training_mode": "fixed_optimizer_updates",
        "candidate": {
            **CANDIDATE,
            "max_steps": MAX_STEPS,
            "scaling_flops_budget": 0.0,
            "scaling_tokens_per_param": 0.0,
        },
        "heldout_evaluation_completed_updates": list(
            HELDOUT_EVALUATION_COMPLETED_UPDATES
        ),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "recovery_checkpoint_steps": list(RECOVERY_CHECKPOINT_STEPS),
        "periodic_validation": PERIODIC_VALIDATION,
        "wandb_required": True,
        "wandb_online_requires_authentication": True,
        "dynamics_authorized": False,
        "terminal_state": "awaiting_visual_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = build_plan()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
