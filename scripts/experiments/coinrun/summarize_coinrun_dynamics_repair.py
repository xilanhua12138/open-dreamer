#!/usr/bin/env python3
"""Summarize the preregistered CoinRun dynamics reference repair."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dreamer.coinrun_dynamics_repair import assess_dynamics_repair


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="CR-DYN-0010")
    parser.add_argument(
        "--repair-arm",
        default="final-policy-medium-reference-repair",
    )
    parser.add_argument("--baseline-shortcut", type=Path, required=True)
    parser.add_argument("--baseline-actions", type=Path, required=True)
    parser.add_argument("--repair-shortcut", type=Path, required=True)
    parser.add_argument("--repair-diffusion", type=Path, required=True)
    parser.add_argument("--repair-actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_shortcut = load_json(args.baseline_shortcut)
    baseline_actions = load_json(args.baseline_actions)
    repair_shortcut = load_json(args.repair_shortcut)
    repair_diffusion = load_json(args.repair_diffusion)
    repair_actions = load_json(args.repair_actions)
    assessment = assess_dynamics_repair(
        shortcut_metrics=repair_shortcut,
        action_metrics=repair_actions,
        min_mean_frame_psnr_db=22.0,
        min_horizon_16_psnr_db=20.0,
        min_shuffled_advantage_db=0.25,
        min_noop_advantage_db=0.50,
    )

    payload = {
        "schema_version": "1.0",
        "experiment_id": args.experiment_id,
        "generated_at": datetime.now().astimezone().isoformat(),
        "baseline": {
            "experiment_id": "CR-DYN-0009",
            "arm": "final_only-medium",
            "shortcut": baseline_shortcut,
            "action_conditioning": baseline_actions,
        },
        "repair": {
            "arm": args.repair_arm,
            "shortcut": repair_shortcut,
            "diffusion_256_step": repair_diffusion,
            "action_conditioning": repair_actions,
        },
        "deltas": {
            "shortcut_mean_frame_psnr_db": (
                float(repair_shortcut["mean_frame_psnr_db"])
                - float(baseline_shortcut["mean_frame_psnr_db"])
            ),
            "shortcut_horizon_16_psnr_db": (
                float(repair_shortcut["psnr_by_horizon_db"]["16"])
                - float(baseline_shortcut["psnr_by_horizon_db"]["16"])
            ),
        },
        "preregistered_assessment": assessment,
        "visual_review_required": True,
        "claim_boundary": (
            "This exploratory bundle changes record length, sequence schedule, "
            "training budget and diffusion discretization together. It tests "
            "whether a reference-like recipe repairs the failed baseline; it "
            "does not identify a single causal factor."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
