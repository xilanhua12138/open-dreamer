#!/usr/bin/env python3
"""Assemble descriptive PPO-stage and dynamics-scale comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreamer.experiment_runtime import atomic_write_json


STAGES = {
    "ppo01p05m": 1_048_576,
    "ppo06p29m": 6_291_456,
    "ppo12p58m": 12_582_912,
    "ppo25p17m": 25_165_824,
}
SCALES = {
    "tiny": 155_840,
    "small": 545_920,
    "medium": 3_931_392,
    "large": 12_902_784,
}


def load_metrics(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return {
        "metrics_path": str(path),
        "mean_video_psnr_db": metrics["mean_video_psnr_db"],
        "mean_frame_psnr_db": metrics["mean_frame_psnr_db"],
        "mean_ssim": metrics["mean_ssim"],
        "psnr_by_horizon_db": metrics["psnr_by_horizon_db"],
        "ssim_by_horizon": metrics["ssim_by_horizon"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage-output", type=Path, required=True)
    parser.add_argument("--scale-output", type=Path, required=True)
    args = parser.parse_args()

    stage_rows = []
    for stage, env_steps in STAGES.items():
        path = (
            args.run_root
            / "stage-ablation"
            / "eval"
            / f"{stage}-medium"
            / "metrics.json"
        )
        stage_rows.append(
            {
                "stage": stage,
                "ppo_checkpoint_completed_env_steps": env_steps,
                "dynamics_scale": "medium",
                "dynamics_parameters": SCALES["medium"],
                **load_metrics(path),
            }
        )

    scale_rows = []
    for scale, parameters in SCALES.items():
        if scale == "medium":
            path = (
                args.run_root
                / "stage-ablation"
                / "eval"
                / "ppo25p17m-medium"
                / "metrics.json"
            )
        else:
            path = (
                args.run_root
                / "scale-ablation"
                / "eval"
                / f"ppo25p17m-{scale}"
                / "metrics.json"
            )
        scale_rows.append(
            {
                "scale": scale,
                "dynamics_parameters": parameters,
                "ppo_checkpoint_completed_env_steps": STAGES["ppo25p17m"],
                **load_metrics(path),
            }
        )

    shared = {
        "schema_version": "1.0",
        "tokenizer": "n16.6m EMA at 20,000 updates",
        "tokenizer_parameters": 14_912_320,
        "training_updates_per_arm": 20_000,
        "evaluation": {
            "policy_checkpoint_completed_env_steps": 25_165_824,
            "level_range": [10_000, 10_500],
            "seed": 4_242,
            "videos": 32,
            "context_frames": 16,
            "predicted_frames": 16,
            "metric_target": "tokenizer_decoded_ground_truth",
        },
    }
    atomic_write_json(
        args.stage_output,
        {
            **shared,
            "experiment_id": "CR-DYN-0006",
            "controlled_variable": "ppo_collection_checkpoint_stage",
            "rows": stage_rows,
        },
    )
    atomic_write_json(
        args.scale_output,
        {
            **shared,
            "experiment_id": "CR-DYN-0007",
            "controlled_variable": "dynamics_parameter_scale",
            "rows": scale_rows,
        },
    )


if __name__ == "__main__":
    main()
