#!/usr/bin/env python3
"""Summarize checkpoint-mixture and selected-mixture scale experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dreamer.coinrun_checkpoint_mixture import (
    MIXTURE_SHARDS_BY_STAGE,
    records_by_stage,
)
from dreamer.coinrun_dynamics_selection import select_checkpoint_mixture
from dreamer.experiment_runtime import atomic_write_json


SCALES = {
    "tiny": 155_840,
    "small": 545_920,
    "medium": 3_931_392,
    "large": 12_902_784,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mixture-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--scale-output", type=Path)
    return parser.parse_args()


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
    args = parse_args()
    mixture_rows = []
    for mixture_name in MIXTURE_SHARDS_BY_STAGE:
        metrics_path = (
            args.run_root
            / "mixture-ablation"
            / "eval"
            / f"{mixture_name}-medium"
            / "metrics.json"
        )
        mixture_rows.append(
            {
                "mixture": mixture_name,
                "records_by_stage": records_by_stage(mixture_name),
                **load_metrics(metrics_path),
            }
        )
    selection = select_checkpoint_mixture(mixture_rows)
    atomic_write_json(
        args.mixture_output,
        {
            "schema_version": "1.0",
            "experiment_id": "CR-DYN-0008",
            "controlled_variable": "PPO checkpoint mixture ratio",
            "rows": mixture_rows,
        },
    )
    atomic_write_json(args.selection_output, selection)

    if args.scale_output is not None:
        selected = str(selection["selected_mixture"])
        scale_rows = []
        for scale, parameters in SCALES.items():
            metrics_path = (
                args.run_root
                / "scale-ablation"
                / "eval"
                / f"{selected}-{scale}"
                / "metrics.json"
            )
            scale_rows.append(
                {
                    "mixture": selected,
                    "scale": scale,
                    "parameters": parameters,
                    **load_metrics(metrics_path),
                }
            )
        atomic_write_json(
            args.scale_output,
            {
                "schema_version": "1.0",
                "experiment_id": "CR-DYN-0009",
                "selected_by": "CR-DYN-0008",
                "selected_mixture": selected,
                "controlled_variable": "dynamics parameter scale",
                "rows": scale_rows,
            },
        )

    print(
        json.dumps(
            {
                "mixture_output": str(args.mixture_output),
                "selection_output": str(args.selection_output),
                "scale_output": (
                    str(args.scale_output)
                    if args.scale_output is not None
                    else None
                ),
                "selected_mixture": selection["selected_mixture"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
