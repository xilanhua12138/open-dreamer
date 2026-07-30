#!/usr/bin/env python3
"""Select the best completed CoinRun dynamics checkpoint by held-out PSNR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted((args.experiment / "eval").glob("*/metrics.json")):
        name = metrics_path.parent.name
        checkpoint = args.experiment / "runs" / name / "checkpoints"
        if not checkpoint.is_dir():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": name,
                "checkpoint": str(checkpoint),
                "mean_frame_psnr_db": metrics["mean_frame_psnr_db"],
                "mean_ssim": metrics["mean_ssim"],
            }
        )
    if not rows:
        raise FileNotFoundError(f"No completed evaluations under {args.experiment / 'eval'}")

    rows.sort(key=lambda row: (row["mean_frame_psnr_db"], row["mean_ssim"]), reverse=True)
    payload = {"best": rows[0], "candidates": rows}
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
