#!/usr/bin/env python3
"""Rank a complete CoinRun tokenizer scale sweep by retained quality metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_candidate(candidates: list[dict]) -> dict:
    if len(candidates) < 2:
        raise ValueError("at least two tokenizer candidates are required")
    ordered = sorted(candidates, key=lambda candidate: candidate["parameters"])
    baseline = ordered[0]
    baseline_metrics = baseline["metrics"]

    evaluated = []
    for candidate in ordered:
        metrics = candidate["metrics"]
        improvements = {
            "clean_psnr_db": (
                metrics["ema_clean_psnr"]
                - baseline_metrics["ema_clean_psnr"]
            ),
            "edge_psnr_db": (
                metrics["ema_clean_edge_psnr"]
                - baseline_metrics["ema_clean_edge_psnr"]
            ),
            "temporal_change_psnr_db": (
                metrics["ema_clean_temporal_change_psnr"]
                - baseline_metrics["ema_clean_temporal_change_psnr"]
            ),
        }
        quality_score = (
            candidate["metrics"]["ema_clean_psnr"]
            + candidate["metrics"]["ema_clean_edge_psnr"]
            + candidate["metrics"]["ema_clean_temporal_change_psnr"]
        )
        evaluated.append(
            {
                **candidate,
                "improvements_vs_smallest_db": improvements,
                "quality_score": quality_score,
            }
        )

    ranking = sorted(
        evaluated,
        key=lambda candidate: (
            -candidate["quality_score"],
            candidate["parameters"],
        ),
    )
    return {
        "selection_basis": (
            "highest_unweighted_sum_of_clean_edge_and_temporal_change_psnr"
        ),
        "selected": ranking[0],
        "baseline": evaluated[0],
        "ranking": ranking,
        "candidates": evaluated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        action="append",
        required=True,
        help="Candidate mapping in NAME=PATH form.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    probe_by_name = {
        candidate["name"]: candidate for candidate in probe["candidates"]
    }
    candidates = []
    for mapping in args.metrics:
        name, separator, raw_path = mapping.partition("=")
        if not separator or name not in probe_by_name:
            raise ValueError(f"invalid candidate metrics mapping: {mapping}")
        evaluation = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        candidates.append(
            {
                "name": name,
                "parameters": probe_by_name[name]["parameters"],
                "training_flops_budget": probe_by_name[name][
                    "training_flops_budget"
                ],
                "metrics_path": raw_path,
                "metrics": evaluation["metrics"],
            }
        )

    result = select_candidate(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
