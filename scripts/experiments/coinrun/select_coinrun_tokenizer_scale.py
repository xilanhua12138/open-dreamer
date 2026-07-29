#!/usr/bin/env python3
"""Select the smallest CoinRun tokenizer that clears preregistered gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ABSOLUTE_CLEAN_PSNR_DB = 26.0
MIN_CLEAN_IMPROVEMENT_DB = 1.0
MIN_EDGE_IMPROVEMENT_DB = 0.75
MIN_TEMPORAL_IMPROVEMENT_DB = 0.75


def select_candidate(candidates: list[dict]) -> dict:
    if len(candidates) < 2:
        raise ValueError("at least two tokenizer candidates are required")
    ordered = sorted(candidates, key=lambda candidate: candidate["parameters"])
    baseline = ordered[0]
    baseline_metrics = baseline["metrics"]

    evaluated = []
    for candidate in ordered:
        metrics = candidate["metrics"]
        gates = {
            "absolute_clean_psnr": (
                metrics["ema_clean_psnr"] >= ABSOLUTE_CLEAN_PSNR_DB
            ),
            "clean_improvement": (
                metrics["ema_clean_psnr"]
                - baseline_metrics["ema_clean_psnr"]
                >= MIN_CLEAN_IMPROVEMENT_DB
            ),
            "edge_improvement": (
                metrics["ema_clean_edge_psnr"]
                - baseline_metrics["ema_clean_edge_psnr"]
                >= MIN_EDGE_IMPROVEMENT_DB
            ),
            "temporal_improvement": (
                metrics["ema_clean_temporal_change_psnr"]
                - baseline_metrics["ema_clean_temporal_change_psnr"]
                >= MIN_TEMPORAL_IMPROVEMENT_DB
            ),
        }
        evaluated.append({**candidate, "gates": gates, "eligible": all(gates.values())})

    eligible = [candidate for candidate in evaluated if candidate["eligible"]]
    best_quality = max(
        evaluated,
        key=lambda candidate: (
            candidate["metrics"]["ema_clean_psnr"]
            + candidate["metrics"]["ema_clean_edge_psnr"]
            + candidate["metrics"]["ema_clean_temporal_change_psnr"]
        ),
    )
    return {
        "quality_gate_passed": bool(eligible),
        "selected": eligible[0] if eligible else None,
        "best_quality": best_quality,
        "baseline": evaluated[0],
        "thresholds": {
            "absolute_clean_psnr_db": ABSOLUTE_CLEAN_PSNR_DB,
            "min_clean_improvement_db": MIN_CLEAN_IMPROVEMENT_DB,
            "min_edge_improvement_db": MIN_EDGE_IMPROVEMENT_DB,
            "min_temporal_improvement_db": MIN_TEMPORAL_IMPROVEMENT_DB,
        },
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
