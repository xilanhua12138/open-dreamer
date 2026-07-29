#!/usr/bin/env python3
"""Compare one tokenizer extension arm against a byte-aligned held-out baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = (
    "dataset",
    "seed",
    "batch_size",
    "frames",
    "batches",
    "num_clips",
    "completed_updates",
)
METRIC_FIELDS = {
    "ema_clean": "ema_clean_psnr",
    "ema_masked": "ema_masked_psnr",
    "ema_clean_edge": "ema_clean_edge_psnr",
    "ema_clean_temporal_change": "ema_clean_temporal_change_psnr",
}
SCORE_FIELDS = (
    "ema_clean",
    "ema_clean_edge",
    "ema_clean_temporal_change",
)


def _identity(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in IDENTITY_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"held-out metrics missing identity fields: {missing}")
    return {field: payload[field] for field in IDENTITY_FIELDS}


def _metrics(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("metrics")
    if not isinstance(raw, dict):
        raise ValueError("held-out metrics must contain a metrics object")
    missing = [field for field in METRIC_FIELDS.values() if field not in raw]
    if missing:
        raise ValueError(f"held-out metrics missing primary metrics: {missing}")
    return {
        name: float(raw[field])
        for name, field in METRIC_FIELDS.items()
    }


def compare_metrics(
    *,
    baseline_name: str,
    baseline: dict[str, Any],
    candidate_name: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_identity = _identity(baseline)
    candidate_identity = _identity(candidate)
    if candidate_identity != baseline_identity:
        raise ValueError(
            "evaluation identity differs between baseline and candidate: "
            f"{baseline_identity!r} != {candidate_identity!r}"
        )
    baseline_metrics = _metrics(baseline)
    candidate_metrics = _metrics(candidate)
    deltas = {
        name: candidate_metrics[name] - baseline_metrics[name]
        for name in METRIC_FIELDS
    }
    baseline_score = sum(baseline_metrics[name] for name in SCORE_FIELDS)
    candidate_score = sum(candidate_metrics[name] for name in SCORE_FIELDS)
    return {
        "schema_version": "1.0",
        "comparison_type": "descriptive_fixed_20k_extension",
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "evaluation_identity": baseline_identity,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_minus_baseline_psnr_db": deltas,
        "descriptive_score_definition": list(SCORE_FIELDS),
        "baseline_descriptive_score": baseline_score,
        "candidate_descriptive_score": candidate_score,
        "descriptive_score_delta": candidate_score - baseline_score,
        "quality_claim": "awaiting_visual_review",
        "dynamics_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_metrics.read_text(encoding="utf-8"))
    payload = compare_metrics(
        baseline_name=args.baseline_name,
        baseline=baseline,
        candidate_name=args.candidate_name,
        candidate=candidate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
