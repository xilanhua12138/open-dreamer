"""Deterministic selection rules for sequential dynamics experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dreamer.coinrun_checkpoint_mixture import MIXTURE_SHARDS_BY_STAGE


def select_checkpoint_mixture(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select by preregistered lexicographic held-out rollout quality."""

    expected = set(MIXTURE_SHARDS_BY_STAGE)
    row_by_name = {str(row.get("mixture")): row for row in rows}
    if set(row_by_name) != expected:
        raise ValueError(
            f"mixture rows must be exactly {sorted(expected)}, "
            f"got {sorted(row_by_name)}"
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["mean_frame_psnr_db"]),
            -float(row["mean_ssim"]),
            -float(row["psnr_by_horizon_db"]["16"]),
            str(row["mixture"]),
        ),
    )
    winner = ranked[0]
    return {
        "schema_version": "1.0",
        "experiment_id": "CR-DYN-0008",
        "selection_rule": (
            "descending mean_frame_psnr_db, then descending mean_ssim, "
            "then descending horizon-16 PSNR, then mixture name"
        ),
        "selected_mixture": winner["mixture"],
        "ranked_mixtures": [row["mixture"] for row in ranked],
        "selected_metrics": {
            "mean_frame_psnr_db": winner["mean_frame_psnr_db"],
            "mean_ssim": winner["mean_ssim"],
            "horizon_16_psnr_db": winner["psnr_by_horizon_db"]["16"],
        },
    }
