"""Cross-split validation for CoinRun dynamics datasets."""

from __future__ import annotations

from typing import Any


def _level_interval(metadata: dict[str, Any]) -> tuple[int, int]:
    start = int(metadata.get("start_level", -1))
    count = int(metadata.get("num_levels", 0))
    if start < 0 or count <= 0:
        raise ValueError(
            f"invalid level interval start_level={start}, num_levels={count}"
        )
    return start, start + count


def validate_dataset_pair(
    *,
    train: dict[str, Any],
    evaluation: dict[str, Any],
) -> list[str]:
    errors = []
    train_start, train_stop = _level_interval(train)
    eval_start, eval_stop = _level_interval(evaluation)
    if max(train_start, eval_start) < min(train_stop, eval_stop):
        errors.append("train/eval level ranges overlap")
    train_policy = train.get("policy")
    eval_policy = evaluation.get("policy")
    if not isinstance(train_policy, dict) or not isinstance(eval_policy, dict):
        errors.append("train/eval policy metadata is missing")
    elif train_policy.get("checkpoint_sha256") != eval_policy.get(
        "checkpoint_sha256"
    ):
        errors.append("train/eval policy checkpoint SHA256 differs")
    if train.get("tree_sha256") == evaluation.get("tree_sha256"):
        errors.append("train/eval dataset tree SHA256 is identical")
    for key in ("env", "action_policy", "action_space", "distribution_mode"):
        if train.get(key) != evaluation.get(key):
            errors.append(f"train/eval {key} differs")
    return errors
