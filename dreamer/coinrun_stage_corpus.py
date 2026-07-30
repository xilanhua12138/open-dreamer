"""Validation rules for the PPO-stage CoinRun dynamics corpus."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dreamer.coinrun import COINRUN_ACTION_DIM, COINRUN_NOOP_ACTION


PPO_STAGE_ENV_STEPS: dict[str, int] = {
    "ppo01p05m": 1_048_576,
    "ppo06p29m": 6_291_456,
    "ppo12p58m": 12_582_912,
    "ppo25p17m": 25_165_824,
}


def _level_range(metadata: Mapping[str, Any]) -> tuple[int, int]:
    start = int(metadata["start_level"])
    return start, start + int(metadata["num_levels"])


def _validate_common(
    *,
    label: str,
    metadata: Mapping[str, Any],
    expected_records: int,
    expected_seed: int,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "experiment_id": "CR-DYN-0006",
        "env": "coinrun",
        "records": expected_records,
        "frames_per_record": 64,
        "seed": expected_seed,
        "distribution_mode": "easy",
        "action_policy": "ppo",
        "transition_alignment": (
            "raw_video[t]=observation_t, actions[t]=action_t, "
            "rewards[t]=reward observed after action_t"
        ),
        "consumer": "action_conditioned_dynamics_only",
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            errors.append(
                f"{label}: {key}={metadata.get(key)!r}, "
                f"expected {expected_value!r}"
            )

    action_space = metadata.get("action_space", {})
    if action_space.get("categorical_action_dim") != COINRUN_ACTION_DIM:
        errors.append(f"{label}: invalid categorical action dimension")
    if action_space.get("categorical_noop_action") != COINRUN_NOOP_ACTION:
        errors.append(f"{label}: invalid categorical no-op action")

    policy = metadata.get("policy", {})
    if policy.get("temperature") != 1.0:
        errors.append(f"{label}: policy temperature must be 1.0")
    if policy.get("exploration_epsilon") != 0.05:
        errors.append(f"{label}: policy exploration_epsilon must be 0.05")
    if policy.get("deterministic") is not False:
        errors.append(f"{label}: policy collection must be stochastic")
    if not metadata.get("tree_sha256"):
        errors.append(f"{label}: missing dataset tree_sha256")
    return errors


def validate_stage_corpus(
    *,
    training: Mapping[str, Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> list[str]:
    """Validate a four-stage training corpus and one fixed final-policy eval set."""

    errors: list[str] = []
    expected_stages = set(PPO_STAGE_ENV_STEPS)
    if set(training) != expected_stages:
        errors.append(
            "training stage keys must be exactly "
            f"{sorted(expected_stages)}, got {sorted(training)}"
        )
        return errors

    tree_hashes: set[str] = set()
    for stage, expected_steps in PPO_STAGE_ENV_STEPS.items():
        metadata = training[stage]
        errors.extend(
            _validate_common(
                label=stage,
                metadata=metadata,
                expected_records=2_048,
                expected_seed=20_240,
            )
        )
        if _level_range(metadata) != (0, 200):
            errors.append(f"{stage}: training level range must be [0, 200)")
        observed_steps = metadata.get("policy", {}).get(
            "checkpoint_completed_env_steps"
        )
        if observed_steps != expected_steps:
            errors.append(
                f"{stage}: checkpoint_completed_env_steps={observed_steps}, "
                f"expected {expected_steps}"
            )
        tree_hash = metadata.get("tree_sha256")
        if tree_hash in tree_hashes:
            errors.append(f"{stage}: duplicate dataset tree_sha256 {tree_hash}")
        elif tree_hash:
            tree_hashes.add(str(tree_hash))

    errors.extend(
        _validate_common(
            label="evaluation",
            metadata=evaluation,
            expected_records=512,
            expected_seed=30_240,
        )
    )
    if _level_range(evaluation) != (10_000, 10_500):
        errors.append("evaluation: level range must be [10000, 10500)")
    eval_steps = evaluation.get("policy", {}).get(
        "checkpoint_completed_env_steps"
    )
    if eval_steps != PPO_STAGE_ENV_STEPS["ppo25p17m"]:
        errors.append(
            "evaluation: checkpoint must be the final 25,165,824-step policy"
        )
    eval_hash = evaluation.get("tree_sha256")
    if eval_hash in tree_hashes:
        errors.append("evaluation: dataset tree hash overlaps a training corpus")

    train_start, train_end = (0, 200)
    eval_start, eval_end = _level_range(evaluation)
    if max(train_start, eval_start) < min(train_end, eval_end):
        errors.append("training and evaluation level ranges overlap")
    return errors
