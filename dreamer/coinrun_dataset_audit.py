"""Pure validation helpers for CoinRun dynamics datasets."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_action_range(
    actions: np.ndarray,
    *,
    action_dim: int,
) -> list[str]:
    actions = np.asarray(actions)
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    outside = sorted(
        {
            int(action)
            for action in actions.reshape(-1)
            if int(action) < 0 or int(action) >= action_dim
        }
    )
    return (
        [f"actions outside [0, {action_dim}): {outside}"]
        if outside
        else []
    )


def validate_record_terminals(
    terminals: np.ndarray,
    *,
    expected_frames: int,
) -> list[str]:
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    if terminals.shape != (expected_frames,):
        return [
            f"terminals shape={terminals.shape}, expected={(expected_frames,)}"
        ]
    if np.any(terminals[:-1]):
        return ["terminal flag appears before the final record frame"]
    return []


def validate_ppo_metadata(metadata: dict[str, Any]) -> list[str]:
    action_policy = metadata.get("action_policy")
    if action_policy not in {"ppo", "ppo_mixture"}:
        return []
    errors = []
    policy = metadata.get("policy")
    if not isinstance(policy, dict):
        policy = {}
    if action_policy == "ppo":
        digest = str(policy.get("checkpoint_sha256", ""))
        if not SHA256.fullmatch(digest):
            errors.append(
                "PPO dataset checkpoint_sha256 must be 64 lowercase hex characters"
            )
        try:
            completed_env_steps = int(
                policy.get("checkpoint_completed_env_steps", 0)
            )
        except (TypeError, ValueError):
            completed_env_steps = 0
        if completed_env_steps <= 0:
            errors.append(
                "PPO dataset checkpoint_completed_env_steps must be positive"
            )
    else:
        source_policies = policy.get("source_policies")
        if not isinstance(source_policies, dict) or not source_policies:
            errors.append("PPO mixture dataset must declare source_policies")
        else:
            for source_name, source in source_policies.items():
                if not isinstance(source, dict):
                    errors.append(
                        f"PPO mixture source {source_name} must be an object"
                    )
                    continue
                digest = str(source.get("checkpoint_sha256", ""))
                if not SHA256.fullmatch(digest):
                    errors.append(
                        f"PPO mixture source {source_name} checkpoint_sha256 "
                        "must be 64 lowercase hex characters"
                    )
                try:
                    completed_env_steps = int(
                        source.get("checkpoint_completed_env_steps", 0)
                    )
                except (TypeError, ValueError):
                    completed_env_steps = 0
                if completed_env_steps <= 0:
                    errors.append(
                        f"PPO mixture source {source_name} "
                        "checkpoint_completed_env_steps must be positive"
                    )
    alignment = str(metadata.get("transition_alignment", ""))
    if not all(
        marker in alignment
        for marker in ("observation_t", "action_t", "after action_t")
    ):
        errors.append("PPO dataset transition_alignment is not action-aligned")
    boundary = str(metadata.get("episode_boundary_policy", ""))
    if not all(marker in boundary for marker in ("never cross", "terminal")):
        errors.append(
            "PPO dataset must declare episode-safe boundary handling"
        )
    if metadata.get("terminals_in_records") is not True:
        errors.append(
            "PPO dataset must store terminal flags in every record"
        )
    return errors
