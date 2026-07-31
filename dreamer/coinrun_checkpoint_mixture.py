"""Pure protocol helpers for PPO-checkpoint mixture datasets."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any


PPO_STAGE_ENV_STEPS: dict[str, int] = {
    "ppo01p05m": 1_048_576,
    "ppo06p29m": 6_291_456,
    "ppo12p58m": 12_582_912,
    "ppo25p17m": 25_165_824,
}

SOURCE_RECORDS_PER_STAGE = 2_048
RECORDS_PER_SHARD = 256
MIXTURE_RECORDS = 2_048
MIXTURE_SELECTION_SEED = 40_240

# Shard-aligned ratios avoid rewriting records and preserve byte identity.
MIXTURE_SHARDS_BY_STAGE: dict[str, dict[str, int]] = {
    "final_only": {
        "ppo01p05m": 0,
        "ppo06p29m": 0,
        "ppo12p58m": 0,
        "ppo25p17m": 8,
    },
    "uniform": {
        "ppo01p05m": 2,
        "ppo06p29m": 2,
        "ppo12p58m": 2,
        "ppo25p17m": 2,
    },
    "recency_weighted": {
        "ppo01p05m": 1,
        "ppo06p29m": 1,
        "ppo12p58m": 2,
        "ppo25p17m": 4,
    },
}


def records_by_stage(mixture_name: str) -> dict[str, int]:
    """Return exact record counts for a registered shard-aligned mixture."""

    try:
        shards = MIXTURE_SHARDS_BY_STAGE[mixture_name]
    except KeyError as error:
        raise ValueError(f"unknown checkpoint mixture {mixture_name!r}") from error
    return {
        stage: shard_count * RECORDS_PER_SHARD
        for stage, shard_count in shards.items()
    }


def _stable_stage_seed(*, seed: int, stage: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stage}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def select_stage_shards(
    *,
    stage: str,
    shard_names: Sequence[str],
    count: int,
    seed: int = MIXTURE_SELECTION_SEED,
) -> list[str]:
    """Select a deterministic prefix from one stable stage permutation."""

    if stage not in PPO_STAGE_ENV_STEPS:
        raise ValueError(f"unknown PPO stage {stage!r}")
    expected_shards = SOURCE_RECORDS_PER_STAGE // RECORDS_PER_SHARD
    if len(shard_names) != expected_shards:
        raise ValueError(
            f"{stage} must have {expected_shards} source shards, "
            f"got {len(shard_names)}"
        )
    if len(set(shard_names)) != len(shard_names):
        raise ValueError(f"{stage} source shard names must be unique")
    if not 0 <= count <= expected_shards:
        raise ValueError(
            f"{stage} requested shard count must be in [0, {expected_shards}], "
            f"got {count}"
        )
    permutation = sorted(shard_names)
    random.Random(_stable_stage_seed(seed=seed, stage=stage)).shuffle(
        permutation
    )
    return sorted(permutation[:count])


def validate_mixture_protocol() -> list[str]:
    """Validate the registered ratios before any dataset materialization."""

    errors: list[str] = []
    expected_stages = set(PPO_STAGE_ENV_STEPS)
    for mixture_name, shards_by_stage in MIXTURE_SHARDS_BY_STAGE.items():
        if set(shards_by_stage) != expected_stages:
            errors.append(
                f"{mixture_name}: stage keys must be exactly "
                f"{sorted(expected_stages)}"
            )
            continue
        if any(
            not isinstance(count, int) or count < 0
            for count in shards_by_stage.values()
        ):
            errors.append(f"{mixture_name}: shard counts must be non-negative ints")
        total_records = sum(shards_by_stage.values()) * RECORDS_PER_SHARD
        if total_records != MIXTURE_RECORDS:
            errors.append(
                f"{mixture_name}: records={total_records}, "
                f"expected {MIXTURE_RECORDS}"
            )
    return errors


def validate_mixture_metadata(
    *,
    mixtures: Mapping[str, Mapping[str, Any]],
    source_training: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Validate materialized mixture identities against the source pools."""

    errors = validate_mixture_protocol()
    expected_mixtures = set(MIXTURE_SHARDS_BY_STAGE)
    if set(mixtures) != expected_mixtures:
        errors.append(
            "mixture keys must be exactly "
            f"{sorted(expected_mixtures)}, got {sorted(mixtures)}"
        )
        return errors
    if set(source_training) != set(PPO_STAGE_ENV_STEPS):
        errors.append("source training stages do not match the registered stages")
        return errors

    observed_tree_hashes: set[str] = set()
    for mixture_name, expected_shards in MIXTURE_SHARDS_BY_STAGE.items():
        metadata = mixtures[mixture_name]
        expected_common = {
            "experiment_id": "CR-DYN-0008",
            "env": "coinrun",
            "records": MIXTURE_RECORDS,
            "frames_per_record": 64,
            "distribution_mode": "easy",
            "action_policy": "ppo_mixture",
            "selection_seed": MIXTURE_SELECTION_SEED,
            "start_level": 0,
            "num_levels": 200,
            "consumer": "action_conditioned_dynamics_only",
        }
        for key, expected_value in expected_common.items():
            if metadata.get(key) != expected_value:
                errors.append(
                    f"{mixture_name}: {key}={metadata.get(key)!r}, "
                    f"expected {expected_value!r}"
                )

        mixture = metadata.get("mixture")
        if not isinstance(mixture, Mapping):
            errors.append(f"{mixture_name}: missing mixture metadata")
            continue
        if mixture.get("name") != mixture_name:
            errors.append(f"{mixture_name}: mixture name mismatch")
        observed_shards = mixture.get("shards_by_stage")
        if observed_shards != expected_shards:
            errors.append(
                f"{mixture_name}: shards_by_stage={observed_shards!r}, "
                f"expected {expected_shards!r}"
            )
        expected_records = records_by_stage(mixture_name)
        if mixture.get("records_by_stage") != expected_records:
            errors.append(
                f"{mixture_name}: records_by_stage does not match protocol"
            )

        policy = metadata.get("policy")
        if not isinstance(policy, Mapping):
            errors.append(f"{mixture_name}: missing policy metadata")
            continue
        source_policies = policy.get("source_policies")
        if not isinstance(source_policies, Mapping):
            errors.append(f"{mixture_name}: missing source_policies")
            continue
        if set(source_policies) != set(PPO_STAGE_ENV_STEPS):
            errors.append(f"{mixture_name}: source policy stages differ")
            continue
        for stage, expected_steps in PPO_STAGE_ENV_STEPS.items():
            source = source_training[stage]
            observed = source_policies[stage]
            if not isinstance(observed, Mapping):
                errors.append(f"{mixture_name}/{stage}: invalid source policy")
                continue
            if (
                observed.get("checkpoint_completed_env_steps")
                != expected_steps
            ):
                errors.append(
                    f"{mixture_name}/{stage}: checkpoint step mismatch"
                )
            if observed.get("checkpoint_sha256") != source.get("policy", {}).get(
                "checkpoint_sha256"
            ):
                errors.append(
                    f"{mixture_name}/{stage}: checkpoint SHA256 mismatch"
                )
            if observed.get("source_tree_sha256") != source.get("tree_sha256"):
                errors.append(
                    f"{mixture_name}/{stage}: source tree SHA256 mismatch"
                )
            if observed.get("records") != expected_records[stage]:
                errors.append(f"{mixture_name}/{stage}: record count mismatch")

        source_shards = metadata.get("source_shards")
        if not isinstance(source_shards, list):
            errors.append(f"{mixture_name}: source_shards must be a list")
        elif len(source_shards) != MIXTURE_RECORDS // RECORDS_PER_SHARD:
            errors.append(
                f"{mixture_name}: source_shards has {len(source_shards)} "
                f"entries, expected {MIXTURE_RECORDS // RECORDS_PER_SHARD}"
            )
        else:
            observed_counts = {stage: 0 for stage in PPO_STAGE_ENV_STEPS}
            for entry in source_shards:
                if not isinstance(entry, Mapping):
                    errors.append(
                        f"{mixture_name}: source_shards entry must be an object"
                    )
                    continue
                stage = entry.get("stage")
                if stage not in observed_counts:
                    errors.append(
                        f"{mixture_name}: unknown source shard stage {stage!r}"
                    )
                    continue
                observed_counts[str(stage)] += 1
                if entry.get("sha256") not in {
                    shard.get("sha256")
                    for shard in source_training[str(stage)].get(
                        "shard_entries", []
                    )
                }:
                    errors.append(
                        f"{mixture_name}/{stage}: source shard SHA256 not found"
                    )
            if observed_counts != expected_shards:
                errors.append(
                    f"{mixture_name}: materialized shard counts "
                    f"{observed_counts!r}, expected {expected_shards!r}"
                )

        tree_hash = metadata.get("tree_sha256")
        if not tree_hash:
            errors.append(f"{mixture_name}: missing tree_sha256")
        elif tree_hash in observed_tree_hashes:
            errors.append(
                f"{mixture_name}: duplicate mixture tree SHA256 {tree_hash}"
            )
        else:
            observed_tree_hashes.add(str(tree_hash))
    return errors
