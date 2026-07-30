#!/usr/bin/env python3
"""Materialize auditable shard-aligned mixtures of PPO checkpoint data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import grain

from dreamer.coinrun import COINRUN_ACTION_DIM, COINRUN_NOOP_ACTION
from dreamer.coinrun_checkpoint_mixture import (
    MIXTURE_RECORDS,
    MIXTURE_SELECTION_SEED,
    MIXTURE_SHARDS_BY_STAGE,
    PPO_STAGE_ENV_STEPS,
    RECORDS_PER_SHARD,
    records_by_stage,
    select_stage_shards,
    validate_mixture_protocol,
)
from dreamer.experiment_runtime import atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-id", default="CR-DYN-0008")
    parser.add_argument("--selection-seed", type=int, default=MIXTURE_SELECTION_SEED)
    return parser.parse_args()


def _tree_sha256(entries: list[dict[str, Any]]) -> str:
    lines = [
        f"{entry['name']} {entry['bytes']} {entry['sha256']}"
        for entry in entries
    ]
    return hashlib.sha256(
        ("\n".join(lines) + "\n").encode("utf-8")
    ).hexdigest()


def _load_source_metadata(source_root: Path) -> dict[str, dict[str, Any]]:
    return {
        stage: json.loads(
            (source_root / stage / "metadata.json").read_text(encoding="utf-8")
        )
        for stage in PPO_STAGE_ENV_STEPS
    }


def _validate_source_shard(
    *,
    path: Path,
    expected_sha256: str,
) -> tuple[int, str]:
    if not path.is_file():
        raise ValueError(f"source shard does not exist: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"source shard SHA256 mismatch for {path}: "
            f"{observed_sha256} != {expected_sha256}"
        )
    records = len(grain.sources.ArrayRecordDataSource([str(path)]))
    if records != RECORDS_PER_SHARD:
        raise ValueError(
            f"source shard {path} contains {records} records, "
            f"expected {RECORDS_PER_SHARD}"
        )
    return path.stat().st_size, observed_sha256


def _materialize_mixture(
    *,
    mixture_name: str,
    source_root: Path,
    output_root: Path,
    source_metadata: dict[str, dict[str, Any]],
    experiment_id: str,
    selection_seed: int,
) -> dict[str, Any]:
    output_dir = output_root / mixture_name
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        shards = sorted(output_dir.glob("shard-*.array_record"))
        if len(shards) != MIXTURE_RECORDS // RECORDS_PER_SHARD:
            raise ValueError(
                f"{output_dir} has metadata but {len(shards)} shards; "
                f"expected {MIXTURE_RECORDS // RECORDS_PER_SHARD}"
            )
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(output_dir.iterdir())
    if existing_files:
        raise ValueError(
            f"{output_dir} contains partial mixture data without metadata"
        )

    shards_by_stage = MIXTURE_SHARDS_BY_STAGE[mixture_name]
    source_shards: list[dict[str, Any]] = []
    shard_entries: list[dict[str, Any]] = []
    output_index = 0
    for stage in PPO_STAGE_ENV_STEPS:
        metadata = source_metadata[stage]
        entries = metadata.get("shard_entries")
        if not isinstance(entries, list):
            raise ValueError(f"{stage} metadata is missing shard_entries")
        entry_by_name = {
            str(entry["name"]): entry
            for entry in entries
            if isinstance(entry, dict) and "name" in entry
        }
        selected = select_stage_shards(
            stage=stage,
            shard_names=sorted(entry_by_name),
            count=shards_by_stage[stage],
            seed=selection_seed,
        )
        for source_name in selected:
            source_path = source_root / stage / source_name
            entry = entry_by_name[source_name]
            size, digest = _validate_source_shard(
                path=source_path,
                expected_sha256=str(entry["sha256"]),
            )
            output_name = f"shard-{output_index:05d}.array_record"
            output_path = output_dir / output_name
            output_path.symlink_to(source_path.resolve())
            shard_entries.append(
                {
                    "name": output_name,
                    "bytes": size,
                    "sha256": digest,
                }
            )
            source_shards.append(
                {
                    "output_name": output_name,
                    "stage": stage,
                    "source_name": source_name,
                    "source_uri": str(source_path.resolve()),
                    "bytes": size,
                    "sha256": digest,
                }
            )
            output_index += 1

    expected_shards = MIXTURE_RECORDS // RECORDS_PER_SHARD
    if output_index != expected_shards:
        raise ValueError(
            f"{mixture_name} materialized {output_index} shards, "
            f"expected {expected_shards}"
        )

    exact_records = records_by_stage(mixture_name)
    metadata = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "env": "coinrun",
        "records": MIXTURE_RECORDS,
        "frames_per_record": 64,
        "seed": 20_240,
        "selection_seed": selection_seed,
        "start_level": 0,
        "num_levels": 200,
        "level_range": [0, 200],
        "distribution_mode": "easy",
        "shards": expected_shards,
        "tree_sha256": _tree_sha256(shard_entries),
        "shard_entries": shard_entries,
        "action_space": {
            "categorical_action_dim": COINRUN_ACTION_DIM,
            "categorical_noop_action": COINRUN_NOOP_ACTION,
        },
        "action_policy": "ppo_mixture",
        "policy": {
            "temperature": 1.0,
            "exploration_epsilon": 0.05,
            "deterministic": False,
            "source_policies": {
                stage: {
                    "checkpoint": source_metadata[stage]["policy"][
                        "checkpoint"
                    ],
                    "checkpoint_sha256": source_metadata[stage]["policy"][
                        "checkpoint_sha256"
                    ],
                    "checkpoint_completed_env_steps": PPO_STAGE_ENV_STEPS[
                        stage
                    ],
                    "source_tree_sha256": source_metadata[stage][
                        "tree_sha256"
                    ],
                    "records": exact_records[stage],
                }
                for stage in PPO_STAGE_ENV_STEPS
            },
        },
        "mixture": {
            "name": mixture_name,
            "shards_by_stage": shards_by_stage,
            "records_by_stage": exact_records,
            "selection_method": (
                "deterministic nested prefix of a stable per-stage shard "
                "permutation"
            ),
        },
        "source_shards": source_shards,
        "transition_alignment": (
            "raw_video[t]=observation_t, actions[t]=action_t, "
            "rewards[t]=reward observed after action_t"
        ),
        "episode_boundary_policy": (
            "records never cross auto-reset boundaries; partial chunks are "
            "discarded when terminals[t] is true"
        ),
        "terminals_in_records": True,
        "consumer": "action_conditioned_dynamics_only",
        "excluded_downstream_stages": [
            "behavior_cloning",
            "policy_training_inside_world_model",
        ],
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def main() -> None:
    args = parse_args()
    protocol_errors = validate_mixture_protocol()
    if protocol_errors:
        raise ValueError(f"invalid mixture protocol: {protocol_errors}")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_metadata = _load_source_metadata(source_root)
    mixtures = {
        mixture_name: _materialize_mixture(
            mixture_name=mixture_name,
            source_root=source_root,
            output_root=output_root,
            source_metadata=source_metadata,
            experiment_id=args.experiment_id,
            selection_seed=args.selection_seed,
        )
        for mixture_name in MIXTURE_SHARDS_BY_STAGE
    }
    print(json.dumps(mixtures, indent=2), flush=True)


if __name__ == "__main__":
    main()
