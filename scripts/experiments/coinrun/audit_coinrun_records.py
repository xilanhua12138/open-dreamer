#!/usr/bin/env python3
"""Audit CoinRun ArrayRecords and emit a reproducible dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import grain
import numpy as np

from dreamer.coinrun import (
    COINRUN_ACTION_DIM,
    COINRUN_CONTROL_ACTIONS,
    COINRUN_NOOP_ACTION,
)
from dreamer.coinrun_dataset_audit import (
    validate_action_range,
    validate_ppo_metadata,
    validate_record_terminals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-to-check", type=int, default=256)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    metadata_path = args.dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shards = sorted(args.dataset.glob("shard-*.array_record"))
    if not shards:
        raise ValueError(f"no ArrayRecord shards in {args.dataset}")
    source = grain.sources.ArrayRecordDataSource([str(path) for path in shards])
    expected_records = int(metadata["records"])
    errors: list[str] = []
    if len(source) != expected_records:
        errors.append(
            f"metadata records={expected_records}, ArrayRecord records={len(source)}"
        )

    action_space = metadata.get("action_space", {})
    if action_space.get("categorical_action_dim") != COINRUN_ACTION_DIM:
        errors.append(f"unexpected action dimension: {action_space}")
    if action_space.get("categorical_noop_action") != COINRUN_NOOP_ACTION:
        errors.append(f"unexpected no-op action: {action_space}")
    errors.extend(validate_ppo_metadata(metadata))

    sample_count = min(max(1, args.records_to_check), len(source))
    sample_indices = np.linspace(
        0,
        len(source) - 1,
        num=sample_count,
        dtype=np.int64,
    )
    observed_actions: set[int] = set()
    repeated_transitions = 0
    total_transitions = 0
    records_with_reward = 0
    expected_frames = int(metadata["frames_per_record"])
    expected_video_bytes = expected_frames * 64 * 64 * 3

    for index in sample_indices:
        record = pickle.loads(source[int(index)])
        sequence_length = int(record["sequence_length"])
        actions = np.asarray(record["actions"], dtype=np.int32).reshape(-1)
        rewards = np.asarray(record["rewards"], dtype=np.float32).reshape(-1)
        if sequence_length != expected_frames:
            errors.append(
                f"record {index}: sequence_length={sequence_length}, "
                f"expected={expected_frames}"
            )
        if len(record["raw_video"]) != expected_video_bytes:
            errors.append(
                f"record {index}: raw_video bytes={len(record['raw_video'])}, "
                f"expected={expected_video_bytes}"
            )
        if actions.shape != (expected_frames,):
            errors.append(
                f"record {index}: actions shape={actions.shape}, "
                f"expected={(expected_frames,)}"
            )
            continue
        if rewards.shape != (expected_frames,):
            errors.append(
                f"record {index}: rewards shape={rewards.shape}, "
                f"expected={(expected_frames,)}"
            )
        action_errors = validate_action_range(
            actions,
            action_dim=COINRUN_ACTION_DIM,
        )
        errors.extend(
            f"record {index}: {message}" for message in action_errors
        )
        if metadata.get("action_policy") in {"ppo", "ppo_mixture"}:
            if "terminals" not in record:
                errors.append(f"record {index}: missing terminals")
            else:
                terminal_errors = validate_record_terminals(
                    np.asarray(record["terminals"]),
                    expected_frames=expected_frames,
                )
                errors.extend(
                    f"record {index}: {message}"
                    for message in terminal_errors
                )
        observed_actions.update(int(action) for action in actions)
        repeated_transitions += int(np.sum(actions[1:] == actions[:-1]))
        total_transitions += max(0, actions.size - 1)
        records_with_reward += int(np.any(rewards != 0))

    if metadata.get("action_policy") == "structured":
        unexpected = observed_actions - set(COINRUN_CONTROL_ACTIONS)
        if unexpected:
            errors.append(f"structured policy emitted unexpected actions: {unexpected}")
        missing = set(COINRUN_CONTROL_ACTIONS) - observed_actions
        if missing:
            errors.append(f"structured policy sample missed actions: {missing}")

    persistence_ratio = repeated_transitions / max(1, total_transitions)
    if metadata.get("action_policy") == "structured" and persistence_ratio < 0.70:
        errors.append(
            f"structured macro persistence {persistence_ratio:.4f} is below 0.70"
        )

    shard_entries = []
    tree_lines = []
    for path in shards:
        digest = sha256_file(path)
        size = path.stat().st_size
        shard_entries.append(
            {"name": path.name, "bytes": size, "sha256": digest}
        )
        tree_lines.append(f"{path.name} {size} {digest}")
    tree_sha256 = hashlib.sha256(
        ("\n".join(tree_lines) + "\n").encode("utf-8")
    ).hexdigest()

    payload = {
        "dataset": str(args.dataset),
        "valid": not errors,
        "errors": errors,
        "metadata": metadata,
        "array_record_records": len(source),
        "audited_records": sample_count,
        "observed_actions": sorted(observed_actions),
        "macro_persistence_ratio": persistence_ratio,
        "records_with_nonzero_reward": records_with_reward,
        "metadata_sha256": sha256_file(metadata_path),
        "tree_sha256": tree_sha256,
        "shards": shard_entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
