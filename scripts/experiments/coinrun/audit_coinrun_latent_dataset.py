#!/usr/bin/env python3
"""Audit a CoinRun latent dataset against its immutable raw source."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from dreamer.coinrun_latent_dataset import validate_coinrun_latent_pair
from dreamer.experiment_runtime import atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--latent-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-to-check", type=int, default=0)
    return parser.parse_args()


def _tree_sha256(paths: list[Path]) -> str:
    lines = [
        f"{path.name} {path.stat().st_size} {sha256_file(path)}"
        for path in sorted(paths)
    ]
    return hashlib.sha256(
        ("\n".join(lines) + "\n").encode("utf-8")
    ).hexdigest()


def main() -> None:
    args = parse_args()
    import grain

    from dreamer.data.serialization import deserialize_msgpack_record

    raw_metadata = json.loads(
        (args.raw_dataset / "metadata.json").read_text(encoding="utf-8")
    )
    latent_metadata_path = args.latent_dataset / "metadata.json"
    latent_metadata = json.loads(
        latent_metadata_path.read_text(encoding="utf-8")
    )
    raw_paths = sorted(args.raw_dataset.glob("shard-*.array_record"))
    latent_paths = sorted(args.latent_dataset.glob("shard-*.array_record"))
    raw_source = grain.sources.ArrayRecordDataSource(
        [str(path) for path in raw_paths]
    )
    latent_source = grain.sources.ArrayRecordDataSource(
        [str(path) for path in latent_paths]
    )

    errors: list[str] = []
    if len(raw_source) != len(latent_source):
        errors.append(
            f"record count mismatch raw={len(raw_source)} "
            f"latent={len(latent_source)}"
        )
    if len(latent_source) != int(latent_metadata["records"]):
        errors.append("latent metadata record count mismatch")
    raw_tree = _tree_sha256(raw_paths)
    latent_tree = _tree_sha256(latent_paths)
    if raw_tree != raw_metadata["tree_sha256"]:
        errors.append("raw tree differs from raw metadata")
    if raw_tree != latent_metadata["source"]["raw_tree_sha256"]:
        errors.append("latent metadata points to a different raw tree")
    if latent_tree != latent_metadata["tree_sha256"]:
        errors.append("latent tree differs from latent metadata")

    pair_audits: list[dict[str, Any]] = []
    check_count = (
        min(args.records_to_check, len(latent_source))
        if args.records_to_check > 0
        else len(latent_source)
    )
    indices = (
        np.linspace(
            0,
            len(latent_source) - 1,
            num=check_count,
            dtype=np.int64,
        )
        if check_count
        else np.asarray([], dtype=np.int64)
    )
    for raw_index in indices:
        index = int(raw_index)
        raw_record = pickle.loads(raw_source[index])
        latent_record = deserialize_msgpack_record(latent_source[index])
        try:
            pair_audit = validate_coinrun_latent_pair(
                raw_record,
                latent_record,
            )
            source = latent_record.get("source", {})
            if source.get("record_index") != index:
                raise ValueError(
                    f"latent source record_index={source.get('record_index')} "
                    f"does not match {index}"
                )
            if source.get("raw_tree_sha256") != raw_tree:
                raise ValueError("latent record points to a different raw tree")
            pair_audits.append(pair_audit)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"record {index}: {error}")

    shapes = sorted(
        {tuple(audit["latent_shape"]) for audit in pair_audits}
    )
    mean = np.asarray(latent_metadata["latent"]["mean"], dtype=np.float64)
    std = np.asarray(latent_metadata["latent"]["std"], dtype=np.float64)
    if mean.shape != (16,) or std.shape != (16,):
        errors.append(
            f"latent stats must each have shape (16,), got {mean.shape}/{std.shape}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        errors.append("latent stats contain non-finite values")
    if np.any(std <= 0):
        errors.append("latent std must be positive")

    payload = {
        "schema_version": "1.0",
        "experiment_id": "CR-DYN-0011",
        "valid": not errors,
        "errors": errors,
        "raw_dataset": str(args.raw_dataset),
        "latent_dataset": str(args.latent_dataset),
        "raw_tree_sha256": raw_tree,
        "latent_tree_sha256": latent_tree,
        "latent_metadata_sha256": sha256_file(latent_metadata_path),
        "raw_records": len(raw_source),
        "latent_records": len(latent_source),
        "audited_records": check_count,
        "latent_shapes": [list(shape) for shape in shapes],
        "all_actions_aligned": not errors and bool(pair_audits),
        "all_rewards_aligned": not errors and bool(pair_audits),
        "all_terminals_aligned": not errors and bool(pair_audits),
        "tokenizer": latent_metadata["tokenizer"],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
