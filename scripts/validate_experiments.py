#!/usr/bin/env python3
"""Validate the repository's append-only experiment ledger using stdlib only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
INDEX = EXPERIMENTS / "index.json"

EXPERIMENT_ID = re.compile(r"^[A-Z0-9]+(?:-[A-Z0-9]+)+-\d{4}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")

EXECUTION_STATUSES = {
    "planned",
    "queued",
    "running",
    "completed",
    "failed",
    "aborted",
    "blocked",
    "invalid",
}
SCIENTIFIC_STATUSES = {
    "supports_hypothesis",
    "rejects_hypothesis",
    "inconclusive",
    "not_evaluated",
}
CLAIM_LEVELS = {
    "none",
    "smoke_test",
    "pipeline_closure",
    "internal_result",
    "partial_reproduction",
    "strict_reproduction",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\bgh[opsu]_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Alibaba access key key name": re.compile(
        r'"(?:access_key_secret|accesskeysecret)"\s*:', re.IGNORECASE
    ),
}


class DuplicateKey(ValueError):
    pass


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{path.relative_to(ROOT)}: possible {label}")
        value = json.loads(text, object_pairs_hook=no_duplicate_keys)
    except (OSError, json.JSONDecodeError, DuplicateKey) as exc:
        errors.append(f"{path.relative_to(ROOT)}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.relative_to(ROOT)}: top-level JSON must be an object")
        return None
    return value


def require_keys(
    value: dict[str, Any], keys: set[str], path: Path, errors: list[str]
) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        errors.append(
            f"{path.relative_to(ROOT)}: missing required keys {', '.join(missing)}"
        )


def validate_artifacts(
    artifacts: Any, path: Path, execution_status: str, errors: list[str]
) -> None:
    if not isinstance(artifacts, list):
        errors.append(f"{path.relative_to(ROOT)}: artifacts must be a list")
        return
    for index, artifact in enumerate(artifacts):
        prefix = f"{path.relative_to(ROOT)}: artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for key in ("name", "uri", "kind"):
            if not artifact.get(key):
                errors.append(f"{prefix}.{key} is required")
        digest = artifact.get("sha256")
        if digest is not None and not SHA256.fullmatch(str(digest)):
            errors.append(f"{prefix}.sha256 must be 64 lowercase hex characters")
        if artifact.get("retention") == "git":
            uri = artifact.get("uri")
            if not isinstance(uri, str):
                errors.append(f"{prefix}: git-retained artifact needs a repository URI")
                continue
            artifact_path = ROOT / uri
            if not artifact_path.is_file():
                errors.append(f"{prefix}: git-retained artifact does not exist")
            elif digest is None:
                errors.append(f"{prefix}: git-retained artifact requires sha256")
            else:
                actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                if actual != digest:
                    errors.append(
                        f"{prefix}: sha256 mismatch, recorded {digest}, actual {actual}"
                    )
        if execution_status in {"completed", "failed", "aborted", "invalid"}:
            if artifact.get("retention") == "external" and digest is None:
                errors.append(f"{prefix}: external retained artifacts require sha256")


def validate_events(path: Path, experiment_id: str, errors: list[str]) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path.relative_to(ROOT)}: {exc}")
        return 0
    if not lines:
        errors.append(f"{path.relative_to(ROOT)}: must contain at least one event")
        return 0
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line, object_pairs_hook=no_duplicate_keys)
        except (json.JSONDecodeError, DuplicateKey) as exc:
            errors.append(f"{path.relative_to(ROOT)}:{line_number}: {exc}")
            continue
        if not isinstance(event, dict):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: event must be an object"
            )
            continue
        for key in ("at", "type", "message", "evidence"):
            if key not in event:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_number}: missing {key}"
                )
        if "experiment_id" in event and event["experiment_id"] != experiment_id:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: experiment_id mismatch"
            )
        if not isinstance(event.get("evidence"), list):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: evidence must be a list"
            )
    return len(lines)


def validate_experiment(
    directory: Path, index_entry: dict[str, Any], errors: list[str]
) -> tuple[str, str, str]:
    experiment_id = directory.name
    relative = directory.relative_to(ROOT)
    if not EXPERIMENT_ID.fullmatch(experiment_id):
        errors.append(f"{relative}: invalid experiment ID")

    expected_files = ("README.md", "manifest.json", "results.json", "events.jsonl")
    for filename in expected_files:
        if not (directory / filename).is_file():
            errors.append(f"{relative}: missing {filename}")

    manifest_path = directory / "manifest.json"
    results_path = directory / "results.json"
    if not manifest_path.is_file() or not results_path.is_file():
        return "invalid", "not_evaluated", "none"

    manifest = load_json(manifest_path, errors)
    results = load_json(results_path, errors)
    if manifest is None or results is None:
        return "invalid", "not_evaluated", "none"

    require_keys(
        manifest,
        {
            "schema_version",
            "experiment_id",
            "retrospective",
            "created_at",
            "owner",
            "title",
            "kind",
            "question",
            "reason",
            "hypothesis",
            "falsification",
            "baseline_id",
            "depends_on",
            "controlled_variable",
            "constants",
            "source",
            "data",
            "compute",
            "protocol",
            "success_criteria",
            "planned_outputs",
        },
        manifest_path,
        errors,
    )
    require_keys(
        results,
        {
            "schema_version",
            "experiment_id",
            "record_revision",
            "execution_status",
            "scientific_status",
            "claim_level",
            "started_at",
            "ended_at",
            "summary",
            "observations",
            "comparisons",
            "deviations",
            "failures",
            "conclusion",
            "artifacts",
            "revision_history",
        },
        results_path,
        errors,
    )

    for payload_path, payload in ((manifest_path, manifest), (results_path, results)):
        if payload.get("schema_version") != "1.0":
            errors.append(f"{payload_path.relative_to(ROOT)}: unsupported schema_version")
        if payload.get("experiment_id") != experiment_id:
            errors.append(f"{payload_path.relative_to(ROOT)}: experiment_id mismatch")

    if not isinstance(manifest.get("retrospective"), bool):
        errors.append(
            f"{manifest_path.relative_to(ROOT)}: retrospective must be boolean"
        )
    if not isinstance(manifest.get("success_criteria"), list) or not manifest.get(
        "success_criteria"
    ):
        errors.append(
            f"{manifest_path.relative_to(ROOT)}: success_criteria must be non-empty"
        )
    if not isinstance(manifest.get("depends_on"), list):
        errors.append(f"{manifest_path.relative_to(ROOT)}: depends_on must be a list")

    source = manifest.get("source")
    if not isinstance(source, dict):
        errors.append(f"{manifest_path.relative_to(ROOT)}: source must be an object")
    else:
        commit = source.get("commit")
        if commit is not None and not COMMIT.fullmatch(str(commit)):
            errors.append(f"{manifest_path.relative_to(ROOT)}: invalid source commit")
        if not isinstance(source.get("dirty"), bool):
            errors.append(f"{manifest_path.relative_to(ROOT)}: source.dirty must be bool")
        if source.get("dirty") and not SHA256.fullmatch(
            str(source.get("patch_sha256", ""))
        ):
            errors.append(
                f"{manifest_path.relative_to(ROOT)}: dirty source requires patch_sha256"
            )
        runner = source.get("runner")
        runner_digest = source.get("runner_sha256")
        if runner is not None:
            runner_path = ROOT / str(runner)
            if not runner_path.is_file():
                errors.append(
                    f"{manifest_path.relative_to(ROOT)}: source runner does not exist"
                )
            elif not SHA256.fullmatch(str(runner_digest)):
                errors.append(
                    f"{manifest_path.relative_to(ROOT)}: source runner needs sha256"
                )
            else:
                actual = hashlib.sha256(runner_path.read_bytes()).hexdigest()
                if actual != runner_digest:
                    errors.append(
                        f"{manifest_path.relative_to(ROOT)}: source runner sha256 "
                        f"mismatch, recorded {runner_digest}, actual {actual}"
                    )

    execution = str(results.get("execution_status"))
    scientific = str(results.get("scientific_status"))
    claim = str(results.get("claim_level"))
    if execution not in EXECUTION_STATUSES:
        errors.append(f"{results_path.relative_to(ROOT)}: invalid execution_status")
    if scientific not in SCIENTIFIC_STATUSES:
        errors.append(f"{results_path.relative_to(ROOT)}: invalid scientific_status")
    if claim not in CLAIM_LEVELS:
        errors.append(f"{results_path.relative_to(ROOT)}: invalid claim_level")
    if not isinstance(results.get("record_revision"), int) or results.get(
        "record_revision", 0
    ) < 1:
        errors.append(f"{results_path.relative_to(ROOT)}: record_revision must be >= 1")

    conclusion = results.get("conclusion")
    if not isinstance(conclusion, dict):
        errors.append(f"{results_path.relative_to(ROOT)}: conclusion must be an object")
    else:
        for key in ("observed", "interpretation", "not_established", "decision"):
            if not isinstance(conclusion.get(key), list):
                errors.append(
                    f"{results_path.relative_to(ROOT)}: conclusion.{key} must be a list"
                )

    if execution == "completed" and not results.get("observations"):
        errors.append(
            f"{results_path.relative_to(ROOT)}: completed run needs observations"
        )
    if execution in {"completed", "failed", "aborted", "invalid"}:
        if not results.get("started_at") or not results.get("ended_at"):
            errors.append(
                f"{results_path.relative_to(ROOT)}: terminal run needs start/end times"
            )

    validate_artifacts(results.get("artifacts"), results_path, execution, errors)
    if (directory / "events.jsonl").is_file():
        validate_events(directory / "events.jsonl", experiment_id, errors)

    readme_path = directory / "README.md"
    if readme_path.is_file() and experiment_id not in readme_path.read_text(
        encoding="utf-8"
    ):
        errors.append(f"{readme_path.relative_to(ROOT)}: must mention {experiment_id}")

    expected_path = f"experiments/{experiment_id}"
    if index_entry.get("path") != expected_path:
        errors.append(f"{INDEX.relative_to(ROOT)}: wrong path for {experiment_id}")
    for field, actual in (
        ("execution_status", execution),
        ("scientific_status", scientific),
        ("claim_level", claim),
    ):
        if index_entry.get(field) != actual:
            errors.append(
                f"{INDEX.relative_to(ROOT)}: {experiment_id} {field} disagrees with results"
            )
    if index_entry.get("depends_on") != manifest.get("depends_on"):
        errors.append(
            f"{INDEX.relative_to(ROOT)}: {experiment_id} depends_on disagrees with manifest"
        )
    return execution, scientific, claim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    errors: list[str] = []

    index = load_json(INDEX, errors)
    if index is None:
        print("\n".join(errors), file=sys.stderr)
        return 1
    entries = index.get("entries")
    if not isinstance(entries, list):
        print("experiments/index.json: entries must be a list", file=sys.stderr)
        return 1

    by_id: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"experiments/index.json: entries[{position}] must be object")
            continue
        experiment_id = entry.get("experiment_id")
        if not isinstance(experiment_id, str):
            errors.append(
                f"experiments/index.json: entries[{position}] missing experiment_id"
            )
            continue
        if experiment_id in by_id:
            errors.append(f"experiments/index.json: duplicate {experiment_id}")
        by_id[experiment_id] = entry

    directories = sorted(
        path
        for path in EXPERIMENTS.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    directory_ids = {path.name for path in directories}
    index_ids = set(by_id)
    for missing in sorted(index_ids - directory_ids):
        errors.append(f"experiments/index.json: directory missing for {missing}")
    for missing in sorted(directory_ids - index_ids):
        errors.append(f"experiments/index.json: entry missing for {missing}")

    for directory in directories:
        entry = by_id.get(directory.name)
        if entry is not None:
            validate_experiment(directory, entry, errors)

    known_ids = index_ids
    for experiment_id, entry in by_id.items():
        dependencies = entry.get("depends_on")
        if isinstance(dependencies, list):
            for dependency in dependencies:
                if dependency not in known_ids:
                    errors.append(
                        f"experiments/index.json: {experiment_id} has unknown dependency "
                        f"{dependency}"
                    )

    if errors:
        print(f"Experiment ledger validation failed ({len(errors)} errors):")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Experiment ledger valid: {len(directories)} experiments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
