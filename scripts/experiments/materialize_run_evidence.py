#!/usr/bin/env python3
"""Materialize machine-owned run evidence into an experiment ledger entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dreamer.experiment_runtime import append_jsonl, atomic_write_json, sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _artifact(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "uri": str(path.relative_to(run_dir)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _render_markdown(run_name: str, evidence: dict[str, Any]) -> str:
    lines = [
        f"<!-- BEGIN GENERATED RUN {run_name} -->",
        f"### Generated run evidence: `{run_name}`",
        "",
        f"- Run ID: `{evidence['run_id']}`",
        f"- State: `{evidence['state']}`",
        f"- Last completed updates: `{evidence.get('last_completed_updates')}`",
        f"- Source commit: `{evidence.get('source_commit')}`",
        "",
        "| Prefix | Last metrics |",
        "|---|---|",
    ]
    for prefix, metrics in sorted(evidence["last_metrics"].items()):
        rendered = ", ".join(f"`{key}={value}`" for key, value in sorted(metrics.items()))
        lines.append(f"| `{prefix}` | {rendered} |")
    lines.extend(
        [
            "",
            f"Machine-readable evidence: `{evidence['ledger_uri']}`.",
            f"<!-- END GENERATED RUN {run_name} -->",
        ]
    )
    return "\n".join(lines)


def _replace_generated_section(readme: str, run_name: str, section: str) -> str:
    begin = f"<!-- BEGIN GENERATED RUN {run_name} -->"
    end = f"<!-- END GENERATED RUN {run_name} -->"
    if begin in readme:
        before, remainder = readme.split(begin, maxsplit=1)
        _, after = remainder.split(end, maxsplit=1)
        return before.rstrip() + "\n\n" + section + after
    return readme.rstrip() + "\n\n" + section + "\n"


def materialize_run_evidence(
    *,
    experiment_id: str,
    run_name: str,
    run_dir: Path,
    experiment_dir: Path,
    at: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    experiment_dir = Path(experiment_dir)
    runtime_path = run_dir / "runtime-identity.json"
    state_path = run_dir / "run-state.json"
    if not runtime_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"{run_dir}: runtime-identity.json and run-state.json are required"
        )
    runtime = _load_json(runtime_path)
    state = _load_json(state_path)
    metrics_rows = _load_jsonl(run_dir / "metrics.jsonl")
    last_metrics: dict[str, dict[str, Any]] = {}
    for row in metrics_rows:
        prefix = row.get("prefix")
        metrics = row.get("metrics")
        if isinstance(prefix, str) and isinstance(metrics, dict):
            last_metrics[prefix] = metrics

    validation_rows = []
    for path in sorted((run_dir / "validation").glob("updates-*/metrics.json")):
        validation_rows.append(_load_json(path))

    artifact_paths = [
        path
        for path in (
            runtime_path,
            state_path,
            run_dir / "metrics.jsonl",
            run_dir / "telemetry.jsonl",
            run_dir / "artifacts.jsonl",
            run_dir / "validation-set.json",
            run_dir / "wandb-run.json",
        )
        if path.is_file()
    ]
    artifact_paths.extend(
        path
        for pattern in (
            "runtime-identities/*",
            ".hydra/config.yaml",
            ".hydra/overrides.yaml",
            ".hydra/hydra.yaml",
            "validation/updates-*/*",
        )
        for path in sorted(run_dir.glob(pattern))
        if path.is_file() and path not in artifact_paths
    )
    run_artifacts = [_artifact(path, run_dir) for path in artifact_paths]
    input_fingerprint = hashlib.sha256(
        json.dumps(
            run_artifacts,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    ledger_uri = f"experiments/{experiment_id}/raw/{run_name}-run-evidence.json"
    evidence_path = experiment_dir / "raw" / f"{run_name}-run-evidence.json"
    previous_evidence = (
        _load_json(evidence_path) if evidence_path.is_file() else None
    )
    generated_at = at or datetime.now(timezone.utc).isoformat()
    if (
        at is None
        and previous_evidence is not None
        and previous_evidence.get("input_fingerprint") == input_fingerprint
    ):
        generated_at = previous_evidence["generated_at"]
    evidence = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "run_name": run_name,
        "generated_at": generated_at,
        "input_fingerprint": input_fingerprint,
        "run_id": runtime.get("run_id"),
        "attempt_id": runtime.get("attempt_id"),
        "source_commit": runtime.get("source", {}).get("commit"),
        "config_sha256": runtime.get("config_sha256"),
        "state": state.get("state"),
        "last_completed_updates": state.get("last_completed_updates"),
        "last_metrics": last_metrics,
        "validation": validation_rows,
        "run_artifacts": run_artifacts,
        "ledger_uri": ledger_uri,
    }
    atomic_write_json(evidence_path, evidence)
    evidence_sha256 = sha256_file(evidence_path)

    results_path = experiment_dir / "results.json"
    results = _load_json(results_path)
    observation_name = f"generated_run_evidence:{run_name}"
    observation = {
        "name": observation_name,
        "run_id": evidence["run_id"],
        "state": evidence["state"],
        "last_completed_updates": evidence["last_completed_updates"],
        "uri": ledger_uri,
        "sha256": evidence_sha256,
    }
    artifact = {
        "name": f"generated run evidence: {run_name}",
        "kind": "json",
        "uri": ledger_uri,
        "retention": "git",
        "sha256": evidence_sha256,
    }

    observations = [
        item
        for item in results.setdefault("observations", [])
        if item.get("name") != observation_name
    ]
    artifacts = [
        item
        for item in results.setdefault("artifacts", [])
        if item.get("name") != artifact["name"]
    ]
    new_observations = [*observations, observation]
    new_artifacts = [*artifacts, artifact]
    changed = (
        new_observations != results["observations"]
        or new_artifacts != results["artifacts"]
    )
    results["observations"] = new_observations
    results["artifacts"] = new_artifacts
    if changed:
        results["record_revision"] = int(results.get("record_revision", 0)) + 1
        results.setdefault("revision_history", []).append(
            {
                "revision": results["record_revision"],
                "at": evidence["generated_at"],
                "reason": (
                    f"Materialized machine-owned runtime, telemetry, metric and "
                    f"validation evidence for {run_name}."
                ),
            }
        )
        atomic_write_json(results_path, results)
        append_jsonl(
            experiment_dir / "events.jsonl",
            {
                "at": evidence["generated_at"],
                "type": "run_evidence_materialized",
                "experiment_id": experiment_id,
                "message": (
                    f"Materialized structured run evidence for {run_name} "
                    f"at state {evidence['state']}."
                ),
                "evidence": [
                    f"{ledger_uri} sha256 {evidence_sha256}",
                    f"input fingerprint {input_fingerprint}",
                ],
            },
        )

    readme_path = experiment_dir / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    generated = _render_markdown(run_name, evidence)
    updated_readme = _replace_generated_section(readme, run_name, generated)
    if updated_readme != readme:
        readme_path.write_text(updated_readme, encoding="utf-8")
    return evidence_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    path = materialize_run_evidence(
        experiment_id=args.experiment_id,
        run_name=args.run_name,
        run_dir=args.run_dir,
        experiment_dir=args.experiment_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
