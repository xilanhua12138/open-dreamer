#!/usr/bin/env python3
"""Run one command and always materialize any durable experiment evidence."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from scripts.experiments.materialize_run_evidence import materialize_run_evidence


def run_recorded(
    *,
    experiment_id: str,
    run_name: str,
    run_dir: Path,
    experiment_dir: Path,
    command: Sequence[str],
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    if not command:
        raise ValueError("a command is required after --")
    completed = command_runner(list(command), check=False)
    runtime_path = Path(run_dir) / "runtime-identity.json"
    state_path = Path(run_dir) / "run-state.json"
    if runtime_path.is_file() and state_path.is_file():
        materialize_run_evidence(
            experiment_id=experiment_id,
            run_name=run_name,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
        )
    elif completed.returncode == 0:
        raise RuntimeError(
            "recorded command succeeded without runtime-identity.json and "
            f"run-state.json in {run_dir}"
        )
    return int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    raise SystemExit(
        run_recorded(
            experiment_id=args.experiment_id,
            run_name=args.run_name,
            run_dir=args.run_dir,
            experiment_dir=args.experiment_dir,
            command=command,
        )
    )


if __name__ == "__main__":
    main()
