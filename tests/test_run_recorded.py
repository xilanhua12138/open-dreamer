from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.experiments.run_recorded import run_recorded


class RecordedRunTests(unittest.TestCase):
    def test_success_without_runtime_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = mock.Mock(
                return_value=subprocess.CompletedProcess(["train"], returncode=0)
            )

            with self.assertRaisesRegex(RuntimeError, "runtime-identity.json"):
                run_recorded(
                    experiment_id="CR-TOK-9999",
                    run_name="arm",
                    run_dir=root / "run",
                    experiment_dir=root / "experiment",
                    command=["train"],
                    command_runner=runner,
                )

            runner.assert_called_once_with(["train"], check=False)

    def test_failed_run_still_materializes_available_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            experiment_dir = root / "experiment"
            run_dir.mkdir()
            experiment_dir.mkdir()
            (run_dir / "runtime-identity.json").write_text(
                json.dumps({"run_id": "run-1"}),
                encoding="utf-8",
            )
            (run_dir / "run-state.json").write_text(
                json.dumps({"state": "FAILED"}),
                encoding="utf-8",
            )
            runner = mock.Mock(
                return_value=subprocess.CompletedProcess(["train"], returncode=7)
            )
            with mock.patch(
                "scripts.experiments.run_recorded.materialize_run_evidence"
            ) as materialize:
                returncode = run_recorded(
                    experiment_id="CR-TOK-9999",
                    run_name="arm",
                    run_dir=run_dir,
                    experiment_dir=experiment_dir,
                    command=["train"],
                    command_runner=runner,
                )

            self.assertEqual(returncode, 7)
            materialize.assert_called_once_with(
                experiment_id="CR-TOK-9999",
                run_name="arm",
                run_dir=run_dir,
                experiment_dir=experiment_dir,
            )


if __name__ == "__main__":
    unittest.main()
