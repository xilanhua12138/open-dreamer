from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import validate_experiments as ledger


class ExperimentLedgerValidatorTests(unittest.TestCase):
    def test_load_json_accepts_unique_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "record.json"
            path.write_text('{"experiment_id":"CR-DYN-0001"}\n', encoding="utf-8")
            errors: list[str] = []

            with mock.patch.object(ledger, "ROOT", root):
                result = ledger.load_json(path, errors)

            self.assertEqual(result, {"experiment_id": "CR-DYN-0001"})
            self.assertEqual(errors, [])

    def test_load_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "record.json"
            path.write_text('{"status":"running","status":"completed"}\n', encoding="utf-8")
            errors: list[str] = []

            with mock.patch.object(ledger, "ROOT", root):
                result = ledger.load_json(path, errors)

            self.assertIsNone(result)
            self.assertEqual(len(errors), 1)
            self.assertIn("duplicate JSON key: status", errors[0])

    def test_git_artifact_hash_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "experiments" / "raw.json"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b'{"metric": 1}\n')
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            artifacts = [
                {
                    "name": "raw metrics",
                    "kind": "json",
                    "uri": "experiments/raw.json",
                    "retention": "git",
                    "sha256": digest,
                }
            ]
            errors: list[str] = []

            with mock.patch.object(ledger, "ROOT", root):
                ledger.validate_artifacts(
                    artifacts, root / "results.json", "completed", errors
                )

            self.assertEqual(errors, [])

    def test_git_artifact_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "experiments" / "raw.json"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b'{"metric": 1}\n')
            artifacts = [
                {
                    "name": "raw metrics",
                    "kind": "json",
                    "uri": "experiments/raw.json",
                    "retention": "git",
                    "sha256": "0" * 64,
                }
            ]
            errors: list[str] = []

            with mock.patch.object(ledger, "ROOT", root):
                ledger.validate_artifacts(
                    artifacts, root / "results.json", "completed", errors
                )

            self.assertEqual(len(errors), 1)
            self.assertIn("sha256 mismatch", errors[0])

    def test_terminal_external_artifact_without_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = [
                {
                    "name": "checkpoint archive",
                    "kind": "tar.gz",
                    "uri": "s3://bucket/checkpoint.tar.gz",
                    "retention": "external",
                    "sha256": None,
                }
            ]
            errors: list[str] = []

            with mock.patch.object(ledger, "ROOT", root):
                ledger.validate_artifacts(
                    artifacts, root / "results.json", "aborted", errors
                )

            self.assertEqual(errors, [
                "results.json: artifacts[0]: external retained artifacts require sha256"
            ])

    def test_event_evidence_must_be_a_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "events.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "at": "2026-07-29T00:00:00+08:00",
                        "type": "completed",
                        "message": "done",
                        "evidence": "metrics.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            errors: list[str] = []

            with mock.patch.object(ledger, "ROOT", root):
                count = ledger.validate_events(path, "CR-DYN-0001", errors)

            self.assertEqual(count, 1)
            self.assertEqual(errors, ["events.jsonl:1: evidence must be a list"])


if __name__ == "__main__":
    unittest.main()
