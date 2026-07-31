from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.experiments.materialize_run_evidence import materialize_run_evidence


class MaterializeRunEvidenceTests(unittest.TestCase):
    def test_materialization_updates_machine_owned_evidence_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            experiment_dir = root / "experiments" / "CR-TOK-9999"
            run_dir.mkdir(parents=True)
            (experiment_dir / "raw").mkdir(parents=True)
            (run_dir / "runtime-identity.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "attempt_id": "attempt-1",
                        "source": {"commit": "a" * 40},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run-state.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "state": "COMPLETED",
                        "last_completed_updates": 20_000,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "at": "2026-07-29T00:00:00+00:00",
                        "step": 19_999,
                        "completed_updates": 20_000,
                        "prefix": "train/",
                        "metrics": {"loss": 0.25},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            identity_dir = run_dir / "runtime-identities"
            identity_dir.mkdir()
            (identity_dir / "attempt-1.source.patch").write_text(
                "diff --git a/model.py b/model.py\n",
                encoding="utf-8",
            )
            results_path = experiment_dir / "results.json"
            results_path.write_text(
                json.dumps(
                    {
                        "record_revision": 1,
                        "observations": [],
                        "artifacts": [],
                        "revision_history": [],
                    }
                ),
                encoding="utf-8",
            )
            readme_path = experiment_dir / "README.md"
            readme_path.write_text("# CR-TOK-9999\n", encoding="utf-8")

            first = materialize_run_evidence(
                experiment_id="CR-TOK-9999",
                run_name="n1p1m-seed0",
                run_dir=run_dir,
                experiment_dir=experiment_dir,
                at="2026-07-29T00:01:00+00:00",
            )
            second = materialize_run_evidence(
                experiment_id="CR-TOK-9999",
                run_name="n1p1m-seed0",
                run_dir=run_dir,
                experiment_dir=experiment_dir,
                at="2026-07-29T00:01:00+00:00",
            )

            self.assertEqual(first, second)
            results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(results["record_revision"], 2)
            self.assertEqual(len(results["observations"]), 1)
            self.assertEqual(len(results["artifacts"]), 1)
            self.assertEqual(len(results["revision_history"]), 1)
            self.assertEqual(
                results["observations"][0]["last_completed_updates"], 20_000
            )
            readme = readme_path.read_text(encoding="utf-8")
            self.assertEqual(readme.count("BEGIN GENERATED RUN n1p1m-seed0"), 1)
            events = [
                json.loads(line)
                for line in (experiment_dir / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["type"], "run_evidence_materialized")
            evidence = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(evidence["last_metrics"]["train/"]["loss"], 0.25)
            self.assertIn(
                "runtime-identities/attempt-1.source.patch",
                [artifact["uri"] for artifact in evidence["run_artifacts"]],
            )


if __name__ == "__main__":
    unittest.main()
