from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dreamer.experiment_runtime import RunRecorder, collect_runtime_identity


class ExperimentRuntimeTests(unittest.TestCase):
    def test_runtime_identity_is_stable_and_does_not_copy_proxy_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            environment = {
                "CUDA_VISIBLE_DEVICES": "0",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "HTTPS_PROXY": "http://user:secret@example.invalid:7890",
                "WANDB_API_KEY": "must-not-be-recorded",
            }
            with (
                mock.patch(
                    "dreamer.experiment_runtime._collect_git_identity",
                    return_value={"commit": "a" * 40, "dirty": False},
                ),
                mock.patch(
                    "dreamer.experiment_runtime._collect_accelerator_identity",
                    return_value={"available": False, "gpus": []},
                ),
                mock.patch(
                    "dreamer.experiment_runtime._collect_package_versions",
                    return_value={"jax": "0.test"},
                ),
            ):
                first = collect_runtime_identity(
                    run_dir,
                    config={"max_steps": 20},
                    environ=environment,
                    argv=["train.py", "token=value"],
                )
                second = collect_runtime_identity(
                    run_dir,
                    config={"max_steps": 20},
                    environ=environment,
                    argv=["train.py", "token=value"],
                )

            self.assertEqual(first["run_id"], second["run_id"])
            self.assertNotEqual(first["attempt_id"], second["attempt_id"])
            self.assertEqual(
                first["environment"]["values"],
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                },
            )
            self.assertEqual(first["environment"]["proxy_configured"], True)
            serialized = json.dumps(first)
            self.assertNotIn("must-not-be-recorded", serialized)
            self.assertNotIn("user:secret", serialized)
            self.assertEqual(first["command"], ["train.py", "token=<redacted>"])

    def test_runtime_identity_saves_tracked_patch_and_hashes_untracked_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            run_dir = root / "run"
            repository.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=repository,
                check=True,
            )
            tracked = repository / "tracked.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "initial"],
                cwd=repository,
                check=True,
            )
            tracked.write_text("VALUE = 2\n", encoding="utf-8")
            (repository / "untracked.py").write_text(
                "EXPERIMENT = True\n",
                encoding="utf-8",
            )

            with (
                mock.patch(
                    "dreamer.experiment_runtime._collect_accelerator_identity",
                    return_value={"available": False, "gpus": []},
                ),
                mock.patch(
                    "dreamer.experiment_runtime._collect_package_versions",
                    return_value={"jax": "0.test"},
                ),
            ):
                identity = collect_runtime_identity(
                    run_dir,
                    config={"max_steps": 20},
                    environ={},
                    repository=repository,
                )

            source = identity["source"]
            patch_path = run_dir / source["patch"]["uri"]
            self.assertTrue(patch_path.is_file())
            self.assertIn("-VALUE = 1", patch_path.read_text(encoding="utf-8"))
            self.assertIn("+VALUE = 2", patch_path.read_text(encoding="utf-8"))
            self.assertEqual(source["untracked_files"][0]["path"], "untracked.py")
            self.assertEqual(
                len(source["untracked_files"][0]["sha256"]),
                64,
            )
            self.assertFalse(source["reproducible_from_recorded_source"])

    def test_run_recorder_writes_metrics_telemetry_and_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            recorder = RunRecorder(
                run_dir=run_dir,
                max_steps=10,
                progress_interval_seconds=0,
                system_metrics_interval_seconds=60,
            )
            with (
                mock.patch(
                    "dreamer.experiment_runtime.collect_runtime_identity",
                    return_value={
                        "schema_version": "1.0",
                        "run_id": "run-1",
                        "attempt_id": "attempt-1",
                    },
                ),
                mock.patch.object(
                    recorder,
                    "_collect_system_metrics",
                    return_value={"disk_free_bytes": 123},
                ),
            ):
                recorder.start(config={"max_steps": 10})
                recorder.log_metrics(
                    step=4,
                    prefix="train/",
                    metrics={"loss": 1.25, "finite": True},
                )
                recorder.finish()

            metrics = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
            telemetry = [
                json.loads(line)
                for line in (run_dir / "telemetry.jsonl").read_text().splitlines()
            ]
            state = json.loads((run_dir / "run-state.json").read_text())

            self.assertEqual(metrics[0]["completed_updates"], 5)
            self.assertEqual(metrics[0]["metrics"], {"finite": True, "loss": 1.25})
            self.assertEqual(
                [event["type"] for event in telemetry],
                ["run_started", "progress", "system", "run_completed"],
            )
            self.assertEqual(telemetry[1]["completed_updates"], 5)
            self.assertIsNone(telemetry[1]["steps_per_second"])
            self.assertIsNone(telemetry[1]["eta_seconds"])
            self.assertEqual(state["state"], "COMPLETED")
            self.assertEqual(state["last_completed_updates"], 5)

    def test_run_recorder_preserves_failure_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = RunRecorder(Path(temporary), max_steps=10)
            with mock.patch(
                "dreamer.experiment_runtime.collect_runtime_identity",
                return_value={
                    "schema_version": "1.0",
                    "run_id": "run-1",
                    "attempt_id": "attempt-1",
                },
            ):
                recorder.start(config={})
                recorder.finish(RuntimeError("boom"))

            state = json.loads(
                (Path(temporary) / "run-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["state"], "FAILED")
            self.assertEqual(state["failure"]["type"], "RuntimeError")
            self.assertEqual(state["failure"]["message"], "boom")

    def test_run_recorder_emits_rate_and_eta_after_second_progress_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            recorder = RunRecorder(
                run_dir=run_dir,
                max_steps=10,
                progress_interval_seconds=0,
                system_metrics_interval_seconds=60,
            )
            with (
                mock.patch(
                    "dreamer.experiment_runtime.collect_runtime_identity",
                    return_value={
                        "schema_version": "1.0",
                        "run_id": "run-1",
                        "attempt_id": "attempt-1",
                    },
                ),
                mock.patch(
                    "dreamer.experiment_runtime.time.monotonic",
                    side_effect=[100.0, 102.0],
                ),
                mock.patch.object(
                    recorder,
                    "_collect_system_metrics",
                    return_value={"disk_free_bytes": 123},
                ),
            ):
                recorder.start(config={"max_steps": 10})
                recorder.log_metrics(step=3, prefix="train/", metrics={"loss": 2.0})
                recorder.log_metrics(step=5, prefix="train/", metrics={"loss": 1.0})
                recorder.finish()

            telemetry = [
                json.loads(line)
                for line in (run_dir / "telemetry.jsonl").read_text().splitlines()
            ]
            progress = [event for event in telemetry if event["type"] == "progress"]
            self.assertEqual(len(progress), 2)
            self.assertEqual(progress[1]["steps_per_second"], 1.0)
            self.assertEqual(progress[1]["eta_seconds"], 4.0)

    def test_progress_heartbeat_is_independent_from_metric_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            recorder = RunRecorder(
                run_dir=run_dir,
                max_steps=10,
                progress_interval_seconds=0,
                system_metrics_interval_seconds=60,
            )
            with (
                mock.patch(
                    "dreamer.experiment_runtime.collect_runtime_identity",
                    return_value={
                        "schema_version": "1.0",
                        "run_id": "run-1",
                        "attempt_id": "attempt-1",
                    },
                ),
                mock.patch(
                    "dreamer.experiment_runtime.time.monotonic",
                    return_value=100.0,
                ),
                mock.patch.object(
                    recorder,
                    "_collect_system_metrics",
                    return_value={"disk_free_bytes": 123},
                ),
            ):
                recorder.start(config={"max_steps": 10})
                recorder.observe_progress(step=6)
                recorder.finish()

            state = json.loads(
                (run_dir / "run-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["last_completed_updates"], 7)
            self.assertFalse((run_dir / "metrics.jsonl").exists())

    def test_system_telemetry_accepts_unsupported_nvidia_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = RunRecorder(Path(temporary), max_steps=10)
            with mock.patch(
                "dreamer.experiment_runtime._run_command",
                return_value="0, N/A, 123, 24564, [N/A], [Not Supported]",
            ):
                metrics = recorder._collect_system_metrics()

            self.assertIsNone(metrics["gpus"][0]["utilization_percent"])
            self.assertEqual(metrics["gpus"][0]["memory_used_mb"], 123.0)
            self.assertIsNone(metrics["gpus"][0]["temperature_c"])
            self.assertIsNone(metrics["gpus"][0]["power_w"])


if __name__ == "__main__":
    unittest.main()
