from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from dreamer.configs import LoggerConfig
from dreamer.logging import Logger, WandbLogger, build_logger


class StructuredLoggingTests(unittest.TestCase):
    def test_primary_process_detection_supports_jax_without_initialized_api(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch(
                    "dreamer.logging.jax.distributed",
                    new=types.SimpleNamespace(),
                ),
                mock.patch.dict("os.environ", {"RANK": "0"}, clear=False),
            ):
                logger = build_logger(
                    LoggerConfig(use_wandb=False),
                    config={},
                    dir=temporary,
                )

            self.assertIsNotNone(logger.recorder)

    def test_non_primary_process_does_not_open_shared_recorder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch(
                    "dreamer.logging.jax.distributed.is_initialized",
                    return_value=False,
                ),
                mock.patch.dict("os.environ", {"RANK": "1"}, clear=False),
            ):
                logger = build_logger(
                    LoggerConfig(use_wandb=False),
                    config={},
                    dir=temporary,
                )

            self.assertIsNone(logger.recorder)

    def test_local_logger_always_persists_structured_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = Logger(
                LoggerConfig(log_every=5, max_steps=10),
                config={"run_name": "test"},
                dir=temporary,
            )
            with mock.patch(
                "dreamer.experiment_runtime.collect_runtime_identity",
                return_value={
                    "schema_version": "1.0",
                    "run_id": "run-1",
                    "attempt_id": "attempt-1",
                },
            ):
                with logger:
                    logger.log(5, {"loss": 2.5})

            rows = [
                json.loads(line)
                for line in (Path(temporary) / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(rows[0]["step"], 5)
            self.assertEqual(rows[0]["prefix"], "train/")
            self.assertEqual(rows[0]["metrics"]["loss"], 2.5)

    def test_local_logger_records_existing_media_as_hashed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            media = run_dir / "validation.mp4"
            media.write_bytes(b"video")
            logger = Logger(LoggerConfig(), config={}, dir=temporary)
            with mock.patch(
                "dreamer.experiment_runtime.collect_runtime_identity",
                return_value={
                    "schema_version": "1.0",
                    "run_id": "run-1",
                    "attempt_id": "attempt-1",
                },
            ):
                with logger:
                    logger.log_video(3, "validation/reconstruction", media)

            rows = [
                json.loads(line)
                for line in (run_dir / "artifacts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(rows[0]["key"], "validation/reconstruction")
            self.assertEqual(rows[0]["bytes"], 5)
            self.assertEqual(len(rows[0]["sha256"]), 64)

    def test_wandb_logger_reuses_durable_run_id_and_keeps_local_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            init_calls = []
            logged = []
            finished = []

            class FakeRun:
                id = "run-1"
                name = "observed-run"
                entity = "team"
                project = "open-dreamer"
                url = "https://wandb.invalid/run-1"

                def log(self, payload, step):
                    logged.append((payload, step))

            fake_wandb = types.SimpleNamespace(
                init=lambda **kwargs: init_calls.append(kwargs) or FakeRun(),
                finish=lambda **kwargs: finished.append(kwargs),
                Image=lambda value, caption=None: (value, caption),
                Video=lambda value, **kwargs: (value, kwargs),
            )
            config = LoggerConfig(
                run_name="observed-run",
                use_wandb=True,
                wandb_entity="team",
                wandb_project="open-dreamer",
                wandb_group="CR-TOK-9999",
                wandb_tags=["coinrun", "tokenizer"],
                wandb_mode="offline",
                max_steps=10,
            )
            logger = WandbLogger(config, config={"seed": 0}, dir=temporary)
            with (
                mock.patch.dict(sys.modules, {"wandb": fake_wandb}),
                mock.patch(
                    "dreamer.experiment_runtime.collect_runtime_identity",
                    return_value={
                        "schema_version": "1.0",
                        "run_id": "run-1",
                        "attempt_id": "attempt-1",
                    },
                ),
            ):
                with logger:
                    logger.log_metrics(4, {"loss": 1.0})

            self.assertEqual(init_calls[0]["id"], "run-1")
            self.assertEqual(init_calls[0]["resume"], "allow")
            self.assertEqual(init_calls[0]["mode"], "offline")
            self.assertEqual(logged[0], ({"train/loss": 1.0, "train/step": 4}, 4))
            self.assertEqual(finished, [{"exit_code": 0}])
            self.assertTrue((Path(temporary) / "metrics.jsonl").is_file())
            wandb_identity = json.loads(
                (Path(temporary) / "wandb-run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(wandb_identity["url"], "https://wandb.invalid/run-1")

    def test_wandb_finish_failure_is_preserved_in_local_run_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            class FakeRun:
                id = "run-1"
                name = "observed-run"
                entity = "team"
                project = "open-dreamer"
                url = "https://wandb.invalid/run-1"

                def log(self, payload, step):
                    del payload, step

            def fail_finish(**kwargs):
                del kwargs
                raise RuntimeError("wandb finish failed")

            fake_wandb = types.SimpleNamespace(
                init=lambda **kwargs: FakeRun(),
                finish=fail_finish,
            )
            logger = WandbLogger(
                LoggerConfig(
                    run_name="observed-run",
                    use_wandb=True,
                    wandb_project="open-dreamer",
                    max_steps=10,
                ),
                config={},
                dir=temporary,
            )
            with (
                mock.patch.dict(sys.modules, {"wandb": fake_wandb}),
                mock.patch(
                    "dreamer.experiment_runtime.collect_runtime_identity",
                    return_value={
                        "schema_version": "1.0",
                        "run_id": "run-1",
                        "attempt_id": "attempt-1",
                    },
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "wandb finish failed"):
                    with logger:
                        logger.log_metrics(0, {"loss": 1.0})

            state = json.loads(
                (Path(temporary) / "run-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["state"], "FAILED")
            self.assertEqual(state["failure"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
