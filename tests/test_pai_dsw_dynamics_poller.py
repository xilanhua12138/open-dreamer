import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "operations"
    / "poll_pai_dsw_dynamics.py"
)
SPEC = importlib.util.spec_from_file_location("pai_dsw_poller", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
poller_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = poller_module
SPEC.loader.exec_module(poller_module)


def instance_payload(status: str) -> str:
    return json.dumps(
        {
            "InstanceId": poller_module.INSTANCE_ID,
            "Status": status,
            "EcsSpec": poller_module.EXPECTED_SPEC,
            "RequestedResource": {"GPUType": poller_module.EXPECTED_GPU},
        }
    )


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, args, input_text, timeout_seconds):
        self.calls.append((list(args), input_text, timeout_seconds))
        if not self.responses:
            raise AssertionError(f"unexpected command: {args}")
        expected_action, response = self.responses.pop(0)
        rendered = " ".join(args)
        if expected_action not in rendered:
            raise AssertionError(
                f"expected {expected_action!r} in command {rendered!r}"
            )
        return response


def ok_json(payload):
    return poller_module.CommandResult(0, json.dumps(payload), "")


class PaiDswDynamicsPollerTest(unittest.TestCase):
    def make_poller(self, fake_runner):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp_path = Path(temp_dir.name)
        aliyun_config_path = temp_path / "aliyun-config.json"
        aliyun_config_path.write_text(
            json.dumps(
                {
                    "profiles": [
                        {
                            "name": poller_module.PROFILE,
                            "access_key_id": "test-access-key-id",
                            "access_key_secret": "test-access-key-secret",
                            "sts_token": "test-sts-token",
                            "region_id": poller_module.REGION,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        config = poller_module.Config(
            state_dir=temp_path,
            start_wait_seconds=0,
            status_poll_seconds=0,
            aliyun_config_path=aliyun_config_path,
        )
        return poller_module.Poller(
            config,
            command_runner=fake_runner,
            sleeper=lambda _: None,
            monotonic=lambda: 0,
        )

    def test_failed_instance_requests_exactly_one_start(self):
        fake = FakeRunner(
            [
                (
                    "get-instance",
                    poller_module.CommandResult(
                        0, instance_payload("Failed"), ""
                    ),
                ),
                ("start-instance", ok_json({"Success": True, "RequestId": "r1"})),
                (
                    "get-instance",
                    poller_module.CommandResult(
                        0, instance_payload("Failed"), ""
                    ),
                ),
            ]
        )

        outcome = self.make_poller(fake).run_cycle()

        self.assertEqual(outcome, "capacity_unavailable")
        start_calls = [
            call for call in fake.calls if "start-instance" in call[0]
        ]
        self.assertEqual(len(start_calls), 1)

    def test_starting_instance_never_requests_another_start(self):
        fake = FakeRunner(
            [
                (
                    "get-instance",
                    poller_module.CommandResult(
                        0, instance_payload("Starting"), ""
                    ),
                ),
                (
                    "get-instance",
                    poller_module.CommandResult(
                        0, instance_payload("Failed"), ""
                    ),
                ),
            ]
        )

        outcome = self.make_poller(fake).run_cycle()

        self.assertEqual(outcome, "capacity_unavailable")
        self.assertFalse(
            any("start-instance" in call[0] for call in fake.calls)
        )

    def test_wrong_gpu_identity_blocks_before_start(self):
        payload = json.loads(instance_payload("Failed"))
        payload["RequestedResource"]["GPUType"] = "NVIDIA A100"
        fake = FakeRunner(
            [
                (
                    "get-instance",
                    poller_module.CommandResult(0, json.dumps(payload), ""),
                )
            ]
        )
        monitor = self.make_poller(fake)

        with self.assertRaisesRegex(
            poller_module.PollerError, "identity mismatch"
        ):
            monitor.run_cycle()

        self.assertEqual(len(fake.calls), 1)

    def test_running_instance_with_blocked_preflight_sets_no_timer(self):
        fake = FakeRunner(
            [
                (
                    "get-instance",
                    poller_module.CommandResult(
                        0, instance_payload("Running"), ""
                    ),
                ),
                (
                    "proxyclient config",
                    poller_module.CommandResult(0, "", ""),
                ),
                (
                    "ssh",
                    poller_module.CommandResult(
                        20, "PREFLIGHT_BLOCKED gpu_busy:123\n", ""
                    ),
                ),
            ]
        )

        outcome = self.make_poller(fake).run_cycle()

        self.assertEqual(outcome, "blocked")
        self.assertFalse(
            any("shutdown-timer" in " ".join(call[0]) for call in fake.calls)
        )

    def test_new_dynamics_launch_gets_twelve_hour_timer(self):
        fake = FakeRunner(
            [
                (
                    "get-instance",
                    poller_module.CommandResult(
                        0, instance_payload("Running"), ""
                    ),
                ),
                (
                    "proxyclient config",
                    poller_module.CommandResult(0, "", ""),
                ),
                (
                    "ssh",
                    poller_module.CommandResult(
                        0, "DYNAMICS_STARTED pid=4321\n", ""
                    ),
                ),
                (
                    "delete-instance-shutdown-timer",
                    poller_module.CommandResult(0, "{}", ""),
                ),
                (
                    "create-instance-shutdown-timer",
                    ok_json({"Success": True}),
                ),
            ]
        )

        outcome = self.make_poller(fake).run_cycle()

        self.assertEqual(outcome, "started")
        proxy_call = next(
            call for call in fake.calls if "proxyclient" in call[0]
        )
        self.assertEqual(proxy_call[0], ["proxyclient", "config"])
        self.assertEqual(
            proxy_call[1],
            "\n".join(
                [
                    "",
                    poller_module.REGION,
                    "test-access-key-id",
                    "test-access-key-secret",
                    "test-sts-token",
                    "",
                ]
            ),
        )
        timer_call = next(
            call
            for call in fake.calls
            if "create-instance-shutdown-timer" in call[0]
        )
        self.assertIn(str(poller_module.SHUTDOWN_12H_MS), timer_call[0])

    def test_existing_dynamics_retains_timer_above_four_hours(self):
        remaining = poller_module.REFRESH_THRESHOLD_MS + 1
        fake = FakeRunner(
            [
                (
                    "get-instance",
                    poller_module.CommandResult(
                        0, instance_payload("Running"), ""
                    ),
                ),
                (
                    "proxyclient config",
                    poller_module.CommandResult(0, "", ""),
                ),
                (
                    "ssh",
                    poller_module.CommandResult(
                        0, "DYNAMICS_ALREADY_RUNNING pid=4321\n", ""
                    ),
                ),
                (
                    "get-instance-shutdown-timer",
                    ok_json({"RemainingTimeInMs": remaining}),
                ),
            ]
        )

        outcome = self.make_poller(fake).run_cycle()

        self.assertEqual(outcome, "running")
        self.assertFalse(
            any(
                "create-instance-shutdown-timer" in call[0]
                for call in fake.calls
            )
        )

    def test_missing_sts_token_blocks_before_proxyclient_or_ssh(self):
        fake = FakeRunner(
            [
                (
                    "get-instance",
                    poller_module.CommandResult(
                        0, instance_payload("Running"), ""
                    ),
                )
            ]
        )
        monitor = self.make_poller(fake)
        assert monitor.config.aliyun_config_path is not None
        monitor.config.aliyun_config_path.write_text(
            json.dumps(
                {
                    "profiles": [
                        {
                            "name": poller_module.PROFILE,
                            "access_key_id": "test-access-key-id",
                            "access_key_secret": "test-access-key-secret",
                            "sts_token": "",
                            "region_id": poller_module.REGION,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            poller_module.PollerError, "missing refreshed STS fields: sts_token"
        ):
            monitor.run_cycle()

        self.assertEqual(len(fake.calls), 1)

    def test_proxyclient_failure_does_not_expose_credentials(self):
        fake = FakeRunner(
            [
                (
                    "get-instance",
                    poller_module.CommandResult(
                        0, instance_payload("Running"), ""
                    ),
                ),
                (
                    "proxyclient config",
                    poller_module.CommandResult(
                        1,
                        "",
                        "bad config test-access-key-secret test-sts-token",
                    ),
                ),
            ]
        )
        monitor = self.make_poller(fake)

        with self.assertRaises(poller_module.PollerError) as raised:
            monitor.run_cycle()

        error = str(raised.exception)
        self.assertIn("sensitive command output redacted", error)
        self.assertNotIn("test-access-key-secret", error)
        self.assertNotIn("test-sts-token", error)

    def test_remote_script_contains_all_frozen_inputs_and_atomic_pid(self):
        script = poller_module.remote_launch_script()

        self.assertIn(poller_module.REMOTE_COMMIT, script)
        self.assertIn(poller_module.REMOTE_RUNNER_SHA256, script)
        self.assertIn("env-steps-001048576.msgpack", script)
        self.assertIn("env-steps-006291456.msgpack", script)
        self.assertIn("env-steps-012582912.msgpack", script)
        self.assertIn("env-steps-025165824.msgpack", script)
        self.assertIn("n16p6m/checkpoints/19999/_CHECKPOINT_METADATA", script)
        self.assertIn('mv "$pid_tmp" "$RUN_ROOT/pipeline.pid"', script)
        self.assertIn("WANDB_MODE=offline", script)


if __name__ == "__main__":
    unittest.main()
