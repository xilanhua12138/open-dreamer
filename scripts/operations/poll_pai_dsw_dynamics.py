#!/usr/bin/env python3
"""One-shot PAI-DSW capacity poller for the frozen CoinRun dynamics run.

The process is intentionally one-shot so launchd can call it frequently without
leaving a sleeping process behind. A local advisory lock prevents overlap.
Only the existing A10 instance may be started. When it becomes ready, a remote
preflight atomically starts the preregistered CR-DYN-0008/0009 pipeline.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


INSTANCE_ID = "dsw-ljvgtooejpvpxmhhc6"
REGION = "cn-shanghai"
PROFILE = "open-dreamer"
EXPECTED_GPU = "NVIDIA A10"
EXPECTED_SPEC = "ecs.gn7i-c8g1.2xlarge"
SSH_ALIAS = "dsw-ljvgtooejpvpxmhhc6"

REMOTE_ROOT = "/mnt/workspace/open-dreamer-dynamics-checkpoint-mixtures"
REMOTE_RUN_ROOT = (
    f"{REMOTE_ROOT}/logs/coinrun-dynamics-checkpoint-mixtures-v1"
)
REMOTE_DATA_ROOT = "/mnt/workspace/datasets/coinrun-ppo-checkpoint-mixtures-v1"
REMOTE_COMMIT = "8711cf8e4100e614ff8f0e406838b72ef8ca02b4"
REMOTE_RUNNER = (
    "scripts/experiments/coinrun/"
    "run_coinrun_dynamics_checkpoint_mixtures.sh"
)
REMOTE_RUNNER_SHA256 = (
    "80258b1a02125ef02b0104b7837e626c32db00cc240f2305f115c3f0195abc18"
)
REMOTE_DEMO_ROOT = "/mnt/workspace/open-dreamer-dynamics-live-demo"
REMOTE_DEMO_COMMIT = "369907703807021ee6c2450d052a0ea78eb79400"
REMOTE_DYNAMICS_PYTHON = (
    "/mnt/workspace/open-dreamer-tokenizer-quality-first/.venv/bin/python"
)
REMOTE_EVAL_DATA = f"{REMOTE_DATA_ROOT}/eval-final-policy"
LOCAL_DEMO_URL = "http://127.0.0.1:7860"

SHUTDOWN_12H_MS = 12 * 60 * 60 * 1000
SHUTDOWN_8H_MS = 8 * 60 * 60 * 1000
REFRESH_THRESHOLD_MS = 4 * 60 * 60 * 1000


class PollerError(RuntimeError):
    """A bounded operational error that should be recorded and retried later."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Config:
    state_dir: Path
    start_wait_seconds: int = 90
    status_poll_seconds: int = 10
    command_timeout_seconds: int = 45
    aliyun_config_path: Path | None = None


CommandRunner = Callable[[Sequence[str], str | None, int], CommandResult]


def default_command_runner(
    args: Sequence[str], input_text: str | None, timeout_seconds: int
) -> CommandResult:
    completed = subprocess.run(
        list(args),
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


class Poller:
    def __init__(
        self,
        config: Config,
        *,
        command_runner: CommandRunner = default_command_runner,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.command_runner = command_runner
        self.sleeper = sleeper
        self.monotonic = monotonic
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.config.state_dir / "events.jsonl"

    def emit(self, event: str, **fields: object) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def run_command(
        self,
        args: Sequence[str],
        *,
        input_text: str | None = None,
        allow_failure: bool = False,
        timeout_seconds: int | None = None,
        redact_failure_detail: bool = False,
    ) -> CommandResult:
        result = self.command_runner(
            args,
            input_text,
            timeout_seconds or self.config.command_timeout_seconds,
        )
        if result.returncode != 0 and not allow_failure:
            detail = (
                "sensitive command output redacted"
                if redact_failure_detail
                else (result.stderr or result.stdout).strip()
            )
            raise PollerError(
                f"command failed rc={result.returncode}: {args[0]} "
                f"{args[1] if len(args) > 1 else ''}: {detail[:500]}"
            )
        return result

    @staticmethod
    def aliyun_args(action: str, *extra: str) -> list[str]:
        return [
            "aliyun",
            "pai-dsw",
            action,
            "--region",
            REGION,
            "--instance-id",
            INSTANCE_ID,
            "--profile",
            PROFILE,
            *extra,
        ]

    def get_instance(self) -> dict[str, object]:
        result = self.run_command(self.aliyun_args("get-instance"))
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise PollerError("get-instance returned non-JSON output") from exc
        if payload.get("InstanceId") != INSTANCE_ID:
            raise PollerError("get-instance returned the wrong instance")
        gpu = (payload.get("RequestedResource") or {}).get("GPUType")
        spec = payload.get("EcsSpec")
        if gpu != EXPECTED_GPU or spec != EXPECTED_SPEC:
            raise PollerError(
                f"instance identity mismatch gpu={gpu!r} spec={spec!r}"
            )
        return payload

    def request_start_once(self) -> None:
        result = self.run_command(self.aliyun_args("start-instance"))
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise PollerError("start-instance returned non-JSON output") from exc
        if payload.get("Success") is not True:
            raise PollerError("start-instance did not report success")
        self.emit("start_requested", request_id=payload.get("RequestId"))

    def wait_for_running(self) -> str:
        deadline = self.monotonic() + self.config.start_wait_seconds
        last_status = "Starting"
        first = True
        while first or self.monotonic() < deadline:
            first = False
            payload = self.get_instance()
            last_status = str(payload.get("Status"))
            self.emit("status_observed", status=last_status)
            if last_status in {"Running", "Failed", "Stopped"}:
                return last_status
            self.sleeper(self.config.status_poll_seconds)
        return last_status

    def get_shutdown_remaining_ms(self) -> int | None:
        result = self.run_command(
            self.aliyun_args("get-instance-shutdown-timer"),
            allow_failure=True,
        )
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        remaining = payload.get("RemainingTimeInMs")
        return int(remaining) if remaining is not None else None

    def set_shutdown_timer(self, remaining_ms: int) -> None:
        self.run_command(
            self.aliyun_args("delete-instance-shutdown-timer"),
            allow_failure=True,
        )
        result = self.run_command(
            self.aliyun_args(
                "create-instance-shutdown-timer",
                "--remaining-time-in-ms",
                str(remaining_ms),
            )
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise PollerError(
                "create-instance-shutdown-timer returned non-JSON output"
            ) from exc
        if payload.get("Success") is not True:
            raise PollerError("shutdown timer creation did not report success")
        self.emit("shutdown_timer_set", remaining_ms=remaining_ms)

    def ensure_running_timer(self) -> None:
        remaining = self.get_shutdown_remaining_ms()
        if remaining is None or remaining < REFRESH_THRESHOLD_MS:
            self.set_shutdown_timer(SHUTDOWN_8H_MS)
        else:
            self.emit("shutdown_timer_retained", remaining_ms=remaining)

    def refresh_proxyclient_credentials(self) -> None:
        aliyun_config_path = (
            self.config.aliyun_config_path
            or Path.home() / ".aliyun" / "config.json"
        )
        try:
            aliyun_config = json.loads(
                aliyun_config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise PollerError(
                f"cannot read Aliyun profile config: {aliyun_config_path}"
            ) from exc

        profile = next(
            (
                item
                for item in aliyun_config.get("profiles", [])
                if item.get("name") == PROFILE
            ),
            None,
        )
        if profile is None:
            raise PollerError(f"Aliyun profile {PROFILE!r} is missing")

        required_fields = (
            "access_key_id",
            "access_key_secret",
            "sts_token",
        )
        missing = [
            field for field in required_fields if not profile.get(field)
        ]
        if missing:
            raise PollerError(
                "Aliyun profile is missing refreshed STS fields: "
                + ", ".join(missing)
            )
        profile_region = str(profile.get("region_id") or REGION)
        if profile_region != REGION:
            raise PollerError(
                f"Aliyun profile region mismatch: {profile_region!r}"
            )

        credential_input = "\n".join(
            [
                "",
                profile_region,
                str(profile["access_key_id"]),
                str(profile["access_key_secret"]),
                str(profile["sts_token"]),
                "",
            ]
        )
        self.run_command(
            ["proxyclient", "config"],
            input_text=credential_input,
            redact_failure_detail=True,
        )
        self.emit(
            "proxy_credentials_refreshed",
            sts_expiration=profile.get("sts_expiration"),
        )

    def validate_local_demo(self, *, perform_step: bool) -> dict[str, object] | None:
        health_result = self.run_command(
            [
                "curl",
                "-fsS",
                "--max-time",
                "10",
                f"{LOCAL_DEMO_URL}/health",
            ],
            allow_failure=True,
        )
        if health_result.returncode != 0:
            return None
        try:
            health = json.loads(health_result.stdout)
        except json.JSONDecodeError:
            return None
        if health.get("ok") is not True or not health.get("model"):
            return None

        result: dict[str, object] = {
            "model": str(health["model"]),
            "health_ok": True,
        }
        if perform_step:
            step_result = self.run_command(
                [
                    "curl",
                    "-fsS",
                    "--max-time",
                    "90",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    '{"action":4}',
                    f"{LOCAL_DEMO_URL}/api/step",
                ],
                allow_failure=True,
                timeout_seconds=95,
            )
            if step_result.returncode != 0:
                return None
            try:
                step = json.loads(step_result.stdout)
            except json.JSONDecodeError:
                return None
            frame = step.get("frame")
            if (
                step.get("action_id") != 4
                or not isinstance(step.get("step"), int)
                or not isinstance(frame, str)
                or not frame.startswith("data:image/png;base64,")
            ):
                return None
            result.update(
                {
                    "step": int(step["step"]),
                    "step_action_id": int(step["action_id"]),
                    "latency_ms": step.get("latency_ms"),
                }
            )
        return result

    def ensure_local_demo_tunnel(self) -> str:
        ready_path = self.config.state_dir / "LIVE_DEMO_READY.json"
        existing_ready = ready_path.is_file()
        validation = self.validate_local_demo(perform_step=not existing_ready)
        if validation is None:
            port_check = self.run_command(
                [
                    "/usr/sbin/lsof",
                    "-nP",
                    "-iTCP:7860",
                    "-sTCP:LISTEN",
                ],
                allow_failure=True,
            )
            if port_check.returncode == 0:
                raise PollerError(
                    "local port 7860 is occupied but demo health validation failed"
                )
            self.run_command(
                [
                    "ssh",
                    "-f",
                    "-N",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-L",
                    "7860:127.0.0.1:7860",
                    SSH_ALIAS,
                ],
                timeout_seconds=30,
            )
            for _ in range(12):
                self.sleeper(5)
                validation = self.validate_local_demo(perform_step=True)
                if validation is not None:
                    break
        if validation is None:
            raise PollerError("local demo tunnel did not pass health and step checks")

        if not existing_ready:
            payload = {
                "schema_version": "1.0",
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "url": LOCAL_DEMO_URL,
                **validation,
            }
            temp_path = ready_path.with_suffix(".json.tmp")
            temp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temp_path.replace(ready_path)
            self.emit("local_demo_verified", **validation)
        else:
            self.emit("local_demo_health_retained", model=validation["model"])
        return "demo_ready"

    def remote_preflight_and_launch(self) -> str:
        self.refresh_proxyclient_credentials()
        result = self.run_command(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                SSH_ALIAS,
                "bash -s",
            ],
            input_text=remote_launch_script(),
            allow_failure=True,
            timeout_seconds=90,
        )
        output = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        marker = next(
            (
                line.strip()
                for line in output.splitlines()
                if line.startswith("DYNAMICS_")
                or line.startswith("PREFLIGHT_BLOCKED")
            ),
            "",
        )
        if result.returncode not in {0, 20}:
            raise PollerError(
                f"remote preflight failed rc={result.returncode}: {output[:800]}"
            )
        if not marker:
            raise PollerError(
                f"remote preflight returned no status marker: {output[:800]}"
            )
        if marker.startswith("PREFLIGHT_BLOCKED"):
            self.emit("remote_preflight_blocked", detail=marker)
            return "blocked"
        if marker.startswith("DYNAMICS_STARTED"):
            self.emit("dynamics_started", detail=marker)
            self.set_shutdown_timer(SHUTDOWN_12H_MS)
            return "started"
        if marker.startswith("DYNAMICS_ALREADY_RUNNING"):
            self.emit("dynamics_already_running", detail=marker)
            self.ensure_running_timer()
            return "running"
        if marker.startswith("DYNAMICS_ALREADY_COMPLETE"):
            self.emit("dynamics_already_complete", detail=marker)
            return "complete"
        if marker.startswith("DYNAMICS_COMPLETE_DEMO_STARTED"):
            self.emit("dynamics_complete_demo_started", detail=marker)
            self.set_shutdown_timer(SHUTDOWN_12H_MS)
            return "demo_started"
        if marker.startswith("DYNAMICS_COMPLETE_DEMO_WARMING"):
            self.emit("dynamics_complete_demo_warming", detail=marker)
            return "demo_warming"
        if marker.startswith("DYNAMICS_COMPLETE_DEMO_READY"):
            self.emit("dynamics_complete_demo_ready", detail=marker)
            return self.ensure_local_demo_tunnel()
        raise PollerError(f"unrecognized remote marker: {marker}")

    def run_cycle(self) -> str:
        payload = self.get_instance()
        status = str(payload.get("Status"))
        self.emit("cycle_started", status=status)

        if status in {"Failed", "Stopped"}:
            self.request_start_once()
            status = self.wait_for_running()
        elif status == "Starting":
            status = self.wait_for_running()

        if status == "Running":
            return self.remote_preflight_and_launch()
        if status == "Failed":
            self.emit("capacity_unavailable")
            return "capacity_unavailable"
        if status == "Stopped":
            self.emit("start_did_not_progress")
            return "stopped"

        self.emit("instance_not_ready", status=status)
        return "starting"


def remote_launch_script() -> str:
    checkpoint_root = (
        "/mnt/workspace/open-dreamer-ppo-parity/logs/"
        "coinrun-ppo-official-parity-v1/ppo-train/checkpoints"
    )
    tokenizer_metadata = (
        "/mnt/workspace/open-dreamer-tokenizer-quality-first/logs/"
        "coinrun-tokenizer-quality-first-20k-20260729/runs/n16p6m/"
        "checkpoints/19999/_CHECKPOINT_METADATA"
    )
    checkpoints = " ".join(
        f'"{checkpoint_root}/env-steps-{step:09d}.msgpack"'
        for step in (1048576, 6291456, 12582912, 25165824)
    )
    return f"""\
set -Eeuo pipefail
ROOT="{REMOTE_ROOT}"
RUN_ROOT="{REMOTE_RUN_ROOT}"
DATA_ROOT="{REMOTE_DATA_ROOT}"
RUNNER="{REMOTE_RUNNER}"
EXPECTED_HEAD="{REMOTE_COMMIT}"
EXPECTED_RUNNER_SHA="{REMOTE_RUNNER_SHA256}"
DEMO_ROOT="{REMOTE_DEMO_ROOT}"
DEMO_EXPECTED_HEAD="{REMOTE_DEMO_COMMIT}"
DEMO_PYTHON="{REMOTE_DYNAMICS_PYTHON}"
DEMO_DATA="{REMOTE_EVAL_DATA}"
DEMO_PID_FILE="$RUN_ROOT/live-demo.pid"
DEMO_LOG_FILE="$RUN_ROOT/live-demo.log"
DEMO_READY_FILE="$RUN_ROOT/LIVE_DEMO_READY.json"
DEMO_SELECTION="$RUN_ROOT/live-demo-selection.json"

blocked() {{
  printf 'PREFLIGHT_BLOCKED %s\\n' "$1"
  exit 20
}}

test -d "$ROOT/.git" || test -f "$ROOT/.git" || blocked "missing_worktree"
cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD" || blocked "head_mismatch"
test "$(sha256sum "$RUNNER" | awk '{{print $1}}')" = "$EXPECTED_RUNNER_SHA" ||
  blocked "runner_hash_mismatch"
test -r experiments/CR-DYN-0008/manifest.json || blocked "missing_manifest_0008"
test -r experiments/CR-DYN-0009/manifest.json || blocked "missing_manifest_0009"
test -x /mnt/workspace/open-dreamer-ppo-collector/.venv-ppo/bin/python ||
  blocked "missing_ppo_python"
test -x /mnt/workspace/open-dreamer-tokenizer-quality-first/.venv/bin/python ||
  blocked "missing_dynamics_python"
test -f "{tokenizer_metadata}" || blocked "missing_tokenizer_checkpoint"
for checkpoint in {checkpoints}; do
  test -s "$checkpoint" || blocked "missing_ppo_checkpoint:$checkpoint"
done

start_or_check_demo() {{
  test -d "$DEMO_ROOT/.git" || test -f "$DEMO_ROOT/.git" ||
    blocked "missing_demo_worktree"
  test "$(git -C "$DEMO_ROOT" rev-parse HEAD)" = "$DEMO_EXPECTED_HEAD" ||
    blocked "demo_head_mismatch"
  test -z "$(git -C "$DEMO_ROOT" status --porcelain)" ||
    blocked "dirty_demo_worktree"
  test -x "$DEMO_PYTHON" || blocked "missing_demo_python"
  test -s "$DEMO_DATA/metadata.json" || blocked "missing_demo_dataset"

  demo_pid=""
  if test -f "$DEMO_PID_FILE"; then
    demo_pid="$(tr -cd '0-9' < "$DEMO_PID_FILE" 2>/dev/null || true)"
  fi
  if test -n "$demo_pid" && kill -0 "$demo_pid" 2>/dev/null; then
    if test -s "$DEMO_READY_FILE"; then
      model="$(
        "$DEMO_PYTHON" -c \
          'import json,sys; print(json.load(open(sys.argv[1]))["model"])' \
          "$DEMO_READY_FILE"
      )"
      printf 'DYNAMICS_COMPLETE_DEMO_READY model=%s pid=%s\\n' \
        "$model" "$demo_pid"
      return
    fi

    health_json="$RUN_ROOT/live-demo-health.json"
    step_json="$RUN_ROOT/live-demo-step.json"
    if curl -fsS --max-time 15 \
         http://127.0.0.1:7860/health >"$health_json" &&
       curl -fsS --max-time 90 \
         -H 'Content-Type: application/json' \
         -d '{{"action":4}}' \
         http://127.0.0.1:7860/api/step >"$step_json" &&
       "$DEMO_PYTHON" - "$health_json" "$step_json" "$DEMO_READY_FILE" <<'PY'
import json
import sys
from datetime import datetime, timezone

health = json.load(open(sys.argv[1], encoding="utf-8"))
step = json.load(open(sys.argv[2], encoding="utf-8"))
frame = step.get("frame")
assert health.get("ok") is True
assert health.get("model")
assert step.get("action_id") == 4
assert isinstance(step.get("step"), int)
assert isinstance(frame, str) and frame.startswith("data:image/png;base64,")
payload = {{
    "schema_version": "1.0",
    "verified_at": datetime.now(timezone.utc).isoformat(),
    "model": health["model"],
    "step": step["step"],
    "step_action_id": step["action_id"],
    "latency_ms": step.get("latency_ms"),
}}
with open(sys.argv[3] + ".tmp", "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\\n")
import os
os.replace(sys.argv[3] + ".tmp", sys.argv[3])
PY
    then
      model="$(
        "$DEMO_PYTHON" -c \
          'import json,sys; print(json.load(open(sys.argv[1]))["model"])' \
          "$DEMO_READY_FILE"
      )"
      printf 'DYNAMICS_COMPLETE_DEMO_READY model=%s pid=%s\\n' \
        "$model" "$demo_pid"
    else
      printf 'DYNAMICS_COMPLETE_DEMO_WARMING pid=%s\\n' "$demo_pid"
    fi
    return
  fi
  test ! -f "$DEMO_PID_FILE" || blocked "stale_demo_pid:$demo_pid"

  "$DEMO_PYTHON" \
    "$DEMO_ROOT/scripts/experiments/coinrun/select_best_coinrun_checkpoint.py" \
    --experiment "$RUN_ROOT/scale-ablation" \
    --output "$DEMO_SELECTION" >/dev/null
  model="$(
    "$DEMO_PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["best"]["name"])' \
      "$DEMO_SELECTION"
  )"
  checkpoint="$(
    "$DEMO_PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["best"]["checkpoint"])' \
      "$DEMO_SELECTION"
  )"
  test -d "$checkpoint" || blocked "missing_demo_checkpoint:$checkpoint"

  nohup env \
    PYTHONPATH="$DEMO_ROOT" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$DEMO_PYTHON" \
    "$DEMO_ROOT/scripts/experiments/coinrun/live_coinrun_demo.py" \
      --checkpoint "$checkpoint" \
      --dataset "$DEMO_DATA" \
      --model-name "$model" \
      --context 16 \
      --denoise-steps 4 \
      --host 127.0.0.1 \
      --port 7860 \
      >"$DEMO_LOG_FILE" 2>&1 < /dev/null &
  demo_pid=$!
  demo_pid_tmp="$DEMO_PID_FILE.tmp.$$"
  printf '%s\\n' "$demo_pid" >"$demo_pid_tmp"
  mv "$demo_pid_tmp" "$DEMO_PID_FILE"
  sleep 3
  kill -0 "$demo_pid" 2>/dev/null || blocked "demo_died_after_launch"
  printf 'DYNAMICS_COMPLETE_DEMO_STARTED pid=%s model=%s\\n' \
    "$demo_pid" "$model"
}}

mkdir -p "$RUN_ROOT"
existing_pid=""
if test -f "$RUN_ROOT/pipeline.pid"; then
  existing_pid="$(
    tr -cd '0-9' < "$RUN_ROOT/pipeline.pid" 2>/dev/null || true
  )"
  if test -n "$existing_pid" && kill -0 "$existing_pid" 2>/dev/null; then
    printf 'DYNAMICS_ALREADY_RUNNING pid=%s\\n' "$existing_pid"
    exit 0
  fi
fi
if test -f "$RUN_ROOT/STATUS" &&
   grep -q ' COMPLETE CHECKPOINT_MIXTURE_AND_DYNAMICS_SCALE_ABLATIONS$' \
     "$RUN_ROOT/STATUS"; then
  unexpected_runtime_dirty_path="$(
    git status --porcelain |
      cut -c4- |
      grep -Ev '^experiments/CR-DYN-000(8|9)/' |
      head -n 1 || true
  )"
  test -z "$unexpected_runtime_dirty_path" ||
    blocked "unexpected_runtime_dirty_path:$unexpected_runtime_dirty_path"
  start_or_check_demo
  exit 0
fi
test ! -f "$RUN_ROOT/pipeline.pid" || blocked "stale_pipeline_pid"
test -z "$(git status --porcelain)" || blocked "dirty_worktree"

gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
  sed '/^[[:space:]]*$/d' || true)"
test -z "$gpu_pids" || blocked "gpu_busy:$gpu_pids"

while IFS= read -r pid_file; do
  test "$pid_file" = "$RUN_ROOT/pipeline.pid" && continue
  pid="$(tr -cd '0-9' < "$pid_file" 2>/dev/null || true)"
  if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then
    blocked "other_pipeline:$pid_file:$pid"
  fi
done < <(
  find /mnt/workspace -type f -name pipeline.pid \
    \\( -path '*ppo*' -o -path '*dynamics*' \\) 2>/dev/null
)

nohup env \
  OPEN_DREAMER_ROOT="$ROOT" \
  OPEN_DREAMER_RUN_ROOT="$RUN_ROOT" \
  OPEN_DREAMER_DATA_ROOT="$DATA_ROOT" \
  WANDB_MODE=offline \
  bash "$RUNNER" >"$RUN_ROOT/bootstrap.log" 2>&1 < /dev/null &
pipeline_pid=$!
pid_tmp="$RUN_ROOT/pipeline.pid.tmp.$$"
printf '%s\\n' "$pipeline_pid" >"$pid_tmp"
mv "$pid_tmp" "$RUN_ROOT/pipeline.pid"
sleep 3
kill -0 "$pipeline_pid" 2>/dev/null || blocked "pipeline_died_after_launch"
printf 'DYNAMICS_STARTED pid=%s\\n' "$pipeline_pid"
"""


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--start-wait-seconds", type=int, default=90)
    parser.add_argument("--status-poll-seconds", type=int, default=10)
    parser.add_argument("--command-timeout-seconds", type=int, default=45)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = Config(
        state_dir=args.state_dir.expanduser().resolve(),
        start_wait_seconds=args.start_wait_seconds,
        status_poll_seconds=args.status_poll_seconds,
        command_timeout_seconds=args.command_timeout_seconds,
    )
    config.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = config.state_dir / "poller.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        poller = Poller(config)
        try:
            outcome = poller.run_cycle()
        except (PollerError, subprocess.TimeoutExpired, OSError) as exc:
            poller.emit("cycle_error", error=str(exc)[:1000])
            return 1
        poller.emit("cycle_finished", outcome=outcome)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
