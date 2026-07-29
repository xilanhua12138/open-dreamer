"""Structured runtime provenance and telemetry for durable experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0"
SAFE_ENVIRONMENT_KEYS = (
    "CONDA_DEFAULT_ENV",
    "CUDA_VERSION",
    "CUDA_VISIBLE_DEVICES",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_PLATFORM_NAME",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "WANDB_MODE",
)
PROXY_ENVIRONMENT_KEYS = (
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
)
SENSITIVE_ARGUMENT_MARKERS = (
    "api_key",
    "apikey",
    "access_key",
    "password",
    "secret",
    "token",
)
PACKAGE_NAMES = (
    "array-record",
    "flax",
    "grain",
    "hydra-core",
    "imageio",
    "jax",
    "jaxlib",
    "jaxlpips",
    "numpy",
    "optax",
    "orbax-checkpoint",
    "wandb",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 5.0,
) -> str | None:
    try:
        result = subprocess.run(
            list(args),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _optional_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"", "n/a", "[n/a]", "not supported", "[not supported]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _collect_git_identity(
    repository: Path,
    *,
    run_dir: Path,
    attempt_id: str,
) -> dict[str, Any]:
    root_text = _run_command(("git", "rev-parse", "--show-toplevel"), cwd=repository)
    if root_text is None:
        return {"available": False}
    root = Path(root_text)
    commit = _run_command(("git", "rev-parse", "HEAD"), cwd=root)
    branch = _run_command(("git", "branch", "--show-current"), cwd=root)
    status = _run_command(("git", "status", "--porcelain"), cwd=root)
    diff = _run_command(("git", "diff", "--binary", "HEAD"), cwd=root, timeout=20)
    patch = None
    if diff:
        patch_path = (
            run_dir / "runtime-identities" / f"{attempt_id}.source.patch"
        )
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(diff + "\n", encoding="utf-8")
        patch = {
            "uri": str(patch_path.relative_to(run_dir)),
            "sha256": sha256_file(patch_path),
            "bytes": patch_path.stat().st_size,
        }
    untracked_output = _run_command(
        ("git", "ls-files", "--others", "--exclude-standard"),
        cwd=root,
    )
    untracked_files = []
    for relative in sorted((untracked_output or "").splitlines()):
        path = root / relative
        if path.is_file():
            untracked_files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    source_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "commit": commit,
                "tracked_patch_sha256": patch["sha256"] if patch else None,
                "untracked_files": untracked_files,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "available": True,
        "repository_root": str(root),
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status),
        "reproducible_from_recorded_source": not untracked_files,
        "patch": patch,
        "untracked_files": untracked_files,
        "source_fingerprint_sha256": source_fingerprint,
    }


def _collect_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _collect_accelerator_identity() -> dict[str, Any]:
    query = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    gpus = []
    if query:
        for line in query.splitlines():
            fields = [field.strip() for field in line.split(",", maxsplit=4)]
            if len(fields) == 5:
                index, gpu_uuid, name, driver, memory_total_mb = fields
                gpus.append(
                    {
                        "index": int(index),
                        "uuid": gpu_uuid,
                        "name": name,
                        "driver_version": driver,
                        "memory_total_mb": int(memory_total_mb),
                    }
                )
    try:
        import jax

        jax_devices = [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "process_index": int(device.process_index),
            }
            for device in jax.devices()
        ]
    except Exception as exc:  # Runtime identity must survive partial GPU setup.
        jax_devices = [{"error_type": type(exc).__name__, "error": str(exc)}]
    return {
        "available": bool(gpus or jax_devices),
        "gpus": gpus,
        "jax_devices": jax_devices,
        "cuda_toolkit": _run_command(("nvcc", "--version")),
    }


def _redact_argv(argv: Sequence[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        key, separator, _ = argument.partition("=")
        lowered = key.lower()
        if any(marker in lowered for marker in SENSITIVE_ARGUMENT_MARKERS):
            if separator:
                result.append(f"{key}=<redacted>")
            else:
                result.append(argument)
                redact_next = True
        else:
            result.append(argument)
    return result


def _get_or_create_run_id(run_dir: Path) -> str:
    path = run_dir / "run-id.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
        raise ValueError(f"{path}: run_id must be a non-empty string")
    run_id = str(uuid.uuid4())
    atomic_write_json(path, {"schema_version": SCHEMA_VERSION, "run_id": run_id})
    return run_id


def collect_runtime_identity(
    run_dir: Path,
    *,
    config: Any,
    environ: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
    repository: Path | None = None,
) -> dict[str, Any]:
    """Collect one immutable process-attempt identity and write it atomically."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    environ = os.environ if environ is None else environ
    repository = (
        Path(__file__).resolve().parents[1] if repository is None else repository
    )
    run_id = _get_or_create_run_id(run_dir)
    attempt_id = str(uuid.uuid4())
    config_safe = _json_safe(config)
    canonical_config = json.dumps(
        config_safe, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    lockfiles = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = repository / name
        if path.is_file():
            lockfiles[name] = sha256_file(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "collected_at": utc_now(),
        "command": _redact_argv(sys.argv if argv is None else argv),
        "config_sha256": hashlib.sha256(canonical_config).hexdigest(),
        "source": _collect_git_identity(
            repository,
            run_dir=run_dir,
            attempt_id=attempt_id,
        ),
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": platform.node(),
        },
        "packages": _collect_package_versions(),
        "accelerator": _collect_accelerator_identity(),
        "dependency_files": lockfiles,
        "environment": {
            "values": {
                key: environ[key]
                for key in SAFE_ENVIRONMENT_KEYS
                if environ.get(key) is not None
            },
            "proxy_configured": any(environ.get(key) for key in PROXY_ENVIRONMENT_KEYS),
        },
    }
    attempt_path = run_dir / "runtime-identities" / f"{attempt_id}.json"
    atomic_write_json(attempt_path, payload)
    atomic_write_json(run_dir / "runtime-identity.json", payload)
    return payload


class RunRecorder:
    """Durable local run state shared by local and remote loggers."""

    def __init__(
        self,
        run_dir: Path,
        *,
        max_steps: int | None,
        progress_interval_seconds: float = 30.0,
        system_metrics_interval_seconds: float = 60.0,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.progress_interval_seconds = max(
            0.0, float(progress_interval_seconds)
        )
        self.system_metrics_interval_seconds = max(
            0.0, float(system_metrics_interval_seconds)
        )
        self.identity: dict[str, Any] | None = None
        self.run_id: str | None = None
        self.attempt_id: str | None = None
        self._last_system_metrics_at = float("-inf")
        self._last_completed_updates: int | None = None
        self._last_progress_at: float | None = None
        self._last_progress_updates: int | None = None
        self._finished = False

    @property
    def state_path(self) -> Path:
        return self.run_dir / "run-state.json"

    def start(self, *, config: Any) -> None:
        if self.identity is not None:
            raise RuntimeError("RunRecorder.start() may only be called once")
        self.identity = collect_runtime_identity(self.run_dir, config=config)
        self.run_id = str(self.identity["run_id"])
        self.attempt_id = str(self.identity["attempt_id"])
        atomic_write_json(self.run_dir / "runtime-identity.json", self.identity)
        attempt_path = (
            self.run_dir / "runtime-identities" / f"{self.attempt_id}.json"
        )
        atomic_write_json(attempt_path, self.identity)
        previous = {}
        if self.state_path.is_file():
            previous = json.loads(self.state_path.read_text(encoding="utf-8"))
        started_at = previous.get("started_at") or utc_now()
        self._last_completed_updates = previous.get("last_completed_updates")
        state = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "state": "RUNNING",
            "started_at": started_at,
            "attempt_started_at": utc_now(),
            "updated_at": utc_now(),
            "max_steps": self.max_steps,
            "last_step": previous.get("last_step"),
            "last_completed_updates": self._last_completed_updates,
            "failure": None,
        }
        atomic_write_json(self.state_path, state)
        self._telemetry("run_started", state="RUNNING")

    def _require_started(self) -> None:
        if self.identity is None or self.run_id is None or self.attempt_id is None:
            raise RuntimeError("RunRecorder.start() must run before recording")

    def _telemetry(self, event_type: str, **fields: Any) -> None:
        self._require_started()
        append_jsonl(
            self.run_dir / "telemetry.jsonl",
            {
                "schema_version": SCHEMA_VERSION,
                "at": utc_now(),
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                "type": event_type,
                **fields,
            },
        )

    def _write_progress_state(self, *, step: int, completed_updates: int) -> None:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state.update(
            {
                "state": "RUNNING",
                "updated_at": utc_now(),
                "last_step": step,
                "last_completed_updates": completed_updates,
            }
        )
        atomic_write_json(self.state_path, state)

    def log_metrics(
        self,
        *,
        step: int,
        prefix: str,
        metrics: Mapping[str, Any],
    ) -> None:
        self._require_started()
        completed_updates = step + 1
        clean_metrics = {
            str(key): _json_safe(value)
            for key, value in sorted(metrics.items(), key=lambda item: item[0])
        }
        append_jsonl(
            self.run_dir / "metrics.jsonl",
            {
                "schema_version": SCHEMA_VERSION,
                "at": utc_now(),
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                "step": step,
                "completed_updates": completed_updates,
                "prefix": prefix,
                "metrics": clean_metrics,
            },
        )
        self.observe_progress(step=step)

    def observe_progress(self, *, step: int) -> None:
        """Persist a rate-limited heartbeat independent of metric frequency."""

        self._require_started()
        completed_updates = step + 1
        now = time.monotonic()
        is_final = self.max_steps is not None and completed_updates >= self.max_steps
        if (
            self._last_progress_at is not None
            and now - self._last_progress_at < self.progress_interval_seconds
            and not is_final
        ):
            return
        if (
            self._last_progress_updates is None
            or completed_updates > self._last_progress_updates
        ):
            steps_per_second = None
            eta_seconds = None
            if (
                self._last_progress_at is not None
                and self._last_progress_updates is not None
                and now > self._last_progress_at
            ):
                steps_per_second = (
                    completed_updates - self._last_progress_updates
                ) / (now - self._last_progress_at)
                if (
                    self.max_steps is not None
                    and steps_per_second is not None
                    and steps_per_second > 0
                    and completed_updates < self.max_steps
                ):
                    eta_seconds = (
                        self.max_steps - completed_updates
                    ) / steps_per_second
            self._telemetry(
                "progress",
                step=step,
                completed_updates=completed_updates,
                max_steps=self.max_steps,
                steps_per_second=steps_per_second,
                eta_seconds=eta_seconds,
            )
            self._last_progress_at = now
            self._last_progress_updates = completed_updates
            self._last_completed_updates = completed_updates
            self._write_progress_state(
                step=step,
                completed_updates=completed_updates,
            )
        if now - self._last_system_metrics_at >= self.system_metrics_interval_seconds:
            self._telemetry(
                "system",
                step=step,
                completed_updates=completed_updates,
                metrics=self._collect_system_metrics(),
            )
            self._last_system_metrics_at = now

    def register_artifact(self, *, step: int, key: str, path: Path) -> dict[str, Any]:
        self._require_started()
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"artifact does not exist: {path}")
        try:
            uri = str(path.relative_to(self.run_dir.resolve()))
        except ValueError:
            uri = str(path)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "at": utc_now(),
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "step": step,
            "completed_updates": step + 1,
            "key": key,
            "uri": uri,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        append_jsonl(self.run_dir / "artifacts.jsonl", payload)
        return payload

    def _collect_system_metrics(self) -> dict[str, Any]:
        disk = shutil.disk_usage(self.run_dir)
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(max_rss * (1 if sys.platform == "darwin" else 1024))
        payload: dict[str, Any] = {
            "process_max_rss_bytes": rss_bytes,
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
        }
        query = _run_command(
            (
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            )
        )
        if query:
            gpu_rows = []
            for line in query.splitlines():
                fields = [field.strip() for field in line.split(",", maxsplit=5)]
                if len(fields) != 6:
                    continue
                index, utilization, used, total, temperature, power = fields
                gpu_rows.append(
                    {
                        "index": int(index),
                        "utilization_percent": _optional_float(utilization),
                        "memory_used_mb": _optional_float(used),
                        "memory_total_mb": _optional_float(total),
                        "temperature_c": _optional_float(temperature),
                        "power_w": _optional_float(power),
                    }
                )
            payload["gpus"] = gpu_rows
        return payload

    def finish(self, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._require_started()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        terminal_state = "FAILED" if error is not None else "COMPLETED"
        failure = (
            {
                "type": type(error).__name__,
                "message": str(error),
            }
            if error is not None
            else None
        )
        state.update(
            {
                "state": terminal_state,
                "updated_at": utc_now(),
                "ended_at": utc_now(),
                "failure": failure,
            }
        )
        atomic_write_json(self.state_path, state)
        self._telemetry(
            "run_failed" if error is not None else "run_completed",
            state=terminal_state,
            last_completed_updates=self._last_completed_updates,
            failure=failure,
        )
        self._finished = True
