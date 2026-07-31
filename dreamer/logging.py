from pathlib import Path
from typing import Any, Dict, Literal, Optional
import os
import re

import jax

from dreamer.configs import LoggerConfig
from dreamer.experiment_runtime import RunRecorder, atomic_write_json


class Logger:
    """Durable local logger that can be mirrored by a remote backend."""
    def __init__(
        self,
        logger_cfg: LoggerConfig,
        max_steps: Optional[int] = None,
        config: Any = None,
        dir: Optional[str] = None,
    ):
        self.log_every = logger_cfg.log_every
        self.max_steps = logger_cfg.max_steps if max_steps is None else max_steps
        self.config = config
        self.dir = dir
        self._initialized = False
        self.recorder = (
            RunRecorder(
                Path(dir),
                max_steps=self.max_steps,
                progress_interval_seconds=(
                    logger_cfg.telemetry_progress_every_seconds
                ),
                system_metrics_interval_seconds=(
                    logger_cfg.telemetry_system_every_seconds
                ),
            )
            if dir is not None
            else None
        )

    def should_log(self, step: int) -> bool:
        if self.log_every <= 0:
            return False
        is_periodic = (step % self.log_every == 0)
        is_last = (self.max_steps is not None) and (step == self.max_steps - 1)
        return is_periodic or is_last

    def __enter__(self):
        if self.recorder is not None:
            self.recorder.start(config=self.config)
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.recorder is not None:
            self.recorder.finish(exc_val)
        self._initialized = False
        return False

    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "train/",
    ):
        clean_metrics = self._convert_metrics(metrics)
        if self.recorder is not None:
            self.recorder.log_metrics(
                step=step,
                prefix=prefix,
                metrics=clean_metrics,
            )

    def observe_step(self, step: int) -> None:
        if self.recorder is not None:
            self.recorder.observe_progress(step=step)

    def log_image(
        self,
        step: int,
        key: str,
        image: Any,
        caption: Optional[str] = None,
    ) -> Path | None:
        if self.recorder is None:
            return Path(image) if isinstance(image, (str, Path)) else None
        if isinstance(image, (str, Path)):
            path = Path(image)
        else:
            import imageio.v3 as iio
            import numpy as np

            safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", key).strip("-")
            path = Path(self.dir) / "media" / safe_key / f"step-{step:08d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(path, np.asarray(image))
        self.recorder.register_artifact(step=step, key=key, path=path)
        return path

    def log_video(
        self,
        step: int,
        key: str,
        video_path: Path,
        format: Literal["gif", "mp4", "webm", "ogg"] | None = "mp4",
        **kwargs,
    ) -> Path:
        path = Path(video_path)
        if self.recorder is not None:
            self.recorder.register_artifact(step=step, key=key, path=path)
        return path

    def log(self, step: int, metrics: Dict[str, Any], pbar: Any = None, prefix: str = "train/", float_fmt: str = ".4f", pbar_filter: Optional[str] = None):
        if not self.should_log(step):
            return

        # Convert JAX arrays to Python scalars
        clean_metrics = self._convert_metrics(metrics)

        # Update tqdm progress bar (all loggers support this)
        if pbar is not None:
            self._update_pbar(pbar, clean_metrics, float_fmt, pbar_filter)

        # Log to backend
        self.log_metrics(step, clean_metrics, prefix)

    def _convert_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JAX arrays and other types to Python scalars."""
        clean_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                v = v.item()

            if isinstance(v, (float, int)):
                clean_metrics[k] = v
            else:
                try:
                    clean_metrics[k] = float(v)
                except (ValueError, TypeError):
                    clean_metrics[k] = v
        return clean_metrics

    def _update_pbar(
        self,
        pbar: Any,
        metrics: Dict[str, Any],
        float_fmt: str,
        pbar_filter: Optional[str],
    ):
        """Update tqdm progress bar with filtered metrics."""
        filtered_metrics = metrics

        # Apply regex filter if provided
        if pbar_filter is not None:
            pattern = re.compile(pbar_filter)
            filtered_metrics = {k: v for k, v in metrics.items() if pattern.search(k)}

        # Format values for display
        postfix_data = {}
        for k, v in filtered_metrics.items():
            if isinstance(v, float):
                postfix_data[k] = f"{v:{float_fmt}}"
            else:
                postfix_data[k] = str(v)

        pbar.set_postfix(**postfix_data)


class WandbLogger(Logger):
    def __init__(
        self,
        logger_cfg: LoggerConfig,
        config: Any = None,
        dir: Optional[str] = None,
    ):
        super().__init__(
            logger_cfg=logger_cfg,
            config=config,
            dir=dir,
        )
        self.entity = logger_cfg.wandb_entity
        self.project = logger_cfg.wandb_project
        self.name = logger_cfg.run_name
        self.group = logger_cfg.wandb_group
        self.tags = list(logger_cfg.wandb_tags)
        self.mode = logger_cfg.wandb_mode
        self._run = None

    def __enter__(self):
        import wandb
        super().__enter__()
        try:
            self._run = wandb.init(
                entity=self.entity,
                project=self.project,
                name=self.name,
                group=self.group,
                tags=self.tags,
                mode=self.mode,
                id=self.recorder.run_id if self.recorder is not None else None,
                resume="allow",
                config=self.config,
                dir=self.dir,
                save_code=True,
            )
            if self.dir is not None:
                wandb_identity_path = Path(self.dir) / "wandb-run.json"
                atomic_write_json(
                    wandb_identity_path,
                    {
                        "schema_version": "1.0",
                        "run_id": self._run.id,
                        "name": self._run.name,
                        "entity": self._run.entity,
                        "project": self._run.project,
                        "group": self.group,
                        "mode": self.mode,
                        "url": self._run.url,
                    },
                )
                if self.recorder is not None:
                    self.recorder.register_artifact(
                        step=-1,
                        key="runtime/wandb_run",
                        path=wandb_identity_path,
                    )
        except BaseException as exc:
            if self.recorder is not None:
                self.recorder.finish(exc)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        finish_error = None
        if self._run is not None:
            import wandb
            try:
                wandb.finish(exit_code=1 if exc_val is not None else 0)
            except BaseException as error:
                finish_error = error
        effective_error = exc_val if exc_val is not None else finish_error
        super().__exit__(
            type(effective_error) if effective_error is not None else None,
            effective_error,
            (
                effective_error.__traceback__
                if effective_error is not None
                else exc_tb
            ),
        )
        self._run = None
        if finish_error is not None and exc_val is None:
            raise finish_error
        return False

    def log_metrics(self, step: int, metrics: Dict[str, Any], prefix: str = "train/"):
        super().log_metrics(step, metrics, prefix)
        clean_metrics = self._convert_metrics(metrics)
        wandb_data = {f"{prefix}{k}": v for k, v in clean_metrics.items()}
        wandb_data[f"{prefix}step"] = step
        self._run.log(wandb_data, step=step)

    def log_image(self, step: int, key: str, image: Any, caption: Optional[str] = None):
        import wandb
        import numpy as np
        local_path = super().log_image(step, key, image, caption)
        wandb_image = (
            str(local_path)
            if local_path is not None
            else np.asarray(image)
        )
        self._run.log(
            {key: wandb.Image(wandb_image, caption=caption)},
            step=step,
        )
        return local_path

    def log_video(self, step: int, key: str, video_path: Path, format: Literal["gif", "mp4", "webm", "ogg"] | None = "mp4", **kwargs):
        import wandb
        local_path = super().log_video(
            step,
            key,
            video_path,
            format=format,
            **kwargs,
        )
        self._run.log(
            {key: wandb.Video(str(local_path), format=format, **kwargs)},
            step=step,
        )
        return local_path


def build_logger(
    logger_cfg: LoggerConfig,
    config: Any = None,
    dir: Optional[str] = None,
) -> Logger:
    def _is_primary_process() -> bool:
        is_initialized = getattr(jax.distributed, "is_initialized", None)
        if is_initialized is not None and is_initialized():
            return jax.process_index() == 0
        rank_env = (
            os.environ.get("JAX_PROCESS_INDEX")
            or os.environ.get("RANK")
            or os.environ.get("SLURM_PROCID")
            or os.environ.get("OMPI_COMM_WORLD_RANK")
            or "0"
        )
        return int(rank_env) == 0

    is_primary_process = _is_primary_process()
    use_wandb = logger_cfg.use_wandb and is_primary_process
    if use_wandb:
        return WandbLogger(
            logger_cfg=logger_cfg,
            config=config,
            dir=dir,
        )
    else:
        return Logger(
            logger_cfg=logger_cfg,
            config=config,
            dir=dir if is_primary_process else None,
        )
