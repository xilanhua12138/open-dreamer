"""Fixed held-out validation helpers for dynamics training."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from dreamer.data import build_iterator


def periodic_rollout_names(*, include_diffusion: bool) -> tuple[str, ...]:
    if include_diffusion:
        return (
            "online_diffusion",
            "ema_diffusion",
            "online_shortcut",
            "ema_shortcut",
        )
    return ("online_shortcut", "ema_shortcut")


def build_fixed_validation_batch(
    dataset_cfg: Any,
    *,
    validation_array_record_path: str,
    validation_seed: int,
    validation_batch_size: int,
    validation_sequence_length: int,
    device: Any,
    dtype: Any,
    iterator_builder: Callable[..., Any] = build_iterator,
) -> dict[str, Any]:
    """Materialize one immutable held-out batch for every periodic evaluation."""

    if not validation_array_record_path:
        raise ValueError("validation_array_record_path must be non-empty")
    if min(validation_batch_size, validation_sequence_length) <= 0:
        raise ValueError(
            "validation_batch_size and validation_sequence_length must be positive"
        )

    validation_cfg = copy.deepcopy(dataset_cfg)
    validation_cfg.array_record_path = validation_array_record_path
    validation_cfg.p_include_reward = 0.0
    validation_cfg.dataloader_cfg.B = validation_batch_size
    validation_cfg.dataloader_cfg.num_workers = 0

    iterator = iterator_builder(
        validation_cfg,
        seed=validation_seed,
        device=device,
        dtype=dtype,
        seq_len=validation_sequence_length,
        return_actions=True,
    )
    return next(iter(iterator))
