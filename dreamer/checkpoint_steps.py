"""Pure helpers for selecting and retaining exact training checkpoints."""

from __future__ import annotations

from collections.abc import Iterable


def resolve_checkpoint_step(
    requested_step: int | None,
    available_steps: Iterable[int],
) -> int:
    """Resolve an explicit checkpoint step, defaulting to the latest one."""
    available = sorted({int(step) for step in available_steps})
    if not available:
        raise FileNotFoundError("No checkpoint found in checkpoint directory")
    if requested_step is None:
        return available[-1]
    requested = int(requested_step)
    if requested not in available:
        formatted = ", ".join(f"{step:,}" for step in available)
        raise FileNotFoundError(
            f"checkpoint step {requested:,} is unavailable; available steps: {formatted}"
        )
    return requested


def normalize_checkpoint_save_steps(
    max_steps: int,
    requested_steps: Iterable[int],
) -> list[int]:
    """Validate explicit zero-based steps and always retain the final update."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    final_step = max_steps - 1
    normalized = {int(step) for step in requested_steps}
    normalized.add(final_step)
    invalid = sorted(step for step in normalized if step < 0 or step > final_step)
    if invalid:
        raise ValueError(
            f"checkpoint steps must be within [0, {final_step:,}], got {invalid}"
        )
    return sorted(normalized)
