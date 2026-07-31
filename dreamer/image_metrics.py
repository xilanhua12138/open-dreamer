"""Region-aware image metrics for small, high-background video frames."""

from __future__ import annotations

import math

import numpy as np


def _as_unit_float(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    return np.clip(array.astype(np.float32), 0.0, 1.0)


def edge_mask(target: np.ndarray, *, threshold: float) -> np.ndarray:
    """Return pixels adjacent to a horizontal or vertical luminance edge."""

    target_01 = _as_unit_float(target)
    if target_01.ndim < 4 or target_01.shape[-1] != 3:
        raise ValueError(f"expected (..., H, W, 3), got {target_01.shape}")
    gray = target_01.mean(axis=-1)
    horizontal = np.zeros_like(gray)
    vertical = np.zeros_like(gray)
    horizontal[..., :-1] = np.abs(gray[..., 1:] - gray[..., :-1])
    vertical[..., :-1, :] = np.abs(gray[..., 1:, :] - gray[..., :-1, :])
    return np.maximum(horizontal, vertical) >= threshold


def temporal_change_mask(target: np.ndarray, *, threshold: float) -> np.ndarray:
    """Return pixels whose RGB value changes from the preceding frame."""

    target_01 = _as_unit_float(target)
    if target_01.ndim != 5 or target_01.shape[-1] != 3:
        raise ValueError(f"expected (B, T, H, W, 3), got {target_01.shape}")
    mask = np.zeros(target_01.shape[:-1], dtype=bool)
    mask[:, 1:] = (
        np.max(np.abs(target_01[:, 1:] - target_01[:, :-1]), axis=-1)
        >= threshold
    )
    return mask


def region_squared_error(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int]:
    """Return normalized RGB squared-error sum and selected scalar count."""

    prediction_01 = _as_unit_float(prediction)
    target_01 = _as_unit_float(target)
    mask = np.asarray(mask, dtype=bool)
    if prediction_01.shape != target_01.shape:
        raise ValueError(
            f"prediction/target shape mismatch: "
            f"{prediction_01.shape} vs {target_01.shape}"
        )
    if mask.shape != target_01.shape[:-1]:
        raise ValueError(
            f"mask shape {mask.shape} does not match {target_01.shape[:-1]}"
        )
    count = int(mask.sum()) * target_01.shape[-1]
    if count == 0:
        raise ValueError("region mask selected no pixels")
    error = (prediction_01 - target_01) ** 2
    return float(error[mask].sum()), count


def psnr_from_squared_error(squared_error: float, count: int) -> float:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    mse = squared_error / count
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))
