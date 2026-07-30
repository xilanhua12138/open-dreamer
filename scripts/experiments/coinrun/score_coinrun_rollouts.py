#!/usr/bin/env python3
"""Score generated CoinRun rollouts against tokenizer-decoded held-out targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy.ndimage import uniform_filter


def read_video(path: Path) -> np.ndarray:
    return np.stack(list(iio.imiter(path, plugin="FFMPEG"))).astype(np.float64) / 255.0


def frame_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Match the common 7x7, channel-wise SSIM calculation."""
    win_size = 7
    covariance_norm = win_size**2 / (win_size**2 - 1)
    ux = uniform_filter(x, size=(win_size, win_size, 1), mode="reflect")
    uy = uniform_filter(y, size=(win_size, win_size, 1), mode="reflect")
    uxx = uniform_filter(x * x, size=(win_size, win_size, 1), mode="reflect")
    uyy = uniform_filter(y * y, size=(win_size, win_size, 1), mode="reflect")
    uxy = uniform_filter(x * y, size=(win_size, win_size, 1), mode="reflect")
    vx = covariance_norm * (uxx - ux * ux)
    vy = covariance_norm * (uyy - uy * uy)
    vxy = covariance_norm * (uxy - ux * uy)
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
        (ux * ux + uy * uy + c1) * (vx + vy + c2)
    )
    pad = (win_size - 1) // 2
    return float(score[pad:-pad, pad:-pad].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pred_paths = sorted(args.video_dir.glob("pred_*.mp4"))
    gt_paths = sorted(args.video_dir.glob("gt_decoded_*.mp4"))
    if not pred_paths or len(pred_paths) != len(gt_paths):
        raise ValueError(
            f"Expected matching pred/gt_decoded videos, got "
            f"{len(pred_paths)}/{len(gt_paths)}"
        )

    per_video = []
    all_frame_psnr = []
    all_frame_ssim = []
    for pred_path, gt_path in zip(pred_paths, gt_paths, strict=True):
        pred = read_video(pred_path)[args.context :]
        gt = read_video(gt_path)[args.context :]
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: {pred_path} {pred.shape} vs {gt.shape}")
        mse_by_frame = np.mean((pred - gt) ** 2, axis=(1, 2, 3))
        psnr_by_frame = 10.0 * np.log10(1.0 / np.maximum(mse_by_frame, 1e-12))
        ssim_by_frame = np.asarray(
            [frame_ssim(pred_frame, gt_frame) for pred_frame, gt_frame in zip(pred, gt)]
        )
        all_frame_psnr.append(psnr_by_frame)
        all_frame_ssim.append(ssim_by_frame)
        per_video.append(
            {
                "video": pred_path.stem.removeprefix("pred_"),
                "psnr_db": float(
                    10.0 * np.log10(1.0 / np.maximum(mse_by_frame.mean(), 1e-12))
                ),
                "mean_frame_psnr_db": float(psnr_by_frame.mean()),
                "mean_ssim": float(ssim_by_frame.mean()),
            }
        )

    psnr = np.stack(all_frame_psnr)
    ssim = np.stack(all_frame_ssim)
    windows = [n for n in (1, 3, 8, 16) if n <= psnr.shape[1]]
    payload = {
        "video_dir": str(args.video_dir),
        "num_videos": len(per_video),
        "context_frames": args.context,
        "predicted_frames": int(psnr.shape[1]),
        "mean_video_psnr_db": float(np.mean([row["psnr_db"] for row in per_video])),
        "mean_frame_psnr_db": float(psnr.mean()),
        "mean_ssim": float(ssim.mean()),
        "psnr_by_horizon_db": {
            str(n): float(psnr[:, :n].mean()) for n in windows
        },
        "ssim_by_horizon": {
            str(n): float(ssim[:, :n].mean()) for n in windows
        },
        "per_video": per_video,
    }
    output = args.output or args.video_dir.parent.parent / "metrics.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
