"""Held-out evaluation helpers for the real-environment CoinRun PPO policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dreamer.coinrun import COINRUN_ACTION_DIM
from dreamer.coinrun_ppo import Gym3VectorEnvAdapter


@dataclass(frozen=True)
class CompletedEpisode:
    env_index: int
    episode_return: float
    length: int


class EpisodeStatisticsTracker:
    """Accumulate action-aligned rewards until the action ends an episode."""

    def __init__(self, *, num_envs: int) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.num_envs = num_envs
        self.active_returns = np.zeros(num_envs, dtype=np.float64)
        self.active_lengths = np.zeros(num_envs, dtype=np.int64)

    def observe(
        self,
        *,
        rewards: np.ndarray,
        terminals: np.ndarray,
    ) -> list[CompletedEpisode]:
        rewards = np.asarray(rewards, dtype=np.float64)
        terminals = np.asarray(terminals, dtype=bool)
        expected = (self.num_envs,)
        if rewards.shape != expected or terminals.shape != expected:
            raise ValueError(
                f"rewards and terminals must have shape {expected}, got "
                f"{rewards.shape} and {terminals.shape}"
            )
        self.active_returns += rewards
        self.active_lengths += 1
        completed = [
            CompletedEpisode(
                env_index=int(index),
                episode_return=float(self.active_returns[index]),
                length=int(self.active_lengths[index]),
            )
            for index in np.flatnonzero(terminals)
        ]
        self.active_returns[terminals] = 0.0
        self.active_lengths[terminals] = 0
        return completed


def validate_level_split(
    *,
    train_start_level: int,
    train_num_levels: int,
    eval_start_level: int,
    eval_num_levels: int,
) -> None:
    values = {
        "train_start_level": train_start_level,
        "eval_start_level": eval_start_level,
    }
    if any(value < 0 for value in values.values()):
        raise ValueError(f"level starts must be non-negative: {values}")
    if train_num_levels <= 0 or eval_num_levels < 0:
        raise ValueError(
            "train_num_levels must be positive and eval_num_levels must be "
            "non-negative: "
            f"{{'train_num_levels': {train_num_levels}, "
            f"'eval_num_levels': {eval_num_levels}}}"
        )
    if eval_num_levels == 0:
        return
    train_levels = range(
        train_start_level,
        train_start_level + train_num_levels,
    )
    eval_levels = range(
        eval_start_level,
        eval_start_level + eval_num_levels,
    )
    if max(train_levels.start, eval_levels.start) < min(
        train_levels.stop,
        eval_levels.stop,
    ):
        raise ValueError(
            "training and held-out evaluation level ranges overlap: "
            f"[{train_levels.start}, {train_levels.stop}) and "
            f"[{eval_levels.start}, {eval_levels.stop})"
        )


def summarize_episodes(episodes: list[CompletedEpisode]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("at least one completed episode is required")
    returns = np.asarray(
        [episode.episode_return for episode in episodes],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [episode.length for episode in episodes],
        dtype=np.float64,
    )
    return {
        "episodes": len(episodes),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "return_p05": float(np.quantile(returns, 0.05)),
        "return_p95": float(np.quantile(returns, 0.95)),
        "mean_episode_length": float(np.mean(lengths)),
        "successes": int(np.sum(returns > 0.0)),
        "success_rate": float(np.mean(returns > 0.0)),
    }


def evaluate_policy(
    *,
    env: Any,
    policy: Any,
    num_episodes: int,
    max_vector_steps: int | None = None,
    max_visual_episodes: int = 4,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Evaluate a frozen policy and retain a bounded set of complete videos."""

    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    if max_visual_episodes < 0:
        raise ValueError(
            f"max_visual_episodes must be non-negative, got {max_visual_episodes}"
        )
    adapter = Gym3VectorEnvAdapter(env, action_dim=COINRUN_ACTION_DIM)
    tracker = EpisodeStatisticsTracker(num_envs=adapter.num_envs)
    max_vector_steps = (
        max_vector_steps
        if max_vector_steps is not None
        else max(1_000, ceil(num_episodes / adapter.num_envs) * 1_000)
    )
    if max_vector_steps <= 0:
        raise ValueError("max_vector_steps must be positive")

    frame_buffers: list[list[np.ndarray]] = [
        [] for _ in range(adapter.num_envs)
    ]
    completed: list[CompletedEpisode] = []
    videos: list[np.ndarray] = []
    action_counts = np.zeros(COINRUN_ACTION_DIM, dtype=np.int64)
    vector_steps = 0
    while len(completed) < num_episodes and vector_steps < max_vector_steps:
        current = adapter.current
        actions = np.asarray(
            policy.sample(current.observation, current.first),
            dtype=np.int32,
        )
        action_counts += np.bincount(actions, minlength=COINRUN_ACTION_DIM)
        transition = adapter.step(actions)
        vector_steps += 1
        for env_index in range(adapter.num_envs):
            frame_buffers[env_index].append(
                transition.observation[env_index].copy()
            )
        new_episodes = tracker.observe(
            rewards=transition.rewards,
            terminals=transition.terminals,
        )
        for episode in new_episodes:
            if len(videos) < max_visual_episodes:
                videos.append(
                    np.stack(frame_buffers[episode.env_index]).astype(np.uint8)
                )
            frame_buffers[episode.env_index].clear()
            completed.append(episode)
            if len(completed) == num_episodes:
                break
    if len(completed) < num_episodes:
        raise RuntimeError(
            f"only completed {len(completed)}/{num_episodes} episodes within "
            f"{max_vector_steps} vector steps"
        )

    summary = summarize_episodes(completed[:num_episodes])
    summary.update(
        {
            "vector_steps": vector_steps,
            "environment_transitions": vector_steps * adapter.num_envs,
            "action_counts": {
                str(action): int(count)
                for action, count in enumerate(action_counts)
                if count
            },
        }
    )
    return summary, videos


def sample_video_frames(
    video: np.ndarray,
    *,
    max_frames: int,
) -> np.ndarray:
    """Keep the full episode time span while bounding stored visual frames."""

    video = np.asarray(video)
    if video.ndim < 1 or video.shape[0] <= 0:
        raise ValueError(f"video must contain frames, got {video.shape}")
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")
    if video.shape[0] <= max_frames:
        return video
    indices = np.linspace(
        0,
        video.shape[0] - 1,
        num=max_frames,
        dtype=np.int64,
    )
    return video[indices]


def write_evaluation_media(
    *,
    videos: list[np.ndarray],
    output_dir: Path,
    completed_env_steps: int,
    fps: int = 15,
    max_frames_per_video: int = 256,
) -> list[Path]:
    """Write deterministic GIFs plus a first/middle/last-frame contact sheet."""

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not videos:
        return []
    import imageio.v3 as iio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    sheet_rows = []
    for index, video in enumerate(videos):
        video = sample_video_frames(
            np.asarray(video, dtype=np.uint8),
            max_frames=max_frames_per_video,
        )
        if video.ndim != 4 or video.shape[1:] != (64, 64, 3):
            raise ValueError(
                f"evaluation video must have shape (time, 64, 64, 3), got {video.shape}"
            )
        gif_path = (
            output_dir
            / f"env-steps-{completed_env_steps:09d}-episode-{index:02d}.gif"
        )
        iio.imwrite(gif_path, video, duration=1.0 / fps, loop=0)
        paths.append(gif_path)
        sheet_rows.append(
            np.concatenate(
                [
                    video[0],
                    video[len(video) // 2],
                    video[-1],
                ],
                axis=1,
            )
        )
    sheet_path = (
        output_dir
        / f"env-steps-{completed_env_steps:09d}-contact-sheet.png"
    )
    iio.imwrite(sheet_path, np.concatenate(sheet_rows, axis=0))
    paths.append(sheet_path)
    return paths
