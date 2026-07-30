#!/usr/bin/env python3
"""Train a PPO policy in real CoinRun for later world-model data collection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from dreamer.coinrun import COINRUN_ACTION_DIM
from dreamer.coinrun_ppo import (
    CoinRunActorCritic,
    Gym3VectorEnvAdapter,
    PPOCoinRunPolicy,
    PPOHyperparameters,
    load_train_state,
    make_ppo_minibatch_step,
    save_train_state,
    update_ppo,
)
from dreamer.coinrun_ppo_evaluation import (
    EpisodeStatisticsTracker,
    evaluate_policy,
    validate_level_split,
    write_evaluation_media,
)
from dreamer.coinrun_ppo_training import (
    PPOTrainConfig,
    RewardNormalizer,
    build_training_batch,
    make_action_sampler,
)
from dreamer.configs import LoggerConfig
from dreamer.experiment_runtime import atomic_write_json
from dreamer.logging import build_logger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="coinrun-ppo-seed0")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--total-env-steps", type=int, default=25_165_824)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--adam-epsilon", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--reward-clip", type=float, default=10.0)
    parser.add_argument(
        "--reward-normalization-gamma",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--advantage-normalization",
        choices=("batch", "minibatch"),
        default="minibatch",
    )
    parser.add_argument(
        "--backbone-kernel-init",
        choices=("orthogonal_sqrt2", "orthogonal_gain1", "glorot_uniform"),
        default="glorot_uniform",
    )
    parser.add_argument("--residual-branch-scale", type=float, default=1.0)
    parser.add_argument(
        "--residual-last-kernel-init",
        choices=("same", "zeros"),
        default="same",
    )
    parser.add_argument("--residual-skip-init", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-start-level", type=int, default=0)
    parser.add_argument("--train-num-levels", type=int, default=200)
    parser.add_argument("--eval-start-level", type=int, default=0)
    parser.add_argument("--eval-num-levels", type=int, default=0)
    parser.add_argument("--distribution-mode", default="easy")
    parser.add_argument(
        "--checkpoint-every-env-steps",
        type=int,
        default=1_048_576,
    )
    parser.add_argument(
        "--evaluation-every-env-steps",
        type=int,
        default=1_048_576,
    )
    parser.add_argument("--evaluation-episodes", type=int, default=128)
    parser.add_argument("--evaluation-envs", type=int, default=16)
    parser.add_argument("--evaluation-seed", type=int, default=4_242)
    parser.add_argument("--evaluation-max-vector-steps", type=int, default=40_000)
    parser.add_argument(
        "--evaluation-policy",
        choices=("deterministic_argmax", "stochastic"),
        default="stochastic",
    )
    parser.add_argument(
        "--final-evaluation-episodes",
        type=int,
        default=512,
        help="Run a separate final evaluation; zero explicitly disables it.",
    )
    parser.add_argument("--visual-episodes", type=int, default=4)
    parser.add_argument("--visual-fps", type=int, default=15)
    parser.add_argument("--visual-max-frames", type=int, default=256)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--wandb-project", default="open-dreamer")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="CR-PPO-0001")
    return parser.parse_args(argv)


def _make_env(
    *,
    num_envs: int,
    start_level: int,
    num_levels: int,
    distribution_mode: str,
    seed: int,
) -> Any:
    try:
        from procgen import ProcgenGym3Env
    except ImportError as error:
        raise RuntimeError(
            "Procgen is required for real CoinRun PPO training. Install the "
            "repository's CoinRun runtime before launching this script."
        ) from error
    env = ProcgenGym3Env(
        num=num_envs,
        env_name="coinrun",
        start_level=start_level,
        num_levels=num_levels,
        distribution_mode=distribution_mode,
        rand_seed=seed,
    )
    action_dim = int(env.ac_space.eltype.n)
    if action_dim != COINRUN_ACTION_DIM:
        raise ValueError(
            f"expected CoinRun action dimension {COINRUN_ACTION_DIM}, "
            f"got {action_dim}"
        )
    return env


def _validate_interval(
    *,
    name: str,
    value: int,
    batch_size: int,
    total_env_steps: int,
) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if value % batch_size:
        raise ValueError(
            f"{name}={value} must be divisible by rollout batch size "
            f"{batch_size}"
        )
    if total_env_steps % value:
        raise ValueError(
            f"total_env_steps={total_env_steps} must be divisible by "
            f"{name}={value}"
        )


def _checkpoint_metadata(
    *,
    config: PPOTrainConfig,
    completed_env_steps: int,
    update_index: int,
    policy_key: jax.Array,
    reward_normalizer: RewardNormalizer,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "architecture": "impala_cnn",
        "action_dim": COINRUN_ACTION_DIM,
        "completed_env_steps": completed_env_steps,
        "completed_ppo_updates": update_index + 1,
        "train_config": config.to_dict(),
        "policy_prng_key": np.asarray(policy_key, dtype=np.uint32).tolist(),
        "reward_normalizer": reward_normalizer.state_dict(),
        "resume_semantics": (
            "Fault-only resume restores model, optimizer, PRNG and reward "
            "statistics. Procgen process state is not checkpointed, so the "
            "vector environment restarts from the recorded train split."
        ),
        "downstream_scope": (
            "Freeze checkpoint and collect action-conditioned trajectories "
            "for dynamics. No behavior-cloning policy is part of this run."
        ),
    }


def _restore_fault_checkpoint(
    *,
    path: Path,
    state: train_state.TrainState,
    config: PPOTrainConfig,
    reward_normalizer: RewardNormalizer,
) -> tuple[train_state.TrainState, int, jax.Array]:
    restored, metadata = load_train_state(path, state)
    if metadata.get("architecture") != "impala_cnn":
        raise ValueError("resume checkpoint architecture mismatch")
    if int(metadata.get("action_dim", -1)) != COINRUN_ACTION_DIM:
        raise ValueError("resume checkpoint action-space mismatch")
    if metadata.get("train_config") != config.to_dict():
        raise ValueError(
            "fault-only resume requires an identical PPOTrainConfig"
        )
    completed_env_steps = int(metadata.get("completed_env_steps", -1))
    if completed_env_steps < 0 or completed_env_steps % config.batch_size:
        raise ValueError("resume checkpoint has invalid completed_env_steps")
    reward_normalizer.load_state_dict(metadata["reward_normalizer"])
    policy_key = jnp.asarray(metadata["policy_prng_key"], dtype=jnp.uint32)
    if policy_key.shape != (2,):
        raise ValueError("resume checkpoint has invalid policy PRNG key")
    return restored, completed_env_steps, policy_key


def _write_checkpoint(
    *,
    state: train_state.TrainState,
    config: PPOTrainConfig,
    completed_env_steps: int,
    update_index: int,
    policy_key: jax.Array,
    reward_normalizer: RewardNormalizer,
    checkpoint_dir: Path,
) -> Path:
    checkpoint_path = (
        checkpoint_dir / f"env-steps-{completed_env_steps:09d}.msgpack"
    )
    save_train_state(
        checkpoint_path,
        state,
        metadata=_checkpoint_metadata(
            config=config,
            completed_env_steps=completed_env_steps,
            update_index=update_index,
            policy_key=policy_key,
            reward_normalizer=reward_normalizer,
        ),
    )
    atomic_write_json(
        checkpoint_dir / "latest.json",
        {
            "schema_version": "1.0",
            "completed_env_steps": completed_env_steps,
            "checkpoint": checkpoint_path.name,
        },
    )
    return checkpoint_path


def _evaluate(
    *,
    state: train_state.TrainState,
    args: argparse.Namespace,
    config: PPOTrainConfig,
    completed_env_steps: int,
    run_dir: Path,
    logger: Any,
    num_episodes: int | None = None,
    milestone_root: str = "validation",
) -> dict[str, Any]:
    evaluation_env = _make_env(
        num_envs=args.evaluation_envs,
        start_level=args.eval_start_level,
        num_levels=args.eval_num_levels,
        distribution_mode=args.distribution_mode,
        seed=args.evaluation_seed,
    )
    policy = PPOCoinRunPolicy(
        model=CoinRunActorCritic(
            action_dim=COINRUN_ACTION_DIM,
            backbone_kernel_init=config.backbone_kernel_init,
            residual_branch_scale=config.residual_branch_scale,
            residual_last_kernel_init=config.residual_last_kernel_init,
            residual_skip_init=config.residual_skip_init,
        ),
        params=state.params,
        metadata={
            "architecture": "impala_cnn",
            "action_dim": COINRUN_ACTION_DIM,
            "completed_env_steps": completed_env_steps,
        },
        num_envs=args.evaluation_envs,
        seed=args.evaluation_seed,
        deterministic=args.evaluation_policy == "deterministic_argmax",
    )
    summary, videos = evaluate_policy(
        env=evaluation_env,
        policy=policy,
        num_episodes=(
            args.evaluation_episodes if num_episodes is None else num_episodes
        ),
        max_vector_steps=args.evaluation_max_vector_steps,
        max_visual_episodes=args.visual_episodes,
    )
    milestone_dir = (
        run_dir / milestone_root / f"env-steps-{completed_env_steps:09d}"
    )
    evaluation_level_range = (
        None
        if args.eval_num_levels == 0
        else [
            args.eval_start_level,
            args.eval_start_level + args.eval_num_levels,
        ]
    )
    summary.update(
        {
            "schema_version": "1.0",
            "completed_env_steps": completed_env_steps,
            "policy": args.evaluation_policy,
            "seed": args.evaluation_seed,
            "start_level": args.eval_start_level,
            "num_levels": args.eval_num_levels,
            "distribution_mode": args.distribution_mode,
            "train_level_range": [
                config.start_level,
                config.start_level + config.num_levels,
            ],
            "evaluation_level_range": evaluation_level_range,
            "evaluation_distribution": (
                "full_distribution"
                if args.eval_num_levels == 0
                else "fixed_level_range"
            ),
        }
    )
    summary_path = milestone_dir / "metrics.json"
    atomic_write_json(summary_path, summary)
    logger.recorder.register_artifact(
        step=(completed_env_steps // config.batch_size) - 1,
        key="evaluation/metrics",
        path=summary_path,
    )
    for path in write_evaluation_media(
        videos=videos,
        output_dir=milestone_dir,
        completed_env_steps=completed_env_steps,
        fps=args.visual_fps,
        max_frames_per_video=args.visual_max_frames,
    ):
        if path.suffix == ".gif":
            logger.log_video(
                step=(completed_env_steps // config.batch_size) - 1,
                key=f"evaluation/{path.stem}",
                video_path=path,
                format="gif",
                fps=args.visual_fps,
            )
        else:
            logger.log_image(
                step=(completed_env_steps // config.batch_size) - 1,
                key="evaluation/contact_sheet",
                image=path,
            )
    logger.log_metrics(
        step=(completed_env_steps // config.batch_size) - 1,
        prefix="evaluation/",
        metrics={
            key: value
            for key, value in summary.items()
            if isinstance(value, (int, float))
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    config = PPOTrainConfig(
        total_env_steps=args.total_env_steps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_clip_epsilon=args.value_clip_epsilon,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        reward_clip=args.reward_clip,
        reward_normalization_gamma=args.reward_normalization_gamma,
        advantage_normalization=args.advantage_normalization,
        backbone_kernel_init=args.backbone_kernel_init,
        residual_branch_scale=args.residual_branch_scale,
        residual_last_kernel_init=args.residual_last_kernel_init,
        residual_skip_init=args.residual_skip_init,
        seed=args.seed,
        start_level=args.train_start_level,
        num_levels=args.train_num_levels,
        distribution_mode=args.distribution_mode,
    )
    config.validate()
    validate_level_split(
        train_start_level=args.train_start_level,
        train_num_levels=args.train_num_levels,
        eval_start_level=args.eval_start_level,
        eval_num_levels=args.eval_num_levels,
    )
    for name, value in (
        ("checkpoint_every_env_steps", args.checkpoint_every_env_steps),
        ("evaluation_every_env_steps", args.evaluation_every_env_steps),
    ):
        _validate_interval(
            name=name,
            value=value,
            batch_size=config.batch_size,
            total_env_steps=config.total_env_steps,
        )
    if min(
        args.evaluation_episodes,
        args.evaluation_envs,
        args.evaluation_max_vector_steps,
        args.visual_fps,
        args.visual_max_frames,
    ) <= 0:
        raise ValueError("evaluation counts and visual_fps must be positive")
    if args.final_evaluation_episodes < 0:
        raise ValueError("final_evaluation_episodes must be non-negative")
    if args.visual_episodes < 0:
        raise ValueError("visual_episodes must be non-negative")

    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = _make_env(
        num_envs=config.num_envs,
        start_level=config.start_level,
        num_levels=config.num_levels,
        distribution_mode=config.distribution_mode,
        seed=config.seed,
    )
    adapter = Gym3VectorEnvAdapter(env, action_dim=COINRUN_ACTION_DIM)
    model = CoinRunActorCritic(
        action_dim=COINRUN_ACTION_DIM,
        backbone_kernel_init=config.backbone_kernel_init,
        residual_branch_scale=config.residual_branch_scale,
        residual_last_kernel_init=config.residual_last_kernel_init,
        residual_skip_init=config.residual_skip_init,
    )
    parameter_key, policy_key = jax.random.split(
        jax.random.PRNGKey(config.seed)
    )
    params = model.init(
        parameter_key,
        jnp.asarray(adapter.current.observation),
    )["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(
                config.learning_rate,
                eps=config.adam_epsilon,
            ),
        ),
    )
    reward_normalizer = RewardNormalizer(
        num_envs=config.num_envs,
        gamma=config.reward_normalization_gamma,
        clip=config.reward_clip,
    )
    completed_env_steps = 0
    if args.resume is not None:
        state, completed_env_steps, policy_key = _restore_fault_checkpoint(
            path=args.resume,
            state=state,
            config=config,
            reward_normalizer=reward_normalizer,
        )
    if completed_env_steps >= config.total_env_steps:
        raise ValueError(
            "resume checkpoint is already at or beyond total_env_steps"
        )

    logger_config = LoggerConfig(
        run_name=args.run_name,
        use_wandb=args.wandb_mode != "disabled",
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_tags=["coinrun", "ppo", args.experiment_id, f"seed{config.seed}"],
        wandb_mode="offline" if args.wandb_mode == "disabled" else args.wandb_mode,
        log_every=1,
        max_steps=config.num_updates,
        telemetry_progress_every_seconds=30.0,
        telemetry_system_every_seconds=60.0,
    )
    runtime_config = {
        "schema_version": "1.0",
        "experiment_id": args.experiment_id,
        "ppo": config.to_dict(),
        "evaluation": {
            "every_env_steps": args.evaluation_every_env_steps,
            "episodes": args.evaluation_episodes,
            "envs": args.evaluation_envs,
            "seed": args.evaluation_seed,
            "start_level": args.eval_start_level,
            "num_levels": args.eval_num_levels,
            "policy": args.evaluation_policy,
            "final_episodes": args.final_evaluation_episodes,
            "max_vector_steps": args.evaluation_max_vector_steps,
            "visual_episodes": args.visual_episodes,
            "visual_fps": args.visual_fps,
            "visual_max_frames": args.visual_max_frames,
        },
        "checkpoint_every_env_steps": args.checkpoint_every_env_steps,
        "resume": str(args.resume.resolve()) if args.resume else None,
        "wandb_mode": args.wandb_mode,
        "downstream_scope": {
            "collect_action_conditioned_trajectories": (
                args.experiment_id == "CR-PPO-0001"
            ),
            "behavior_cloning": False,
            "policy_training_inside_world_model": False,
        },
    }
    hyperparameters = PPOHyperparameters(
        clip_epsilon=config.clip_epsilon,
        value_clip_epsilon=config.value_clip_epsilon,
        value_coefficient=config.value_coefficient,
        entropy_coefficient=config.entropy_coefficient,
        minibatch_size=config.minibatch_size,
        update_epochs=config.update_epochs,
        advantage_normalization=config.advantage_normalization,
    )
    minibatch_step = make_ppo_minibatch_step(
        apply_fn=state.apply_fn,
        hyperparameters=hyperparameters,
    )
    action_sampler = make_action_sampler(apply_fn=state.apply_fn)
    value_apply = jax.jit(
        lambda params, observations: state.apply_fn(
            {"params": params},
            observations,
        )[1]
    )
    episode_tracker = EpisodeStatisticsTracker(num_envs=config.num_envs)
    start_time = time.monotonic()
    initial_completed_env_steps = completed_env_steps

    with build_logger(
        logger_config,
        config=runtime_config,
        dir=str(run_dir),
    ) as logger:
        if completed_env_steps == 0:
            _evaluate(
                state=state,
                args=args,
                config=config,
                completed_env_steps=0,
                run_dir=run_dir,
                logger=logger,
            )
        while completed_env_steps < config.total_env_steps:
            update_index = completed_env_steps // config.batch_size
            observation_rows = []
            action_rows = []
            log_prob_rows = []
            value_rows = []
            normalized_reward_rows = []
            terminal_rows = []
            raw_rewards = []
            completed_episodes = []
            for _ in range(config.rollout_steps):
                current = adapter.current
                actions, log_probs, values, policy_key = action_sampler(
                    state.params,
                    jnp.asarray(current.observation),
                    policy_key,
                )
                actions_np = np.asarray(jax.device_get(actions), dtype=np.int32)
                transition = adapter.step(actions_np)
                normalized_rewards = reward_normalizer.normalize(
                    transition.rewards,
                    transition.terminals,
                )
                observation_rows.append(transition.observation)
                action_rows.append(actions_np)
                log_prob_rows.append(np.asarray(jax.device_get(log_probs)))
                value_rows.append(np.asarray(jax.device_get(values)))
                normalized_reward_rows.append(normalized_rewards)
                terminal_rows.append(transition.terminals)
                raw_rewards.append(transition.rewards)
                completed_episodes.extend(
                    episode_tracker.observe(
                        rewards=transition.rewards,
                        terminals=transition.terminals,
                    )
                )

            next_value = value_apply(
                state.params,
                jnp.asarray(adapter.current.observation),
            )
            batch = build_training_batch(
                observations=jnp.asarray(np.stack(observation_rows)),
                actions=jnp.asarray(np.stack(action_rows)),
                old_log_probs=jnp.asarray(np.stack(log_prob_rows)),
                values=jnp.asarray(np.stack(value_rows)),
                rewards=jnp.asarray(np.stack(normalized_reward_rows)),
                terminals=jnp.asarray(np.stack(terminal_rows)),
                next_value=next_value,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            policy_key, update_key = jax.random.split(policy_key)
            state, update_metrics = update_ppo(
                state=state,
                batch=batch,
                key=update_key,
                hyperparameters=hyperparameters,
                minibatch_step=minibatch_step,
            )
            completed_env_steps += config.batch_size
            elapsed = max(time.monotonic() - start_time, 1e-9)
            raw_reward_array = np.asarray(raw_rewards, dtype=np.float32)
            episode_returns = np.asarray(
                [episode.episode_return for episode in completed_episodes],
                dtype=np.float64,
            )
            train_metrics: dict[str, float | int] = {
                **update_metrics,
                "completed_env_steps": completed_env_steps,
                "environment_steps_per_second": (
                    (completed_env_steps - initial_completed_env_steps) / elapsed
                ),
                "raw_reward_mean": float(np.mean(raw_reward_array)),
                "raw_reward_nonzero_fraction": float(
                    np.mean(raw_reward_array != 0.0)
                ),
                "episodes_completed_in_rollout": len(completed_episodes),
                "reward_normalizer_variance": (
                    reward_normalizer.running.variance
                ),
            }
            if episode_returns.size:
                train_metrics.update(
                    {
                        "episode_return_mean": float(
                            np.mean(episode_returns)
                        ),
                        "episode_success_rate": float(
                            np.mean(episode_returns > 0.0)
                        ),
                    }
                )
            logger.log_metrics(
                step=update_index,
                prefix="train/",
                metrics=train_metrics,
            )

            if completed_env_steps % args.checkpoint_every_env_steps == 0:
                checkpoint_path = _write_checkpoint(
                    state=state,
                    config=config,
                    completed_env_steps=completed_env_steps,
                    update_index=update_index,
                    policy_key=policy_key,
                    reward_normalizer=reward_normalizer,
                    checkpoint_dir=checkpoint_dir,
                )
                logger.recorder.register_artifact(
                    step=update_index,
                    key="checkpoint",
                    path=checkpoint_path,
                )
            if completed_env_steps % args.evaluation_every_env_steps == 0:
                _evaluate(
                    state=state,
                    args=args,
                    config=config,
                    completed_env_steps=completed_env_steps,
                    run_dir=run_dir,
                    logger=logger,
                )

        final_update_index = (completed_env_steps // config.batch_size) - 1
        if args.final_evaluation_episodes:
            _evaluate(
                state=state,
                args=args,
                config=config,
                completed_env_steps=completed_env_steps,
                run_dir=run_dir,
                logger=logger,
                num_episodes=args.final_evaluation_episodes,
                milestone_root="final-validation",
            )
        final_checkpoint = (
            checkpoint_dir
            / f"env-steps-{completed_env_steps:09d}.msgpack"
        )
        if not final_checkpoint.is_file():
            raise RuntimeError(
                f"final checkpoint was not written: {final_checkpoint}"
            )
        logger.recorder.register_artifact(
            step=final_update_index,
            key="checkpoint/final",
            path=final_checkpoint,
        )
        atomic_write_json(
            run_dir / "TRAINING_COMPLETE.json",
            {
                "schema_version": "1.0",
                "completed_env_steps": completed_env_steps,
                "checkpoint": str(final_checkpoint),
                "downstream_next_step": (
                    "Compare this frozen policy with the public Procgen "
                    "CoinRun reference before authorizing downstream use."
                ),
            },
        )


if __name__ == "__main__":
    main()
