from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state

from dreamer.coinrun import COINRUN_ACTION_DIM
from dreamer.coinrun_ppo import (
    CoinRunActorCritic,
    PPOBatch,
    PPOCoinRunPolicy,
    PPOHyperparameters,
    ResidualBlock,
    categorical_entropy,
    categorical_log_prob,
    compute_gae,
    load_train_state,
    make_ppo_minibatch_step,
    make_minibatch_indices,
    ppo_loss,
    save_train_state,
    update_ppo,
)


class CoinRunPPOCoreTests(unittest.TestCase):
    def test_gae_stops_at_episode_boundaries(self) -> None:
        advantages, returns = compute_gae(
            rewards=jnp.asarray([[1.0], [2.0], [3.0]]),
            values=jnp.asarray([[0.5], [1.0], [1.5]]),
            next_value=jnp.asarray([9.0]),
            terminals=jnp.asarray([[False], [True], [False]]),
            gamma=0.9,
            gae_lambda=0.95,
        )

        np.testing.assert_allclose(
            np.asarray(advantages[:, 0]),
            np.asarray([2.255, 1.0, 9.6]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(returns[:, 0]),
            np.asarray([2.755, 2.0, 11.1]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_gae_rejects_mismatched_rollout_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            compute_gae(
                rewards=jnp.ones((4, 2)),
                values=jnp.ones((4, 3)),
                next_value=jnp.ones((2,)),
                terminals=jnp.zeros((4, 2), dtype=bool),
                gamma=0.99,
                gae_lambda=0.95,
            )

    def test_categorical_helpers_match_known_distribution(self) -> None:
        logits = jnp.asarray([[0.0, np.log(3.0)]])
        actions = jnp.asarray([1])

        log_prob = categorical_log_prob(logits, actions)
        entropy = categorical_entropy(logits)

        self.assertAlmostEqual(float(log_prob[0]), float(np.log(0.75)), places=6)
        self.assertAlmostEqual(
            float(entropy[0]),
            float(-(0.25 * np.log(0.25) + 0.75 * np.log(0.75))),
            places=6,
        )

    def test_clipped_surrogate_uses_pessimistic_bound(self) -> None:
        result = ppo_loss(
            new_log_probs=jnp.log(jnp.asarray([2.0, 0.5])),
            old_log_probs=jnp.zeros((2,)),
            advantages=jnp.asarray([1.0, -1.0]),
            new_values=jnp.asarray([0.5, -0.5]),
            old_values=jnp.zeros((2,)),
            returns=jnp.asarray([1.0, -1.0]),
            entropy=jnp.zeros((2,)),
            clip_epsilon=0.2,
            value_clip_epsilon=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.0,
        )

        self.assertAlmostEqual(float(result.policy_loss), -0.2, places=6)
        self.assertAlmostEqual(float(result.value_loss), 0.32, places=6)
        self.assertAlmostEqual(float(result.clip_fraction), 1.0, places=6)
        self.assertGreater(float(result.approx_kl), 0.0)
        self.assertAlmostEqual(
            float(result.total_loss),
            float(result.policy_loss + 0.5 * result.value_loss),
            places=6,
        )

    def test_minibatch_indices_cover_each_sample_once_per_epoch(self) -> None:
        batches = make_minibatch_indices(
            key=jax.random.PRNGKey(7),
            batch_size=12,
            minibatch_size=3,
            update_epochs=2,
        )

        self.assertEqual(batches.shape, (2, 4, 3))
        for epoch in np.asarray(batches):
            self.assertEqual(sorted(epoch.reshape(-1).tolist()), list(range(12)))

    def test_minibatch_indices_reject_non_divisible_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            make_minibatch_indices(
                key=jax.random.PRNGKey(0),
                batch_size=10,
                minibatch_size=4,
                update_epochs=1,
            )

    def test_actor_critic_outputs_coinrun_action_logits_and_scalar_values(self) -> None:
        model = CoinRunActorCritic(action_dim=COINRUN_ACTION_DIM)
        observations = jnp.zeros((3, 64, 64, 3), dtype=jnp.uint8)
        variables = model.init(jax.random.PRNGKey(1), observations)

        logits, values = model.apply(variables, observations)

        self.assertEqual(logits.shape, (3, COINRUN_ACTION_DIM))
        self.assertEqual(values.shape, (3,))
        self.assertTrue(np.isfinite(np.asarray(logits)).all())
        self.assertTrue(np.isfinite(np.asarray(values)).all())

    def test_checkpoint_round_trip_restores_step_params_and_optimizer(self) -> None:
        model = CoinRunActorCritic(action_dim=COINRUN_ACTION_DIM)
        observations = jnp.zeros((2, 64, 64, 3), dtype=jnp.uint8)
        params = model.init(jax.random.PRNGKey(2), observations)["params"]
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adam(3e-4),
        ).replace(step=17)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.msgpack"
            metadata = {
                "schema_version": "1.0",
                "action_dim": COINRUN_ACTION_DIM,
                "completed_env_steps": 4096,
            }
            save_train_state(checkpoint, state, metadata=metadata)
            restored, restored_metadata = load_train_state(checkpoint, state)

        self.assertEqual(int(restored.step), 17)
        self.assertEqual(restored_metadata, metadata)
        leaves = jax.tree.leaves(
            jax.tree.map(
                lambda left, right: np.asarray(left) - np.asarray(right),
                state.params,
                restored.params,
            )
        )
        self.assertTrue(all(np.allclose(leaf, 0.0) for leaf in leaves))

    def test_ppo_update_changes_parameters_and_reports_exact_minibatch_count(
        self,
    ) -> None:
        class TinyActorCritic(nn.Module):
            @nn.compact
            def __call__(self, observations):
                hidden = nn.Dense(8)(observations)
                hidden = nn.tanh(hidden)
                return nn.Dense(3)(hidden), nn.Dense(1)(hidden)[..., 0]

        model = TinyActorCritic()
        observations = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 32.0
        params = model.init(jax.random.PRNGKey(3), observations)["params"]
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(3e-4, eps=1e-5),
            ),
        )
        logits, values = model.apply({"params": params}, observations)
        actions = jnp.asarray([0, 1, 2, 0, 1, 2, 0, 1])
        old_log_probs = categorical_log_prob(logits, actions)
        batch = PPOBatch(
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            old_values=values,
            advantages=jnp.asarray([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -1.5]),
            returns=values + jnp.asarray(
                [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -1.5]
            ),
        )

        updated, metrics = update_ppo(
            state=state,
            batch=batch,
            key=jax.random.PRNGKey(4),
            hyperparameters=PPOHyperparameters(
                minibatch_size=4,
                update_epochs=2,
            ),
        )

        self.assertEqual(int(updated.step), 4)
        self.assertEqual(metrics["optimizer_minibatches"], 4)
        for name in (
            "total_loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
        ):
            self.assertTrue(np.isfinite(metrics[name]), msg=name)
        differences = jax.tree.leaves(
            jax.tree.map(
                lambda left, right: np.asarray(left) - np.asarray(right),
                state.params,
                updated.params,
            )
        )
        self.assertTrue(any(np.any(np.abs(value) > 0.0) for value in differences))

    def test_official_parity_normalizes_advantages_inside_each_minibatch(
        self,
    ) -> None:
        captured_advantages: list[np.ndarray] = []

        def capture_minibatch(state, minibatch):
            captured_advantages.append(np.asarray(minibatch.advantages))
            zero = jnp.asarray(0.0)
            from dreamer.coinrun_ppo import PPOLosses

            return state, PPOLosses(
                total_loss=zero,
                policy_loss=zero,
                value_loss=zero,
                entropy=zero,
                approx_kl=zero,
                clip_fraction=zero,
            )

        batch = PPOBatch(
            observations=jnp.arange(16, dtype=jnp.float32).reshape(8, 2),
            actions=jnp.zeros((8,), dtype=jnp.int32),
            old_log_probs=jnp.zeros((8,)),
            old_values=jnp.zeros((8,)),
            advantages=jnp.asarray(
                [-10.0, -3.0, -1.0, 0.0, 2.0, 4.0, 9.0, 20.0]
            ),
            returns=jnp.zeros((8,)),
        )

        update_ppo(
            state=object(),
            batch=batch,
            key=jax.random.PRNGKey(101),
            hyperparameters=PPOHyperparameters(
                minibatch_size=4,
                update_epochs=2,
                advantage_normalization="minibatch",
            ),
            minibatch_step=capture_minibatch,
        )

        self.assertEqual(len(captured_advantages), 4)
        for advantages in captured_advantages:
            self.assertAlmostEqual(float(np.mean(advantages)), 0.0, places=6)
            self.assertAlmostEqual(float(np.std(advantages)), 1.0, places=6)

    def test_actor_critic_accepts_official_glorot_backbone_initialization(
        self,
    ) -> None:
        observations = jnp.zeros((2, 64, 64, 3), dtype=jnp.uint8)
        official = CoinRunActorCritic(
            action_dim=COINRUN_ACTION_DIM,
            backbone_kernel_init="glorot_uniform",
        )
        legacy = CoinRunActorCritic(
            action_dim=COINRUN_ACTION_DIM,
            backbone_kernel_init="orthogonal_sqrt2",
        )

        official_params = official.init(
            jax.random.PRNGKey(102),
            observations,
        )["params"]
        legacy_params = legacy.init(
            jax.random.PRNGKey(102),
            observations,
        )["params"]
        official_kernel = np.asarray(
            official_params["ImpalaConvSequence_0"]["Conv_0"]["kernel"]
        )
        legacy_kernel = np.asarray(
            legacy_params["ImpalaConvSequence_0"]["Conv_0"]["kernel"]
        )

        self.assertEqual(official_kernel.shape, legacy_kernel.shape)
        self.assertFalse(np.array_equal(official_kernel, legacy_kernel))
        self.assertTrue(np.isfinite(official_kernel).all())

    def test_actor_critic_accepts_unit_gain_orthogonal_initialization(
        self,
    ) -> None:
        observations = jnp.zeros((2, 64, 64, 3), dtype=jnp.uint8)
        unit_gain = CoinRunActorCritic(
            action_dim=COINRUN_ACTION_DIM,
            backbone_kernel_init="orthogonal_gain1",
        )

        params = unit_gain.init(
            jax.random.PRNGKey(103),
            observations,
        )["params"]
        kernel = np.asarray(
            params["ImpalaConvSequence_0"]["Conv_0"]["kernel"]
        )

        self.assertEqual(kernel.shape, (3, 3, 3, 16))
        self.assertTrue(np.isfinite(kernel).all())

    def test_residual_branch_scale_changes_only_the_residual_contribution(
        self,
    ) -> None:
        inputs = jnp.arange(2 * 8 * 8 * 4, dtype=jnp.float32).reshape(
            2, 8, 8, 4
        ) / 100.0
        full = ResidualBlock(
            channels=4,
            backbone_kernel_init="orthogonal_sqrt2",
            residual_branch_scale=1.0,
        )
        scaled = ResidualBlock(
            channels=4,
            backbone_kernel_init="orthogonal_sqrt2",
            residual_branch_scale=0.25,
        )
        variables = full.init(jax.random.PRNGKey(104), inputs)

        full_output = full.apply(variables, inputs)
        scaled_output = scaled.apply(variables, inputs)

        np.testing.assert_allclose(
            np.asarray(scaled_output - inputs),
            0.25 * np.asarray(full_output - inputs),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_zero_last_residual_kernel_starts_as_exact_identity(self) -> None:
        inputs = jnp.arange(2 * 8 * 8 * 4, dtype=jnp.float32).reshape(
            2, 8, 8, 4
        ) / 100.0
        block = ResidualBlock(
            channels=4,
            backbone_kernel_init="orthogonal_sqrt2",
            residual_last_kernel_init="zeros",
        )
        variables = block.init(jax.random.PRNGKey(105), inputs)

        output = block.apply(variables, inputs)

        np.testing.assert_array_equal(np.asarray(output), np.asarray(inputs))
        np.testing.assert_array_equal(
            np.asarray(variables["params"]["Conv_1"]["kernel"]),
            np.zeros_like(np.asarray(variables["params"]["Conv_1"]["kernel"])),
        )

    def test_skip_init_starts_as_identity_with_learnable_zero_gate(self) -> None:
        inputs = jnp.arange(2 * 8 * 8 * 4, dtype=jnp.float32).reshape(
            2, 8, 8, 4
        ) / 100.0
        block = ResidualBlock(
            channels=4,
            backbone_kernel_init="orthogonal_sqrt2",
            residual_skip_init=True,
        )
        variables = block.init(jax.random.PRNGKey(106), inputs)

        output = block.apply(variables, inputs)

        np.testing.assert_array_equal(np.asarray(output), np.asarray(inputs))
        self.assertEqual(float(variables["params"]["skip_init_gain"]), 0.0)

    def test_compiled_minibatch_step_can_be_reused_across_ppo_updates(
        self,
    ) -> None:
        class TinyActorCritic(nn.Module):
            @nn.compact
            def __call__(self, observations):
                hidden = nn.Dense(8)(observations)
                return nn.Dense(3)(hidden), nn.Dense(1)(hidden)[..., 0]

        model = TinyActorCritic()
        observations = jnp.arange(32, dtype=jnp.float32).reshape(8, 4)
        params = model.init(jax.random.PRNGKey(13), observations)["params"]
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adam(3e-4),
        )
        logits, values = model.apply({"params": params}, observations)
        actions = jnp.asarray([0, 1, 2, 0, 1, 2, 0, 1])
        batch = PPOBatch(
            observations=observations,
            actions=actions,
            old_log_probs=categorical_log_prob(logits, actions),
            old_values=values,
            advantages=jnp.linspace(-1.0, 1.0, 8),
            returns=values + 1.0,
        )
        hyperparameters = PPOHyperparameters(
            minibatch_size=4,
            update_epochs=1,
        )
        minibatch_step = make_ppo_minibatch_step(
            apply_fn=state.apply_fn,
            hyperparameters=hyperparameters,
        )

        state, _ = update_ppo(
            state=state,
            batch=batch,
            key=jax.random.PRNGKey(14),
            hyperparameters=hyperparameters,
            minibatch_step=minibatch_step,
        )
        state, _ = update_ppo(
            state=state,
            batch=batch,
            key=jax.random.PRNGKey(15),
            hyperparameters=hyperparameters,
            minibatch_step=minibatch_step,
        )

        self.assertEqual(int(state.step), 4)

    def test_frozen_policy_checkpoint_samples_reproducibly_and_in_bounds(
        self,
    ) -> None:
        model = CoinRunActorCritic(action_dim=COINRUN_ACTION_DIM)
        observations = jnp.zeros((2, 64, 64, 3), dtype=jnp.uint8)
        params = model.init(jax.random.PRNGKey(5), observations)["params"]
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adam(3e-4),
        )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.msgpack"
            save_train_state(
                checkpoint,
                state,
                metadata={
                    "schema_version": "1.0",
                    "action_dim": COINRUN_ACTION_DIM,
                    "architecture": "impala_cnn",
                    "completed_env_steps": 1000,
                },
            )
            left = PPOCoinRunPolicy.from_checkpoint(
                checkpoint,
                num_envs=2,
                seed=99,
            )
            right = PPOCoinRunPolicy.from_checkpoint(
                checkpoint,
                num_envs=2,
                seed=99,
            )
            left_actions = left.sample(
                np.zeros((2, 64, 64, 3), dtype=np.uint8),
                np.ones((2,), dtype=bool),
            )
            right_actions = right.sample(
                np.zeros((2, 64, 64, 3), dtype=np.uint8),
                np.ones((2,), dtype=bool),
            )

        self.assertEqual(left_actions.tolist(), right_actions.tolist())
        self.assertTrue(np.all(left_actions >= 0))
        self.assertTrue(np.all(left_actions < COINRUN_ACTION_DIM))


if __name__ == "__main__":
    unittest.main()
