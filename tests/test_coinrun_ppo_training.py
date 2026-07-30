from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.coinrun_ppo_training import (
    PPOTrainConfig,
    RewardNormalizer,
    build_training_batch,
    make_action_sampler,
    sample_actions_and_values,
)


class CoinRunPPOTrainingTests(unittest.TestCase):
    def test_config_derives_exact_update_and_minibatch_counts(self) -> None:
        config = PPOTrainConfig(
            total_env_steps=1_024,
            num_envs=8,
            rollout_steps=16,
            num_minibatches=4,
            update_epochs=3,
            reward_normalization_gamma=0.99,
            advantage_normalization="minibatch",
            backbone_kernel_init="glorot_uniform",
        )

        config.validate()

        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.minibatch_size, 32)
        self.assertEqual(config.num_updates, 8)
        self.assertEqual(config.optimizer_minibatches_per_update, 12)
        self.assertEqual(config.reward_normalization_gamma, 0.99)
        self.assertEqual(config.advantage_normalization, "minibatch")
        self.assertEqual(config.backbone_kernel_init, "glorot_uniform")

    def test_config_rejects_inexact_environment_step_budget(self) -> None:
        config = PPOTrainConfig(
            total_env_steps=1_000,
            num_envs=8,
            rollout_steps=16,
        )

        with self.assertRaisesRegex(ValueError, "divisible"):
            config.validate()

    def test_config_rejects_unknown_reference_recipe_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "advantage_normalization"):
            PPOTrainConfig(advantage_normalization="global-ish").validate()

        with self.assertRaisesRegex(ValueError, "backbone_kernel_init"):
            PPOTrainConfig(backbone_kernel_init="mystery").validate()

    def test_reward_normalizer_resets_discounted_return_after_terminal(self) -> None:
        normalizer = RewardNormalizer(num_envs=2, gamma=0.9, clip=10.0)

        first = normalizer.normalize(
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([False, True]),
        )
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(
            normalizer.discounted_returns,
            np.asarray([1.0, 0.0]),
        )

        normalizer.normalize(
            np.asarray([1.0, 1.0], dtype=np.float32),
            np.asarray([False, False]),
        )
        np.testing.assert_allclose(
            normalizer.discounted_returns,
            np.asarray([1.9, 1.0]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_training_batch_flattens_time_and_env_without_losing_alignment(
        self,
    ) -> None:
        observations = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
        actions = jnp.asarray([[0, 1, 2], [2, 1, 0]])
        old_log_probs = jnp.zeros((2, 3))
        values = jnp.asarray([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
        rewards = jnp.ones((2, 3))
        terminals = jnp.asarray(
            [[False, True, False], [False, False, True]]
        )

        batch = build_training_batch(
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            values=values,
            rewards=rewards,
            terminals=terminals,
            next_value=jnp.asarray([4.0, 5.0, 6.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )

        self.assertEqual(batch.observations.shape, (6, 4))
        self.assertEqual(batch.actions.tolist(), [0, 1, 2, 2, 1, 0])
        self.assertEqual(batch.old_values.tolist(), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        self.assertEqual(batch.advantages.shape, (6,))
        self.assertEqual(batch.returns.shape, (6,))

    def test_policy_sampling_reports_log_prob_and_value_for_same_actions(
        self,
    ) -> None:
        def apply_fn(variables, observations):
            del variables
            logits = jnp.tile(jnp.asarray([[0.0, 1.0, -1.0]]), (observations.shape[0], 1))
            values = jnp.arange(observations.shape[0], dtype=jnp.float32)
            return logits, values

        actions, log_probs, values, next_key = sample_actions_and_values(
            apply_fn=apply_fn,
            params={},
            observations=jnp.zeros((4, 2)),
            key=jax.random.PRNGKey(12),
        )

        self.assertEqual(actions.shape, (4,))
        self.assertEqual(log_probs.shape, (4,))
        self.assertEqual(values.tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertTrue(np.all(np.asarray(actions) >= 0))
        self.assertTrue(np.all(np.asarray(actions) < 3))
        self.assertFalse(np.array_equal(np.asarray(next_key), np.asarray(jax.random.PRNGKey(12))))

    def test_compiled_action_sampler_can_be_reused_with_new_params_and_key(
        self,
    ) -> None:
        def apply_fn(variables, observations):
            bias = variables["params"]["bias"]
            logits = jnp.tile(bias[None], (observations.shape[0], 1))
            values = jnp.sum(observations, axis=-1)
            return logits, values

        sampler = make_action_sampler(apply_fn=apply_fn)
        observations = jnp.ones((2, 3))
        first = sampler(
            {"bias": jnp.asarray([0.0, 1.0, -1.0])},
            observations,
            jax.random.PRNGKey(20),
        )
        second = sampler(
            {"bias": jnp.asarray([1.0, 0.0, -1.0])},
            observations,
            first[3],
        )

        self.assertEqual(first[0].shape, (2,))
        self.assertEqual(first[1].shape, (2,))
        self.assertEqual(first[2].tolist(), [3.0, 3.0])
        self.assertFalse(np.array_equal(np.asarray(first[3]), np.asarray(second[3])))


if __name__ == "__main__":
    unittest.main()
