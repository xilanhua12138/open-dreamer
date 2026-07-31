from __future__ import annotations

import unittest

import jax.numpy as jnp

from dreamer.actions import Actions, shift_actions


class CoinRunActionContractTests(unittest.TestCase):
    def test_shift_actions_uses_explicit_procgen_noop(self) -> None:
        actions = Actions(categorical=jnp.asarray([[1, 7, 5]], dtype=jnp.int32))

        shifted = shift_actions(
            actions,
            categorical_action_dim=15,
            categorical_noop_action=4,
        )

        self.assertEqual(shifted.categorical.tolist(), [[4, 1, 7]])

    def test_shift_actions_rejects_noop_outside_action_space(self) -> None:
        actions = Actions(categorical=jnp.asarray([[1, 7, 5]], dtype=jnp.int32))

        with self.assertRaisesRegex(ValueError, "categorical_noop_action"):
            shift_actions(
                actions,
                categorical_action_dim=15,
                categorical_noop_action=15,
            )

    def test_shift_actions_rejects_missing_noop_for_categorical_actions(self) -> None:
        actions = Actions(categorical=jnp.asarray([[1, 7, 5]], dtype=jnp.int32))

        with self.assertRaisesRegex(ValueError, "categorical_noop_action"):
            shift_actions(
                actions,
                categorical_action_dim=15,
                categorical_noop_action=None,
            )


if __name__ == "__main__":
    unittest.main()
