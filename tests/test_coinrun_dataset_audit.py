from __future__ import annotations

import unittest

import numpy as np

from dreamer.coinrun_dataset_audit import (
    validate_action_range,
    validate_ppo_metadata,
    validate_record_terminals,
)


class CoinRunDatasetAuditTests(unittest.TestCase):
    def test_ppo_metadata_requires_checkpoint_identity_and_alignment(self) -> None:
        errors = validate_ppo_metadata(
            {
                "action_policy": "ppo",
                "policy": {
                    "checkpoint_sha256": "a" * 64,
                    "checkpoint_completed_env_steps": 25_165_824,
                },
                "transition_alignment": (
                    "raw_video[t]=observation_t, actions[t]=action_t, "
                    "rewards[t]=reward observed after action_t"
                ),
                "episode_boundary_policy": (
                    "records never cross auto-reset boundaries; partial "
                    "chunks are discarded on terminal"
                ),
                "terminals_in_records": True,
            }
        )

        self.assertEqual(errors, [])

    def test_ppo_metadata_rejects_unverifiable_checkpoint(self) -> None:
        errors = validate_ppo_metadata(
            {
                "action_policy": "ppo",
                "policy": {
                    "checkpoint_sha256": "short",
                    "checkpoint_completed_env_steps": 0,
                },
                "transition_alignment": "ambiguous",
                "episode_boundary_policy": "unknown",
                "terminals_in_records": False,
            }
        )

        self.assertEqual(
            errors,
            [
                "PPO dataset checkpoint_sha256 must be 64 lowercase hex characters",
                "PPO dataset checkpoint_completed_env_steps must be positive",
                "PPO dataset transition_alignment is not action-aligned",
                "PPO dataset must declare episode-safe boundary handling",
                "PPO dataset must store terminal flags in every record",
            ],
        )

    def test_ppo_mixture_metadata_requires_each_checkpoint_identity(
        self,
    ) -> None:
        metadata = {
            "action_policy": "ppo_mixture",
            "policy": {
                "source_policies": {
                    "early": {
                        "checkpoint_sha256": "a" * 64,
                        "checkpoint_completed_env_steps": 1_048_576,
                    },
                    "final": {
                        "checkpoint_sha256": "b" * 64,
                        "checkpoint_completed_env_steps": 25_165_824,
                    },
                }
            },
            "transition_alignment": (
                "raw_video[t]=observation_t, actions[t]=action_t, "
                "rewards[t]=reward observed after action_t"
            ),
            "episode_boundary_policy": (
                "records never cross auto-reset boundaries; partial chunks "
                "are discarded when terminals[t] is true"
            ),
            "terminals_in_records": True,
        }

        self.assertEqual(validate_ppo_metadata(metadata), [])

        metadata["policy"]["source_policies"]["early"][
            "checkpoint_sha256"
        ] = "bad"
        self.assertEqual(
            validate_ppo_metadata(metadata),
            [
                "PPO mixture source early checkpoint_sha256 must be 64 "
                "lowercase hex characters"
            ],
        )

    def test_action_range_rejects_negative_and_action_dim_endpoint(self) -> None:
        self.assertEqual(
            validate_action_range(np.asarray([0, 14, -1, 15]), action_dim=15),
            ["actions outside [0, 15): [-1, 15]"],
        )

    def test_terminal_may_only_appear_at_final_record_frame(self) -> None:
        self.assertEqual(
            validate_record_terminals(
                np.asarray([False, False, True]),
                expected_frames=3,
            ),
            [],
        )
        self.assertEqual(
            validate_record_terminals(
                np.asarray([False, True, False]),
                expected_frames=3,
            ),
            ["terminal flag appears before the final record frame"],
        )


if __name__ == "__main__":
    unittest.main()
