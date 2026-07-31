from __future__ import annotations

import unittest

from dreamer.coinrun_dataset_pair import validate_dataset_pair


def metadata(
    *,
    start_level: int,
    num_levels: int,
    tree_sha256: str,
    checkpoint_sha256: str = "a" * 64,
) -> dict:
    return {
        "env": "coinrun",
        "action_policy": "ppo",
        "start_level": start_level,
        "num_levels": num_levels,
        "tree_sha256": tree_sha256,
        "action_space": {
            "categorical_action_dim": 15,
            "categorical_noop_action": 4,
        },
        "policy": {
            "checkpoint_sha256": checkpoint_sha256,
        },
    }


class CoinRunDatasetPairTests(unittest.TestCase):
    def test_disjoint_pair_with_same_policy_is_valid(self) -> None:
        errors = validate_dataset_pair(
            train=metadata(
                start_level=20_000,
                num_levels=4_096,
                tree_sha256="b" * 64,
            ),
            evaluation=metadata(
                start_level=30_000,
                num_levels=512,
                tree_sha256="c" * 64,
            ),
        )

        self.assertEqual(errors, [])

    def test_pair_rejects_overlap_checkpoint_mismatch_and_same_tree(self) -> None:
        errors = validate_dataset_pair(
            train=metadata(
                start_level=20_000,
                num_levels=4_096,
                tree_sha256="b" * 64,
            ),
            evaluation=metadata(
                start_level=24_000,
                num_levels=512,
                tree_sha256="b" * 64,
                checkpoint_sha256="c" * 64,
            ),
        )

        self.assertEqual(
            errors,
            [
                "train/eval level ranges overlap",
                "train/eval policy checkpoint SHA256 differs",
                "train/eval dataset tree SHA256 is identical",
            ],
        )


if __name__ == "__main__":
    unittest.main()
