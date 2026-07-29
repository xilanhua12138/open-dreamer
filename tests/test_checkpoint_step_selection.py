from __future__ import annotations

import unittest

from dreamer.checkpoint_steps import (
    normalize_checkpoint_save_steps,
    resolve_checkpoint_step,
)


class CheckpointStepSelectionTests(unittest.TestCase):
    def test_none_selects_latest_available_checkpoint(self) -> None:
        self.assertEqual(resolve_checkpoint_step(None, [2_499, 4_999, 9_999]), 9_999)

    def test_requested_checkpoint_must_exist(self) -> None:
        self.assertEqual(resolve_checkpoint_step(4_999, [2_499, 4_999, 9_999]), 4_999)
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint step 5,000"):
            resolve_checkpoint_step(5_000, [2_499, 4_999, 9_999])

    def test_empty_checkpoint_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "No checkpoint"):
            resolve_checkpoint_step(None, [])

    def test_explicit_milestones_and_final_step_are_retained(self) -> None:
        self.assertEqual(
            normalize_checkpoint_save_steps(20_000, [2_499, 4_999, 9_999]),
            [2_499, 4_999, 9_999, 19_999],
        )

    def test_out_of_range_save_step_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"within \[0, 19,999\]"):
            normalize_checkpoint_save_steps(20_000, [20_000])


if __name__ == "__main__":
    unittest.main()
