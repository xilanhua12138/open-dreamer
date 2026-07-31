from __future__ import annotations

import unittest

from scripts.experiments.coinrun.tokenizer_quality_first_protocol import build_plan


class TokenizerQualityFirstProtocolTests(unittest.TestCase):
    def test_all_five_scales_receive_the_same_20k_update_budget(self) -> None:
        plan = build_plan()

        self.assertEqual(
            [candidate["name"] for candidate in plan["candidates"]],
            ["n0.17m", "n1.1m", "n3.7m", "n8.6m", "n16.6m"],
        )
        self.assertEqual(
            [candidate["max_steps"] for candidate in plan["candidates"]],
            [20_000, 20_000, 20_000, 20_000, 20_000],
        )
        self.assertTrue(
            all(candidate["scaling_flops_budget"] == 0.0 for candidate in plan["candidates"])
        )
        self.assertTrue(
            all(
                candidate["scaling_tokens_per_param"] == 0.0
                for candidate in plan["candidates"]
            )
        )

    def test_milestones_are_exact_completed_update_counts(self) -> None:
        plan = build_plan()

        self.assertEqual(plan["evaluation_completed_updates"], [2_500, 5_000, 10_000, 20_000])
        self.assertEqual(plan["checkpoint_steps"], [2_499, 4_999, 9_999, 19_999])
        self.assertEqual(plan["recovery_checkpoint_steps"], [14_999])

    def test_dynamics_remains_blocked_until_visual_review(self) -> None:
        plan = build_plan()

        self.assertFalse(plan["dynamics_authorized"])
        self.assertEqual(plan["terminal_state"], "awaiting_visual_review")


if __name__ == "__main__":
    unittest.main()
