from __future__ import annotations

import unittest
from pathlib import Path

from scripts.experiments.coinrun.tokenizer_28p7_protocol import build_plan


class Tokenizer28p7ProtocolTests(unittest.TestCase):
    def test_single_extension_arm_preserves_fixed_20k_comparison_contract(self) -> None:
        plan = build_plan()

        self.assertEqual(plan["experiment_id"], "CR-TOK-0004")
        self.assertEqual(plan["baseline_experiment_id"], "CR-TOK-0003")
        self.assertEqual(
            plan["candidate"],
            {
                "name": "n28.7m",
                "directory": "n28p7m",
                "depth": 6,
                "d_model": 384,
                "expected_parameters": 25_564_032,
                "max_steps": 20_000,
                "scaling_flops_budget": 0.0,
                "scaling_tokens_per_param": 0.0,
            },
        )
        self.assertEqual(
            plan["heldout_evaluation_completed_updates"],
            [2_500, 5_000, 10_000, 20_000],
        )
        self.assertEqual(
            plan["checkpoint_steps"],
            [2_499, 4_999, 9_999, 19_999],
        )

    def test_periodic_validation_and_downstream_guard_are_explicit(self) -> None:
        plan = build_plan()

        self.assertEqual(plan["periodic_validation"]["every_steps"], 2_500)
        self.assertEqual(plan["periodic_validation"]["seed"], 4_242)
        self.assertEqual(plan["periodic_validation"]["frames"], 16)
        self.assertEqual(plan["periodic_validation"]["batch_size"], 8)
        self.assertEqual(plan["periodic_validation"]["batches"], 2)
        self.assertFalse(plan["dynamics_authorized"])
        self.assertEqual(plan["terminal_state"], "awaiting_visual_review")

    def test_runner_uses_recorded_runtime_wandb_and_fixed_validation(self) -> None:
        runner = Path(
            "scripts/experiments/coinrun/run_coinrun_tokenizer_28p7_fixed20k.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("scripts/experiments/run_recorded.py", runner)
        self.assertIn("use_wandb=true", runner)
        self.assertIn("logger.wandb_group=CR-TOK-0004", runner)
        self.assertIn("validation.enabled=true", runner)
        self.assertIn("validation.every_steps=2500", runner)
        self.assertIn("validation.seed=4242", runner)
        self.assertIn("dynamics_blocked", runner)


if __name__ == "__main__":
    unittest.main()
