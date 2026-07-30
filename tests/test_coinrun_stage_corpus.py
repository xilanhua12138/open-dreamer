from __future__ import annotations

import copy
import unittest

from dreamer.coinrun_stage_corpus import (
    PPO_STAGE_ENV_STEPS,
    validate_stage_corpus,
)


def metadata(
    *,
    stage: str,
    steps: int,
    records: int = 2_048,
    seed: int = 20_240,
    start_level: int = 0,
    num_levels: int = 200,
) -> dict:
    return {
        "experiment_id": "CR-DYN-0006",
        "env": "coinrun",
        "records": records,
        "frames_per_record": 64,
        "seed": seed,
        "start_level": start_level,
        "num_levels": num_levels,
        "distribution_mode": "easy",
        "action_policy": "ppo",
        "action_space": {
            "categorical_action_dim": 15,
            "categorical_noop_action": 4,
        },
        "policy": {
            "checkpoint_completed_env_steps": steps,
            "temperature": 1.0,
            "exploration_epsilon": 0.05,
            "deterministic": False,
        },
        "transition_alignment": (
            "raw_video[t]=observation_t, actions[t]=action_t, "
            "rewards[t]=reward observed after action_t"
        ),
        "consumer": "action_conditioned_dynamics_only",
        "tree_sha256": f"tree-{stage}",
    }


class CoinRunStageCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = {
            stage: metadata(stage=stage, steps=steps)
            for stage, steps in PPO_STAGE_ENV_STEPS.items()
        }
        self.evaluation = metadata(
            stage="evaluation",
            steps=PPO_STAGE_ENV_STEPS["ppo25p17m"],
            records=512,
            seed=30_240,
            start_level=10_000,
            num_levels=500,
        )

    def test_accepts_exact_four_stage_corpus_and_fixed_final_policy_eval(
        self,
    ) -> None:
        self.assertEqual(
            validate_stage_corpus(
                training=self.training,
                evaluation=self.evaluation,
            ),
            [],
        )

    def test_rejects_wrong_checkpoint_stage(self) -> None:
        broken = copy.deepcopy(self.training)
        broken["ppo06p29m"]["policy"][
            "checkpoint_completed_env_steps"
        ] = 1_048_576

        errors = validate_stage_corpus(
            training=broken,
            evaluation=self.evaluation,
        )

        self.assertIn(
            "ppo06p29m: checkpoint_completed_env_steps=1048576, "
            "expected 6291456",
            errors,
        )

    def test_rejects_eval_overlap_and_action_contract_drift(self) -> None:
        broken_eval = copy.deepcopy(self.evaluation)
        broken_eval["start_level"] = 100
        broken_eval["action_space"]["categorical_noop_action"] = 8

        errors = validate_stage_corpus(
            training=self.training,
            evaluation=broken_eval,
        )

        self.assertIn("evaluation: invalid categorical no-op action", errors)
        self.assertIn(
            "evaluation: level range must be [10000, 10500)",
            errors,
        )
        self.assertIn("training and evaluation level ranges overlap", errors)


if __name__ == "__main__":
    unittest.main()
