from __future__ import annotations

import copy
import unittest

from dreamer.coinrun_checkpoint_mixture import (
    MIXTURE_RECORDS,
    MIXTURE_SELECTION_SEED,
    MIXTURE_SHARDS_BY_STAGE,
    PPO_STAGE_ENV_STEPS,
    records_by_stage,
    select_stage_shards,
    validate_mixture_metadata,
    validate_mixture_protocol,
)


def source_metadata(stage: str, steps: int) -> dict:
    return {
        "tree_sha256": f"source-tree-{stage}",
        "policy": {
            "checkpoint_completed_env_steps": steps,
            "checkpoint_sha256": f"checkpoint-{stage}",
        },
        "shard_entries": [
            {
                "name": f"shard-{index:05d}.array_record",
                "sha256": f"{stage}-shard-{index}",
            }
            for index in range(8)
        ],
    }


def mixture_metadata(
    mixture_name: str,
    sources: dict[str, dict],
) -> dict:
    shards_by_stage = MIXTURE_SHARDS_BY_STAGE[mixture_name]
    records = records_by_stage(mixture_name)
    source_shards = []
    output_index = 0
    for stage, count in shards_by_stage.items():
        for shard_index in range(count):
            source_shards.append(
                {
                    "output_name": f"shard-{output_index:05d}.array_record",
                    "stage": stage,
                    "source_name": f"shard-{shard_index:05d}.array_record",
                    "sha256": f"{stage}-shard-{shard_index}",
                }
            )
            output_index += 1
    return {
        "experiment_id": "CR-DYN-0008",
        "env": "coinrun",
        "records": MIXTURE_RECORDS,
        "frames_per_record": 64,
        "distribution_mode": "easy",
        "action_policy": "ppo_mixture",
        "selection_seed": MIXTURE_SELECTION_SEED,
        "start_level": 0,
        "num_levels": 200,
        "consumer": "action_conditioned_dynamics_only",
        "mixture": {
            "name": mixture_name,
            "shards_by_stage": shards_by_stage,
            "records_by_stage": records,
        },
        "policy": {
            "source_policies": {
                stage: {
                    "checkpoint_completed_env_steps": steps,
                    "checkpoint_sha256": sources[stage]["policy"][
                        "checkpoint_sha256"
                    ],
                    "source_tree_sha256": sources[stage]["tree_sha256"],
                    "records": records[stage],
                }
                for stage, steps in PPO_STAGE_ENV_STEPS.items()
            }
        },
        "source_shards": source_shards,
        "tree_sha256": f"mixture-tree-{mixture_name}",
    }


class CoinRunCheckpointMixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = {
            stage: source_metadata(stage, steps)
            for stage, steps in PPO_STAGE_ENV_STEPS.items()
        }
        self.mixtures = {
            name: mixture_metadata(name, self.sources)
            for name in MIXTURE_SHARDS_BY_STAGE
        }

    def test_registered_mixtures_hold_total_records_constant(self) -> None:
        self.assertEqual(validate_mixture_protocol(), [])
        self.assertEqual(
            {
                name: sum(records_by_stage(name).values())
                for name in MIXTURE_SHARDS_BY_STAGE
            },
            {
                "final_only": 2_048,
                "uniform": 2_048,
                "recency_weighted": 2_048,
            },
        )

    def test_stage_selection_is_deterministic_and_nested(self) -> None:
        shard_names = [
            f"shard-{index:05d}.array_record" for index in range(8)
        ]

        selected_two = select_stage_shards(
            stage="ppo12p58m",
            shard_names=shard_names,
            count=2,
        )
        selected_four = select_stage_shards(
            stage="ppo12p58m",
            shard_names=list(reversed(shard_names)),
            count=4,
        )

        self.assertEqual(len(selected_two), 2)
        self.assertEqual(len(selected_four), 4)
        self.assertTrue(set(selected_two).issubset(selected_four))
        self.assertEqual(
            selected_two,
            select_stage_shards(
                stage="ppo12p58m",
                shard_names=shard_names,
                count=2,
            ),
        )

    def test_accepts_exact_materialized_mixture_metadata(self) -> None:
        self.assertEqual(
            validate_mixture_metadata(
                mixtures=self.mixtures,
                source_training=self.sources,
            ),
            [],
        )

    def test_rejects_ratio_and_checkpoint_identity_drift(self) -> None:
        broken = copy.deepcopy(self.mixtures)
        broken["uniform"]["mixture"]["records_by_stage"]["ppo01p05m"] = 256
        broken["recency_weighted"]["policy"]["source_policies"][
            "ppo25p17m"
        ]["checkpoint_sha256"] = "wrong"

        errors = validate_mixture_metadata(
            mixtures=broken,
            source_training=self.sources,
        )

        self.assertIn(
            "uniform: records_by_stage does not match protocol",
            errors,
        )
        self.assertIn(
            "recency_weighted/ppo25p17m: checkpoint SHA256 mismatch",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
