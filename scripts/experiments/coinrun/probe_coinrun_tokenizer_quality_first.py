#!/usr/bin/env python3
"""Probe exact model sizes for the fixed-20k tokenizer quality sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
from flax import nnx

from dreamer.models import Tokenizer
from dreamer.parallel import build_parallel
from dreamer.utils import count_parameters_by_component
from scripts.experiments.coinrun.probe_coinrun_tokenizer_scale_v2 import build_config
from scripts.experiments.coinrun.tokenizer_quality_first_protocol import (
    CANDIDATES,
    MAX_STEPS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mesh, _, mesh_rules = build_parallel("data")
    rows = []
    with jax.set_mesh(mesh):
        for candidate in CANDIDATES:
            model = Tokenizer(
                build_config(candidate),
                mesh_rules=mesh_rules,
                rngs=nnx.Rngs(0),
            )
            parameters = int(count_parameters_by_component(model)["total"])
            flops_per_step = int(
                model.estimate_flops(batch_size=128, seq_length=16)
            )
            rows.append(
                {
                    **candidate,
                    "parameters": parameters,
                    "batch_size": 128,
                    "sequence_length": 16,
                    "n_latents": 16,
                    "d_bottleneck": 16,
                    "training_mode": "fixed_optimizer_updates",
                    "training_flops_budget": 0.0,
                    "allocated_steps": MAX_STEPS,
                    "flops_per_step": flops_per_step,
                    "estimated_training_flops": flops_per_step * MAX_STEPS,
                }
            )

    payload = {"schema_version": "1.0", "candidates": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
