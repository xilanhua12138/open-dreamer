#!/usr/bin/env python3
"""Probe the exact local parameter and FLOPs identity for the 28.7M label."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
from flax import nnx

from dreamer.models import Tokenizer
from dreamer.parallel import build_parallel
from dreamer.utils import count_parameters_by_component
from scripts.experiments.coinrun.probe_coinrun_tokenizer_scale_v2 import (
    build_config,
)
from scripts.experiments.coinrun.tokenizer_28p7_protocol import (
    CANDIDATE,
    MAX_STEPS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mesh, _, mesh_rules = build_parallel("data")
    with jax.set_mesh(mesh):
        model = Tokenizer(
            build_config(CANDIDATE),
            mesh_rules=mesh_rules,
            rngs=nnx.Rngs(0),
        )
        parameters = int(count_parameters_by_component(model)["total"])
        flops_per_step = int(
            model.estimate_flops(batch_size=128, seq_length=16)
        )
    expected_parameters = int(CANDIDATE["expected_parameters"])
    if parameters != expected_parameters:
        raise ValueError(
            "28.7M-label parameter identity changed: "
            f"expected {expected_parameters}, observed {parameters}"
        )
    payload = {
        "schema_version": "1.0",
        "candidate": {
            **CANDIDATE,
            "parameters": parameters,
            "batch_size": 128,
            "sequence_length": 16,
            "n_latents": 16,
            "d_bottleneck": 16,
            "training_mode": "fixed_optimizer_updates",
            "allocated_steps": MAX_STEPS,
            "flops_per_step": flops_per_step,
            "estimated_training_flops": flops_per_step * MAX_STEPS,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
