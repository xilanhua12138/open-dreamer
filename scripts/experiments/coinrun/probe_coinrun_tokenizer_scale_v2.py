#!/usr/bin/env python3
"""Probe exact sizes and native step allocations for CoinRun tokenizer v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
from flax import nnx

from dreamer.configs import (
    DecoderModelConfig,
    EncoderModelConfig,
    TokenizerModelConfig,
)
from dreamer.models import Tokenizer
from dreamer.parallel import build_parallel
from dreamer.utils import count_parameters_by_component


CANDIDATES = (
    {
        "name": "n0.17m",
        "depth": 1,
        "d_model": 64,
        "training_flops_budget": 1e16,
        "phase": "primary_fixed_flops",
    },
    {
        "name": "n1.1m",
        "depth": 2,
        "d_model": 128,
        "training_flops_budget": 1e16,
        "phase": "primary_fixed_flops",
    },
    {
        "name": "n3.7m",
        "depth": 3,
        "d_model": 192,
        "training_flops_budget": 1e16,
        "phase": "primary_fixed_flops",
    },
    {
        "name": "n8.6m",
        "depth": 4,
        "d_model": 256,
        "training_flops_budget": 2e16,
        "phase": "unconditional_capacity_sweep",
    },
    {
        "name": "n16.6m",
        "depth": 5,
        "d_model": 320,
        "training_flops_budget": 4e16,
        "phase": "unconditional_capacity_sweep",
    },
)


def build_config(candidate: dict) -> TokenizerModelConfig:
    depth = candidate["depth"]
    d_model = candidate["d_model"]
    common = {
        "depth": depth,
        "d_model": d_model,
        "n_heads": d_model // 64,
        "n_kv_heads": 1,
        "n_latents": 16,
        "d_bottleneck": 16,
        "patch_size": 8,
        "dropout_rate": 0.0,
        "qk_norm_type": "qknorm",
        "time_every": depth,
        "time_layer_offset": depth - 1,
        "context_length": 16,
        "dtype": "bfloat16",
        "param_dtype": "float32",
    }
    return TokenizerModelConfig(
        encoder=EncoderModelConfig(**common),
        decoder=DecoderModelConfig(
            **common,
            d_patch=8 * 8 * 3,
            H=64,
            W=64,
        ),
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
                    "flops_per_step": flops_per_step,
                    "allocated_steps": int(
                        candidate["training_flops_budget"] / flops_per_step
                    ),
                }
            )

    payload = {"schema_version": "1.0", "candidates": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
