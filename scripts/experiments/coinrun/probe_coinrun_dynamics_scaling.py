#!/usr/bin/env python3
"""Report parameter and native FLOPs estimates for CoinRun dynamics candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
from flax import nnx

from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.configs import DynamicsModelConfig
from dreamer.models import Dynamics
from dreamer.parallel import build_parallel
from dreamer.utils import count_parameters_by_component


CANDIDATES = (
    {
        "name": "tiny",
        "depth": 2,
        "d_model": 64,
        "n_heads": 1,
        "n_kv_heads": 1,
        "n_register": 8,
    },
    {
        "name": "small",
        "depth": 2,
        "d_model": 128,
        "n_heads": 2,
        "n_kv_heads": 1,
        "n_register": 16,
    },
    {
        "name": "medium",
        "depth": 4,
        "d_model": 256,
        "n_heads": 4,
        "n_kv_heads": 1,
        "n_register": 32,
    },
    {
        "name": "large",
        "depth": 6,
        "d_model": 384,
        "n_heads": 6,
        "n_kv_heads": 1,
        "n_register": 32,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--budget", type=float, default=1e15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = json.loads(args.latent_stats.read_text(encoding="utf-8"))
    mesh, _, mesh_rules = build_parallel("data")
    rows = []

    with jax.set_mesh(mesh):
        tokenizer_bundle = TokenizerCheckpointBundle.from_pretrained(
            str(args.tokenizer_checkpoint),
            mesh_rules=mesh_rules,
        )
        tokenizer = tokenizer_bundle.tokenizer
        tokenizer_training_flops = tokenizer.estimate_flops(
            batch_size=args.batch_size,
            seq_length=args.frames,
        )
        encoder_flops = tokenizer_training_flops // 6
        n_latents = tokenizer.cfg.encoder.n_latents

        for index, candidate in enumerate(CANDIDATES):
            cfg = DynamicsModelConfig(
                d_bottleneck=16,
                depth=candidate["depth"],
                d_model=candidate["d_model"],
                n_heads=candidate["n_heads"],
                n_kv_heads=candidate["n_kv_heads"],
                packing_factor=2,
                n_register=candidate["n_register"],
                qk_norm_type="qknorm",
                time_every=2,
                time_layer_offset=1,
                mlp_ratio=4,
                dropout_rate=0.0,
                k_max=8,
                context_length=64,
                dtype="bfloat16",
                param_dtype="float32",
                num_binary_actions=0,
                categorical_action_dim=15,
                continuous_action_dim=0,
                latent_mean=tuple(stats["mean"]),
                latent_std=tuple(stats["std"]),
            )
            model = Dynamics(
                cfg,
                mesh_rules=mesh_rules,
                rngs=nnx.Rngs(jax.random.key(100 + index)),
            )
            params = count_parameters_by_component(model)["total"]
            dynamics_flops = model.estimate_flops(
                batch_size=args.batch_size,
                seq_length=args.frames,
                n_latents=n_latents,
            )
            native_flops_per_step = dynamics_flops + encoder_flops
            rows.append(
                {
                    **candidate,
                    "params": params,
                    "dynamics_flops_per_step": dynamics_flops,
                    "encoder_flops_per_step": encoder_flops,
                    "native_flops_per_step": native_flops_per_step,
                    "budget": args.budget,
                    "steps": int(args.budget / native_flops_per_step),
                }
            )

    payload = {
        "tokenizer_checkpoint": str(args.tokenizer_checkpoint),
        "tokenizer_params": count_parameters_by_component(
            tokenizer_bundle.tokenizer
        )["total"],
        "batch_size": args.batch_size,
        "frames": args.frames,
        "budget": args.budget,
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
