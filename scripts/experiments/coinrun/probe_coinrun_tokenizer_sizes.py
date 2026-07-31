#!/usr/bin/env python3
"""Print parameter and FLOP counts for candidate tiny CoinRun tokenizers."""

from __future__ import annotations

import itertools

import jax
from flax import nnx

from dreamer.configs import DecoderModelConfig, EncoderModelConfig, TokenizerModelConfig
from dreamer.models import Tokenizer
from dreamer.parallel import build_parallel
from dreamer.utils import count_parameters_by_component


def build_config(
    *,
    depth: int,
    d_model: int,
    n_latents: int,
    d_bottleneck: int,
) -> TokenizerModelConfig:
    common = {
        "depth": depth,
        "d_model": d_model,
        "n_heads": max(1, d_model // 64),
        "n_kv_heads": 1,
        "n_latents": n_latents,
        "d_bottleneck": d_bottleneck,
        "patch_size": 8,
        "dropout_rate": 0.0,
        "qk_norm_type": "qknorm",
        "time_every": max(1, depth),
        "time_layer_offset": max(0, depth - 1),
        "context_length": 16,
        "dtype": "bfloat16",
        "param_dtype": "float32",
    }
    encoder = EncoderModelConfig(**common)
    decoder = DecoderModelConfig(
        **common,
        d_patch=8 * 8 * 3,
        H=64,
        W=64,
    )
    return TokenizerModelConfig(encoder=encoder, decoder=decoder)


def main() -> None:
    mesh, _, mesh_rules = build_parallel("data")
    with jax.set_mesh(mesh):
        for depth, d_model, n_latents, d_bottleneck in itertools.product(
            (1, 2),
            (64, 96, 128),
            (8, 16, 32),
            (8, 16),
        ):
            if d_model % 64 != 0:
                continue
            model = Tokenizer(
                build_config(
                    depth=depth,
                    d_model=d_model,
                    n_latents=n_latents,
                    d_bottleneck=d_bottleneck,
                ),
                mesh_rules=mesh_rules,
                rngs=nnx.Rngs(0),
            )
            params = int(count_parameters_by_component(model)["total"])
            flops_b8_t16 = model.estimate_flops(batch_size=8, seq_length=16)
            flops_b128_t16 = model.estimate_flops(batch_size=128, seq_length=16)
            print(
                f"depth={depth} d_model={d_model} n_latents={n_latents} "
                f"d_bottleneck={d_bottleneck} params={params} "
                f"flops_b8_t16={flops_b8_t16} "
                f"steps_1e16_b8={int(1e16 / flops_b8_t16)} "
                f"flops_b128_t16={flops_b128_t16} "
                f"steps_1e16_b128={int(1e16 / flops_b128_t16)}"
            )


if __name__ == "__main__":
    main()
