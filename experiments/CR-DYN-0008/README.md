# CR-DYN-0008 — PPO-checkpoint mixture ablation

## Question

When total data, tokenizer, medium dynamics model, optimizer budget and held-out
futures are fixed, which mixture of trajectories from four PPO checkpoints
produces the best CoinRun world-model rollout quality?

## Design

The source checkpoints are `1.05M / 6.29M / 12.58M / 25.17M` completed
environment transitions. Four source pools are collected with identical
settings. Three 2,048-record training corpora are then materialized:

| Mixture | 1.05M | 6.29M | 12.58M | 25.17M |
|---|---:|---:|---:|---:|
| `final_only` | 0 | 0 | 0 | 2,048 |
| `uniform` | 512 | 512 | 512 | 512 |
| `recency_weighted` | 256 | 256 | 512 | 1,024 |

Mixtures use symlinks to byte-identical source shards. The shard-aligned ratios
avoid rewriting trajectories, while a stable per-stage permutation and nested
prefixes ensure that overlapping stages reuse the same records.

Every arm trains the same 3,931,392-parameter medium dynamics model from
scratch for 20,000 updates with the 14,912,320-parameter 16.6M EMA tokenizer.
All three arms are evaluated on the same final-policy held-out futures. Only
after all arms finish does the frozen rule rank mean-frame PSNR, mean SSIM and
horizon-16 PSNR lexicographically.

## Observed

The frozen branch is now checked out cleanly at
`/mnt/workspace/open-dreamer-dynamics-checkpoint-mixtures`. Runner SHA256,
16 targeted tests, Python dependency imports and all four PPO plus 16.6M
tokenizer checkpoint inputs passed CPU preflight.

No source-pool collection, mixture materialization, dynamics update or
evaluation has started. At the launch check, `CR-PPO-0005` used 12,600 MiB of
the A10 at 97% utilization, so this experiment is blocked rather than
concurrently launched.

## Interpretation

None yet.

## Not established

- Which checkpoint mixture is best.
- Whether the recency-weighted mixture improves action-conditioned prediction.
- Whether any resulting checkpoint is interactively controllable.

## Decision

Recheck all GPU processes after `CR-PPO-0005` exits, then launch once. No arm
may be skipped based on an interim quality result.
