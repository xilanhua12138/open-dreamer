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
evaluation has started. The first launch check found `CR-PPO-0005` using
12,600 MiB of the A10 at 97% utilization, so this experiment was correctly
blocked rather than launched concurrently. After PPO stopped, exactly one
restart request was sent to the same A10; at 18:55:21 Asia/Shanghai the provider
reported `Sales of this resource are temporarily suspended in the specified
zone` and the instance returned to `Failed`.

At 22:11 Asia/Shanghai the frequent launch poller observed the same A10 as
`Running`. Its remote preflight then failed twice with
`InvalidSecurityToken.Expired` from the local ProxyClient credential. The
poller therefore did not attempt a dynamics launch. Until that temporary
credential is refreshed, remote GPU and pipeline state remain unverified.

## Interpretation

None yet.

## Not established

- Which checkpoint mixture is best.
- Whether the recency-weighted mixture improves action-conditioned prediction.
- Whether any resulting checkpoint is interactively controllable.

## Decision

Do not restart the now-Running instance, change specification, create a
replacement instance or bypass the exclusive poller. Refresh only the expired
local ProxyClient temporary credential from the existing `open-dreamer` OAuth
profile. The poller must then repeat the full GPU/PID/source/checkpoint
preflight before its single launch. No arm may be skipped based on an interim
quality result.
