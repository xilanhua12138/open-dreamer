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

The first launch check found `CR-PPO-0005` using
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

At 01:48:23 on 2026-07-31 the poller refreshed the temporary credential,
repeated the frozen source/checkpoint/GPU preflight and launched exactly one
serial pipeline as PID `929`. It set a 12-hour shutdown timer. Source-pool
collection advanced through all four PPO checkpoints and the runner entered
`final_only-medium`; at 02:01:20 the arm was at `3,711/20,000` updates,
approximately 9.25 steps/s, 99% GPU utilization and 2,344/23,028 MiB.

`final_only-medium` reached its exact terminal state at 02:38:06 with
20,000 completed updates and last step 19,999. Its fixed 32-video held-out
evaluation produced 16.165151 dB mean-video PSNR, 17.073096 dB mean-frame
PSNR and 0.713450 mean SSIM. Horizon 1/3/8/16 PSNR was
22.593902/20.605716/18.681997/17.073096 dB. The source metric file has SHA256
`60c6ada3...1bf707`; the run-state file has SHA256
`4e202149...42b39`.

The runner then entered `uniform-medium`. Its structured run state reached
11,190/20,000 updates at 03:06:24 while the A10 remained at 99% utilization
and 2,364/23,028 MiB. PID `929` remains the sole active pipeline.

`uniform-medium` subsequently completed its exact 20,000-update budget at
03:26:16. Its aligned 32-video held-out evaluation produced 15.627570 dB
mean-video PSNR, 16.567942 dB mean-frame PSNR and 0.697699 mean SSIM.
Horizon 1/3/8/16 PSNR was
21.918747/19.861178/17.937733/16.567942 dB. Final-only provisionally leads
uniform by 0.505154 dB mean-frame PSNR and 0.015751 mean SSIM.

The runner then entered `recency_weighted-medium`; at 04:05:48 its structured
state was 16,654/20,000 updates. PID `929` remained alive and the A10 reported
99% utilization with 2,370/23,028 MiB allocated.

## Interpretation

Two held-out arms are now valid and final-only is ahead of uniform on every
frozen selection key. This is descriptive only: the experiment still cannot
select a mixture or begin the scale sweep until recency-weighted completes.

## Not established

- Which checkpoint mixture is best.
- Whether the recency-weighted mixture improves action-conditioned prediction.
- Whether any resulting checkpoint is interactively controllable.

## Decision

Continue PID `929` through all three mixture arms. Only the frozen complete-arm
selection may choose the CR-DYN-0009 corpus; no arm may be skipped based on an
interim quality result. The five-minute poller must recognize the owned PID and
must not start a second training process.
