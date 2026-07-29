# CR-DYN-0005 — 4/16/32 history-context ablation

## Question

For one selected checkpoint and exactly the same future frames/actions, how does conditioning on 4, 16, or 32 history frames affect 16-frame CoinRun rollout quality?

## Why this experiment

The completed dynamics runs evaluate only four context frames, while the trained model supports longer context. A useful context ablation must not accidentally change the predicted future when history length changes.

## Hypothesis and falsifier

- Hypothesis: longer history improves or preserves PSNR/SSIM because it disambiguates velocity and recent action effects.
- Falsified if: 16/32 frames consistently reduce the controlled metrics or the result is non-monotonic beyond measurement noise.

## Controlled design

- One checkpoint selected after `CR-DYN-0004` by held-out mean-frame PSNR, SSIM tiebreaker.
- Contexts: 4, 16, 32 frames.
- Horizon: the same next 16 frames in every arm.
- The same episodes, future actions, start positions, sampler, denoise steps, seed and PRNG usage.
- Each example uses a 48-frame window so context grows backward while the future endpoint remains fixed.
- Metrics are computed from raw frames before MP4 encoding.

If any future frame hash differs between arms, mark the experiment `invalid`.

## Results

| Context frames | PSNR@1 | PSNR@3 | PSNR@8 | PSNR@16 | SSIM@16 |
|---:|---:|---:|---:|---:|---:|
| 4 | 22.64 | 20.91 | 18.29 | 16.88 | 0.569 |
| 16 | 22.69 | 20.83 | 18.73 | **17.15** | **0.592** |
| 32 | 22.44 | 20.51 | 18.54 | 17.02 | 0.589 |

### Observed

- Medium was selected once after the large-inclusive capacity comparison.
- The `gt_decoded` future SHA256 was exactly `4345583fcef0e07ae8fa46ae46a53242c642ca2e8c1f94dea893b470a677ee3f` in the 4/16/32 raw arrays, so the result is valid under the frozen-future contract.
- Context 16 beat context 4 by `0.26713 dB` PSNR and `0.02270` SSIM.
- Context 32 also beat context 4, but was slightly below context 16.

### Interpretation

Longer history helped relative to four frames, supporting the hypothesis, but gains plateaued: the longest tested history was not the best. Context 16 is the practical choice for this checkpoint.

### Not established

- Whether the small 16-versus-32 difference is stable across seeds.
- Whether context 16 remains best for other checkpoints, data scales, or horizons.
- Subjective action controllability.

### Decision

Use medium with 16 context frames in the live demo. Preserve the same-future array-hash check for any rerun.
