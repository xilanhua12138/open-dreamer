# CR-DYN-0003 — fixed-20k capacity ablation

## Question

When tiny, small, and medium CoinRun dynamics models receive the same 20,000-step curriculum, does held-out rollout quality improve with capacity?

## Why this experiment

`CR-DYN-0002` confounded model capacity with a 32× optimizer-step range and did not finish tiny. This experiment holds optimization steps and curriculum constant so the primary changed variable is model capacity.

## Hypothesis and falsifier

- Hypothesis: with adequate and equal optimization, held-out rollout PSNR and SSIM improve monotonically from tiny to small to medium.
- Falsified if: either primary metric fails to improve monotonically, training is unstable, or a candidate does not complete the shared protocol.

## Controlled design

- Fixed: tokenizer, train/eval data, batch 32 × 64 frames, 20,000 steps, flow-only through step 10,000, bootstrap fraction 0.25 afterward, Muon, WSD, OT coupling, EMA shortcut sampling, eight held-out trajectories, context 4, horizon 16, seed 4242.
- Changed: depth/width/head/register configuration and therefore parameter count.
- Sizes: 155,840 / 545,920 / 3,931,392 parameters.
- Metric target: tokenizer-decoded ground truth.

## Results

| Model | Parameters | PSNR@1 | PSNR@3 | PSNR@8 | PSNR@16 | SSIM@16 |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 155,840 | 14.97 | 14.81 | 13.15 | 12.77 | 0.337 |
| small | 545,920 | 18.68 | 16.40 | 14.41 | 14.17 | 0.465 |
| medium | 3,931,392 | 20.73 | 19.23 | 16.89 | 16.02 | 0.542 |

### Observed

- Both mean-frame PSNR and SSIM improved monotonically with capacity.
- Medium improved PSNR@16 by `1.8495 dB` over small and `3.2499 dB` over tiny.
- Total training/evaluation wall time was about 79 minutes on one A10.

### Interpretation

At this data scale and shared 20k-step curriculum, the 474-step medium result in `CR-DYN-0002` was dominated by insufficient optimization. Capacity helps under the corrected protocol.

### Not established

- Compute-optimal scaling. The larger models consumed more FLOPs at fixed steps.
- Statistical significance across seeds; only seed 4242 was used for evaluation.
- Official Minecraft or CoinRun reproduction.
- Playable control quality; trajectories are random-action and no policy was trained.

### Decision

Add one larger model at the identical 20k protocol to see whether the monotonic trend continues, then select the best checkpoint by held-out mean-frame PSNR with SSIM as a tiebreaker.
