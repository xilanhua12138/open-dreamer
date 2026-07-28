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

## Current result

### Observed

The evaluator and orchestration scripts are prepared, but no selected large-inclusive checkpoint exists because `CR-DYN-0004` has not run. There are no 4/16/32 metrics.

### Interpretation

The experiment is dependency-blocked, not scientifically negative.

### Not established

- Whether four frames are enough.
- Whether longer context helps.
- Which context should be used in the live demo.

### Decision

After the large run, select one checkpoint once, freeze the future manifest, run all contexts without changing the checkpoint, and save hashes that prove future identity.
