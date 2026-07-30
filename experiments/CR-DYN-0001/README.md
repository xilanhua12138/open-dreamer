# CR-DYN-0001 — minimal complete action-conditioned CoinRun world model

## Question

Can OpenDreamer's tokenizer plus action-conditioned latent dynamics complete the full CoinRun data→training→held-out rollout chain at a small scale?

## Why this experiment

A tokenizer that reconstructs frames is not yet a world model. The missing proof was whether historical latents plus actions could generate a multi-frame future and be decoded back into coherent video.

## Hypothesis and falsifier

- Hypothesis: a 545,920-parameter dynamics transformer trained for 15,000 steps can learn enough local CoinRun motion to produce finite, temporally continuous 16-frame held-out rollouts.
- Falsified if: the action-conditioned trainer cannot run on non-Minecraft actions, the rollout path fails, metrics are non-finite, or predictions collapse to unrelated/static frames.

## Controlled design

- Dataset: 2,048 train and 256 held-out trajectories, each 64 frames, random Procgen actions.
- Dynamics: depth 2, `d_model=128`, two query heads, one KV head, 16 registers.
- Training: batch 32 × 64 frames; 15,000 steps; shortcut bootstrap begins at step 5,000 with fraction 0.25.
- Held-out rollout: four context frames and 16 predicted frames.
- Metric target: tokenizer-decoded ground truth, so the reported held-out scores isolate dynamics error.

## Results

### Observed

- The end-to-end pipeline completed on one A10.
- Final training rollout PSNR was finite for online/EMA and diffusion/shortcut sampling.
- Four held-out trajectories achieved mean dynamics-only PSNR `16.6342 dB` and mean SSIM `0.6513`.
- Per-trajectory PSNR was `19.06`, `16.75`, `17.28`, and `13.45 dB`.

### Interpretation

The small model learned temporal continuity and a usable action-conditioned latent prediction path. This establishes a minimal world-model pipeline, not a high-quality simulator.

### Not established

- Reproduction of an official CoinRun dynamics result; no official minimal CoinRun dynamics config/checkpoint/metric was released.
- A playable CoinRun sandbox.
- A trained policy or agent. The data uses random actions and the experiment trains dynamics only.
- Long-horizon stability beyond 16 predicted frames.

### Decision

Use the same dataset/evaluator for capacity experiments, while separating tokenizer quality from dynamics-only rollout quality.
