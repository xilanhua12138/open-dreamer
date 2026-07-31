# CR-DYN-0012 — CoinRun 52.80M dynamics xlarge

## Question

Does increasing only dynamics capacity from 3.93M to 52.80M improve the
complete 200k offline-latent CoinRun recipe?

## Why this experiment

CR-DYN-0011 uses about 4.4 GiB on the existing 23 GiB A10. Earlier capacity
work stopped at 12.90M and used the rejected 20k, 64-frame protocol, so it does
not answer whether a substantially larger dynamics model helps the repaired
long-record recipe. This experiment starts only after CR-DYN-0011 finishes and
nano-dreamer validates the compatible Tokenizer → PPO → Dynamics chain.

## Hypothesis and falsifier

- Hypothesis: a 52,801,152-parameter dynamics transformer improves long-range
  rollout fidelity without reducing action sensitivity.
- Falsified if: batch 16 does not fit one A10; terminal evaluators are missing;
  shortcut mean/horizon-16 PSNR fails to improve by 0.5 dB over CR-DYN-0011;
  action deltas miss their absolute gates; or visual review remains unstable.

## Controlled design

- Baseline: CR-DYN-0011 medium, 3,931,136 parameters.
- Changed: dynamics depth/width/heads only, to depth 9, width 640, 10 heads.
- Held constant: tokenizer, PPO checkpoint, raw/latent dataset trees, 200k
  updates, batch 16, 64/128 schedule, Muon, LR, k-max, bootstrap and all
  terminal futures/evaluators.
- Known risk: activation memory scales much faster than parameter bytes. The
  4.4 GiB medium observation does not prove xlarge batch 16 fits.

## Results

### Observed

CPU construction counted exactly 52,801,152 parameters. No GPU compile,
optimizer update or evaluation has run.

### Interpretation

The selected architecture is close to the requested 50M class and is 13.4×
the medium dynamics size. Fit and quality remain unknown.

### Not established

- It is not established that dynamics must be larger than tokenizer.
- It is not established that unused VRAM implies capacity is the current
  quality bottleneck.
- It is not established that xlarge fits batch 16 or improves rollouts.

### Decision

After nano validation passes and the GPU is idle, run one batch-16 compile and
optimizer-update preflight. On success, start the exact from-scratch 200k run.
On OOM, retain the failure and do not silently change the protocol.
