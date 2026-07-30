# CR-PPO-0003 — CoinRun PPO one-factor bug screen

## Question

Which individual old-recipe difference, if any, independently reproduces the early policy-quality collapse?

## Why this experiment

`CR-PPO-0002` passed the public-curve control, but it changed multiple variables together. That proves the package matters; it does not prove every difference is a bug.

## Hypothesis and falsifier

- Hypothesis: at least one single reversion lowers 6.29M-step final-256 mean return by `>=1.0` or success rate by `>=0.10`.
- Falsified if every arm remains within both margins of a same-run reference repeat.

## Controlled design

- Baseline: a fresh reference-aligned repeat.
- Changed: exactly one of training levels, reward-normalizer gamma, advantage normalization location or backbone initializer.
- Held constant: seed, total steps, model widths, optimizer, rollout, Procgen runtime and full-distribution stochastic evaluation.
- Known limitation: this is a 6.29M-step screen under one seed, not a final 25M multi-seed attribution.

## Results

### Observed

Not run.

### Interpretation

None.

### Not established

- Which setting is causal.
- That a weaker setting is intrinsically invalid rather than less sample-efficient.

### Decision

Run all five arms serially. Do not collect trajectories or start dynamics.
