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

The clean frozen source `9cb7581` started once on the existing A10 at
`2026-07-30 16:04:08 Asia/Shanghai`. The serial runner entered the fresh
`reference` arm with PID `98113`; no previous PPO process or compute
application was alive before launch.

All five arms completed at exactly 6,291,456 transitions:

| Arm | Final-256 mean | Success | Delta mean | Delta success |
|---|---:|---:|---:|---:|
| reference | 7.7344 | 77.34% | 0.0000 | 0.00 pp |
| levels500 | 8.3594 | 83.59% | +0.6250 | +6.25 pp |
| reward_gamma0999 | 8.3594 | 83.59% | +0.6250 | +6.25 pp |
| batch_advantage | 7.7734 | 77.73% | +0.0391 | +0.39 pp |
| orthogonal_init | 5.3906 | 53.91% | -2.3438 | -23.44 pp |

### Interpretation

Orthogonal-sqrt2 backbone initialization is the only tested setting that
independently reproduced a large early policy-quality drop. The other three
single reverts did not explain the historical collapse.

### Not established

- Whether the same effect persists across seeds or to 25.17M transitions.
- Whether initialization explains the entire historical run gap.
- That orthogonal initialization is a software bug rather than a weaker
  training choice for this architecture and budget.

### Decision

Do not run the all-old combined control. Move compute to a separately
preregistered initializer mechanism and mitigation study. Do not collect
trajectories or start dynamics.
