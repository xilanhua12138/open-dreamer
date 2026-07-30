# CR-PPO-0002 — CoinRun PPO official-recipe parity control

## Question

Does a from-scratch JAX PPO control aligned to the public Procgen easy-200 recipe recover the approximately `8.7–9.0` mean CoinRun return shown near 25M steps in the official paper?

## Why this experiment

`CR-PPO-0001` closed its training and data-audit pipeline, but its larger post-hoc stochastic full-distribution evaluation reached only `4.96` mean return at the final checkpoint and `5.39` at the best preregistered checkpoint. This is too far below the public easy-200 reference curve to dismiss as a 64-episode deterministic evaluation artifact.

Source comparison found concrete recipe mismatches:

- 500 training levels instead of the official easy-200 setup.
- reward-normalizer discount `0.999` instead of the reference wrapper default `0.99`.
- one whole-rollout advantage normalization instead of normalization inside every optimizer minibatch.
- orthogonal `sqrt(2)` IMPALA backbone initialization instead of the `tf.layers` Glorot defaults used by the public `build_impala_cnn`.
- deterministic fixed-level evaluation instead of a stochastic worker over `num_levels=0`.

The new switches preserve `CR-PPO-0001` defaults. Only this experiment enables the parity settings.

## Hypothesis and falsifier

- Hypothesis: the aligned recipe reaches final 512-episode mean return `>= 8.0` and success rate `>= 0.80`.
- Falsified if either threshold is missed after exactly `25,165,824` transitions.

## Controlled design

- Baseline: `CR-PPO-0001`.
- Changed: the five public-reference mismatches listed above.
- Held constant: A10, seed 0, model widths, 64 envs × 256 rollout, 8 minibatches, 3 epochs, optimizer coefficients and total transition budget.
- Known confounders: JAX rather than TensorFlow, Procgen `0.10.7` rather than the 2019 package revision, one seed, and a target read visually from the paper figure.

## Results

### Observed

The preregistration was committed before any GPU result. Execution source `a4d2574` then passed a real CPU-only Procgen `num_levels=0` smoke with the 15-action Glorot model and started one A10 PID at `2026-07-30 14:30:54 +08:00`. Run ID is `a079cdb6-2586-4de5-ad1c-512537184308`; W&B is offline.

The zero-step stochastic full-distribution baseline completed on 128 episodes at mean return `3.125` and success rate `31.25%`. Training then reached 557,056 transitions / 34 updates with finite PPO diagnostics and 99% GPU utilization. This early progress is execution evidence, not the final parity result.

### Interpretation

None.

### Not established

- An exact paper reproduction.
- A reusable dynamics dataset.
- That any single changed setting is independently causal.

### Decision

Continue the single active parity PID. Do not collect trajectories or start dynamics under this experiment.
