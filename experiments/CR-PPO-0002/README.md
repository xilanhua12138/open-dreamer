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

The zero-step stochastic full-distribution baseline completed on 128 episodes at mean return `3.125` and success rate `31.25%`. The run then completed exactly `25,165,824` transitions / `1,536` PPO updates without NaN, OOM or process failure.

The final periodic 128-episode evaluation reached mean return `8.828125` and success rate `88.28125%`. The separately materialized 512-episode final evaluation reached mean return `8.671875`, success rate `86.71875%`, median return `10`, and mean episode length `77.28125`. It therefore passed both preregistered `8.0 / 80%` thresholds.

The final checkpoint is `7,520,613` bytes with SHA256 `8ebfa7a91e5b52ce2d51294fa0a9aa3a520f4bb1f2a6a2d17e5b4e154023e811`. The recorded attempt ended in `COMPLETED`; the outer status explicitly says `WORLD_MODEL_NOT_STARTED`.

### Interpretation

The combined public-recipe alignment recovered the expected public-curve range under this seed and runtime. Compared with the old checkpoint's post-hoc 512-episode stochastic full-distribution result, mean return increased from `4.9609375` to `8.671875` and success rate from `49.609375%` to `86.71875%`.

This result shows that CR-PPO-0001 was not low only because its original evaluation used deterministic actions and a fixed level range. It does not identify one causal bug because training levels, reward-normalizer gamma, advantage normalization and backbone initialization changed together.

### Not established

- Which individual recipe change caused the old training collapse.
- That every non-reference setting is intrinsically invalid or a software bug.
- An exact paper reproduction across TensorFlow/JAX, Procgen versions and seeds.
- A reusable dynamics dataset.
- Authorization to collect replacement trajectories or start dynamics.

### Decision

Use the validated recipe as the default for new from-scratch PPO runs, add a machine-enforced final-policy quality gate before collection, and run single-variable controls before naming one old setting as the training bug. Do not collect trajectories or start dynamics under this experiment.
