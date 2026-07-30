# CR-PPO-0005 — CoinRun PPO initializer mechanism and mitigation

## Why this run exists

`CR-PPO-0003` showed that changing only the unnormalized IMPALA backbone from
Glorot to orthogonal gain `sqrt(2)` reduced final-256 mean return from `7.734375`
to `5.390625` and success from `77.34375%` to `53.90625%` at 6,291,456
transitions. The other three one-factor reversions did not cross the
preregistered collapse threshold.

The current encoder applies the same orthogonal `sqrt(2)` initializer to all
three stem convolutions, twelve residual convolutions and the post-convolution
dense layer. Each of six residual blocks is unnormalized and computes
`x + F(x)` without residual scaling. This experiment tests whether the observed
sample-efficiency loss is associated with initial residual signal and gradient
amplification, and whether architecture-aware initialization can mitigate it.

## Frozen design

Before any new result, freeze:

- a zero-update probe on the same fixed CoinRun observation batch across 16
  parameter seeds;
- the retained `CR-PPO-0003` Glorot and orthogonal-`sqrt(2)` final-256 anchors;
- four new from-scratch 6,291,456-transition arms:
  `orthogonal_gain1`, `orthogonal_sqrt2_depth_scaled`,
  `orthogonal_sqrt2_zero_last`, and `orthogonal_sqrt2_skipinit`;
- identical stochastic, full-distribution, seed-4242, 256-episode final
  evaluation for every training arm.

The depth-scaled arm multiplies each residual branch by `1/sqrt(6)`. The
zero-last arm initializes the second convolution of every residual block to
zero. The SkipInit arm gives every block a learnable scalar gate initialized to
zero. These last two begin as exact identity residual blocks, which is enforced
by tests.

## Claim boundary

This is a one-game, one-training-seed internal mechanism screen. It may justify
a larger multi-seed/multi-game study, but cannot by itself establish a general
PPO result or a publishable method.

Trajectory collection, behavior cloning and dynamics training remain out of
scope.

## Current status

The first attempt stopped after successfully writing the complete 16-seed probe
but before any training step. `run_recorded.py` correctly rejected the probe
because it had not emitted `runtime-identity.json` and `run-state.json`.

The 349,214-byte probe JSON is retained locally with SHA256
`04f205d0...d5aec4`; the remote failed attempt and logs remain intact. A precise
red test reproduced the missing recorder contract, and the fix now makes the
probe use `RunRecorder`, register its JSON artifact, and require a structured
completed state before the runner may advance. Clean pushed recovery source
`855981088b3f5674ff1c7f05ef5da829fc4c0509` passed all 124 tests and is queued;
the scientific protocol is unchanged.
