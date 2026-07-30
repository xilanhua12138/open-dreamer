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
`855981088b3f5674ff1c7f05ef5da829fc4c0509` passed all 124 tests. One recovery
pipeline started at 18:14:55 Asia/Shanghai with PID `110225`.

The fixed probe completed with a single observation SHA256
`fd36a901...04feb`, 16 initialization seeds per arm, finite metrics and a
structured completed run state:

| Arm | Encoder RMS | JVP RMS gain | Global gradient L2 | Last branch / skip |
|---|---:|---:|---:|---:|
| Glorot reference | 0.4960 | 0.6493 | 4.6084 | 0.5380 |
| Orthogonal √2 anchor | 13.0191 | 17.8623 | 3577.4231 | 1.0570 |
| Orthogonal gain 1 | 0.7874 | 1.0729 | 11.0045 | 0.5300 |
| Orthogonal √2, depth-scaled | 2.5210 | 3.4602 | 77.5217 | 0.4308 |
| Orthogonal √2, zero-last | 1.4561 | 2.1225 | 40.1874 | 0.0000 |
| Orthogonal √2, SkipInit | 1.4561 | 2.1225 | 33.6389 | 0.0000 |

Versus Glorot, the unmitigated orthogonal-√2 arm amplified encoder RMS by
`26.2468×`, encoder JVP gain by `27.5101×`, and the synthetic PPO global
gradient by `776.2791×`. Unit-gain orthogonal initialization reduced those
ratios to `1.5875×`, `1.6523×`, and `2.3879×`, respectively. This supports the
initial amplification mechanism; it does not yet establish trained policy
quality.

The first unconditional arm then completed the frozen training and final
evaluation protocol:

| Arm | Mean return | Success | Half-gap threshold | Decision |
|---|---:|---:|---|---|
| Glorot retained anchor | 7.734375 | 77.34375% | — | reference |
| Orthogonal √2 retained anchor | 5.390625 | 53.90625% | — | degraded anchor |
| Orthogonal gain 1 | 6.640625 | 66.40625% | 6.5625 / 65.625% | passes both |

Unit-gain orthogonal initialization recovered `53.3333%` of both retained gaps.
It is therefore a successful partial mitigation, not a full Glorot recovery.
The depth-scaled arm is running, and zero-last plus SkipInit remain
unconditional future arms.

Automatic machine evidence currently writes experiment-ledger-only files into
the execution worktree after each stage. Later runtime identities therefore
report a dirty tree even though no model, trainer, runner or test file changed;
the generated patch is retained and this infrastructure coupling will be
separated after the scientific run.
