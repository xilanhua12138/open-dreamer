# CR-DYN-0010 — CoinRun dynamics reference-recipe repair

## Question

Can one reference-like repair bundle turn the rejected CR-DYN-0009 medium
checkpoint into a higher-quality model whose predictions measurably depend on
the supplied actions?

## Why this experiment

CR-DYN-0009 used 64-frame records and 64-frame training windows. That leaves
exactly one legal crop, so its configured 50% reward-biased sampling could
never change the sampled future. The run also stopped at 20,000 updates with
`k_max=8`, while the repository's general dynamics recipe uses substantially
longer optimization and diffusion schedules.

## Controlled design

- Baseline: reevaluate the rejected `final_only-medium` checkpoint on the same
  fresh held-out futures used by this experiment.
- Fresh data: 4,096 train and 512 held-out final-PPO records, each 160 frames.
- Model: unchanged 3.93M medium dynamics and unchanged 16.6M EMA tokenizer.
- Repair bundle: 64/128-frame windows, 200,000 updates, `k_max=256`, bootstrap
  beginning at 100,000 updates and fixed held-out periodic visualization.
- Action diagnostic: hold context, future target and noise constant; change only
  future actions to batch-shuffled, one-step-shifted or all-noop controls.

This is an exploratory bundle. A successful result says the reference-like
recipe repairs the failed baseline; it does not isolate which change caused
the repair.

## Preregistered acceptance

- EMA shortcut mean-frame PSNR at least 22 dB.
- EMA shortcut horizon-16 PSNR at least 20 dB.
- Aligned actions beat batch-shuffled actions by at least 0.25 dB at horizon 16.
- Aligned actions beat all-noop actions by at least 0.50 dB at horizon 16.
- Direct visual review is still mandatory.

## Current state

Source implementation `2fbfe81ce240f02638fb9815c367af88e999b1c7`
and runner SHA256
`6315df3e4a401455ce2c5c3875855752c487177d5b2e5b3c7ed5194dcb935458`
are frozen. This source includes a pre-execution, protocol-neutral lazy import
fix after the lightweight CI reproduced a missing optional Grain dependency.
After the first same-instance start attempt was blocked by temporary inventory,
the same A10 returned to Running. Frozen remote preflight passed and exactly one
pipeline started at 2026-07-31 11:39:55 Asia/Shanghai. Fresh collection then
completed with 4,096 train and 512 held-out records, each 160 frames. The train,
held-out and pair audits are valid, both corpora expose all 15 Procgen actions,
and action 4 is recorded as no-op. The repaired sampler now has 97 legal
64-frame starts and 33 legal 128-frame starts instead of one inert crop. The
same live pipeline advanced to fixed-future evaluation of the rejected medium
baseline. That retained checkpoint scored 18.402460 dB mean-frame PSNR,
0.720044 mean SSIM and 18.402460 dB at horizon 16. On the 64-video action
subset, aligned actions beat shuffled and all-noop controls by 2.776499 and
2.640131 dB at horizon 16. This controlled metric does not override the prior
direct visual rejection of its temporal/action behavior. Fresh latent
statistics were then computed from 256 videos × 128 frames, and the one
3.93M-parameter repair arm began its preregistered 200,000 updates from scratch
at 2026-07-31 11:48:55 Asia/Shanghai. It completed the first fixed validation
at 10,000 updates and was intentionally stopped at 11,320 updates after
inspection confirmed that each raw-RGB optimizer batch reran the frozen
tokenizer encoder. The incomplete run and its rollout video remain retained as
intermediate evidence; no terminal quality claim is made.

CR-DYN-0011 was preregistered before its execution as the controlled
replacement. It encodes each complete 160-frame record exactly once, audits
the latent record identity plus exact action/reward/terminal alignment, and
preserves the same reward-biased window-selection semantics before training
the same dynamics model and schedule from scratch. No CR-DYN-0010 checkpoint
will be reused.
