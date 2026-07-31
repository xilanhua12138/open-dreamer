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
baseline before beginning the 200k repair training. The Codex heartbeat
`opendreamer-dynamics-repair-monitor` owns subsequent structured monitoring and
evidence updates. No replacement instance or specification change is allowed.
