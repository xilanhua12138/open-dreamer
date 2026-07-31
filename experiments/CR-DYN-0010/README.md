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

Source implementation `1c647df007ebb6d388f081ea95ea07475e683c4e`
and runner SHA256
`6315df3e4a401455ce2c5c3875855752c487177d5b2e5b3c7ed5194dcb935458`
are frozen. Remote preflight and execution remain pending.
