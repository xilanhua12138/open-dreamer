# CR-TOK-0004 — CoinRun tokenizer 28.7M-label fixed-20k extension

## Question

At the same fixed 20,000-update recipe and latent interface as `CR-TOK-0003`, does the local 28.7M-label tokenizer improve held-out reconstruction over the 16.6M-label baseline?

## Why this experiment

The current quality-first sweep intentionally ends at the 16.6M published label. The public CoinRun tokenizer curve also names a 28.7M endpoint, so the local curve is missing its largest scale.

## Hypothesis and falsifier

- Hypothesis: the larger symmetric encoder/decoder backbone improves at least one final EMA clean, edge or temporal-change PSNR metric because the 8.6M arm was still improving at 20,000 updates.
- Falsified if: the aligned 28.7M-label arm completes but improves none of those metrics over the 16.6M-label baseline.

## Controlled design

- Baseline: retained `CR-TOK-0003` n16.6m final checkpoint and held-out evaluation.
- Changed: backbone depth/width from `5 × 320` to `6 × 384`.
- New model identity: 25,564,032 exact local parameters, despite the published `28.7M` label.
- Held constant: immutable train/eval records, 20,000 updates, seed, batch/clip shape, optimizer, loss, EMA, 16 × 16 latent interface and 512-clip held-out evaluation.
- Runtime evidence: structured local identity/state/metrics/telemetry/artifacts plus W&B. W&B may run offline until the DSW host is authenticated; the mode must be reported truthfully.
- Periodic validation: the same 16 fixed validation clips every 2,500 completed updates, with `target | online | EMA` PNG and GIF.

## Results

### Observed

A CPU-only construction probe measured 25,564,032 parameters and 25,728,028,508,160 estimated FLOPs per optimizer update. Source commit `3833b34b46358b0443bb3571a5745c013eb7a7e7` is staged in a detached DSW worktree with runtime dependencies verified. No n28.7m training or held-out evaluation has started.

### Interpretation

None yet.

### Not established

- Whether n28.7m improves reconstruction metrics or visual quality.
- Whether the additional capacity is compute-efficient.
- Whether the tokenizer works with action-conditioned dynamics.

### Decision

Keep this arm queued behind the currently running `CR-TOK-0003` PID on the same A10. Do not run both concurrently and do not start dynamics.
