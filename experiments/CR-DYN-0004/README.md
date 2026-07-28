# CR-DYN-0004 — large fixed-20k capacity extension

## Question

Does the monotonic tiny→small→medium rollout-quality trend from `CR-DYN-0003` continue for a larger dynamics model under the identical 20,000-step curriculum?

## Why this experiment

Three points support a local capacity trend, but the tested range ends at 3.93M parameters. One additional point can show whether quality is still capacity-limited or whether the small-data protocol begins to saturate or destabilize.

## Hypothesis and falsifier

- Hypothesis: large exceeds medium on held-out mean-frame PSNR, with mean SSIM as the tiebreaker.
- Falsified if: large completes the controlled protocol but does not exceed medium PSNR, becomes unstable, or exceeds available A10 memory.

## Controlled design

- Baseline: medium from `CR-DYN-0003`.
- Changed: dynamics architecture to depth 6, `d_model=384`, six query heads, one KV head, 32 registers.
- Fixed: tokenizer, data, batch 32 × 64, 20,000 steps, bootstrap at 10,000, optimizer and schedule, eight held-out videos, context 4, horizon 16, seed 4242, EMA shortcut evaluation.
- Exact parameter count must be written by the probe before training starts.

## Current result

### Observed

- The runner and parameter-probe code are prepared.
- The existing A10 instance stopped at its timer and subsequent starts returned `Failed`; the event reported that sales of the resource were temporarily suspended.
- No large parameter probe output, checkpoint, metric, or rollout exists.

### Interpretation

This is an infrastructure inventory block, not a negative model result.

### Not established

- Exact large parameter count.
- Whether the model fits A10 24 GB.
- Any large-vs-medium quality comparison.

### Decision

Keep retrying only the existing A10 instance. Do not switch to V100/A100 or create another paid instance without explicit authorization. Once it runs, probe parameters first, then execute the frozen protocol once.
