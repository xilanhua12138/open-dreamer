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
- The DSW probe instantiated the frozen architecture at exactly **12,902,784 parameters**. Its retained JSON has SHA256 `3aa59f41e39039d56337173ec587da5b79eb525bcd045d7a7d0611a16b73369f`.

## Current result

### Observed at 2026-07-29 06:49 Asia/Shanghai

- The existing one-A10 instance recovered from the recorded inventory block and reached `Running`.
- A six-hour shutdown timer is due at 2026-07-29 12:45:44 Asia/Shanghai.
- The parameter probe completed before training and recorded 12,902,784 parameters.
- The single extension pipeline entered `TRAINING_LARGE` at 06:49:24; no terminal checkpoint comparison or held-out metric exists yet.

### Interpretation

The infrastructure block is resolved for this attempt. The experiment remains scientifically unevaluated while training is running.

### Not established

- Whether the model fits A10 24 GB.
- Any large-vs-medium quality comparison.

### Decision

Monitor the recorded PID and do not restart a healthy pipeline. On terminal completion or failure, retain the checkpoint/config/metrics evidence and update this record before interpreting the result.
