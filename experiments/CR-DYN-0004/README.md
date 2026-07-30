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

## Results

| Model | Parameters | PSNR@1 | PSNR@3 | PSNR@8 | PSNR@16 | SSIM@16 |
|---|---:|---:|---:|---:|---:|---:|
| medium baseline | 3,931,392 | 20.73 | 19.23 | 16.89 | **16.02** | **0.542** |
| large | 12,902,784 | 19.23 | 17.61 | 16.16 | 15.64 | 0.531 |

### Observed

- The existing one-A10 instance recovered from the recorded inventory block and reached `Running`.
- The parameter probe completed before training and recorded 12,902,784 parameters.
- Large completed all 20,000 steps and identical held-out evaluation.
- Large trailed medium by `0.38095 dB` mean-frame PSNR and `0.01105` mean SSIM.
- The retained checkpoint tree has SHA256 `ce2629b7f3962d79fed0ff7f101549f6620068820aa88c7dec9164ef3499015b`.

### Interpretation

The hypothesis is rejected. Under this small-data fixed-20k protocol, the prior tiny→small→medium monotonic improvement did not extend to large. This identifies a local optimum at medium under the tested recipe, not a universal capacity limit.

### Not established

- Whether the reversal is caused by saturation, overfitting, or a schedule mismatch.
- Multi-seed statistical significance.
- Compute-optimal scaling or official reproduction.

### Decision

Select medium for the fixed-future context ablation and live demo. Before testing a still-larger model, vary data scale, regularization, or schedule rather than spending compute on capacity alone.
