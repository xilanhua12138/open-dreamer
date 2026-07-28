# CR-DYN-0002 — fixed-FLOPs capacity pilot

## Question

Under a very small total budget of `C=1e15`, how does CoinRun dynamics capacity trade against the number of optimizer updates?

## Why this experiment

`CR-DYN-0001` closed one small pipeline but did not show whether dynamics capacity helped. The first scaling design reused the repository's native FLOPs allocator so each model received approximately the same estimated total compute.

## Hypothesis and falsifier

- Hypothesis: within a fixed compute budget, at least one intermediate capacity should outperform both a much smaller and a much larger model because parameter count trades against update count.
- Falsified if: quality improves monotonically with size despite the update imbalance, or no reliable differences are measurable.

## Controlled design

- Fixed: dataset, tokenizer, `C=1e15`, batch 32 × 64 frames, optimizer family, WSD schedule proportions, held-out episodes, context 4, horizon 16, seed 4242.
- Changed: dynamics capacity.
- Compute-derived steps:
  - tiny 155,840 parameters: 15,217 allocated steps.
  - small 545,920 parameters: 4,768 allocated steps.
  - medium 3,931,392 parameters: 474 allocated steps.
- Bootstrap began halfway through each allocated schedule with fraction 0.25.

This design compares recipes at equal estimated compute. It does not isolate capacity alone because optimizer steps differ by more than 32×.

## Results

### Observed

- Medium completed 474 allocated steps and scored mean-frame PSNR `9.8566 dB`, SSIM `0.3666`.
- Small completed 4,768 allocated steps and scored mean-frame PSNR `10.1272 dB`, SSIM `0.1647`.
- Tiny stopped at approximately step 4,452 of 15,217 and was never evaluated.
- The shell wrapper wrote `COMPLETE` when it exited, but the planned three-arm comparison was incomplete.
- Medium did enter bootstrap/shortcut training: `bootstrap_start=237`, and late log lines contain non-zero `boot_mse`.

### Interpretation

The budget was too small for a clean capacity conclusion. Medium received only 474 updates; small had 4,768; tiny was missing. PSNR slightly favored small while SSIM favored medium, so even the two completed arms disagree by metric.

### Not established

- An optimal model size.
- That larger capacity hurts CoinRun dynamics.
- A complete fixed-FLOPs sweep.
- Any comparison at equal optimization progress.

### Decision

Run a corrected capacity ablation with every size receiving the same 20,000-step curriculum. Keep this experiment as evidence that fixed FLOPs and fixed steps answer different questions.

## Correction to the old summary

An earlier retrospective summary said the 474-step medium model “never reached shortcut training.” That is incorrect. The runner and log show bootstrap began at step 237 and `boot_mse` was non-zero. The correct limitation is that it received only about 237 mixed bootstrap steps and remained severely under-optimized relative to later runs.
