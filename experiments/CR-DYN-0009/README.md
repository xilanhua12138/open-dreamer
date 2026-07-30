# CR-DYN-0009 — Selected-mixture dynamics scale sweep

## Question

On the checkpoint mixture selected by the complete `CR-DYN-0008` ablation,
how does CoinRun world-model rollout quality change across 0.16M, 0.55M,
3.93M and 12.90M dynamics parameters at the same 20,000 updates?

## Design

This is a preregistered sequential experiment. It does not choose a mixture
informally: `CR-DYN-0008` must finish all three mixture arms, then its frozen
lexicographic rule writes the selected mixture. Only after that artifact exists
may this experiment start.

Tiny, small and large train from scratch. The byte-identical selected-mixture
medium arm is reused from `CR-DYN-0008`, avoiding a duplicate run without
changing any scientific constant. All scales use the 16.6M EMA tokenizer,
20,000 updates and the same final-policy held-out futures.

## Observed

`CR-DYN-0008` completed all three arms and its frozen rule selected
`final_only` at 04:14:03. The 155,840-parameter `final_only-tiny` arm then
trained from scratch to exactly 20,000 updates and completed at 04:47:34.
Its aligned 32-video evaluation scored 12.052888 dB mean-video PSNR,
12.669427 dB mean-frame PSNR and 0.553509 mean SSIM. Horizon 1/3/8/16 PSNR
was 17.875752/15.739125/13.846570/12.669427 dB.

The runner then entered the 545,920-parameter `final_only-small` arm. At
05:05:49 its structured state was 9,882/20,000 updates, immediately before the
periodic 10k evaluation. PID `929` remained alive.

`final_only-small` completed its exact budget at 05:22:43. Its aligned
32-video evaluation scored 12.497698 dB mean-video PSNR, 13.396642 dB
mean-frame PSNR and 0.580908 mean SSIM. Horizon 1/3/8/16 PSNR was
19.790742/17.537919/15.080334/13.396642 dB.

The 12,902,784-parameter `final_only-large` arm then started from scratch. At
06:07:14 its structured state was 11,207/20,000 updates; PID `929` remained
alive with 99% A10 utilization and 4,426/23,028 MiB allocated.

## Interpretation

Small exceeds tiny by 0.727215 dB mean-frame PSNR and 0.027399 mean SSIM, so
the first required monotonic relation holds. Medium also exceeds small on both
metrics. The overall scale hypothesis still depends on whether large at least
matches the reused medium result.

## Not established

- Whether the corrected capacity curve is monotonic.
- Whether large exceeds medium.
- Whether any scale is good enough for interactive control.

## Decision

Continue the unchanged large arm, reusing the byte-identical selected-medium
result exactly as preregistered. Do not close the scale hypothesis until the
large held-out metric exists.
