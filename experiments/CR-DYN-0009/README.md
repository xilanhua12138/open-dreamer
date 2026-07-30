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

## Interpretation

The tiny result is a valid first scale point and lies 4.403669 dB mean-frame
PSNR and 0.159942 mean SSIM below the reused medium anchor. This does not yet
establish monotonicity because small and large are incomplete.

## Not established

- Whether the corrected capacity curve is monotonic.
- Whether large exceeds medium.
- Whether any scale is good enough for interactive control.

## Decision

Continue the unchanged serial runner through small and large, reusing the
byte-identical selected-medium result exactly as preregistered. Do not interpret
the scale hypothesis until all four points exist.
