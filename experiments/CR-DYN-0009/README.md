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

`final_only-small` completed its exact budget at 05:22:43. Its aligned
32-video evaluation scored 12.497698 dB mean-video PSNR, 13.396642 dB
mean-frame PSNR and 0.580908 mean SSIM. Horizon 1/3/8/16 PSNR was
19.790742/17.537919/15.080334/13.396642 dB.

The 12,902,784-parameter `final_only-large` arm then trained from scratch to
exactly 20,000 updates and completed at 06:42:03. Its aligned evaluation
scored 14.966703 dB mean-video PSNR, 15.723735 dB mean-frame PSNR and
0.674813 mean SSIM. Horizon 1/3/8/16 PSNR was
18.045137/17.798596/16.889218/15.723735 dB.

The reused 3,931,392-parameter `final_only-medium` anchor remained best at
16.165151 dB mean-video PSNR, 17.073096 dB mean-frame PSNR and 0.713450
mean SSIM. The complete descending order was
`medium > large > small > tiny`.

This is a relative ordering inside a poor grid. Direct review of the dependent
demo rejected temporal coherence and action response; 17.073096 dB/0.713450
must not be described as a usable world model.

## Interpretation

Small exceeds tiny by 0.727215 dB mean-frame PSNR and 0.027399 mean SSIM, and
medium exceeds small on both metrics. However, large trails medium by
1.349361 dB mean-frame PSNR, 1.198447 dB mean-video PSNR and 0.038638 mean
SSIM. This violates the preregistered requirement
`tiny < small < medium <= large` on both primary metrics, so the monotonic
scaling hypothesis is rejected under this fixed-20k recipe.

## Not established

- Whether the large model would recover with a different optimization budget.
- Whether additional seeds reproduce the medium-over-large reversal.
- Whether any scale is good enough for interactive control.
- Whether aligned actions beat shuffled, shifted and all-no-op action controls.

## Decision

Record `final_only-medium` as the descriptive winner using the frozen held-out
ordering, but do not reuse it for a usability claim. Follow
[`POSTMORTEM.md`](POSTMORTEM.md): first diagnose action use and sampler versus
checkpoint quality; then recollect longer episode-safe records before a
materially longer `k_max=256` reference run. Do not generalize the result into
“larger models are worse”: it only rejects monotonic improvement for this
one-seed, fixed-20k protocol.
