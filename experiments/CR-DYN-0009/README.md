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

No mixture selection or scale result had been observed at preregistration time.

## Interpretation

None yet.

## Not established

- Which checkpoint mixture will be selected.
- Whether the corrected capacity curve is monotonic.
- Whether large exceeds medium.
- Whether any scale is good enough for interactive control.

## Decision

Remain queued until `CR-DYN-0008` has all three completed arms, valid audits and
a selection artifact.
