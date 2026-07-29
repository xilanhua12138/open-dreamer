# CR-TOK-0002 — CoinRun tokenizer v2 complete five-scale sweep

## Question

What is the smallest CoinRun tokenizer that is visually clear enough for a world model while preserving the fixed latent interface needed by dynamics?

## Why this experiment

`CR-DEMO-0001` proved that a generated PNG and finite PSNR do not imply a usable world model. The 0.17M tokenizer blurred the player and platform detail before dynamics added any error. Training dynamics on that representation would spend GPU time optimizing the wrong bottleneck.

## Hypothesis and falsifier

- Hypothesis: scaling the tokenizer backbone improves full-frame, edge and motion-region reconstruction through at least one of the larger scales.
- Falsified if: the complete five-scale sweep shows no improvement over the fresh `n0.17m` arm, or the larger tokenizers cannot retain the `16 × 16` latent interface needed by dynamics.

## Controlled design

- Baseline: a fresh v2 `n0.17m` arm, with `CR-TOK-0001` retained as historical evidence.
- Changed variable: tokenizer backbone depth/width at local labels `n0.17m / n1.1m / n3.7m / n8.6m / n16.6m`.
- Held constant: the same structured train/eval data, seed, 512 held-out clips, latent interface, optimizer and loss.
- Compute: the first three arms use `1e16` FLOPs each; `n8.6m` uses `2e16` and `n16.6m` uses `4e16` so the largest arms still receive roughly 2.5k optimization steps.
- Known confounder: because the two largest arms receive more FLOPs, all five points form a practical quality/capacity sweep, not one strict five-point iso-FLOPs curve.

## Evaluation

Every scale is trained and evaluated. The final report includes clean, masked, edge and temporal-change PSNR plus an aligned five-scale reconstruction grid. Automatic ranking does not stop training and does not substitute for visual review.

## Results

### Observed

- The exact source commit `5ad134f914c12afa66eda141afd9e446c51d43e3` started on the existing A10 at `2026-07-29T12:49:27+08:00`.
- The 4,096-record train split and 512-record held-out split are level-disjoint and passed audit. Both contain the explicit six-action control set, exceed `0.90` macro persistence and have about `55–56%` completed-episode success.
- The exact model sizes are `163,392 / 1,048,192 / 3,342,528 / 7,734,528 / 14,912,320` parameters. Every arm retains the `16 × 16` latent interface.
- Fresh `n0.17m` completed all 62,675 allocated steps. On 512 held-out clips, EMA clean/masked PSNR was `12.0621 dB`, edge PSNR was `11.9360 dB`, and temporal-change PSNR was `11.4601 dB`.
- While that first arm was still running and before any held-out result existed, the user amended the protocol to require all five scales unconditionally. No completed arm or observed metric influenced this change.
- Fresh `n1.1m` is now training. No larger-arm held-out result exists yet.

### Interpretation

The revised data now contains sustained, successful control trajectories rather than only frame diversity. This is necessary for later dynamics training but does not itself establish tokenizer quality. The first arm's `12.0621 dB` result is substantially below the prior small-run tokenizer, but the v2 data and recipe differ; the complete five-scale curve and retained grids are required before attributing the gap.

### Not established

- Which scale is visually acceptable.
- Whether the selected representation supports action-conditioned dynamics.

### Decision

Train and compare all five tokenizer scales. Keep dynamics blocked until the complete grid receives explicit visual acceptance.
