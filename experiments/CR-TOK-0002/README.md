# CR-TOK-0002 — CoinRun tokenizer v2 unequal-step pilot

## Question

Under the originally declared per-scale FLOPs allocations, how do the first completed CoinRun tokenizer scales behave?

## Why this experiment

`CR-DEMO-0001` proved that a generated PNG and finite PSNR do not imply a usable world model. The 0.17M tokenizer blurred the player and platform detail before dynamics added any error. Training dynamics on that representation would spend GPU time optimizing the wrong bottleneck.

## Hypothesis and falsifier

- Hypothesis: scaling the tokenizer backbone improves full-frame, edge and motion-region reconstruction through at least one of the larger scales.
- Falsified if: the complete five-scale sweep shows no improvement over the fresh `n0.17m` arm, or the larger tokenizers cannot retain the `16 × 16` latent interface needed by dynamics.

## Controlled design

- Baseline: a fresh v2 `n0.17m` arm, with `CR-TOK-0001` retained as historical evidence.
- Changed variable: tokenizer backbone depth/width at local labels `n0.17m / n1.1m / n3.7m / n8.6m / n16.6m`.
- Held constant: the same structured train/eval data, seed, 512 held-out clips, latent interface, optimizer and loss.
- Compute: the declared allocations translate to `62,675 / 9,358 / 2,938 / 2,550 / 2,656` optimizer steps across the five scales.
- Known confounder: capacity and optimization maturity change together. This run cannot measure attainable quality at each scale.

## Evaluation

The original runner exited after three arms through an obsolete quality-gate branch. Those three arms and the incomplete-run evidence are retained; the missing two arms are not filled under this experiment ID.

## Results

### Observed

- The exact source commit `5ad134f914c12afa66eda141afd9e446c51d43e3` started on the existing A10 at `2026-07-29T12:49:27+08:00`.
- The 4,096-record train split and 512-record held-out split are level-disjoint and passed audit. Both contain the explicit six-action control set, exceed `0.90` macro persistence and have about `55–56%` completed-episode success.
- The exact model sizes are `163,392 / 1,048,192 / 3,342,528 / 7,734,528 / 14,912,320` parameters. Every arm retains the `16 × 16` latent interface.
- Fresh `n0.17m` completed 62,675 updates. On 512 held-out clips, EMA clean/masked PSNR was `12.0621 dB`, edge PSNR was `11.9360 dB`, and temporal-change PSNR was `11.4601 dB`.
- Fresh `n1.1m` completed 9,358 updates and reached `27.2563 dB` EMA clean, `24.6312 dB` EMA masked, `20.6026 dB` EMA edge and `20.8226 dB` EMA temporal-change PSNR.
- Fresh `n3.7m` completed only 2,938 updates. Its online clean PSNR was `29.3045 dB`, but its EMA clean PSNR was `26.2075 dB`, a `3.0971 dB` gap.
- While that first arm was still running and before any held-out result existed, the user amended the protocol to require all five scales unconditionally. No completed arm or observed metric influenced this change.
- The initial runner nevertheless exited after the third arm. `n8.6m` and `n16.6m` were never started under `CR-TOK-0002`.

### Interpretation

The 1.1M result shows that the new representation recipe is not universally collapsed. However, the 3.7M online-versus-EMA gap after only 2,938 updates shows why the mixed-step endpoints are not suitable for deciding whether extra capacity helps. The experiment is therefore an optimization-budget diagnostic, not a quality-first scale curve.

### Not established

- Which scale is visually acceptable.
- Whether capacity improves attainable quality at a common optimization horizon.
- Whether the selected representation supports action-conditioned dynamics.

### Decision

Preserve this incomplete run and do not start its old completion runner. `CR-TOK-0003` trains all five scales for exactly 20,000 updates, evaluates four learning-curve milestones, and keeps dynamics blocked until explicit visual acceptance.
