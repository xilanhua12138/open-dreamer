# CR-TOK-0002 — CoinRun tokenizer v2 adaptive scale gate

## Question

What is the smallest CoinRun tokenizer that is visually clear enough for a world model while preserving the fixed latent interface needed by dynamics?

## Why this experiment

`CR-DEMO-0001` proved that a generated PNG and finite PSNR do not imply a usable world model. The 0.17M tokenizer blurred the player and platform detail before dynamics added any error. Training dynamics on that representation would spend GPU time optimizing the wrong bottleneck.

## Hypothesis and falsifier

- Hypothesis: scaling the tokenizer backbone improves full-frame, edge and motion-region reconstruction enough that at least the local `n3.7m` label clears the quality gate.
- Falsified if: no candidate through the conditional `n16.6m` label clears all metric gates, or a qualifying tokenizer cannot retain the `16 × 16` latent interface needed by dynamics.

## Controlled design

- Baseline: a fresh v2 `n0.17m` arm, with `CR-TOK-0001` retained as historical evidence.
- Primary changed variable: tokenizer backbone depth/width at local labels `n0.17m / n1.1m / n3.7m`.
- Primary held constant: the same structured train/eval data, seed, 512 held-out clips, latent interface, optimizer, loss and `1e16` FLOPs per arm.
- Conditional extension: `n8.6m` at `2e16`, then `n16.6m` at `4e16`, only if every smaller arm misses the preregistered gate.
- Known confounder: conditional extensions use larger compute budgets to preserve a useful minimum number of steps; they are quality-search arms and must not be interpreted as part of the fixed-FLOPs primary scaling curve.

## Quality gate

A candidate must:

1. reach at least `26.0 dB` held-out EMA clean PSNR;
2. beat the fresh v2 `n0.17m` arm by at least `1.0 dB` clean PSNR;
3. beat it by at least `0.75 dB` on edge pixels;
4. beat it by at least `0.75 dB` on temporal-change pixels;
5. retain `n_latents=16` and `d_bottleneck=16`;
6. pass explicit visual review before dynamics begins.

## Results

### Observed

- The exact source commit `5ad134f914c12afa66eda141afd9e446c51d43e3` started on the existing A10 at `2026-07-29T12:49:27+08:00`.
- The 4,096-record train split and 512-record held-out split are level-disjoint and passed audit. Both contain the explicit six-action control set, exceed `0.90` macro persistence and have about `55–56%` completed-episode success.
- The exact model sizes are `163,392 / 1,048,192 / 3,342,528 / 7,734,528 / 14,912,320` parameters. Every arm retains the `16 × 16` latent interface.
- Fresh `n0.17m` training is running. No reconstruction metric or visual result exists yet.

### Interpretation

The revised data now contains sustained, successful control trajectories rather than only frame diversity. This is necessary for later dynamics training but does not itself establish tokenizer quality.

### Not established

- Which scale is visually acceptable.
- Whether the selected representation supports action-conditioned dynamics.

### Decision

Block all new dynamics training until this experiment clears both metric and visual gates.
