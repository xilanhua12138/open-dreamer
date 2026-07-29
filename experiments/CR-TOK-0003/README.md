# CR-TOK-0003 — CoinRun tokenizer quality-first fixed-20k sweep

## Question

At the same 20,000 optimizer updates, how do held-out reconstruction quality and convergence change across all five CoinRun tokenizer scales?

## Why this experiment

`CR-TOK-0002` mixed capacity with radically different training horizons: `62,675 / 9,358 / 2,938 / 2,550 / 2,656` allocated updates. That is a compute-efficiency experiment, not a fair test of attainable tokenizer quality. Its 3.7M arm also ended with a `3.0971 dB` online-versus-EMA clean-PSNR gap, consistent with an immature endpoint.

## Hypothesis and falsifier

- Hypothesis: at a shared 20,000-update horizon, at least one larger tokenizer improves final held-out clean, edge or temporal-change reconstruction over the 0.17M baseline, while milestone curves reveal whether apparent regressions are optimization lag or persistent capacity limits.
- Falsified if: none of the four larger scales improves any of those final EMA metrics over 0.17M, or larger models cannot finish all four milestone evaluations with the fixed `16 × 16` latent interface.

## Controlled design

- Baseline: fresh 0.17M at exactly 20,000 updates.
- Changed variable: encoder/decoder depth and width across exact parameter counts `163,392 / 1,048,192 / 3,342,528 / 7,734,528 / 14,912,320`.
- Held constant: structured train/eval records and hashes, seed, batch 128, 16-frame clips, latent interface, loss, optimizer, EMA and held-out clips.
- Training: scaling helpers disabled; every arm runs exactly 20,000 optimizer updates.
- Milestones: checkpoint/evaluate after exactly `2,500 / 5,000 / 10,000 / 20,000` completed updates.
- Resume: only fault recovery inside this new run root; no CR-TOK-0002 checkpoint may seed an arm.

## Results

### Observed

The experiment is preregistered and has not started.

### Interpretation

None yet.

### Not established

- Which scale has the best fixed-20k quality.
- Whether the reconstructed player and platform edges are visually acceptable.
- Whether the selected tokenizer supports action-conditioned dynamics.

### Decision

Bind the exact source commit, launch the single idempotent A10 pipeline, retain every milestone, and keep dynamics blocked until explicit visual review.
