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

The exact source commit `26d9a1290dc7e1a42a98cea62284fe7787bbe4e6` started one pipeline on the existing A10 at `2026-07-29T14:56:16+08:00`. The generated plan confirms `20,000` updates for every arm, disabled FLOPs/tokens-per-param overrides, exact `2,500 / 5,000 / 10,000 / 20,000` evaluation milestones and `dynamics_authorized=false`.

All five fresh scales completed exactly 20,000 updates and all four held-out evaluations on the same 512 clips. Their EMA metrics are:

| Scale | Updates | Clean PSNR | Masked PSNR | Edge PSNR | Temporal-change PSNR |
|---|---:|---:|---:|---:|---:|
| 0.17M | 2,500 | 12.0504 | 12.0504 | 11.9046 | 11.4659 |
| 0.17M | 5,000 | 12.0612 | 12.0612 | 11.9386 | 11.4584 |
| 0.17M | 10,000 | 12.0612 | 12.0612 | 11.9348 | 11.4598 |
| 0.17M | 20,000 | 12.0610 | 12.0610 | 11.9328 | 11.4600 |
| 1.1M | 2,500 | 21.7918 | 21.2389 | 17.0254 | 17.4338 |
| 1.1M | 5,000 | 25.6757 | 24.6136 | 19.5432 | 19.8007 |
| 1.1M | 10,000 | 27.0566 | 25.5887 | 20.5354 | 20.7290 |
| 1.1M | 20,000 | 27.7460 | 24.7005 | 20.9956 | 21.2332 |
| 3.7M | 2,500 | 24.2511 | 23.5876 | 18.7702 | 19.7721 |
| 3.7M | 5,000 | 29.5504 | 28.3669 | 22.3873 | 23.2115 |
| 3.7M | 10,000 | 31.6040 | 29.9676 | 23.8374 | 24.5532 |
| 3.7M | 20,000 | 32.7710 | 29.3187 | 24.5359 | 25.2065 |
| 8.6M | 2,500 | 25.3477 | 24.7050 | 19.7458 | 20.3611 |
| 8.6M | 5,000 | 30.8968 | 29.8279 | 23.2430 | 24.0609 |
| 8.6M | 10,000 | 33.5191 | 32.0974 | 25.0501 | 25.8041 |
| 8.6M | 20,000 | 34.8962 | 32.0022 | 25.9151 | 26.5915 |
| 16.6M | 2,500 | 25.4389 | 24.8707 | 20.0149 | 20.7167 |
| 16.6M | 5,000 | 31.9628 | 30.9707 | 24.0440 | 24.9204 |
| 16.6M | 10,000 | 34.5984 | 33.2668 | 25.8410 | 26.5933 |
| 16.6M | 20,000 | 36.1691 | 33.6156 | 26.7739 | 27.4284 |

The same PID completed all five arms without reusing an old checkpoint. All 20 milestone JSON files, reconstruction PNGs, Hydra configs and checkpoint-tree hashes are retained. The final descriptive score ranks `16.6M > 8.6M > 3.7M > 1.1M > 0.17M`; no quality gate or eligible filter was used.

### Interpretation

Under this fixed-20k single-seed protocol, final EMA clean, edge and temporal-change PSNR improve monotonically with tokenizer capacity. The 16.6M arm still improves from 10k to 20k updates, so the largest arm was not obviously saturated at the earlier milestone. This supports the preregistered capacity-quality hypothesis, but numerical ranking is not visual acceptance.

### Not established

- Whether the reconstructed player and platform edges are visually acceptable.
- Whether the 16.6M tokenizer supports controllable action-conditioned dynamics.
- Whether this single-seed ranking generalizes beyond this dataset and recipe.

### Decision

Proceed to the separately preregistered `CR-TOK-0004` 28.7M-label extension on the same A10. Keep dynamics blocked and retain the aligned final five-scale grid for explicit visual review.
