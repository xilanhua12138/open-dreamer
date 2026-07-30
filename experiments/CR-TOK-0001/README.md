# CR-TOK-0001 — smallest published CoinRun tokenizer scaling point

## Question

Can the repository's smallest published CoinRun tokenizer parameter point and native `1e16` FLOPs schedule run end-to-end on one A10, and what held-out reconstruction quality does it achieve on locally collected CoinRun data?

## Why this experiment

The first minimal tokenizer run used an arbitrary short schedule. It showed that code executed, but could not be compared with the smallest point in the released CoinRun scaling figure. This run aligns the published parameter point and native FLOPs-budget mechanism before investing in a larger sweep.

## Hypothesis and falsifier

- Hypothesis: the approximately `0.17M` point is small enough for an A10 24 GB and should complete the repository's native scaling schedule with finite held-out reconstruction metrics.
- Falsified if: the exact model cannot be instantiated, the FLOPs schedule cannot complete on one A10, or held-out reconstruction is non-finite/collapsed.

## Controlled design

- Public reference: approximately `0.17M`, `1e16` model FLOPs, and an approximately `21.24 dB` value estimated from the released SVG.
- Local model: 163,392 parameters; encoder and decoder each use depth 1, `d_model=64`, one head and 16 latents.
- Training data: 2,048 random-action CoinRun trajectories × 64 frames.
- Evaluation: 256 held-out clips, 16 frames per clip, seed 4242.

The public plot value is an approximate visual reading, not an exact published table entry. The data collection and full sweep recipe were not released, so this is not a strict numerical reproduction.

## Results

### Observed

- Native FLOPs allocation produced 56,928 optimizer steps and completed in 54m22s of training.
- Final training-batch PSNR was `25.0045 dB`.
- Held-out online PSNR: clean `24.2488 dB`, masked `20.2864 dB`.
- Held-out EMA PSNR: clean `24.2447 dB`, masked `20.3353 dB`.
- Peak device-memory field retained by the native result row: 9,326,919,680 bytes.

### Interpretation

The smallest released parameter point and native budget mechanism are operational on one A10. Masked held-out reconstruction is close to the rough public-figure estimate, but the comparison is only directional because the local dataset and unknown release details differ.

### Not established

- Exact reproduction of the paper or public scaling point.
- That `25.0045 dB` is comparable to held-out PSNR; it is a final training-batch metric.
- Any world-model dynamics or action-conditioned prediction quality.
- The earlier conversational number `29.94 dB`; no retained authoritative held-out artifact supports it.

### Decision

Use this checkpoint as the fixed tokenizer for small CoinRun dynamics ablations. Do not scale tokenizer size until the world-model training and held-out rollout protocol is stable.
