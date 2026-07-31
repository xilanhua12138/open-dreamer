# CR-DYN-0011 — CoinRun dynamics offline-latent reference repair

## Question

Can we remove the repeated 16.6M tokenizer encoder from every dynamics update
without changing the CoinRun transition, sampling, model or evaluation
protocol?

## Why this experiment

CR-DYN-0010 trained from raw RGB. At every optimizer update it encoded all
selected frames again even though the tokenizer is frozen. Its first 10k
milestone projected roughly 13 hours for 200k updates. OpenDreamer's general
dynamics recipe instead consumes pre-tokenized latent ArrayRecords.

## Controlled design

- Baseline: the retained CR-DYN-0010 raw-video run through its first 10k fixed
  validation.
- Changed: encode each raw frame exactly once before training and load latent
  records thereafter.
- Held constant: raw train/eval trees, 4,096/512 record counts, 160-frame
  records, action/reward/terminal order, reward-biased crop rule, frozen
  tokenizer, latent normalization, 3.93M dynamics, batch 16, 64/128 schedule,
  optimizer, seed, 200k updates, `k_max=256` and terminal raw-RGB evaluation.
- New audit: compare every latent record to the raw source for actions, rewards,
  terminals, record index and raw-tree identity before training.

## Preregistered acceptance

- Both full train/eval pair audits pass.
- The 10k steady-state update throughput is at least 1.25x CR-DYN-0010.
- The unchanged run reaches 200k and completes shortcut, full diffusion and
  action-corruption evaluations.
- The CR-DYN-0010 absolute quality/action thresholds remain in force.
- Direct visual review is still mandatory.

## Results

### Observed

- Deterministic offline encoding produced 4,096 train and 512 held-out
  160-frame records with latent shape `160 × 16 × 16`.
- Full pair audits passed for every record: exact categorical actions, rewards,
  terminals, record order and raw-tree identities are preserved.
- After a zero-update preprocessing-wrapper repair, the unchanged 3.93M
  dynamics arm started from scratch at 2026-07-31 13:11:38 Asia/Shanghai.
- An external timer interrupted the first attempt after metric update 62,601.
  The explicitly authorized recovery restored this experiment's step-50k
  checkpoint under attempt `9e81f38c-090d-4b59-b11c-669103fa190e`; updates
  50,001–62,601 were replayed and remain recorded as a deviation.
- The recovery completed the unchanged 200,000-update timeline. Final
  checkpoint step 199,999 contains 62 files / 99,518,770 bytes and has
  project-canonical tree SHA256
  `7fa36878716d9d943a947882402a9e27437707f1fbd64022e264a67e3d72485c`.
- Over recorder progress samples from updates 1,001–10,000, median throughput
  was 6.5986 updates/s from offline latents versus 4.0157 updates/s from raw
  RGB: a 1.6432x speedup, above the preregistered 1.25x gate.

Terminal original-RGB metrics:

| Evaluation | Videos | Mean-frame PSNR | Mean-video PSNR | Mean SSIM |
|---|---:|---:|---:|---:|
| Shortcut | 128 | 24.065977 dB | 21.896703 dB | 0.833979 |
| Full diffusion, 256 steps | 8 | 24.053035 dB | 22.448113 dB | 0.846804 |

Fixed-noise action control over 64 held-out futures:

| Future actions | Horizon-16 PSNR | Mean SSIM | Delta vs aligned |
|---|---:|---:|---:|
| Aligned | 26.210389 dB | 0.875959 | — |
| Batch shuffled | 18.404333 dB | 0.687258 | -7.806056 dB |
| One-step shifted | 23.326101 dB | 0.808972 | -2.884288 dB |
| All no-op | 19.729838 dB | 0.709443 | -6.480551 dB |

All preregistered numeric quality and action-use gates passed. Direct visual
review remains mandatory and is not inferred from these metrics.

### Interpretation

Offline latent encoding preserved the audited transition protocol and removed
repeated tokenizer work while materially accelerating the same dynamics
recipe. Terminal shortcut, full-diffusion and action-control metrics support
the preregistered infrastructure hypothesis. The replayed interval is a
fault-recovery deviation, not extra scientific budget.

### Not established

- Visual acceptability or interactive usability of the terminal checkpoint.
- Which individual part of the broader reference-recipe repair caused the
  quality gain.
- Whether a model larger than 3.93M improves this completed recipe.

### Decision

Use this checkpoint and its terminal metrics as the formal Medium reference for
NanoDreamer parity. Keep CR-DYN-0012 XLarge paused until NanoDreamer reproduces
the Tokenizer, PPO, full data/latent audits, Dynamics controls and continuous
demo within its frozen tolerances. Quality remains `awaiting_visual_review`.
