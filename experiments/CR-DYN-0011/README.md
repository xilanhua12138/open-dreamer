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

Not started.

### Interpretation

None.

### Not established

- Offline latent parity.
- Throughput improvement.
- Dynamics quality or usability.

### Decision

Freeze the implementation, publish a clean detached execution worktree, encode
and audit the retained raw corpus, then start the latent-only dynamics run.
