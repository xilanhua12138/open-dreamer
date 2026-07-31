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
- Before an external instance stop, the run emitted structured metrics through
  62,601 completed updates and fixed validation videos through 60,001
  completed updates. The latest durable checkpoint is step 50,000.
- The last pre-stop telemetry reported 5.7366 updates/s, finite loss and an ETA
  of 23,974 seconds. The 60k train metric remained finite at flow MSE
  0.0380859375.
- The same A10 returned to `Running` at 2026-07-31 16:48:40 Asia/Shanghai, but
  the old PID was dead and the GPU was idle. The run is therefore blocked
  pending an explicitly authorized fault resume; it was not silently restarted.
- The user explicitly authorized continuing at 17:16. The same frozen runner
  restored its latest durable step-50k checkpoint as attempt
  `9e81f38c-090d-4b59-b11c-669103fa190e`; no model, data or schedule changed.
- The first retained recovery metric at update 50,401 remained finite
  (`flow_mse=0.0390625`, `grad_norm=2.9029`). Telemetry reported 6.3252
  updates/s, about 4.4/23.0 GiB GPU memory and active GPU compute.
- Because the interruption happened after update 62,601, this recovery must
  replay updates 50,001–62,601. The new attempt ID keeps the overlapping
  metric segments distinguishable; the replay is not counted as additional
  unique optimizer progress.

### Interpretation

The input-representation parity gate has passed and the interrupted run showed
healthy finite optimization through 60k. The authorized recovery is also
healthy, but a complete 200k run and terminal evaluation are still required
before judging the throughput acceptance rule or model quality.

### Not established

- The preregistered throughput speedup relative to CR-DYN-0010.
- Dynamics quality or usability.

### Decision

Continue the explicitly authorized step-50k fault resume under its separate
attempt ID. Preserve both histories and do not start another arm before the
200k terminal evaluations and visual-review evidence are available.
