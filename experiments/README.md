# Experiment ledger

This directory is the durable source of truth for local OpenDreamer experiments. The authoritative machine-readable index is `index.json`; each experiment has a preregistration-style `manifest.json`, observed `results.json`, append-only `events.jsonl`, and a human-readable `README.md`.

The 2026-07-28 records are retrospective backfills from retained Hydra configs, metrics, logs, comparison artifacts, remote paths, and SHA256 hashes. They are explicitly marked `retrospective: true`; later experiments must be registered before GPU time begins.

## Current experiments

| ID | Question | Execution | Scientific result | Claim |
|---|---|---|---|---|
| `CR-TOK-0001` | Can the smallest published CoinRun tokenizer scaling point run end-to-end on one A10? | completed | supports hypothesis | partial reproduction |
| `CR-TOK-0002` | What did the unequal-step tokenizer pilot establish before the obsolete runner exited? | aborted | inconclusive | internal result |
| `CR-TOK-0003` | At fixed 20k updates, how do quality and convergence change across all five tokenizer scales? | running | not evaluated | none |
| `CR-TOK-0004` | At fixed 20k updates, does the 28.7M-label extension improve over 16.6M? | queued | not evaluated | none |
| `CR-DYN-0001` | Can a small action-conditioned CoinRun world-model pipeline close end-to-end? | completed | supports hypothesis | pipeline closure |
| `CR-DYN-0002` | Under `C=1e15`, how does capacity trade against optimizer steps? | aborted | inconclusive | internal result |
| `CR-DYN-0003` | At the same 20k-step curriculum, does held-out rollout quality improve with capacity? | completed | supports hypothesis | internal result |
| `CR-DYN-0004` | Does the fixed-20k capacity trend extend to a larger model? | completed | rejects hypothesis | internal result |
| `CR-DYN-0005` | On one checkpoint and identical futures, how do 4/16/32 history frames affect rollout quality? | completed | supports hypothesis | internal result |
| `CR-DEMO-0001` | Can a user drive the selected CoinRun world model through a low-latency browser demo? | completed | rejects hypothesis | smoke test |
| `CR-PPO-0001` | Can PPO in real CoinRun produce auditable goal-directed trajectories for dynamics? | completed | rejects hypothesis | internal result |
| `CR-PPO-0002` | Does an official-recipe easy-200 parity control recover the public CoinRun curve? | completed | supports hypothesis | partial reproduction |
| `CR-PPO-0003` | Which individual old-recipe difference reproduces the policy-quality collapse? | completed | supports hypothesis | internal result |
| `CR-PPO-0004` | Do the old settings collapse only when combined under an identical evaluator? | aborted | inconclusive | none |
| `CR-PPO-0005` | Does orthogonal √2 amplify an unnormalized residual encoder, and which mitigation recovers PPO? | queued | not evaluated | none |

## Important reading of the chain

`CR-DYN-0002` is intentionally not deleted. Its medium model received only 474 total steps, with bootstrap beginning at step 237, and its PSNR scored below the small model while SSIM scored above it. The corrected `CR-DYN-0003` fixed every model at 20,000 optimizer steps and found monotonic improvement with capacity. The first run is evidence about an extremely small compute budget, not evidence that larger models are worse.

The 12,902,784-parameter large extension completed but scored below medium, so the prior monotonic capacity trend stopped under this fixed-20k recipe. Medium was then evaluated on byte-identical futures with 4/16/32 history frames; context 16 scored best. The browser demo initially passed health and one real inference step, but a second step exposed float32/bfloat16 context-state drift. After preserving the bfloat16 context dtype, health plus two consecutive steps passed through SSH at about 0.67 seconds per cached step.

The subsequent user evaluation still rejected the demo as usable: tokenizer output was unclear and action response was incorrect. A source/runtime audit found a concrete protocol defect: Procgen CoinRun exposes 15 actions, while the repository config declares 16, and the generic action shifter prepends action 8 although Procgen no-op is action 4. Because this shifter is used during dynamics training, the current checkpoint is retained only as pipeline-smoke evidence. The next run must fix and test the action contract before spending more compute on model scale.

`CR-TOK-0002` restarted the representation study but exposed a second confound. Its FLOPs allocations translated to `62,675 / 9,358 / 2,938 / 2,550 / 2,656` optimizer updates, and the obsolete initial runner exited after only the first three arms. Those retained results are optimization-budget diagnostics, not a fair quality curve; the experiment is closed as aborted rather than silently filled under a changed meaning.

`CR-TOK-0003` is the corrected quality-first experiment. It trains the complete local `0.17M / 1.1M / 3.7M / 8.6M / 16.6M` ladder from scratch for exactly 20,000 optimizer updates per arm, with scaling helpers disabled. Every arm is evaluated after 2,500, 5,000, 10,000 and 20,000 completed updates. The final ranking remains descriptive, and dynamics stays blocked until the aligned final grid receives explicit visual acceptance and a later joint-interface smoke test passes.

The first four arms are now durably recorded: 0.17M stayed near 12 dB across all four EMA curves, 1.1M reached final EMA clean / edge / temporal-change PSNR of `27.7460 / 20.9956 / 21.2332 dB`, 3.7M reached `32.7710 / 24.5359 / 25.2065 dB`, and 8.6M reached `34.8962 / 25.9151 / 26.5915 dB`. The pipeline is training fresh 16.6M. These partial metrics do not establish the five-scale ranking or visual acceptability.

`CR-TOK-0004` is preregistered and queued to add the missing published `28.7M` label only after `CR-TOK-0003` terminates. The local depth-6, `d_model=384` implementation has exactly 25,564,032 parameters. Its execution source is frozen at `3833b34`; it keeps the same fixed-20k recipe and held-out identity, adds structured runtime telemetry plus fixed validation media every 2,500 updates, and remains blocked from dynamics pending visual review.

`CR-PPO-0001` completed the missing real-environment trajectory collector described qualitatively by OpenDreamer but absent from its released code. Its final preregistered deterministic policy missed the 50% threshold at `29.6875%`, even though the `22,020,096`-step milestone briefly reached `51.5625%`. The final frozen checkpoint still produced 4,096 train and 512 eval records with `55.6904% / 65.7534%` completed-episode success and valid split/pair audits. A larger post-hoc stochastic full-distribution evaluation raised the final estimate to `4.9609375` mean return, which showed a measurement bias but remained well below the public easy-200 curve.

`CR-PPO-0002` was therefore preregistered before its GPU run. It preserved the 25,165,824-transition budget and architecture widths while aligning 200 training levels, reward-normalizer gamma `0.99`, per-minibatch advantage normalization, Glorot IMPALA backbone initialization and stochastic `num_levels=0` evaluation. The final independent 512-episode evaluation reached `8.671875` mean return and `86.71875%` success, versus the old checkpoint's `4.9609375 / 49.609375%` under the same metric. This supports the combined parity hypothesis but does not identify a single causal bug; one-seed JAX, Procgen-version and framework confounders keep the claim at partial reproduction. No replacement trajectories or dynamics were started.

`CR-PPO-0003` is the preregistered causal screen requested after that result. It repeats the validated recipe to 6,291,456 transitions and then reverts exactly one training factor in each serial arm: 500 levels, reward-normalizer gamma `0.999`, whole-batch advantage normalization or orthogonal initialization. A factor is independently dominant only if final-256 mean return drops at least 1.0 or success rate drops at least 0.10 versus the same-run reference. No arm may collect trajectories or start dynamics.

`CR-PPO-0004` was preregistered as the required follow-up if all single reverts remained healthy, but the final `orthogonal_init` arm independently crossed the collapse threshold. The user therefore cancelled the combined run before execution; it is retained as an aborted scheduling decision, and no GPU time or new result was produced.

`CR-PPO-0005` replaces that lower-information combined control with a mechanism study. It first measures per-block residual/skip scale, encoder JVP gain and synthetic PPO gradient norms over 16 initialization seeds, then trains four targeted mitigations to the same 6.29M-transition budget. It reuses the byte-retained `CR-PPO-0003` Glorot and orthogonal-√2 anchors, but all new arms still run unconditionally; probe metrics are not a quality gate.

## Creating or closing an experiment

Read the complete rules in `../CLAUDE.md`, copy `./_template`, allocate the next ID, and validate:

```bash
uv run --no-project python scripts/validate_experiments.py
```

Do not edit only this table. Update the experiment directory and `index.json` in the same commit.
