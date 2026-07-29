# Experiment ledger

This directory is the durable source of truth for local OpenDreamer experiments. The authoritative machine-readable index is `index.json`; each experiment has a preregistration-style `manifest.json`, observed `results.json`, append-only `events.jsonl`, and a human-readable `README.md`.

The 2026-07-28 records are retrospective backfills from retained Hydra configs, metrics, logs, comparison artifacts, remote paths, and SHA256 hashes. They are explicitly marked `retrospective: true`; later experiments must be registered before GPU time begins.

## Current experiments

| ID | Question | Execution | Scientific result | Claim |
|---|---|---|---|---|
| `CR-TOK-0001` | Can the smallest published CoinRun tokenizer scaling point run end-to-end on one A10? | completed | supports hypothesis | partial reproduction |
| `CR-TOK-0002` | What is the smallest tokenizer scale that clears global, edge, motion and visual gates? | planned | not evaluated | internal result |
| `CR-DYN-0001` | Can a small action-conditioned CoinRun world-model pipeline close end-to-end? | completed | supports hypothesis | pipeline closure |
| `CR-DYN-0002` | Under `C=1e15`, how does capacity trade against optimizer steps? | aborted | inconclusive | internal result |
| `CR-DYN-0003` | At the same 20k-step curriculum, does held-out rollout quality improve with capacity? | completed | supports hypothesis | internal result |
| `CR-DYN-0004` | Does the fixed-20k capacity trend extend to a larger model? | completed | rejects hypothesis | internal result |
| `CR-DYN-0005` | On one checkpoint and identical futures, how do 4/16/32 history frames affect rollout quality? | completed | supports hypothesis | internal result |
| `CR-DEMO-0001` | Can a user drive the selected CoinRun world model through a low-latency browser demo? | completed | rejects hypothesis | smoke test |

## Important reading of the chain

`CR-DYN-0002` is intentionally not deleted. Its medium model received only 474 total steps, with bootstrap beginning at step 237, and its PSNR scored below the small model while SSIM scored above it. The corrected `CR-DYN-0003` fixed every model at 20,000 optimizer steps and found monotonic improvement with capacity. The first run is evidence about an extremely small compute budget, not evidence that larger models are worse.

The 12,902,784-parameter large extension completed but scored below medium, so the prior monotonic capacity trend stopped under this fixed-20k recipe. Medium was then evaluated on byte-identical futures with 4/16/32 history frames; context 16 scored best. The browser demo initially passed health and one real inference step, but a second step exposed float32/bfloat16 context-state drift. After preserving the bfloat16 context dtype, health plus two consecutive steps passed through SSH at about 0.67 seconds per cached step.

The subsequent user evaluation still rejected the demo as usable: tokenizer output was unclear and action response was incorrect. A source/runtime audit found a concrete protocol defect: Procgen CoinRun exposes 15 actions, while the repository config declares 16, and the generic action shifter prepends action 8 although Procgen no-op is action 4. Because this shifter is used during dynamics training, the current checkpoint is retained only as pipeline-smoke evidence. The next run must fix and test the action contract before spending more compute on model scale.

`CR-TOK-0002` therefore blocks new dynamics work and restarts the representation study from scratch. Its primary fixed-FLOPs arms are the local `0.17M / 1.1M / 3.7M` labels on one structured dataset. Larger `8.6M / 16.6M` labels are conditional quality-search arms, not part of the fixed-FLOPs curve. A metric-eligible tokenizer still requires aligned-grid visual acceptance and later joint tokenizer-plus-dynamics inference compatibility before it can be frozen.

## Creating or closing an experiment

Read the complete rules in `../CLAUDE.md`, copy `./_template`, allocate the next ID, and validate:

```bash
uv run --no-project python scripts/validate_experiments.py
```

Do not edit only this table. Update the experiment directory and `index.json` in the same commit.
