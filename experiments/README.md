# Experiment ledger

This directory is the durable source of truth for local OpenDreamer experiments. The authoritative machine-readable index is `index.json`; each experiment has a preregistration-style `manifest.json`, observed `results.json`, append-only `events.jsonl`, and a human-readable `README.md`.

The 2026-07-28 records are retrospective backfills from retained Hydra configs, metrics, logs, comparison artifacts, remote paths, and SHA256 hashes. They are explicitly marked `retrospective: true`; later experiments must be registered before GPU time begins.

## Current experiments

| ID | Question | Execution | Scientific result | Claim |
|---|---|---|---|---|
| `CR-TOK-0001` | Can the smallest published CoinRun tokenizer scaling point run end-to-end on one A10? | completed | supports hypothesis | partial reproduction |
| `CR-DYN-0001` | Can a small action-conditioned CoinRun world-model pipeline close end-to-end? | completed | supports hypothesis | pipeline closure |
| `CR-DYN-0002` | Under `C=1e15`, how does capacity trade against optimizer steps? | aborted | inconclusive | internal result |
| `CR-DYN-0003` | At the same 20k-step curriculum, does held-out rollout quality improve with capacity? | completed | supports hypothesis | internal result |
| `CR-DYN-0004` | Does the fixed-20k capacity trend extend to a larger model? | blocked | not evaluated | none |
| `CR-DYN-0005` | On one checkpoint and identical futures, how do 4/16/32 history frames affect rollout quality? | blocked | not evaluated | none |
| `CR-DEMO-0001` | Can a user drive the selected CoinRun world model through a low-latency browser demo? | blocked | not evaluated | none |

## Important reading of the chain

`CR-DYN-0002` is intentionally not deleted. Its medium model received only 474 total steps, with bootstrap beginning at step 237, and its PSNR scored below the small model while SSIM scored above it. The corrected `CR-DYN-0003` fixed every model at 20,000 optimizer steps and found monotonic improvement with capacity. The first run is evidence about an extremely small compute budget, not evidence that larger models are worse.

`CR-DYN-0004`, `CR-DYN-0005`, and `CR-DEMO-0001` remain blocked because the existing A10 instance could not restart while the resource sale was temporarily suspended. No large checkpoint, context-ablation metric, or working live demo has been produced yet.

## Creating or closing an experiment

Read the complete rules in `../CLAUDE.md`, copy `./_template`, allocate the next ID, and validate:

```bash
uv run --no-project python scripts/validate_experiments.py
```

Do not edit only this table. Update the experiment directory and `index.json` in the same commit.
