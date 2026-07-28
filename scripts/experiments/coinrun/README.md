# CoinRun experiment runners

These are the exact small-scale runner/evaluator scripts retained from the 2026-07-28 experiment chain. Their original SHA256 values are recorded in the corresponding manifests and results.

They intentionally preserve the DSW layout used by the run:

```text
/mnt/workspace/open-dreamer
/mnt/workspace/datasets/coinrun-minimal-complete-20260728
```

Before reuse:

1. Read repository `CLAUDE.md`.
2. Create a new experiment record or document a fault-only resume of an existing ID.
3. Confirm the current source commit and a clean/hashed dirty state.
4. Confirm dataset/checkpoint paths and the existing authorized accelerator.
5. Never reuse an old `STATUS=COMPLETE` marker as scientific evidence; check all expected arms and artifacts.

## Scripts by experiment

| Experiment | Primary scripts |
|---|---|
| `CR-TOK-0001` | `run_coinrun_official_min_scaling.sh`, `eval_coinrun_tokenizer_psnr.py` |
| `CR-DYN-0001` | `generate_coinrun_records.py`, `compute_coinrun_latent_stats.py`, `run_coinrun_minimal_complete.sh`, `evaluate_coinrun_minimal.sh` |
| `CR-DYN-0002` | `probe_coinrun_dynamics_scaling.py`, `run_coinrun_dynamics_scaling.sh`, `score_coinrun_rollouts.py` |
| `CR-DYN-0003` | `run_coinrun_dynamics_fixed20k.sh`, `score_coinrun_rollouts.py` |
| `CR-DYN-0004` | `probe_coinrun_dynamics_scaling.py`, `run_coinrun_large_fixed20k.sh` |
| `CR-DYN-0005` | `select_best_coinrun_checkpoint.py`, `eval_coinrun_context_ablation.py`, `run_coinrun_context_ablation.sh` |
| `CR-DEMO-0001` | `live_coinrun_demo.py`, `run_coinrun_live_demo.sh` |

`run_coinrun_extension_pipeline.sh` orders the large run, checkpoint selection, context ablation, and demo startup. It must remain idempotent through its PID/status/ready files.

The repository-level `scripts/train_dynamics.py` accepts dataset-declared non-negative action dimensions instead of hard-coding Minecraft action counts. This is required for CoinRun's 16-way categorical action space.
