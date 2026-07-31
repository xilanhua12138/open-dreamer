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
6. For new runs, enable the structured recorder and fixed held-out validation
   described in `experiments/OBSERVABILITY.md`. W&B is a mirror; local JSONL and
   hashed media remain mandatory.
7. Launch new formal arms through `scripts/experiments/run_recorded.py`, so a
   successful or failed process exit materializes machine-owned ledger evidence
   without an agent copying metric values.

Do not edit a frozen historical runner merely to add W&B. Create a new
experiment ID/runner or use the observability options before its source commit
is frozen. A future tokenizer runner should normally include:

```text
use_wandb=true
logger.wandb_project=open-dreamer
logger.wandb_group=<experiment-id>
validation.enabled=true
validation.dataset_path=<disjoint-eval-path>
validation.every_steps=<preregistered-interval>
```

## Scripts by experiment

| Experiment | Primary scripts |
|---|---|
| `CR-TOK-0001` | `run_coinrun_official_min_scaling.sh`, `eval_coinrun_tokenizer_psnr.py` |
| `CR-TOK-0002` | `probe_coinrun_tokenizer_scale_v2.py`, `run_coinrun_tokenizer_scale_v2.sh`, `select_coinrun_tokenizer_scale.py` |
| `CR-TOK-0003` | `tokenizer_quality_first_protocol.py`, `probe_coinrun_tokenizer_quality_first.py`, `run_coinrun_tokenizer_quality_first_20k.sh`, `summarize_tokenizer_quality_first.py` |
| `CR-TOK-0004` | `tokenizer_28p7_protocol.py`, `probe_coinrun_tokenizer_28p7.py`, `compare_tokenizer_extension.py`, `run_coinrun_tokenizer_28p7_fixed20k.sh` |
| `CR-DYN-0001` | `generate_coinrun_records.py`, `compute_coinrun_latent_stats.py`, `run_coinrun_minimal_complete.sh`, `evaluate_coinrun_minimal.sh` |
| `CR-DYN-0002` | `probe_coinrun_dynamics_scaling.py`, `run_coinrun_dynamics_scaling.sh`, `score_coinrun_rollouts.py` |
| `CR-DYN-0003` | `run_coinrun_dynamics_fixed20k.sh`, `score_coinrun_rollouts.py` |
| `CR-DYN-0004` | `probe_coinrun_dynamics_scaling.py`, `run_coinrun_large_fixed20k.sh` |
| `CR-DYN-0005` | `select_best_coinrun_checkpoint.py`, `eval_coinrun_context_ablation.py`, `run_coinrun_context_ablation.sh` |
| `CR-DEMO-0001` | `live_coinrun_demo.py`, `run_coinrun_live_demo.sh`, `smoke_live_coinrun_demo.sh` |
| `CR-DEMO-0003` | `live_coinrun_continuous_demo.py` |
| `CR-PPO-0001` | `prepare_coinrun_ppo_runtime.sh`, `train_coinrun_ppo.py`, `collect_coinrun_ppo_records.py`, `audit_coinrun_records.py`, `verify_coinrun_dataset_pair.py`, `run_coinrun_ppo_collector.sh` |

`run_coinrun_extension_pipeline.sh` orders the large run, checkpoint selection, context ablation, and demo startup. It must remain idempotent through its PID/status/ready files.

The repository-level `scripts/train_dynamics.py` accepts dataset-declared non-negative action dimensions instead of hard-coding Minecraft action counts. This is required for CoinRun's 15-way categorical action space.

The live-demo smoke test deliberately executes two consecutive generated steps.
A single successful step does not exercise the dtype of the autoregressively
updated latent context and is insufficient evidence for sustained interaction.

The PPO collector is a data-generation dependency for a later dynamics
experiment. `train_coinrun_ppo.py` learns and validates the policy in real
Procgen CoinRun. `collect_coinrun_ppo_records.py` loads a frozen checkpoint and
writes action-aligned records that cannot cross episode boundaries. The runner
stops after dataset audit; it deliberately does not start a world model or any
additional policy stage.

Procgen 0.10.7 has no CPython 3.11 wheel, while the repository training
environment uses Python 3.11. `prepare_coinrun_ppo_runtime.sh` therefore creates
one explicit Python 3.10 environment containing both Procgen and pinned
CUDA-enabled JAX/Flax/Optax versions. It performs a CPU-only import,
environment, action-space and actor-critic shape smoke test without competing
for the active training GPU. A later launch must separately verify that this
same environment sees the A10 through JAX.
