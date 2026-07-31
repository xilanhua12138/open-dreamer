# Experiment runtime observability

Formal runs use one durable local identity whether W&B is enabled or not. A run
directory contains:

```text
run-id.json                 # Stable across fault-only resumes
runtime-identity.json       # Latest process-attempt identity
runtime-identities/*.json   # Immutable identity for every attempt
runtime-identities/*.patch  # Saved tracked dirty patch for that attempt
run-state.json              # Atomic current state and last completed update
metrics.jsonl               # Structured train/eval/validation metrics
telemetry.jsonl             # Lifecycle, progress/ETA, and system telemetry
artifacts.jsonl             # Media/artifact URI, byte size, and SHA256
wandb-run.json              # W&B mirror identity when enabled
validation-set.json         # Fixed held-out sample identity when enabled
validation/updates-*        # Periodic held-out metrics, PNG, and GIF
```

`runtime-identity.json` records the source commit and saved dirty-patch hash,
dependency-file hashes, selected package versions, Python/platform identity,
JAX devices, NVIDIA driver/GPU identity when available, a redacted command, and
a safe environment allowlist. Untracked source files are content-hashed but not
copied; their presence marks source reproduction incomplete. Proxy values and
credentials are never copied.

`run-state.json` is the machine-owned source for process state. Shell `STATUS`
and PID files may still be used by legacy runners, but new monitors should read
the structured state and require recent progress before deciding that a process
is healthy. Progress heartbeats are sampled independently from metric logging,
so a slow model cannot look healthy merely because its PID still exists.

## Weights & Biases

W&B is an optional online mirror, not the only copy of experiment evidence.
Local JSONL and media are always written first.

Authenticate on each execution machine without committing the key:

```bash
wandb login
```

The recommended project is `open-dreamer`. It can be overridden without source
changes:

```bash
export WANDB_PROJECT=open-dreamer
export WANDB_ENTITY=<team-or-user>
export WANDB_MODE=online
```

Enable W&B and fixed validation in a formal tokenizer runner:

```text
use_wandb=true
logger.wandb_group=CR-TOK-0004
logger.wandb_tags=[coinrun,tokenizer,n8p6m,seed0]
validation.enabled=true
validation.dataset_path=/mnt/workspace/datasets/<dataset>/eval
validation.every_steps=2500
validation.batch_size=8
validation.batches=2
validation.frames=16
validation.seed=4242
```

The validation dataset must be disjoint from training. The loader materializes
the same seed-addressed batches once per process attempt and records their
content SHA256. Each validation milestone reports online/EMA clean PSNR and MSE
and emits a `target | online | EMA` PNG and animated GIF. GIF avoids forking an
ffmpeg subprocess from the multithreaded JAX process. On fault-only resume, the
same dataset, seed, batch shape, and content hash must be recovered.

For a connection smoke test without uploading:

```bash
WANDB_MODE=offline uv run scripts/train_tokenizer.py \
  use_wandb=true \
  logger.wandb_project=open-dreamer \
  max_steps=2
```

An online run must have a non-null URL in `wandb-run.json`. Never describe an
offline run as uploaded; use `wandb sync <offline-run-directory>` after login.

## Running with automatic ledger evidence

Formal runners should execute training through the recorded-run wrapper. It
preserves the command's exit code and materializes evidence after both successful
and failed runs:

```bash
uv run scripts/experiments/run_recorded.py \
  --experiment-id CR-TOK-0004 \
  --run-name n8p6m-seed0 \
  --run-dir /path/to/run \
  --experiment-dir experiments/CR-TOK-0004 \
  -- uv run scripts/train_tokenizer.py \
     hydra.run.dir=/path/to/run
```

A successful command without runtime evidence is rejected. This prevents a
nominally complete arm from silently bypassing provenance collection.

For an already-finished or remotely synchronized run, the materializer can also
be invoked directly:

```bash
uv run scripts/experiments/materialize_run_evidence.py \
  --experiment-id CR-TOK-0004 \
  --run-name n8p6m-seed0 \
  --run-dir /path/to/run \
  --experiment-dir experiments/CR-TOK-0004
```

The command:

1. reads runtime identity, state, structured metrics and validation artifacts;
2. hashes the evidence and writes `raw/<run-name>-run-evidence.json`;
3. upserts one generated observation and artifact into `results.json`;
4. appends one `run_evidence_materialized` event;
5. increments `record_revision` only when the evidence fingerprint changes;
6. replaces one marked generated section in the experiment README.

Human-authored hypothesis, interpretation, claim level, deviations, failures,
and decisions remain manual. Run `scripts/validate_experiments.py` before
committing the materialized ledger.
