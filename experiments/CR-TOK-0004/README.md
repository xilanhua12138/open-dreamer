# CR-TOK-0004 — CoinRun tokenizer 28.7M-label fixed-20k extension

## Question

At the same fixed 20,000-update recipe and latent interface as `CR-TOK-0003`, does the local 28.7M-label tokenizer improve held-out reconstruction over the 16.6M-label baseline?

## Why this experiment

The current quality-first sweep intentionally ends at the 16.6M published label. The public CoinRun tokenizer curve also names a 28.7M endpoint, so the local curve is missing its largest scale.

## Hypothesis and falsifier

- Hypothesis: the larger symmetric encoder/decoder backbone improves at least one final EMA clean, edge or temporal-change PSNR metric because the 8.6M arm was still improving at 20,000 updates.
- Falsified if: the aligned 28.7M-label arm completes but improves none of those metrics over the 16.6M-label baseline.

## Controlled design

- Baseline: retained `CR-TOK-0003` n16.6m final checkpoint and held-out evaluation.
- Changed: backbone depth/width from `5 × 320` to `6 × 384`.
- New model identity: 25,564,032 exact local parameters, despite the published `28.7M` label.
- Held constant: immutable train/eval records, 20,000 updates, seed, batch/clip shape, optimizer, loss, EMA, 16 × 16 latent interface and 512-clip held-out evaluation.
- Runtime evidence: structured local identity/state/metrics/telemetry/artifacts plus W&B. W&B may run offline until the DSW host is authenticated; the mode must be reported truthfully.
- Periodic validation: the same 16 fixed validation clips every 2,500 completed updates, with `target | online | EMA` PNG and GIF.

## Results

### Observed

A CPU-only construction probe measured 25,564,032 parameters and 25,728,028,508,160 estimated FLOPs per optimizer update.

The first launch from source commit `3833b34b46358b0443bb3571a5745c013eb7a7e7` failed before any optimizer update. `build_validation_dataset_config` directly read every dataclass field from the CoinRun Hydra node; that node legitimately omitted `mouse_repr` and relied on the `DatasetConfig` default, producing `ConfigAttributeError`.

The exact missing-default case was added as a regression test and observed failing before the implementation was fixed. Local fix commit `c565f0b` was applied to the detached execution source as `b3eb2af`. The architecture, data, optimizer, update budget and evaluation protocol did not change, and no checkpoint was reused because the failed attempt completed zero updates.

Attempt 02 started from scratch at `2026-07-30T01:07:25+08:00` under PID `68104`. It reached 1,179 recorder-confirmed completed updates before JAX raised `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` while materializing asynchronous train metrics. The process exited, the GPU returned idle, and neither NVIDIA Xid nor kernel OOM evidence was observed. The machine-owned failed-run record and full local/remote log are retained.

Attempt 03 started once at `2026-07-30T01:58:23+08:00` under PID `71283`. It restored this experiment's own checkpoint step 0; no external or earlier-experiment checkpoint was used. The source and scientific protocol are unchanged. W&B remains truthfully offline, and the local structured evidence remains authoritative. The same A10 has a shutdown guard due at `2026-07-30T13:01:15+08:00`.

Attempt 03 passed the first scheduled 2,500-update checkpoint. On the immutable 16-clip periodic validation set (`SHA256 350bd313...0188`), online clean PSNR was `28.1951 dB` and EMA clean PSNR was `24.4368 dB`. The matching target-online-EMA PNG/GIF are retained locally and on DSW. This is a frequent progress diagnostic, not the 512-clip held-out result used for the final comparison.

Attempt 03 subsequently completed exactly 20,000 optimizer updates, all eight fixed-set periodic validations and all four 512-clip held-out evaluations. The final checkpoint is step `19999`; its 47-file, 285,094,215-byte tree has SHA256 `930b975a...fb9b`.

| Completed updates | EMA clean | EMA edge | EMA temporal-change |
|---:|---:|---:|---:|
| 2,500 | 25.6820 dB | 20.2490 dB | 20.8496 dB |
| 5,000 | 32.3405 dB | 24.3265 dB | 25.1582 dB |
| 10,000 | 34.8678 dB | 26.0988 dB | 26.8460 dB |
| 20,000 | 36.8657 dB | 27.0661 dB | 27.6592 dB |

The final n16.6m and n28.7m evaluations match exactly on dataset path, seed, batch size, frame count, batch count, clip count and completed-update horizon. Relative to n16.6m, n28.7m improved final EMA clean / edge / temporal-change PSNR by `+0.6965 / +0.2921 / +0.2308 dB`; EMA masked PSNR improved by `+0.7019 dB`.

### Interpretation

Neither retained failure is a held-out scientific observation about the 28.7M-label tokenizer. The first was a source initialization defect; the second was a recoverable CUDA runtime interruption because it left no kernel Xid/OOM evidence and did not recur after recovery.

The aligned final result supports the preregistered capacity-quality hypothesis: the larger endpoint improved every final EMA comparison metric. The gain is modest relative to the parameter increase, so it does not establish compute efficiency.

### Not established

- Whether the additional capacity is compute-efficient.
- Whether users judge the aligned final grid visually acceptable.
- Whether the tokenizer works with action-conditioned dynamics.

### Decision

Retain n28.7m as the numerically strongest fixed-20k endpoint pending user visual review. Proceed to the separately preregistered real-environment PPO trajectory-collection experiment; do not start dynamics.

<!-- BEGIN GENERATED RUN n28p7m-seed0-attempt-02-cuda-stream-capture -->
### Generated run evidence: `n28p7m-seed0-attempt-02-cuda-stream-capture`

- Run ID: `3d478ca3-6f1d-4068-8d36-6139b28b3a1c`
- State: `FAILED`
- Last completed updates: `1179`
- Source commit: `b3eb2afd37ece2880c60c34762e5cfa427ee19f6`

| Prefix | Last metrics |
|---|---|
| `train/` | `data_tokens_seen=150732800`, `flops_spent=29587232784384000`, `loss=0.037765804678201675`, `lpips=0.03341260552406311`, `lr=0.003000000026077032`, `mse=0.031083282083272934`, `psnr=27.570392608642578`, `rmse=0.17630451917648315`, `total_tokens_seen=188416000` |

Machine-readable evidence: `experiments/CR-TOK-0004/raw/n28p7m-seed0-attempt-02-cuda-stream-capture-run-evidence.json`.
<!-- END GENERATED RUN n28p7m-seed0-attempt-02-cuda-stream-capture -->

<!-- BEGIN GENERATED RUN n28p7m-seed0-final -->
### Generated run evidence: `n28p7m-seed0-final`

- Run ID: `3d478ca3-6f1d-4068-8d36-6139b28b3a1c`
- State: `COMPLETED`
- Last completed updates: `20000`
- Source commit: `b3eb2afd37ece2880c60c34762e5cfa427ee19f6`

| Prefix | Last metrics |
|---|---|
| `train/` | `data_tokens_seen=2621308928`, `flops_spent=514534842134691840`, `loss=0.00302625959739089`, `lpips=0.0012241953518241644`, `lr=1.8739700635705958e-07`, `mse=0.0027814204804599285`, `psnr=38.69329071044922`, `rmse=0.05273917317390442`, `total_tokens_seen=3276636160` |
| `validation/` | `ema_clean_mse=0.0005120024822341899`, `ema_clean_psnr=32.90727933520267`, `num_clips=16`, `online_clean_mse=0.0005134190626752874`, `online_clean_psnr=32.89528010497076` |

Machine-readable evidence: `experiments/CR-TOK-0004/raw/n28p7m-seed0-final-run-evidence.json`.
<!-- END GENERATED RUN n28p7m-seed0-final -->
