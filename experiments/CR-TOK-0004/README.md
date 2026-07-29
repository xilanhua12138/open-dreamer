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

### Interpretation

Neither failure is a held-out scientific observation about the 28.7M-label tokenizer. The first was a source initialization defect; the second is currently classified as a recoverable CUDA runtime interruption because it left no kernel Xid/OOM evidence and did not recur during launch validation.

### Not established

- Whether n28.7m improves reconstruction metrics or visual quality.
- Whether the additional capacity is compute-efficient.
- Whether the tokenizer works with action-conditioned dynamics.

### Decision

Continue the single attempt-03 PID, preserve both earlier failures, and do not start dynamics or PPO while tokenizer training is active.

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
