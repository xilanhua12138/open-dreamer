# CR-DEMO-0001 — action-conditioned CoinRun browser demo

## Question

Can a user press CoinRun action buttons and receive low-latency model-predicted next frames from the selected world-model checkpoint through a local browser?

## Why this experiment

Aggregate metrics do not reveal whether action conditioning feels responsive. A small live interface makes model drift, latency, controllability and failure modes directly inspectable.

## Scope

This demo drives the learned dynamics model. It does not train or run an RL/behavior-cloning agent, and it is not the true Procgen environment.

Planned actions use the retained Procgen mapping:

| Button | Action ID |
|---|---:|
| left | 1 |
| left+jump | 2 |
| noop | 4 |
| jump | 5 |
| right | 7 |
| right+jump | 8 |

The server binds `127.0.0.1:7860`; local access is through an SSH tunnel. The inference sampler uses four denoise steps and context 16 unless `CR-DYN-0005` supports another choice.

## Results

### Observed

- The selected medium checkpoint loaded with context 16 and four denoise steps.
- The first startup exposed an iterable-versus-iterator bug; commit `58758e5` fixed it without rerunning training or evaluation.
- Remote and SSH-forwarded `GET /health` returned `{"ok": true, "model": "medium"}`.
- The original action-7 `POST /api/step` returned a generated PNG frame, but a second generated step failed because the retained latent context had been promoted from `bfloat16` to `float32`.
- Preserving the context dtype before concatenation fixed the mismatch without changing the checkpoint, context length, action mapping or denoise schedule.
- A reset followed by action 7 and action 8 returned generated steps 1 and 2. Their cached inference latencies were `670.9 ms` and `668.1 ms`.
- The recovered process warmed up in `2.8 s` using the persistent JAX compilation cache.
- The local URL is `http://127.0.0.1:7860` until the final user timer at 2026-07-29 15:04:13 Asia/Shanghai.

### Interpretation

The functional two-step demo smoke test now passes, but the low-latency hypothesis remains rejected. It is useful for deliberate sub-second frame-by-frame inspection, not real-time play.

### Not established

- Qualitative control fidelity across every button.
- Autoregressive stability beyond the two-step regression sequence.
- Any trained policy.

### Decision

Give the user the final three-hour evaluation window. Treat latency optimization as a separate experiment before calling the interface real time. Future readiness checks must execute at least two consecutive generated steps; one step cannot exercise the dtype of the updated context.
