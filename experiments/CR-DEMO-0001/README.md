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

## Current result

### Observed

The server and launcher scripts exist, but no live server was started, no tunnel was established, and `/health` or `/api/step` has not been verified.

### Interpretation

The demo is blocked by the same unavailable A10/dependent checkpoint chain as the large and context experiments.

### Not established

- Real-time latency.
- Button-to-motion controllability.
- A working URL.
- Any trained policy.

### Decision

After model selection, start one idempotent server, verify `/health` and one real `/api/step`, then expose only through the local SSH tunnel and retain the exact demo manifest.
