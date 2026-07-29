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
- The user evaluation rejected the result as usable: tokenizer reconstructions were visually unclear and predicted dynamics did not respond correctly to actions.
- The live Procgen environment reports `D15[]`, but `configs/dataset/coinrun.yaml` declares `categorical_action_dim: 16`.
- `scripts/train_dynamics.py` calls `shift_actions` on every batch. Its generic helper prepends `categorical_action_dim // 2`, which is action `8` under the current config; Procgen defines no-op as action `4` and action `8` as `RIGHT+UP`.

### Interpretation

The functional two-step demo smoke test establishes runtime continuity only. It does not establish usable visual quality or controllability. The 0.17M tokenizer already has a weak held-out reconstruction ceiling, the dynamics model adds further error relative to tokenizer-decoded targets, and the training action contract is inconsistent with Procgen. The current checkpoint must therefore be treated as a pipeline smoke artifact rather than an action-controllable world model.

### Not established

- Whether correcting the action cardinality and true no-op ID is sufficient to recover control fidelity after retraining.
- Whether a larger tokenizer is subjectively clear enough for the small CoinRun player and obstacle details.
- Autoregressive stability beyond the two-step regression sequence.
- Any trained policy.

### Decision

Do not call the current demo usable or action-controllable. Before another dynamics run, correct the Procgen action contract and add a same-context, same-randomness counterfactual action test. Evaluate a larger tokenizer separately with foreground/player-region metrics and visual grids. Replace or augment uniformly random trajectories with behavior containing sustained running and jumping before interpreting action conditioning. Do not spend the next GPU budget on another larger dynamics model under the same flawed protocol.
