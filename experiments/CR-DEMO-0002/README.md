# CR-DEMO-0002 — corrected CoinRun dynamics browser demo

## Question

Can the held-out-selected checkpoint from the corrected PPO-mixture and dynamics-scale chain sustain action-conditioned browser inference?

## Why this experiment

CR-DEMO-0001 proved only that the old stack could serve frames. The user rejected its visual clarity and action response, and the retained audit found a broken Procgen action contract. CR-DYN-0008/0009 replace those invalid inputs with the 16.6M EMA tokenizer, parity PPO trajectories, explicit checkpoint mixtures and fixed-20k scale evaluation.

## Hypothesis and falsifier

- Hypothesis: the selected corrected checkpoint becomes healthy and returns two consecutive generated steps through the remote server and local SSH tunnel.
- Falsified if: selection is incomplete, either health/step path fails, dtype drift returns, the tunnel fails, or user review rejects clarity or controllability.

## Controlled design

- Baseline: CR-DEMO-0001.
- Changed: the selected tokenizer/dynamics/data/action-contract stack.
- Held constant: context 16, four denoise steps, fixed held-out final-policy data, port 7860 and the six exposed actions.
- Known confounders: checkpoint scale is selected descriptively on the same held-out protocol and qualitative acceptance remains a single-user review.

## Results

### Observed

The complete scale grid selected `final_only-medium`. At 06:46:49 the
exclusive poller started the remote demo as PID `74923`. The remote server
became healthy and action `4` returned a real PNG as step 1. That step took
9846.2 ms after a 27232.5 ms warmup.

The local SSH tunnel did not complete the required verification. Three
port-forward attempts timed out, the fourth cycle still failed local health
and step checks, and no process listened on local port 7860. At 07:09 the DSW
timer stopped the instance. The poller restarted the same A10 without creating
or changing resources, then safely blocked on persisted stale demo PID
`74923`.

The recovery preserved that entire failed attempt under
`stale-demo-attempts/20260731T023945Z`, then started a fresh remote demo as PID
`968`. Remote no-op returned step 1. A dedicated launchd-managed SSH tunnel
then passed local health, and local action `7` returned the consecutive step 2
at 764.5 ms after the 32.2-second first-load warmup. The verified review URL is
`http://127.0.0.1:7860`.

### Interpretation

Checkpoint loading, the browser access path and two consecutive generated
steps now work. Cached-step latency is interactive at 764.5 ms, while the
first-load path includes a 32.2-second warmup. This is technical readiness,
not evidence that visuals or action semantics are acceptable.

The subsequent direct user review rejected the result: generated motion was
not temporally credible and selecting actions did not produce credible
responses. The demo therefore proves only runtime closure, not a usable world
model.

### Not established

- Whether shortcut sampling, insufficient optimization, short record structure
  or learned action ignorance is the dominant failure.

### Decision

Do not present `final_only-medium` or `http://127.0.0.1:7860` as a usable
world-model demo. Keep the current process only as a rejected diagnostic
artifact during the user-controlled instance window. The training poller has
been removed; only the dedicated SSH tunnel launchd job remains. Diagnose
action use and shortcut/checkpoint quality before another training run.
