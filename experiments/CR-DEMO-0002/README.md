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

### Interpretation

Remote checkpoint loading and one generated step work. The browser access path
and sustained two-step path do not yet work, so this is an infrastructure
blocker rather than a completed interactive evaluation.

### Not established

- Local-tunnel health and a subsequent real generated step.
- Sustained two-step inference.
- User-perceived visual clarity and action responsiveness.

### Decision

Do not advertise `http://127.0.0.1:7860` as usable. Leave restart ownership
with the exclusive poller and resolve its stale-PID lifecycle blocker before
repeating the complete remote-health/step plus local-health/subsequent-step
contract.
