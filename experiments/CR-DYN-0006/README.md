# CR-DYN-0006 — PPO-stage trajectory ablation

> Superseded before execution. No data collection or GPU training ran under
> this ID. The user clarified that the intended variable was the mixture ratio
> of multiple PPO checkpoints inside one corpus, so the corrected experiment is
> `CR-DYN-0008`.

## Question

When tokenizer, dynamics architecture, optimizer budget and held-out futures are fixed, does collecting CoinRun training trajectories from later CR-PPO-0002 checkpoints improve world-model rollout quality?

## Design

The four training corpora come from `1,048,576`, `6,291,456`, `12,582,912` and `25,165,824` PPO environment transitions. Every corpus contains 2,048 episode-safe 64-frame records from levels `[0, 200)`, using the same seed, temperature and exploration epsilon. All four arms train the same 3,931,392-parameter medium dynamics model for 20,000 updates with the 14,912,320-parameter 16.6M EMA tokenizer.

Every arm is scored on one fixed 512-record corpus collected from the final PPO checkpoint on disjoint levels `[10000, 10500)`. The evaluation uses 32 identical held-out futures, 16 context frames and 16 predicted frames.

## Observed

No collection or training result was observed. The experiment was aborted
during preregistration review.

## Interpretation

The isolated-stage design answered a different question from the requested
mixed-checkpoint recipe and therefore must not be executed or interpreted.

## Not established

- Whether later-policy data is better for dynamics.
- Whether policy-stage effects are monotonic.
- Whether any trained checkpoint responds correctly to interactive actions.

## Decision

Retain this record as the rejected design. Execute `CR-DYN-0008` instead.
