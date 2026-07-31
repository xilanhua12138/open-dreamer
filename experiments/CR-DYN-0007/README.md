# CR-DYN-0007 — Final-policy dynamics scale sweep

> Superseded before execution. No GPU arm ran under this ID. The corrected
> `CR-DYN-0009` scale sweep uses the mixture chosen by the fully executed
> `CR-DYN-0008` checkpoint-mixture ablation.

## Question

With the 16.6M EMA tokenizer, corrected 15-action/no-op-4 contract, final CR-PPO-0002 trajectories, equal 20,000-update budgets and identical held-out futures, how does dynamics quality change across 0.16M, 0.55M, 3.93M and 12.90M parameters?

## Design

The sweep reuses the exact final-policy medium arm from CR-DYN-0006 and trains the remaining tiny, small and large arms from scratch. This avoids an unnecessary duplicate GPU run while keeping every scientific constant identical.

Evaluation uses the same fixed final-policy held-out corpus, 32 futures, 16 context frames, 16 predicted frames and seed 4242. The primary metrics compare predictions with tokenizer-decoded ground truth, isolating dynamics error from tokenizer reconstruction error.

## Observed

No scale result was observed. The experiment was aborted during protocol
review.

## Interpretation

The capacity variable remained valid, but hard-coding final-policy-only data
would bypass the newly requested mixture comparison.

## Not established

- Whether the corrected scale curve is monotonic.
- Whether large exceeds medium.
- Whether any scale is good enough for interactive control.

## Decision

Retain this record as the rejected design. Execute `CR-DYN-0008` and then
`CR-DYN-0009`.
