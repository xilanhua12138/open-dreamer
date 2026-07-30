# CR-DYN-0007 — Final-policy dynamics scale sweep

## Question

With the 16.6M EMA tokenizer, corrected 15-action/no-op-4 contract, final CR-PPO-0002 trajectories, equal 20,000-update budgets and identical held-out futures, how does dynamics quality change across 0.16M, 0.55M, 3.93M and 12.90M parameters?

## Design

The sweep reuses the exact final-policy medium arm from CR-DYN-0006 and trains the remaining tiny, small and large arms from scratch. This avoids an unnecessary duplicate GPU run while keeping every scientific constant identical.

Evaluation uses the same fixed final-policy held-out corpus, 32 futures, 16 context frames, 16 predicted frames and seed 4242. The primary metrics compare predictions with tokenizer-decoded ground truth, isolating dynamics error from tokenizer reconstruction error.

## Observed

No new scale result had been observed at preregistration time.

## Interpretation

None yet.

## Not established

- Whether the corrected scale curve is monotonic.
- Whether large exceeds medium.
- Whether any scale is good enough for interactive control.

## Decision

Start only after CR-DYN-0006 has produced a valid corpus audit and completed the final-policy medium arm.
