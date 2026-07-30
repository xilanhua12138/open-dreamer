# CR-PPO-0004 — Historical re-evaluation and all-old combined control

## Question

Does the exact combination of old training settings reproduce collapse under
the aligned evaluator, or was the historical gap caused by evaluation identity
or another historical source/runtime difference?

## Why this experiment

The single-factor screen cannot end with the phrase "probably interactions."
It must first demonstrate that the combined settings actually reproduce the
historical failure.

## Controlled design

- Re-evaluate the retained historical 6.29M checkpoint on 256 stochastic
  full-distribution episodes.
- Train a fresh 6.29M arm with all four old settings combined.
- Compare both with the CR-PPO-0003 reference under the identical evaluator.
- Keep trajectory collection and dynamics disabled.

## Results

### Observed

Aborted before execution at the user's request. No CR-PPO-0004 process,
checkpoint re-evaluation or fresh training arm was started.

### Interpretation

`CR-PPO-0003` already showed that orthogonal-sqrt2 initialization alone crossed
the collapse threshold, so a combined old-settings arm had lower information
value than direct mechanism and mitigation ablations.

### Not established

- That the settings interact.
- That the historical checkpoint is intrinsically weak under the aligned metric.
- That any performance difference is a software bug.

### Decision

Preserve the unrun preregistration and move compute to an initializer mechanism
and mitigation study.
