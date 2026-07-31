# Experiment method

## Purpose

The ledger is designed to turn a sequence of GPU jobs into a chain of scientific decisions. The unit of work is not a training process; it is a falsifiable question plus a controlled comparison.

The style is inspired by the public experimental structure of:

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027), where a mechanism hypothesis is tested through controlled architectural ablations.
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), where simple design choices are isolated and tested under shared backbones and protocols.

This repository does not claim knowledge of any private lab process. It adopts the observable strengths of those papers: simple hypotheses, strong controls, compact tables, honest limitations, and scale only after the mechanism is understood.

## Decision chain

Each experiment should fit this chain:

```text
previous evidence
  -> unresolved question
  -> hypothesis and falsifier
  -> controlled experiment
  -> raw observation
  -> interpretation
  -> bounded claim
  -> next experiment
```

If a run cannot name the previous evidence or the next decision it informs, it is probably an unstructured exploration. Explorations are allowed, but must be labeled `exploratory` and cannot be presented as confirmatory evidence.

## Baseline-first rule

Before adding capacity or complexity:

1. Demonstrate that the minimal end-to-end pipeline works.
2. Establish a stable held-out metric and qualitative artifact.
3. Verify the evaluation target and seed.
4. Change one major variable.
5. Scale only if the smaller experiment distinguishes competing explanations.

Example from the CoinRun experiments:

- The fixed-FLOPs pilot changed model size while compute determined drastically different step counts.
- The larger model received only 474 steps and never reached shortcut training.
- Therefore it could not answer whether capacity helped at adequate optimization.
- The next experiment fixed all models at 20,000 steps, which isolated the capacity trend under a common curriculum.

The failed interpretation is retained because it explains why the corrected experiment exists.

## What counts as controlled

A comparison table must explicitly list:

- changed variable
- fixed variables
- known confounders
- baseline
- evaluation population
- random seed policy
- metric target

If multiple settings differ, describe it as a recipe comparison rather than an ablation.

## Replication vocabulary

Use exact language:

- `smoke_test`: imports, compilation, or one short run works.
- `pipeline_closure`: every intended stage produces an artifact.
- `internal_result`: the local protocol answers a local question.
- `partial_reproduction`: some published scale/config choices match, but material differences remain.
- `strict_reproduction`: public data, recipe, scale, budget, and evaluation protocol match.

When official artifacts or full recipes are absent, strict reproduction may be impossible. State the missing evidence rather than filling gaps with assumptions.
