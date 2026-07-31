# CR-PPO-0001 — Real-CoinRun PPO trajectory collector

## Question

Can a fixed-budget PPO policy trained in real CoinRun produce auditable, episode-safe trajectories that are materially more goal-directed than its untrained initialization for later action-conditioned world-model training?

## Why now

The first dynamics dataset used non-learned action policies, and the resulting interactive world model was rejected because its response to actions was wrong. OpenDreamer describes the same qualitative failure with random CoinRun actions and says a simple RL agent was then used to collect more useful trajectories, but that collector was not released.

This experiment fills only that missing data-collection stage.

## Frozen scope

```text
real CoinRun
  -> PPO actor-critic
  -> freeze final checkpoint
  -> collect disjoint train/eval trajectories
  -> audit action/reward/terminal alignment
  -> later train and evaluate dynamics under a separate experiment
  -> stop
```

There is no behavior-cloning policy and no policy optimization inside the learned world model.

## Controlled design

- Changed variable: PPO training transitions, from the zero-update network through 25,165,824 transitions.
- Fixed training distribution: CoinRun easy, levels `[0, 500)`, seed 0.
- Fixed held-out evaluation: levels `[10000, 10500)`, seed 4242, 64 episodes, deterministic argmax.
- Fixed model: IMPALA-style CNN with 15-way categorical policy and scalar value heads.
- Fixed optimizer recipe: clipped PPO, GAE, normalized/clipped reward, 64 envs × 256 rollout steps, 3 epochs and 8 minibatches.
- Fixed validation cadence: zero-update baseline and every 1,048,576 transitions, each with structured metrics, four full-episode-span GIFs uniformly capped at 256 stored frames each, and a contact sheet.
- Frozen data consumer: action-conditioned dynamics only.

The final stochastic collector uses temperature 1.0 and epsilon 0.05. Every record stores `observation_t`, `action_t`, the reward observed after `action_t`, and terminal flags. Partial chunks are discarded at auto-reset boundaries so a record cannot silently cross episodes.

The 4,096-record dynamics training split is collected from the same 500-level
range on which PPO learns; the 512-record dynamics evaluation split uses the
same disjoint 500-level range used for PPO held-out evaluation. This makes
collector quality measurable on both familiar and unseen levels without
claiming that an unseen-level failure is a dynamics-model failure.

Before seeing any result, the quality targets are fixed at final deterministic
held-out success rate ≥ 50%, an absolute improvement of at least 25 percentage
points over the zero-update network, higher held-out mean return, and at least
30% successful completed episodes in each stochastic collection split. Missing
these targets is a negative result, not a trigger to silently extend the budget.

## Planned evidence

- Stable runtime identity, structured run state, metrics, telemetry and artifact hashes.
- One Python 3.10 runtime with pinned Procgen 0.10.7 and CUDA-enabled JAX 0.4.35; the bootstrap records package versions and runs a CPU-only environment/model smoke before any GPU launch.
- W&B mirror when authenticated, otherwise an honestly labelled offline run.
- Final checkpoint SHA256 and exact completed environment steps.
- Train/eval dataset metadata, action histograms, returns, success rates, shard hashes and tree hashes.
- Full audit of action range, checkpoint provenance, episode boundaries and split disjointness.

## Results

### Observed

The exact Python 3.10 runtime passed a real Procgen action-space/model smoke on CPU and an actor forward smoke on the A10. The full local branch suite has 97 passing tests.

Two compatibility failures occurred before PPO update zero and remain retained: an unconstrained CUDA 12.9 nvcc namespace package was incompatible with JAX 0.4.35, and the shared logger assumed a newer `jax.distributed.is_initialized()` API. Both received failing regression tests before the fixes.

The fixed-budget run completed from clean execution source `ec97611`, run ID `ebf9f582-e3d4-4b5c-9ddb-a78e3244eb14`, attempt ID `8303d788-8966-4850-86a2-d87bc9a47366`, with W&B offline.

The preregistered zero-update held-out baseline had mean return `0.0` and success rate `0.0`. The best deterministic milestone occurred at `22,020,096` transitions with mean return `5.15625` and success rate `51.5625%`; the final checkpoint regressed to mean return `2.96875` and success rate `29.6875%`, so the final 50% threshold was rejected.

The frozen checkpoint SHA256 is `88f3cff15cd46518c14b9add81a913d8a9afcfed1143b234ecabdf8b0b29cbb6`. It produced exactly 4,096 train and 512 eval records. Completed-episode success was `55.6904%` on train collection and `65.7534%` on eval collection; both split audits and the pair audit passed.

A separate post-hoc diagnostic used larger stochastic samples. On 512 episodes over the full Procgen distribution, the best checkpoint reached mean return `5.390625` and the final checkpoint reached `4.9609375`. Thus deterministic 64-episode evaluation did understate policy performance, but measurement alone does not explain the gap to the public easy-200 curve.

### Interpretation

The run is a valid negative policy-quality result under its preregistered final criterion and a valid positive systems/data-audit result.

### Not established

- That PPO-collected trajectories improve dynamics.
- That this reproduces an unpublished OpenDreamer collector recipe.
- Which public-reference recipe mismatch is causal.

### Decision

Keep the audited data and checkpoint, but do not start dynamics from them yet. Run the separately preregistered `CR-PPO-0002` official-recipe parity control from scratch; its scope still excludes trajectory collection, dynamics and imitation learning.
