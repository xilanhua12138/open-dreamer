# CR-DYN-0009 quality postmortem

## Bottom line

`final_only-medium` is only the descriptive winner of a poor fixed-20k grid.
Its 17.073096 dB mean-frame PSNR and 0.713450 mean SSIM do not establish a
usable world model. Direct browser review rejected both temporal coherence and
action response, so this checkpoint must not be described as usable.

The 17.073096 dB figure is cumulative through a 16-frame future. The same
checkpoint falls from 22.594461 dB at horizon 1 to 20.605725 at horizon 3,
18.681995 at horizon 8 and 17.073096 at horizon 16. On normalized pixels,
17.073096 dB corresponds to approximately 0.0196 MSE and 0.140 RMSE.

## What the upstream OpenDreamer material changes

The public OpenDreamer CoinRun write-up says its first random-action dynamics
model was "quite terrible", and that useful trajectories required a simple RL
agent. It also warns that lower flow-matching MSE can coexist with worse
generation, recommends EMA, and reports that optimal-transport pairing plus
v-space loss improved rollout stability.

This experiment already used PPO trajectories, EMA, barycentric OT and v-space
loss. Those ingredients therefore do not explain the remaining collapse by
themselves. The more important gaps are below.

The current public `configs/dynamics.yaml` is a general/Minecraft-oriented
reference, not a disclosed CoinRun recipe. It nevertheless shows the intended
training regime: 200,000 updates, `k_max=256`, bootstrap starting at 100,000
updates, true 256-frame long sequences mixed with packed 64-frame chunks, and a
much larger default transformer. CR-DYN-0008/0009 compressed this to 20,000
updates, `k_max=8`, 64-frame-only records and much smaller models. It was a
pipeline/scaling pilot, not a quality-faithful reproduction.

Sources:

- https://next-state.github.io/open-dreamer/
- https://github.com/next-state/open-dreamer
- upstream `configs/dynamics.yaml`
- upstream `dreamer/data/generate_coinrun_dataset.py`

## Concrete protocol defects

### Reward-biased sampling was inert

The PPO collector wrote records of exactly 64 frames, while dynamics training
requested sequences of exactly 64 frames. In `ProcessRawVideoAndSlice`,
`max_start_idx = episode_len - seq_len`, therefore every record had
`max_start_idx=0`. Setting `p_include_reward=0.5` could not move the slice
toward a reward because there was only one possible slice.

The upstream CoinRun generator defaults to 160-frame chunks. Sampling 64-frame
windows from longer, episode-safe records gives reward bias room to operate and
retains more temporal context.

### Shortcut training was compressed too aggressively

The experiment trained only 20,000 updates with `k_max=8` and began bootstrap
training at 10,000 updates. The public reference uses 200,000 updates,
`k_max=256` and begins bootstrap at 100,000. The demo then used a four-step
shortcut. This makes shortcut quality a likely failure mode even if the
underlying flow objective improved.

### Selection did not test whether actions were used

The held-out selector ranked checkpoints only by aligned PSNR, SSIM and
horizon-16 PSNR. A model can score well on predictable CoinRun motion while
mostly ignoring actions. Direct source tracing and existing alignment tests did
not reveal an observation/action off-by-one error, but the experiment never
measured an aligned-action advantage.

The required diagnostic is fixed context and fixed sampler noise under:

1. aligned actions;
2. actions shuffled across the batch;
3. actions shifted one step;
4. all-no-op actions.

A usable action-conditioned model must beat the corrupted-action controls and
show stable counterfactual divergence for left/right/jump branches.

### The held-out set was too small for a usability claim

The terminal selector used 32 videos and pixel metrics. That is enough to
order this small grid descriptively, but not to establish interactive
controllability or long-horizon coherence. Future evaluation must add more
held-out clips, fixed visual panels/videos, action-use controls and FVD or an
equivalent perceptual temporal metric.

## Repair order

1. Run the no-training action-corruption and sampler/checkpoint diagnostics on
   existing medium and large checkpoints. This separates action ignorance,
   shortcut failure and under-training without spending another training run.
2. Recollect episode-safe 160- or 256-frame PPO records. Preserve exact
   observation/action alignment and verify that reward-biased 64-frame windows
   actually select multiple start positions.
3. Train a medium reference arm with the public `k_max=256` schedule and a
   materially longer budget. Evaluate aligned versus corrupted actions and
   fixed visuals at regular milestones.
4. Only after that arm passes action-use and visual review should another
   dynamics scale sweep or live-demo claim be made.
