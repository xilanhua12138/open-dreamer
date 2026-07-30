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

Queued. The automatic launcher cannot run until CR-DYN-0008/0009 release the only A10.

### Interpretation

None before runtime and visual evidence exists.

### Not established

Health, sustained inference, latency, visual clarity and action response are not yet established.

### Decision

After the full dynamics chain completes, select the best completed scale by held-out mean-frame PSNR then SSIM, start the demo, verify one remote step and one subsequent local-tunnel step, and then ask the user for visual acceptance.
