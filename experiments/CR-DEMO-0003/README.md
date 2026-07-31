# CR-DEMO-0003 — continuous CoinRun world-model browser runtime

## Question

Can the selected CoinRun checkpoint run as a continuously advancing world whose
current action changes when the user holds or releases a control?

## Why this experiment

CR-DEMO-0002 used a blocking request for every generated frame. Each request
also rebuilt and refilled the dynamics context, so the browser froze without
input and release events could be dropped while a request was pending. Its
square image also pushed the controls below the initial viewport. Those are
runtime defects, separate from the already-rejected checkpoint quality.

## Hypothesis and falsifier

- Hypothesis: one context prefill, persistent dynamics and decoder caches, a
  continuous server loop and latest held-input state produce a correct
  interactive runtime.
- Falsified if: no-op frames do not advance autonomously, held input does not
  persist across generated frames, release does not restore no-op, the stream
  fails, or the desktop interface does not fit at 1440x900.

## Controlled design

- Baseline: CR-DEMO-0002.
- Changed: runtime cache lifetime, transport, input protocol and responsive
  browser layout.
- Held constant: `final_only-medium`, 16.6M EMA tokenizer, 16 context frames,
  four denoise steps, seed, held-out dataset and categorical action mapping.
- Known confounders: this runtime repair cannot improve the learned visual
  dynamics or prove that the checkpoint uses actions well.

## Results

### Observed

Remote execution has not started.

### Interpretation

None yet.

### Not established

- Visual coherence.
- Learned action responsiveness.

### Decision

Run all preregistered remote and local-tunnel checks, then ask for direct visual
review without claiming the model itself is fixed.
