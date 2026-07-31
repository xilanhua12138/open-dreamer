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

The new server loaded `final_only-medium` and reported
`continuous-sse-v1`. Remote and local-tunnel verification each observed three
strictly increasing no-op frames without per-frame inference calls. One
held-right update produced at least two later action-7 frames; a newer empty
held state restored action 4. Pause and queued reset both completed without a
cache or dtype failure.

The first remote call paid a one-time 7.1-second JIT compilation cost. After
that, generated frames took 21.4–32.1 ms in the first run and 21.6–25.3 ms in
the saved independent remote run. The local tunnel observed 21.6–25.4 ms.

At 1440x900, the complete 720x720 stage and sidebar fit in the initial
viewport. The document had no horizontal or vertical scroll; the live page
reported 46.28 FPS.

### Interpretation

The runtime now has the intended semantics: the model world advances while a
viewer is subscribed, and held input updates the action used by subsequent
generated frames. Persistent dynamics and decoder caches remove the old
per-request context refill. The responsive layout fixes the oversized frame
that pushed controls below the fold.

This is a runtime and UI result. It does not improve the learned checkpoint.

### Not established

- Visual or temporal coherence of `final_only-medium`.
- Whether the model learned an action effect strong enough to look or feel
  controllable.

### Decision

Keep this runtime and layout as the shell for later checkpoints. Preserve
CR-DEMO-0002's direct user rejection of the current checkpoint; do not describe
this smoke test as a model-quality repair.
