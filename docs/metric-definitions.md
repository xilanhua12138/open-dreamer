# Metric definitions

## PSNR

For normalized images in `[0, 1]`:

```text
PSNR = -10 * log10(MSE)
```

Every PSNR record must state:

- aggregation grain: frame, video, or batch
- target: original RGB or tokenizer-decoded ground truth
- context frames excluded from or included in scoring
- online or EMA weights
- masked or clean tokenizer input

`mean_frame_psnr_db` averages per-frame PSNR values. `mean_video_psnr_db` computes one score per video from its aggregate error and then averages videos. These values need not be equal.

## SSIM

SSIM must state the image range, frame aggregation, evaluation target, and whether context frames are excluded. This repository reports the mean over predicted frames unless a record explicitly says otherwise.

## Horizon metrics

`PSNR@k` and `SSIM@k` mean the aggregate score over the first `k` predicted frames, not only the single frame at index `k`.

## Tokenizer metrics

- `clean`: original unmasked frame passed to the tokenizer.
- `masked`: the training-style masked input is reconstructed.
- `online`: current training weights.
- `ema`: exponential-moving-average weights.
- `training_psnr`: last observed training batch; never substitute for held-out PSNR.

## Dynamics-only metrics

When the target is `tokenizer_decoded_ground_truth`, both the prediction and target pass through the same decoder. This isolates dynamics error but does not measure full pixel fidelity to the original environment.

When the target is `original_rgb`, tokenizer and dynamics errors are combined. Do not compare this number directly against a dynamics-only score.
