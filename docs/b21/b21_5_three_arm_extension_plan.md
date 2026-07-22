# B21.5 matched LF050 extension and three-arm development gate

Status: runner and analyzer ready; GPU extension not yet executed.

## Question

The frozen HIO-at-step-200 replacement policy failed, but a clean-free exact-loss replay of the same HIO candidate rescued 5/40 base failures with zero good25 harms. This experiment asks whether those HIO rescues remain incremental after adding the already frozen LF050 candidate.

## Frozen development cases

Reuse the completed 40-case HIO panel without rerunning base or HIO:

```text
images: 00046 00171 00224 00746 00971
cases per image: 8
base seeds: 7400--7439
measurement seed: 5001
```

Generate only one missing matched candidate per case:

```text
LF050
ann400, diff5
B20_LF_ALPHA=0.50
B20_LF_FRAC=0.35
B20_LF_RADIUS_FRAC=0.12
same initial-noise/base seed as base_full
```

## Frozen clean-free three-arm policy

Use `theta = 0.7`, inherited from the frozen B21.4 margin gate. No threshold tuning is permitted on this panel.

1. Start from base.
2. Select LF iff `loss_lf < loss_base - 0.7`.
3. Select HIO iff `loss_hio < loss_current - 0.7`.

All decisions use exact operator loss only. PSNR is an offline diagnostic.

## Development support gate

Promote the sequential three-arm policy to a fresh image/seed validation only when all hold:

- all 40 matched LF candidates complete;
- HIO adds at least 2 net good25 cases beyond the frozen base+LF gate;
- HIO causes at most 1 good25 harm beyond base+LF;
- HELDOUT4 HIO net beyond base+LF is nonnegative;
- marginal HIO cost remains at most 0.70 of one full base run.

A failure retires HIO as a portfolio arm and retains the base+LF policy. A pass freezes the full policy and moves to fresh images/seeds without changing HIO, LF, injection-step, or margin parameters.
