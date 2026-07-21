# B21.10 hard-case triage plan

Status: zero-GPU analyzer ready; no new GPU experiment authorized yet.

## Motivation

B21.9 rejected Fresh3 as the default fixed budget:

```text
Fresh2 selected good25: 74/80
Fresh3 selected good25: 75/80
Fresh3 incremental rescues: 1
Fresh3 incremental harms:   0
positive images:             1/20
```

Image `66731` remained `0/4` after all three independent restarts and accounts for four of the five remaining Fresh3 failures. This suggests a persistent measurement-level hard case rather than insufficient ordinary restart count.

## Question

Before launching another candidate generator, determine whether the completed Fresh2 outputs contain a useful clean-free signal for identifying cases or images that need a specialized fallback.

The diagnostic uses:

- Fresh2 selected exact operator loss;
- the loss spread between the first two trajectories;
- whether trajectory 2 was accepted;
- image-level aggregates of these clean-free quantities.

PSNR is used only to label historical failures and rescues offline.

## Outputs

The analyzer reports:

- all Fresh2 and Fresh3 failures;
- persistent failures after all three trajectories;
- case-level loss ranking and threshold-capture curves;
- image-level mean/max loss ranks;
- AUC of Fresh2 selected loss for failure and persistent failure;
- whether persistent hard images are among the highest clean-free-loss images.

## Interpretation

This is not a threshold-selection experiment and does not freeze an adaptive policy.

- If persistent hard images rank near the top under clean-free loss, the next GPU experiment may be a small targeted complementary-candidate pilot on those images.
- If they do not, first develop a better clean-free hard-case detector; do not spend a third full-run budget indiscriminately.
- In either case, do not resume blind Fresh4/Fresh5 scaling.
