# B21.9 frozen Fresh3 disjoint validation

Status: runner and analyzer ready; fresh validation not yet executed.

## Frozen method

B21.8 selected the smallest development-qualified restart budget:

```text
K = 3 independent full DAPS trajectories
ann400, diff5
LF disabled
HIO disabled
```

Candidates are added in declared seed order. Starting with trajectory 1, accept a new trajectory iff:

```text
loss_new < loss_current - 0.7
```

Selection uses exact operator loss only. PSNR is offline evaluation.

## Disjoint panel

Use 20 previously untouched IDs from the official FFHQ validation range `60000--69999`:

```text
63678 68922 67092 64050 69441
67673 66511 64116 63199 63135
65317 68111 65656 64471 60067
63319 64542 66731 63368 62957
```

The list is the first 20 SHA-256-ranked unused IDs under salt `B21.9-fresh3-validation-v1`, excluding the ten B21.5/B21.8 images.

Use four cases per image, for 80 total cases.

Fresh locked configuration:

```text
measurement panel seed/tag: 5201
trajectory-1 seeds:         17000--17079
trajectory-2 seeds:         18000--18079
trajectory-3 seeds:         19000--19079
```

Measurements are generated once with the local DAPS preprocessing and phase-retrieval operator, then saved with hashes before any solver output.

## Frozen validation gates

Fresh3 passes only if all hold:

```text
80/80 cases complete with finite metrics
Fresh3 selected good25 >= 74/80
Fresh3 selected-oracle gap <= 2
Fresh3 selected gain beyond Fresh2 >= +4/80
Fresh3 incremental harms <= 1
cumulative selector harms through Fresh3 <= 1
at least 4/20 images have positive Fresh3-vs-Fresh2 net
at most 1/20 images has negative Fresh3-vs-Fresh2 net
every image has Fresh3 selected good25 >= 2/4
```

The `74/80` validation requirement is slightly below the `76/80` development target to avoid requiring an exact replication of the development point estimate. The incremental and image-spread gates separately test whether the third restart contributes beyond Fresh2.

## Decision

- Pass: adopt Fresh3 as the fixed default reliability budget and move to a larger benchmark or an adaptive early-stop design without retuning.
- Fail: retain Fresh2 as the default fixed budget and do not scale blind independent restarts further without a new candidate-generation or adaptive-budget idea.
