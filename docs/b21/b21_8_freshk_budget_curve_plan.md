# B21.8 frozen independent-restart budget curve

Status: runner and analyzer ready; Fresh3/Fresh4 extension not yet executed.

## Motivation

B21.7 established that an independent second DAPS trajectory is better than spending the same full-trajectory budget on LF050:

```text
Fresh2 selected good25:  70/80
Base+LF selected good25: 63/80
Fresh2-only / LF-only:   10 / 3
one-sided exact p:       0.046142578125
```

The matched timing audit found LF/base wall ratios tightly centered at one:

```text
mean:   0.997360
median: 0.998649
range:  [0.976526, 1.010050]
```

Fresh2 is therefore the adopted second-arm allocation. The next question is how many independent full trajectories are needed before validating a fixed restart budget on a new panel.

## Frozen development panel

Reuse the completed ten-image official FFHQ validation-split panel with eight seed cases per image:

```text
images: 68263 63803 66452 63282 66892
        69293 68924 65808 62802 65960
cases:  80
measurement tag: 5101
```

Existing trajectories:

```text
Fresh1: base_full, seeds 10000--10079
Fresh2: base_extra, seeds 13000--13079
```

Generate:

```text
Fresh3 arm: base_extra2, seeds 15000--15079
Fresh4 arm: base_extra3, seeds 16000--16079
```

All four arms use independent full DAPS trajectories with `ann400`, `diff5`, LF disabled, and the same locked measurement within a case.

## Frozen clean-free selector

Start with `base_full`. Add candidates in the declared order. Accept the newly added candidate iff:

```text
loss_new < loss_current - 0.7
```

Selection uses exact operator loss only. PSNR is offline evaluation.

## Outputs

Report cumulative Fresh1--Fresh4 curves for:

- selected good25 and bad25;
- oracle any-good25 and selected-oracle gap;
- incremental selected rescues and harms;
- accepted new candidates;
- minimum selected good25 across the ten image groups;
- historical wall time as a diagnostic, while interpreting K primarily as full-trajectory work equivalents.

## Frozen development target

Choose the smallest `K in {2,3,4}` satisfying all:

```text
selected good25 >= 76/80
selected-oracle gap <= 1
cumulative selected harms <= 1
every image selected good25 >= 6/8
```

This is a development-stage 95% panel-reliability target, not a final population claim.

## Promotion

- If a K qualifies, freeze the smallest qualifying K and validate it unchanged on a disjoint official FFHQ validation panel with new measurements and new seeds.
- If no K qualifies, do not scale independent restarts further by default; return to hard-case candidate generation or an adaptive budget policy.

No threshold, schedule, seed, or target modification is permitted after observing the Fresh3/Fresh4 results.
