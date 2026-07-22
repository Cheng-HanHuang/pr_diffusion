# B21.7 frozen equal-cost Fresh2 versus Base+LF allocation

Status: runner and analyzer ready; independent-restart extension not yet executed.

## Motivation

The fresh three-arm validation rejected HIO as a default portfolio arm. On the same 80 fresh cases:

```text
base:             56/80 good25
base+LF selected: 63/80 good25
three-arm:        67/80 good25
```

HIO is retired because its incremental gain was only +4/80, concentrated on two images, and the full portfolio cost 2.756x base. The retained question is whether LF050 is a better second arm than an independent full DAPS restart at approximately the same two-run budget.

## Frozen panel

Reuse the completed official FFHQ validation-split panel:

```text
images: 68263 63803 66452 63282 66892
        69293 68924 65808 62802 65960
cases per image: 8
total paired cases: 80
measurement tag: 5101
base seeds: 10000--10079
```

Do not regenerate measurements, base candidates, or LF050 candidates.

Generate one missing candidate per case:

```text
base_extra
independent full DAPS ann400, diff5
seeds 13000--13079
B20 LF disabled
```

## Equal-cost policies

### Fresh2

```text
base_full(seed 10000+j)
base_extra(seed 13000+j)
```

Select `base_extra` iff:

```text
loss_extra < loss_base - 0.7
```

### Base+LF

```text
base_full(seed 10000+j)
lf050(same initial seed as base)
```

Select `lf050` iff:

```text
loss_lf < loss_base - 0.7
```

Both selectors are clean-free and use exact operator loss only. PSNR is offline evaluation.

## Primary comparisons

Report for each policy:

- selected good25;
- oracle any-good25;
- selected mean PSNR;
- policy-only good25 wins;
- exact two-sided and directional McNemar tests;
- per-image selected-good net;
- actual marginal and total wall cost.

## Frozen decision rule

Let `delta = Base+LF selected good25 - Fresh2 selected good25`.

```text
Base+LF wins:
  delta >= +4/80
  OR one-sided exact McNemar p < 0.05 favoring Base+LF

Fresh2 wins:
  delta <= -4/80
  OR one-sided exact McNemar p < 0.05 favoring Fresh2

otherwise:
  inconclusive
```

A Base+LF win retains LF050 as the preferred second arm. A Fresh2 win retires LF050 as the default second arm. An inconclusive result keeps both as optional allocation choices without a dominance claim.
