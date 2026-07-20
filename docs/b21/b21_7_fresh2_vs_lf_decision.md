# B21.7 Fresh2 versus Base+LF decision

The 80-case official FFHQ validation-split allocation experiment and the matched runtime calibration both completed without failures.

## Quality result

Using the frozen exact-loss margin selector (`theta=0.7`):

```text
Fresh2 selected good25:   70/80
Base+LF selected good25:  63/80
Fresh2-only wins:          10
Base+LF-only wins:          3
one-sided exact p favoring Fresh2: 0.046142578125
```

The same `+7` advantage appears under oracle any-good25 (`70/80` versus `63/80`), so the result is a candidate-generation difference rather than a selector artifact. Five images favor Fresh2, one favors Base+LF, and four tie.

Under the frozen quality rule, Fresh2 wins and LF050 is retired as the default second arm.

## Matched runtime result

Eight interleaved pairs reran a fresh base trajectory and LF050 under the same current environment, with the same seed within a pair and alternating execution order:

```text
paired mean LF/base:   0.997360
paired median LF/base: 0.998649
paired range:          [0.976526, 1.010050]
base-first mean:       0.994814
LF-first mean:         0.999905
```

The two arms are therefore equivalent in wall time to within about one percent. Execution order did not produce a meaningful effect.

Historical cross-job timing ratios are not used for allocation claims. In particular, the earlier aggregate `base_extra/base = 2.263` field conflicts with both the matched audit and the sampled historical per-case timings, and is treated as cross-run/accounting contamination rather than algorithmic cost.

## Final decision

```text
preferred second arm: independent fresh DAPS restart
retired default arm:  LF050
selector:             frozen exact-loss margin theta=0.7
cost interpretation: Fresh2 and Base+LF are both approximately two full-run equivalents
```

LF050 remains preserved as a diagnostic or optional diversity arm, but there is no evidence to spend the standard second-run budget on it instead of an independent restart.

## Next stage

Build the independent-restart budget curve by adding third and fourth fresh trajectories on the same development panel. Select the smallest restart count that reaches the preregistered reliability target, then validate that count unchanged on a new disjoint official-validation panel.
