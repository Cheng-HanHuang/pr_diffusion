# B21.7 Fresh2 versus Base+LF decision

The 80-case official FFHQ validation-split allocation experiment completed with no failures.

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

Under the frozen quality rule, Fresh2 wins and LF050 should not be the default second arm.

## Runtime caveat

The two second arms have the same declared full-trajectory work (`ann400`, `diff5`), but they were executed in different jobs. Their historical wall times are not comparable:

```text
historical LF wall / historical base wall:      1.025
later base_extra wall / historical base wall:   2.263
```

Thus the quality comparison is valid at fixed trajectory/work budget, but the label `equal wall-clock cost` is not yet supported. The `base_extra` slowdown is likely a run-environment or contention effect, but that must be checked rather than assumed.

## Required next step

Run a small interleaved timing calibration in one job:

- eight frozen cases;
- same seed for fresh-base and LF within each pair;
- both candidates launched under the same current environment;
- alternating base-first and LF-first order;
- four GPUs;
- no new quality claim.

If paired LF/base wall ratios are approximately one, retain the fixed-work conclusion that independent restarts are the preferred second arm and treat the historical `2.263x` number as cross-run timing contamination. Otherwise, investigate the runtime difference before designing the next budget policy.
