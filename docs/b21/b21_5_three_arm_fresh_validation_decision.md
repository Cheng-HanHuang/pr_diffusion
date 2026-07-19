# B21.5 three-arm fresh-validation decision

Status: **HIO portfolio arm rejected after fresh validation**.

## Frozen validation result

The frozen sequential policy was evaluated on ten untouched official FFHQ validation-split images, eight paired cases per image, with new locked measurements and new solver seeds:

```text
base+LF gated good25: 63/80
three-arm gated good25: 67/80
incremental HIO rescues: 4
incremental HIO harms:   0
positive / zero / negative images: 2 / 8 / 0
case-level McNemar p: 0.125
selector capture of oracle incremental HIO: 1.0
```

The HIO gains were concentrated on two images:

```text
63282: +1/8
63803: +3/8
all other images: +0/8
```

The predeclared fresh-validation requirements were not met:

```text
incremental net >= +6/80:                  fail (+4)
positive images >= 5 and negative <= 2:    fail (2 positive)
marginal HIO wall cost <= 0.70 x base:      fail (0.731)
total three-arm wall cost <= 1.75 x base:   fail (2.756)
```

Completion, harm, and selector-capture gates passed. The quality benefit is real but too sparse and too expensive for a default portfolio arm.

## Interpretation

1. HIO-at-step-200 is not a reliable replacement candidate.
2. Exact-loss gating can identify the rare successful HIO rescue: all four oracle incremental HIO successes were selected, with zero good25 harms.
3. The rescue mechanism is image-concentrated rather than broadly general: only two of ten fresh images benefited.
4. The development-panel cost estimate was not portable. The matched fresh run shows LF050 costs approximately one additional full DAPS run (`1.025 x base`) and HIO plus half-DAPS costs `0.731 x base`; therefore the actual three-arm wall budget is about `2.756 x base`.

## Decision

- Mark `WARM_hio_aux_gate` **rejected** as a default portfolio arm.
- Do not tune HIO iterations, injection step, beta, margin, or image-specific triggers on this validation panel.
- Preserve HIO code, measurements, and candidate artifacts for failure forensics only.
- Retain the frozen base+LF policy as the current method candidate.
- Before adopting LF as the default second full-cost arm, run an equal-cost paired comparison against a second independent base restart on the same frozen 80-case panel. That comparison is a new preregistered allocation question and generates only the missing second-base candidates.

Primary artifacts:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_5_three_arm_fresh_val10x8_meas5101/fresh_analysis_theta0.7/fresh_three_arm_rows.csv
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_5_three_arm_fresh_val10x8_meas5101/fresh_analysis_theta0.7/fresh_three_arm_summary_by_image.csv
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_5_three_arm_fresh_val10x8_meas5101/fresh_analysis_theta0.7/fresh_three_arm_verdict.json
```
