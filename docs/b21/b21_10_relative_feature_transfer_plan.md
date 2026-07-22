# B21.10 within-measurement relative-feature transfer

Status: zero-GPU audit ready; no fallback GPU run authorized.

## Motivation

The raw Fresh2 exact loss ranks B21.9 failures perfectly but its absolute development threshold flags 57.5% of validation rows. Dividing by the measurement norm performs worse: validation AUC 0.4595, recall 0.3333, precision 0.0714, and it misses all four persistent `66731` failures.

The next detector screen therefore uses only quantities comparing the two Fresh2 trajectories from the same locked measurement. These quantities do not rely on an absolute residual scale transferring across images.

## Frozen features

For the first two independent trajectories, compute:

- absolute exact-loss gap;
- exact-loss relative gap;
- max/min exact-loss ratio;
- output RMSE disagreement after taking the minimum over identity and 180-degree rotation;
- output mean-absolute disagreement under the same ambiguity handling.

Raw selected exact loss is included only as a reference baseline.

## Transfer protocol

Use B21.8 as development and B21.9 as validation. For every continuous feature independently:

```text
threshold = minimum feature value among development Fresh2 failures
flag iff feature >= threshold
```

This retains 100% development-failure recall by construction. Apply each threshold unchanged to B21.9.

Among relative features only, identify the development-best feature by maximum development specificity at 100% development recall, breaking ties by development AUC. This selection is diagnostic and does not freeze a runtime policy.

## Support rule

A relative feature is strong enough to justify a targeted fallback development pilot only if its unchanged B21.9 result has:

```text
recall >= 0.80
precision >= 0.50
flagged fraction <= 0.25
```

This remains retrospective because B21.9 outcomes have already been viewed. Even a strong result would require a new disjoint panel to validate both detector and fallback.

## Decision

- Strong relative feature: permit a small targeted complementary-candidate development pilot, retaining ordinary Fresh3 as a matched control.
- No strong relative feature: stop threshold mining on these panels, retain Fresh2 as the fixed default, and do not launch a fallback GPU pilot without a genuinely new clean-free detector idea.
