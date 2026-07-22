# B21.10 hard-case detector screen decision

## Scope

This decision closes the retrospective clean-free detector screen performed after the B21.9 Fresh3 validation. The adopted solver budget remains two independent full DAPS trajectories selected by the frozen exact-loss margin rule.

## Detector results

### Raw selected exact loss

- development AUC: `0.9486`
- validation AUC: `1.0000`
- validation recall: `1.0000`
- validation precision: `0.1304`
- validation flagged fraction: `0.5750`

Raw loss ranked the B21.9 failures perfectly, but the absolute development threshold transferred with too many false positives.

### Measurement-norm-normalized residual

- development AUC: `0.7329`
- validation AUC: `0.4595`
- validation recall: `0.3333`
- validation precision: `0.0714`
- validation flagged fraction: `0.3500`

The normalization suppressed all four persistent `66731` failures and is rejected.

### Within-measurement two-run features

The development-selected feature was candidate disagreement RMSE after minimizing over identity and 180-degree rotation.

- development AUC: `0.7843`
- validation AUC: `0.8874`
- validation recall: `0.8333`
- validation precision: `0.2273`
- validation flagged fraction: `0.2750`
- validation failures captured / flagged: `5 / 22`

No screened relative feature met the frozen support rule of recall at least `0.80`, precision at least `0.50`, and flagged fraction at most `0.25`.

## Interpretation

Candidate disagreement is a meaningful retrospective diagnostic, but not a deployable trigger. Weakening the support rule after seeing B21.9 would be post-hoc threshold tuning and would still leave a low-precision fallback policy.

The completed screens also show that detector quality is limited by measurement-level heterogeneity: raw residual scale transfers poorly, measurement-norm normalization removes useful failure information, and within-measurement disagreement remains insufficiently selective.

## Decision

- Retain **Fresh2** as the fixed default budget.
- Reject raw-loss, normalized-residual, and screened relative-feature thresholds as runtime fallback triggers.
- Do not launch LF, HIO, Fresh3, or another complementary arm conditionally on these detectors.
- Stop threshold mining on B21.8/B21.9.
- Preserve candidate disagreement as an offline analysis feature only.
- Proceed to a prospective Fresh2 benchmark on a larger disjoint official-validation image panel.

No adaptive policy is validated by B21.10.
