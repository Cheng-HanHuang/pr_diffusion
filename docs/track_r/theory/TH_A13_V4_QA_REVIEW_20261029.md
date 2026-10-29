# A13 v4 freeze review: deterministic pilot conversion and tail caching

**2026-10-29. Reviewer conclusion: NO OBJECTION** to the disclosed production adjustments, based on the authorized inputs, source inspection and the planner's reported 6,400-case identity test. This is an **audit of packaging/performance**, not a fresh model fit, not another production execution, and not an examination of any held XP-D4 16–63 outcomes.

## Input integrity

The user-uploaded theory_upload_20261029_EVE.zip contained five archive entries: the guidance, deterministic converter, reformatted 16-run pilot CSV, paper snapshot and SHA256SUMS. The **four files listed** in SHA256SUMS were verified against their exact digest *before* reading the guidance. SHA-256 of the reformatted CSV: 12dd28af9406af9f6489aacda4b837385c78a1011627449da1c79da71fa0c484; converter script: 84cbe88075bd62f647badc13e31f4cf76c904bec98e915b4f1f9225bc7a6b0af. The converter itself pins its A9 source CSV SHA 48169156486de4438e715d97a5a3251a728f52e055f135caf14fbcb02fd33e27.

Independent A8-compatible-format verification on authorized uploaded bytes:
- Exactly **100** ImageNet IDs match the previously frozen four-run P1 calibration input (SHA c2177880...), with no missing/extra ID.
- All **100/100 original four-run labels and within-prefix rank permutations** are preserved. First-four physical signed residual values agree to 1e-7; the converter's global 16-run ordering is the stable argsort of 16 recorded signed residuals for all 100 images. The old first-four rank is exactly the restriction of the new order to indices 0–3 for every image.
- Every row has 16 labels and the exact stated category counts; aggregate 1,600 candidates are G378, global O279, partial O14, C750, I179. The converter changes encoding O_g -> O, O_p -> P and schema names, not class counts, known residuals or run identities.
- Consistency of the source A9 counts with converted A8 counts is asserted in converter source; this review does not rerun the original source-to-output conversion without that raw source CSV.

## Cache equivalence argument and limits

At theory 629b1fc, the sole model-facing edit in fit_th1_imagenet_v4.py imports v4_tail_cache (module import installs a replacement for fit_th1_prefix_v1.conditional_old_scores). The cache key is (ranked prefix labels, exact double-precision score weights, tail eps). For the tail arm, it computes the SAME ordered itertools.product statuses, SAME prior powers and SAME weighted_permutation_prob probabilities and normalization as the original; later draws use SAME rng.choice over that cached vector and SAME exponential spacing operations. Reconstructing status bits from the selected integer preserves the product's left-to-right enumeration. All non-tail arms retain the original score law. The cache is pure deterministic memoization and cannot introduce new data dependence or consume future labels. Its memory is bounded across fits by clearing at key changes.

The planner's check_v4_tail_cache_identity.py compares old/new category sequences, every numerical score and the entire generator states for 6,400 calls (4 fits × 4 variants × 2 seeds × 2 passes × 100 images), reporting **bitwise identity**; production was 23x faster on that step. I inspected the source and see **no model-level objection**. This review is **not** an independent rerun of the planner's 6,400-case test; Python/runtime behavior is as stated by the planner and its test receipt.

## Registered immutability

The planner reports:
- v4 code run at theory 629b1fc (9 min 43 s);
- v4 freeze theory 149331c, SHA-256 b0ff25bc98ef7ba294db3f33c4df4f1f9835951890ebd879afedf807c9d93da3;
- UTC freeze 2026-10-29T20:02:30Z; registered prior to XP-D4 unblinding; contract **120 prediction / 24 anatomy / 8 oracle-anchor** records PASS.

Neither packaging change updates the registered priors or their declared **no structural-zero spike** choice. The original v2/v2 E1 are unaffected, and no freeze correction or erratum is indicated. Any later XP-D4 comparison must be interpreted against the ORIGINAL registered v4 values, never this audit note.
