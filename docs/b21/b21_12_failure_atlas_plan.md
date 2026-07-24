# B21.12 zero-GPU failure and selector atlas

Status: implementation ready; descriptive audit only.

## Objective

Describe the residual failure modes of the frozen B21.11 Fresh2 policy and illustrate the selector's successful rescue/protection behavior without running DAPS again or changing any policy decision.

## Source of truth

Use only the completed B21.11 row table:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/
B21_11_fresh2_final_val100_meas5401/analysis_theta0.7/fresh2_final_rows.csv
```

The expected source has 100 distinct official-validation images and the already-validated frozen policy outputs.

## Frozen atlas composition

The atlas must contain exactly 27 disjoint cases:

- 8 Fresh2 selected failures (`fresh2_selected_good25 == 0`);
- 12 Fresh2 rescues (`fresh2_incremental_rescue == 1`);
- 7 protected Fresh1 successes where arm 1 is good, arm 2 is bad, arm 2 is rejected, and the selected output remains good.

Any count mismatch is an error and must stop the audit.

## Required visualizations

For each case, show:

1. FFHQ ground truth;
2. arm 1 in its raw orientation;
3. arm 1 after choosing the better of identity and 180-degree rotation offline;
4. arm 2 in its raw orientation;
5. arm 2 after choosing the better of identity and 180-degree rotation offline.

The selected raw candidate must be visibly marked. Labels must include raw PSNR, exact operator loss, best-orientation PSNR, and whether 180-degree rotation was used.

Produce one per-case PNG and one vertically stacked sheet for each of the three groups.

## Offline ambiguity audit

For each official Fresh2 failure, compute:

- whether the selected candidate reaches good25 after 180-degree alignment to ground truth;
- whether only the unselected candidate reaches good25 after alignment;
- whether neither candidate reaches good25 under identity or 180-degree rotation.

This audit uses ground truth and is interpretation-only. It cannot revise the official B21.11 92/100 raw-PSNR result, change the exact-loss selector, or authorize a runtime orientation resolver.

## Manual descriptive labels

Create a manual-label template for the eight official failures. Objective orientation labels may be prefilled, while subjective visual labels remain blank for human review. Suggested descriptive vocabulary includes:

- rot180/twin ambiguity;
- ghosting or duplicated facial structure;
- diffuse hallucination;
- structural collapse;
- severe color/texture mismatch;
- other.

These labels are for qualitative reporting only. Do not fit a detector, threshold, or fallback policy from this final panel.

## Required artifacts

- `failure_atlas_summary.json`;
- `failure_atlas_rows.csv`;
- one CSV for each atlas group;
- `manual_failure_labels_template.csv`;
- 27 per-case PNG panels;
- three group contact sheets;
- a Markdown report in the output directory.

## Next gate

After the atlas is reviewed, freeze a B22 fixed-baseline comparison plan. Baseline configurations and seeds must be preregistered before any SITCOM or NP GPU launch. B21.12 itself authorizes no GPU experiment.
