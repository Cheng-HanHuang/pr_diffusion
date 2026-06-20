These are the predeclared frozen A14 Branch A policies.

Primary policy:

- `frozen_policy_conservative.json`
- policy family: `consensus_lowfreq_nn`
- role: primary / conservative

Secondary policy:

- `frozen_policy_aggressive.json`
- policy family: `residual_or_lowfreq_nn`
- role: secondary / aggressive / higher-replacement-budget

Both policies were selected from A8+A11 development data only.
No A14 trajectory results were used to choose features or thresholds.

A14 must evaluate these exact JSON files without changing thresholds, feature definitions, directions, or fallback source.
The output-side provenance folder is:

`/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/A14_frozen_policy_config`

The repo-side copies exist so the frozen policy definitions can live in-tree alongside the code and experiment plan:

`/egr/research-pac/huang248/pr_diffusion_repo/configs/branch_A/a14`
