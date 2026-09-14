# B25 — noise-selection bias and phase-retrieval failure mechanisms

Status: **B25.0/B25.1 authorized, PAC inventory gate not yet cleared**.

This stage starts from immutable commit `ed162c2f97430804fddb5d9a0bfec7abde201ca0` on historical branch `codex/b24-bestof4-failure-sweep`. The signed B23.1 ancestor is `27505e6328157ac9296c95dc5e611cbeef80de98`. B24 remains closed under `STOP_B24_METHOD_REFINEMENT` and PR #38 is historical.

B25 branch: `codex/b25-noise-selection-mechanisms`.

## Authorization

B25.0 authorizes inventory, scientific specification, implementation, validation, and a pushed pre-run freeze. B25.1 authorizes the bounded CPU mathematical experiments and CPU analysis of existing allowlisted DEV80 artifacts described in the user authorization.

The first operational gate is a consolidated PAC inventory. Do **not** execute scientific B25 experiments before the inventory output has been returned to the executor and checked.

## Hard restrictions

- No GPU work. Every scientific subprocess must set `CUDA_VISIBLE_DEVICES=""`.
- No pretrained-model inference, including inference on CPU.
- No new FFHQ reconstruction and no new FFHQ measurement generation.
- No loading confirmation305 image, measurement, or reconstruction payloads. Reading the confirmation registry IDs for exclusion checks is allowed.
- No C1-only or otherwise additional image exposure.
- No tuning of B24 NP/EPP/PE3 selectors, schedules, or thresholds.
- No FFHQ method-comparison pilot.
- No NP/SITCOM state transplantation or unsupported cross-family continuation; the B23 compatibility failure remains binding.
- Preserve historical B24/B23 worktrees and outputs. Do not merge, rebase, squash, retarget historical PRs, force-push, or rewrite history.
- Use `/egr/research-pac/huang248`, never `/home`.
- Do not recursively scan large data, environment, model, or output trees.
- Shell launchers for B25 must not use `set -e`, `set -u`, `set -o pipefail`, shell replacement, or an `exit` path intended to close the user's terminal. Report failures and return control instead.

## Required reading order

1. root `AGENTS.md` (historical B23 guidance is subordinate to this B25 authorization for this branch)
2. this file
3. `docs/b24/00_START_HERE.md`
4. `docs/b24/B24_3_FINAL_PLANNER_RETURN.md`
5. `docs/b24/B24_3_FINAL_ARCHIVE_RECORD.md`
6. `docs/planning/02_B23_PAC_EXECUTION_AND_RETURN_PROTOCOL.md`
7. B24 manifests/configs and accepted terminal-artifact records named below

## Pinned B24 sources/provenance to recover

The accepted B24 method-development role split is generated as `B24_METHOD_IMAGE_ROLES.csv` / `.json`, with 80 DEVELOPMENT and 305 CONFIRMATION rows, plus `B24_METHOD_PILOT16.csv`. The PAC pointer used historically is:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_METHOD_STAGE_LATEST_FREEZE.txt`

which points to a directory containing:

- `B24_METHOD_IMAGE_ROLES.csv`
- `B24_METHOD_IMAGE_ROLES.json`
- `B24_METHOD_PILOT16.csv`
- `B24_METHOD_ROLE_FREEZE_SUMMARY.json`
- `SHA256SUMS.txt`

Repository definitions:

- `configs/b24/b24_3_method_dev_spec.json`
- `configs/b24/b24_3_dev80_overnight.json`
- `scripts/b24/freeze_b24_method_roles.py`
- `scripts/b24/freeze_b24_method_roles.sh`
- `scripts/b24/generate_b24_locked_input.py`
- `scripts/b24/launch_b24_3_dev80_overnight.sh`
- `scripts/b24/run_b24_3_dev80_image.py`
- `scripts/b24/run_b24_3_dev80_np.py`
- `scripts/b24/run_b24_3_epp321_refinement.py`
- `scripts/b24/run_b24_3_np_branching.py`
- `scripts/pr_external_difffpr_np_guided_lf_s2_selector.py`
- `scripts/pr_external_difffpr_np_benchmark.py`
- `configs/b23/np1_frozen.yaml`

The corrected B24 closeout source capsule is pinned by `configs/b24/b24_3_zero_gpu_reporting_correction.json` to:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z`

and B25 must use its `DEV80_CLOSEOUT_PER_IMAGE.csv`, `HARD_SUBSET.csv`, Fresh2 records, and associated checksums only as historical records. B25 must separately locate the existing hashed terminal tensor manifests/results referenced by the accepted DEV80 run; it must not regenerate missing candidates.

## Native NP provenance boundary

The B24 method spec freezes parent identity `NP-1` at `configs/b23/np1_frozen.yaml`: 1000 steps, projection start 300, soft candidate count 5, hard candidate count 1, LF score radius 0.6, projection radius 0.2, and LF score mode. B24's DEV80 NP wrapper loads `scripts/b24/run_b24_3_epp321_refinement.py`, which loads `scripts/b24/run_b24_3_np_branching.py`; that runner uses `scripts/pr_external_difffpr_np_guided_lf_s2_selector.py`, which in turn imports the candidate-selection/operator implementation in `scripts/pr_external_difffpr_np_benchmark.py`.

B25 must audit this chain rather than substituting a generic noise-picking implementation. Historical native semantics include candidate zero reusing the previous selected noise when `eps_prev` exists and `K>1`; other candidates are newly sampled Gaussian noise. The selected candidate is the minimum measurement-dependent LF score. B25's simplified independent-proposal experiment must be labeled as a simplification, not as an exact theorem about this incumbent/reuse/projection process.

A verified preprocessing fact to audit in B25.4: `scripts/b24/run_b24_3_dev80_np.py` loads the stored raw measurement and then passes `measurement_raw.clamp_min(0.0)` to the NP context. Whether and how the DAPS/SITCOM paths transform the same stored data must still be recovered before drawing a discrepancy conclusion.

## PAC locations to inventory, not assume

- requested B25 worktree: `/egr/research-pac/huang248/pr_diffusion_b25`
- requested B25 output root: `/egr/research-pac/huang248/outputs/pr_diffusion/b25`
- historical B24 worktree: `/egr/research-pac/huang248/pr_diffusion_b24`
- historical B24 outputs: `/egr/research-pac/huang248/outputs/pr_diffusion/b24`
- FFHQ root: `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024`
- environment root: `/egr/research-pac/huang248/conda-envs`

Do not create or mutate the B25 worktree/output root until their pre-existence and identity have been reported by the inventory gate.

## Next gate

Return the consolidated PAC inventory report and its SHA-256 sidecar to the executor. After it is checked, proceed with B25.0 implementation, literature/equation audit, exact numerical freeze, tests, and a pushed pre-run commit. Only then may B25.1 scientific CPU execution begin.
