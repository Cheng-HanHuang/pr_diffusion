# B25 — noise-selection bias and phase-retrieval failure mechanisms

Status: **B25.0 implemented; PAC inventory cleared; pushed pre-run freeze is the next gate. B25.1 scientific CPU work has not yet run.**

This stage starts from immutable commit `ed162c2f97430804fddb5d9a0bfec7abde201ca0` on historical branch `codex/b24-bestof4-failure-sweep`. The signed B23.1 ancestor is `27505e6328157ac9296c95dc5e611cbeef80de98`. B24 remains closed under `STOP_B24_METHOD_REFINEMENT` and PR #38 is historical.

B25 branch: `codex/b25-noise-selection-mechanisms`. Draft PR: `#39`, based on `codex/b24-bestof4-failure-sweep`.

## Authorization

B25.0 authorizes inventory, scientific specification, implementation, validation, and a pushed pre-run freeze. B25.1 authorizes the bounded CPU mathematical experiments and CPU analysis of existing allowlisted DEV80 artifacts described in the user authorization.

The PAC inventory returned at UTC `20260914T083919Z` with SHA-256 `e780d6bc8412bc8b8d20dbebe8428e4d13ea50254e9431efecfc48eab4128be1` and cleared the identity/path gate. It showed no pre-existing B25 worktree or B25 output root. The B25 worktree may therefore be created only from the fetched B25 branch after rechecking the remote identities.

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
3. `docs/b25/B25_EXECUTOR_CONTRACT.md`
4. `docs/b25/B25_MATHEMATICAL_MODEL.md`
5. `docs/b25/B25_NATIVE_NP_AUDIT.md`
6. `docs/b25/B25_LITERATURE_AND_NOVELTY.md`
7. `docs/b25/B25_CHECKPOINT_REPORT.md`
8. `configs/b25/b25_cpu_spec.json`
9. `configs/b25/b25_synthetic_templates.json`
10. `docs/b24/00_START_HERE.md`
11. `docs/b24/B24_3_FINAL_PLANNER_RETURN.md`
12. `docs/b24/B24_3_FINAL_ARCHIVE_RECORD.md`
13. `docs/planning/02_B23_PAC_EXECUTION_AND_RETURN_PROTOCOL.md`
14. B24 manifests/configs and accepted terminal-artifact records named below

## Pinned B24 sources/provenance

The accepted B24 method-development role split is generated as `B24_METHOD_IMAGE_ROLES.csv` / `.json`, with 80 DEVELOPMENT and 305 CONFIRMATION rows, plus `B24_METHOD_PILOT16.csv`. The PAC inventory resolved the accepted role directory to:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_2_7424_extension_20260907T231303Z/case_freeze/method_stage`

containing the role CSV/JSON, Pilot16 CSV, role-freeze summary, and checksums.

The accepted DEV80 run is:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z`

with `B24_METHOD_DEV80.csv`, `DEV80_MANIFEST.json`, 80 task records, and accepted existing terminal results.

Repository definitions include:

- `configs/b24/b24_3_method_dev_spec.json`
- `configs/b24/b24_3_dev80_overnight.json`
- `scripts/b24/freeze_b24_method_roles.py`
- `scripts/b24/generate_b24_locked_input.py`
- `scripts/b24/launch_b24_3_dev80_overnight.sh`
- `scripts/b24/run_b24_3_dev80_image.py`
- `scripts/b24/run_b24_3_dev80_np.py`
- `scripts/b24/run_b24_3_epp321_refinement.py`
- `scripts/b24/run_b24_3_np_branching.py`
- `scripts/pr_external_difffpr_np_guided_lf_s2_selector.py`
- `scripts/pr_external_difffpr_np_benchmark.py`
- `configs/b23/np1_frozen.yaml`

The original B24 zero-GPU closeout capsule remains immutable at:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z`

The corrected successor that defines the final reporting scope is:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z`

B25.3 uses the corrected successor's copied `DEV80_CLOSEOUT_PER_IMAGE.csv`, `HARD_SUBSET.csv`, Fresh2 records, and associated checksums for the shared-failure definition. It follows terminal paths only through the accepted DEV80 manifests/results and never regenerates a missing candidate.

## Native NP provenance boundary

The B24 method spec freezes parent identity `NP-1` at `configs/b23/np1_frozen.yaml`: 1000 steps, projection start 300, soft candidate count 5, hard candidate count 1, LF score radius 0.6, projection radius 0.2, and LF score mode. B24's DEV80 NP wrapper loads `scripts/b24/run_b24_3_epp321_refinement.py`, which loads `scripts/b24/run_b24_3_np_branching.py`; that runner uses `scripts/pr_external_difffpr_np_guided_lf_s2_selector.py`, which in turn imports the candidate-selection/operator implementation in `scripts/pr_external_difffpr_np_benchmark.py`.

B25 audits this chain rather than substituting a generic noise-picking implementation. Historical native semantics include candidate zero reusing the previous selected noise when `eps_prev` exists and `K>1`; other candidates are newly sampled Gaussian noise. The selected candidate is the minimum measurement-dependent LF score after denoising. B25's simplified independent-proposal experiment is explicitly a simplification, not a theorem about this incumbent/reuse/denoiser/projection process.

A verified preprocessing fact to audit in B25.4: `scripts/b24/run_b24_3_dev80_np.py` verifies the stored raw measurement and then passes `measurement_raw.clamp_min(0.0)` to the NP context. The historical SITCOM wrapper records no measurement preprocessing. The exact pinned DAPS path remains a source-audit question and no DAPS discrepancy is presumed before the B25 source audit.

## B25 implementation entrypoints

- scientific freeze: `configs/b25/b25_cpu_spec.json`
- exact toy construction: `configs/b25/b25_synthetic_templates.json`
- synthetic Experiments 1–2: `scripts/b25/run_b25_synthetic.py`
- DEV80 Experiments 3–4: `scripts/b25/run_b25_dev_diagnostics.py`
- deterministic analysis: `scripts/b25/analyze_b25_results.py`
- integrity tests: `scripts/b25/test_b25.py`
- resource wrapper: `scripts/b25/run_cpu_stage.py`
- launcher: `scripts/b25/launch_b25_cpu.sh`
- worker: `scripts/b25/run_b25_cpu_worker.sh`
- status/resume/stop: `scripts/b25/status_b25_cpu.sh`, `resume_b25_cpu.sh`, `stop_b25_cpu.sh`
- capsule packager: `scripts/b25/package_b25_run.py`

## PAC locations

- B25 worktree: `/egr/research-pac/huang248/pr_diffusion_b25`
- B25 output root: `/egr/research-pac/huang248/outputs/pr_diffusion/b25`
- historical B24 worktree: `/egr/research-pac/huang248/pr_diffusion_b24` (do not update or mutate)
- historical B24 outputs: `/egr/research-pac/huang248/outputs/pr_diffusion/b24`
- FFHQ root: `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024`
- environment root: `/egr/research-pac/huang248/conda-envs`

The B25 launcher explicitly inventories/fetches the branch again, requires a clean B25 worktree, fast-forwards it only to the fetched B25 remote, verifies B24 has not advanced, runs zero-GPU tests, then writes `PRE_RUN_IDENTITY.json` before launching any scientific subprocess.

## Next gate

Create the new B25 worktree from the fetched B25 branch after one more path/ref check, then run `scripts/b25/launch_b25_cpu.sh`. Once `PRE_RUN_IDENTITY.json` is written, that exact commit is the pre-run scientific freeze; resume refuses a different remote or local B25 head. Do not commit result-dependent changes to the B25 branch until the scientific worker has completed or been explicitly stopped.
