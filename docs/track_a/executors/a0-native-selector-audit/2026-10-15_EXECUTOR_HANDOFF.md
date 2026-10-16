# Track A A0 executor handoff — frozen B22/B24 candidate and native-selector audit
Date: 2026-10-15
Assignment: **A0 forensic/data executor** under the autonomous Track A research lead
Repository: https://github.com/Cheng-HanHuang/pr_diffusion
**Only working branch:** `codex/prdiff-track-a-a0-native-selector-audit`
**Handoff source base:** Track A planner `bd83d5093e67f69473c8b7815ad2730b2a5d161a` (includes metaplanner autonomy amendment)
**Initial permissions:** repository source review, source-safe read-only PAC inventory *through user-executed commands*, metadata-only / compact retrospective CPU analysis on already exposed B21/B22/B24 artifacts. **NO new GPU, no diffusion-model inference, no measurement generation, no historical artifact rewrites, no B24 confirmation305 access.**

## 0. Mission, roles, and reading order
You are the **executor**, not a competing planner or a PAC operator. You implement a reproducible metadata audit and historical candidate-level scientific comparison. A research lead will review your artifacts and scientific conclusions, including the decision whether another prospective benchmark is warranted. The user, not ChatGPT, runs any PAC commands.

Read entire documents in this order:
1. `docs/planning/metaplanner/2026-10-15_AUTONOMY_AMENDMENT.md`; `docs/planning/metaplanner/2026-10-15_SHARED_CHARTER.md`; root `AGENTS.md`; live [coordination ledger #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41).
2. `docs/planning/track_a/2026-10-15_TRACK_A_PLANNER_RETURN.md`; `docs/planning/track_a/2026-10-15_METAPLANNER_NEXT_MANDATE.md`; original `docs/planning/track_a/2026-10-15_NP_PUBLICATION_PLANNER_HANDOFF.md`.
3. `docs/b22/b22_3_scientific_closeout.md`, `configs/b22/b22_2_overnight.json`, `scripts/b22/run_b22_2_np_worker.py`, `scripts/b22/run_b22_2_sitcom_worker.py`, `scripts/b22/analyze_b22_3_panel.py`; `docs/b21/b21_11_fresh2_final_benchmark.md`.
4. `docs/b24/00_START_HERE.md`, `docs/b24/B24_3_ZERO_GPU_REPORTING_CORRECTION_RESULT.md`, `docs/b24/B24_3_ZERO_GPU_CLOSEOUT_RESULT.md`, `configs/b24/b24_3_dev80_overnight.json`, `scripts/b24/run_b24_3_dev80_image.py`, `scripts/b24/run_b24_1_method_smoke.py`, `scripts/b24/evaluate_b24_baseline_image.py`, `scripts/b24/run_b24_3_np_branching.py`, `scripts/b24/closeout_b24_3_zero_gpu.py`.
5. `docs/b25/B25_POSTRUN_NATIVE_NP_AUDIT.md`, `docs/b25/B25_FINAL_PLANNER_RETURN.md`, completed B26 return `docs/handoffs/B26_PLANNER_RETURN_2026-10-15.md` on `codex/planner-b26-resume-handoff-20261015` (read-only; not an invitation to rerun). B26 signed-score R had Good25=53/80 vs historical H=55/80: negative result; do not silently substitute R.

Do not ask Track B to perform Track A selector work. This executor must **not** design a new transition/basin-discovery mechanism.

## 1. Data provenance and access firewall
**Exposed historical:**
- B21 Fresh2 100 locked FFHQ `sigma_y=0.05` images/measurements; B22 reuses precisely these 100 measurements.
- B22: 100 common images; 2 DAPS Fresh trajectories (200), 4 sequential-RNG SITCOM (400), 8 NP trajectories (800 = LF four + preprojection S2 four). **Expected 1400 total physical terminals, subject to inventory.**
- B24 DEV80: exactly 20 per pre-exposed screening stratum A/B/C/D; four independent DAPS (320), four independently seeded SITCOM (320), four LF NP roots (320), **expected 960 native baseline candidate terminals**, some Pilot16 NP may be reused from earlier completed artifacts. The B24 DEV80 cohort is **not a representative random test panel**; do not pool directly with B22 or compute a pooled significance test.
- B24 7424-screen records are exposed screening evidence, not a held-out natural distribution. B24 305-confirmation payload/paths/registries are **excluded**, except historical published exclusion proofs where charter explicitly allows. Do not enumerate, open, hash, read, touch, or inspect confirmation305 data.

**PAC locations are historical hints, not verified facts:**
- `/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401/` (historical Fresh2 candidates; analysis `analysis_theta0.7/fresh2_final_rows.csv`)
- `/egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines/` (locate exact validated B22 run through small bounded receipts/manifests, not recursive scan)
- `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z/`
- `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z/`
- track owned **reserved** PAC worktree: `/egr/research-pac/huang248/pr_diffusion_track_a_a0_native_selector` (must check nonexistence/current ownership before creating; do not switch other worktrees)
- track owned **reserved** output: `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a0_native_selector_audit_20261015/`. Never overwrite any existing path.
- Environment binaries under `/egr/research-pac/huang248/conda-envs` only; choose explicit interpreter after targeted inspection, **never default Python** and never `/home`.

Run only user-provided or user-executed bounded commands. No `set -e`, `set -u`, `set -o pipefail`, `exit`, `logout`, broad `pkill`, changing historical git trees, mass `find`/recursive `du`, or package install. Use bounded `ls`, `rg`, known manifest/path reads; no GPU allocation. Always preserve interactive shell and save full logs within new exclusive output if authorized.

## 2. Phase A0.0 — read-only PAC SOURCE + ARTIFACT INVENTORY (FIRST)
Before creating a checkout or output tree, produce **one single safe, read-only PAC terminal block** for the user. It must:
- print selected branch/commit, registered worktrees and status **only for relevant explicit checkout paths** without checkout/reset/stash;
- check exact existence, type and writing conflicts for the reserved track namespace; inspect GitHub source-only diffs and external native repo commit/status/short diff-stat/digest (no cleaning local dirty repos);
- check *bounded* B21 Fresh2, validated B22 full, B24 DEV80/corrected-closeout manifest/index files; locate B22 run by existing manifest pointer and at most shallow known-directory listing;
- inspect **sample key sets, not private image payloads**, in a bounded handful of representative B22/B24 JSON/CSV files to establish candidate-id, selector, PSNR, metric-representation, time, model/operator SHA and receipt availability. Do not inspect GT tensors, FFT values or any 305 confirmation location.
- record actual B24 DAPS/SITCOM frozen tracked diffs and whether source state used for the experiment is exactly reconcilable; **upstream submodule SHA alone is not sufficient**. No copying these dirty diffs into native repos.
- save a compact JSON/Markdown inventory under the approved track-owned output, with checksums of inventory text and source identities, full command/stdout log, and a missing-data table. **The inventory is complete only after user returns real outputs.** No inferred PASS.

**Expected file format clues from audited source**:
- B22: `full/sitcom/rowNNN_ID/candidates/run*_masterseed*/result.json` and `.../policy.json`; `full/np/rowNNN_ID/candidates/cfg*_seed*/result.json` and `.../policy.json`; exact stage/run root must come from receipts.
- B24: `DEV80_MANIFEST.json`, `DEV80_PER_IMAGE.csv`, `workers/gpu*/<task>/IMAGE_COMPLETE.json`; `baseline/DAPS_GROUP.json`, `baseline/SITCOM_GROUP.json`, `baseline/metrics/METRICS.json`; NP `NP4_INDEPENDENT/result.json`, sometimes referenced by Pilot16 reuse instead of duplicated. Actual dynamic paths from manifests, **not folder-name guesses**.
- B24 corrected closeout: `FRESH2_PER_IMAGE.csv`, `FRESH2_SUMMARY.json`; its derivation evaluated exact DAPS signed loss for first two candidates, not necessarily all four.
- B22 SITCOM `selector.correction_norm` step160 exists; its `measurement_residual.normalized_l2` is stored. NP candidate `selector_stats.selector_post_winner_lf_mse_mean` and raw PSNR exist; do not presume signed full terminal residual exists.
- B24 external SITCOM **may not store B22-style step160 correction norm** because it runs a single-trajectory wrapper with `record=False` in ordinary B24. Absence is a scientific missing cell, **not** permission to invent a native K4 selected row.

Deliverable `A0_SOURCE_AND_ARTIFACT_MANIFEST.md` and `A0_SOURCE_AND_ARTIFACT_MANIFEST.json` with explicit **available / unavailable / source uncertain / PAC not checked** statuses per cohort, method, K, field and seed, and exact accepted root/commit/weights/measurement identities. Stop if unexpected file cardinality, observation mismatch, bad completion receipt, or missing output provenance.

## 3. Phase A0.1 — fixed historical comparison engine, **after inventory PASS**
Create a standalone deterministic **metadata-only** Python analysis script in `scripts/track_a/` with tests and a frozen `configs/track_a/` analysis spec. It must ingest only pre-existing archived candidate metrics/selector values/costs by *explicit manifest paths*, no `torch`, no operator import, no `PIL`, no datasets, no FFT, no diffusion model/checkpoint inference. Require finite values, unique (cohort, image, method, seed, configuration, rep), required cardinality, matching GT metric representation, source hashes and a complete ledger. Must **fail closed** on nonpresent selector values; no filling, imputation or selective omission of hard cases.

**Candidate ladders, preserving actual identity:**
- **B22 DAPS:** candidate1 (K1), and two-candidate `Fresh2` historical loss-margin selector; PSNR oracle of K2. Do **not** invent a DAPS K4 or K8 in this cohort.
- **B22 SITCOM:** K1, nested sequential-RNG prefixes K2/K3/K4; original clean-free step160 correction-norm winner at K4; optional *fixed-prefix* K2/K3 correction-norm evaluation allowed as retrospective descriptive sensitivity (not a previously deployed policy). Compare offline GT PSNR-oracle at each prefix.
- **B22 NP:** LF seeds 100,101,102,103 nested K1..K4 and optional independently labeled S2-only K1..K4; B22 `NP-8-RS` is the historical full **4 LF + 4 S2** configuration portfolio, *not* eight IID identical-restarter trajectories. Native trailing LF score min, stable config-order/seed tie. GT PSNR oracle K4 per homogeneous group and mixed K8 separately.
- **B24 DAPS:** independently seeded reps0..3. K1, K2 history-derived Fresh2 margin (only if exact rep0/1 loss is verified), K2/K4 GT oracle. **No historical executable native DAPS K4 policy**: do not label min-final-loss over four as historical native. If all four signed losses are not stored, report unavailable instead of evaluating terminal images.
- **B24 SITCOM:** independently seeded reps0..3, offline GT oracle K1..4; K4 clean-free B22 correction-norm policy is *not* available unless its step160 feature was actually recorded and semantically compatible. Never substitute a low residual without a distinct external-selector diagnostic label.
- **B24 NP:** homogeneous LF four independent roots, native final post-proj LF score at K1..K4 and GT PSNR oracles; do not confuse EPP321/PE3 with normal NP4. Ensure 16 Pilot cases reuse the right frozen results.

**Exact selection implementations:**
1. `oracle_psnr_bestK`: max raw **RGB PSNR** on **same metrics representation**, stable first-run ties; **offline with GT**. SSIM high / LPIPS low are secondary *reported on the PSNR-chosen output*; don't silently choose a different winner by metric.
2. `DAPS Fresh2`: rep1 only when `L_y(rep1)<L_y(rep0)-0.7`; ties stay rep0; signed full native operator loss.
3. `SITCOM B22`: minimum `correction_norm` at anneal step160, run-index ascending tie. This is not a final native `logp`.
4. `NP`: minimum `selector_post_winner_lf_mse_mean` across eligible runs, stable config-order then seed for B22 K8, terminal-index/rep for B24 K4.
5. Optional **signed full-measurement residual** shared diagnostic only if already stored in all compared candidates and same operator/raw observation verified. **Do not secretly evaluate/recompute extra FFTs** under a metadata-only stage. Do not select the most flattering statistic based on PSNR.
6. If `SITCOM` or `DAPS` published-paper main best-of-four differs in metric/quantization, flag instead of claiming numerical exact reproduction; both refer to GT-assisted reporting convention, not deployable selection.

**Analysis outputs**, B22 and B24 **separately**:
- a machine-readable one-row-per-(cohort,image,method,policy,K) `A0_POLICY_ROWS.csv` with source method/ref/candidate IDs and score, selected and oracle PSNR, GT leak tag, measured work, selector-gap;
- per-image `A0_CANDIDATE_ROWS.csv` with seed stream/config tag, metric, cost and fields-present flags, no pixels, no paths to locked 305;
- `A0_METRIC_SUMMARY.csv`: PSNR mean/median/q05/min, Good25, Bad20, selected–oracle gap distribution, bad/threshold discordance, realized GPU-active vs summed candidate wall where known;
- fixed paired-by-image comparisons **only** for same cohort + image/measurement + evaluation representation + budget/K conditioning, paired image-bootstrap CIs, exact McNemar of Good25 rescues/harms; CDF/q05 sensitivity; no pooling B24's stratified DEV80 with B22 random val100;
- `A0_UNSUPPORTED_CELLS.md`: unavailable K/policy/metric, unverified native source identity, uncomputable residuals and why;
- `A0_REPLAY_CHECKS.json` recomputing documented B22 selected Good25 92/93/95, historical B24 Fresh2 Good25=57/80 and NP4 Good25=55/80 **if all required metadata exists**. If mismatch > tolerance or count drift, stop rather than reinterpreting methods.
- cost axis: full actual K count, method-specific UNet calls/backward/FFT if documented, total GPU-active seconds, wall time + concurrency, supported FLOP estimates only where documented and separated from unsupported/custom work. Do not invent exact cross-family FLOP equivalence.

For retrospective same-K or budget comparison, **never treat a method's best-of-K GT oracle as its deployable row**. At K4 on B24, native selected SITCOM/DAPS may be missing; compare **candidate-generation ceilings** to NP4 oracle and present NP4 executable separately with an explicit caveat. No post-hoc code choosing policies via GT.

## 4. Specific scientific questions and decision gates
Return evidence for:
- Does NP actually avoid DAPS/SITCOM candidate-generation catastrophes *at the same K* (oracle-to-oracle) or primarily exploit different selection statistics/population sizes?
- Is the NP native selected/oracle Good25 gap small while **continuous PSNR regret** is large? B22 NP8 selected 95, oracle 96, exact PSNR-oracle matches 26/100; do not conclude near-optimal image quality from Good25 gap alone.
- Does a low **signed terminal Fourier residual** track image PSNR within each family *where already logged*, or is evidence absent/contradictory? No inference that the NP LF selection is an equally appropriate DAPS/SITCOM policy.
- Is there a non-GT clean-free K4/SITCOM baseline available? If not, this is a **priority prospective protocol deficiency**, not permission to invent one.
- Which apparent cost points remain after accounting for actual run count and GPU-active sum? B22 NP8 ≈411.8 GPU-s/image, Fresh2 ≈312.6, SITCOM4S ≈196.1; K2 vs K4 vs K8 **not** compute-matched. B24 supported FLOP estimates are partial, not total.
- Is the B24 DEV80 diagnosis restricted to its balanced strata? Always stratify descriptively and state nonrepresentativeness.

**Executor GO/STOP:** STOP audit when provenance or source mismatch cannot be reconciled, required native selector feature absent (mark cell missing), contamination/confirmation305 risk arises, or a script would need image/GT/operator re-evaluation. Return a negative/missing-data report rather than silently substituting a policy. P0 alone CANNOT declare prospective NP superiority. The Track A lead will decide after your return whether to propose a new fair baseline experiment.

## 5. Source/commit governance and executor return
Branch-only changes under `scripts/track_a/`, `configs/track_a/`, `docs/track_a/executors/a0-native-selector-audit/` and optionally `docs/planning/track_a/` when asked. Do not modify B19–B26 scripts, source, configs, PRs, hidden files, or other track branches. Keep units tests on synthetic in-memory **metadata**; no GPU tests. Commit a frozen pre-analysis source SHA and all schema/test tools; save raw PAC JSON and full logs on PAC only. Repo contains compact analysis tables/metadata if they do not expose sensitive data or giant artifacts.

Before any PAC execution step, check [coordination ledger #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41) and add/update scoped [EXECUTOR] status. CPU-only work can proceed with user-executed bounded commands without a GPU reservation, but **must** record actual paths and source SHA. Before any future GPU, user-launched physical IDs, memory, time, counts, special evidence and study plan are required and must be registered; **not in this A0 scope**.

Commit `A0_EXECUTOR_RETURN.md` under `docs/track_a/executors/a0-native-selector-audit/` with:
(1) branch and exact commit; (2) source/effective dirty-state and model hashes; (3) input cohorts/manifest SHA/seed/candidate cardinality; (4) available vs missing fields; (5) reproduced B22/B24 results; (6) table of GT oracle vs native clean-free selected per observed K (clear unavailable); (7) paired effect/CI and tail; (8) true charged compute; (9) safety/confirmation305 compliance; (10) replay checks and missing artifacts; (11) files/sha256/paths/Drive if appropriate; (12) **GO / NO-GO for further historical work**, not independent permission to publish a claim. Register return commit in issue #41 and send to Track A scientific lead.

## Executor first message
**First response:** confirm reading order and exact branch; summarize source/method semantics, propose **one** consolidated bounded read-only PAC inventory command with full saved log and short terminal summary; **stop after command delivery**. Do not declare inventory PASS, create a checkout, or start a run until the user returns the actual PAC result. Then implement stage A0.1 from verified metadata. No new solver trajectory.
