# Track A A1.3 executor — saved-candidate selection, measurement physics, and failure forensics
**Date:** 2026-10-20 (America/Detroit)  
**Status:** `HANDOFF_COMMITTED / NO_PAC_CPU_RUN / NO_GPU / NO_NEW_DATA`  
**Executor working branch ONLY:** `codex/prdiff-track-a-a13-candidate-audit-20261020`  
**Scientific lead:** Track A planner branch `codex/prdiff-track-a-np-paper-planner-20261015`  
**Starting planner scientific decision commit:** `ac88a3eabc4a58652de4b9913873eb01185d6315`.  
**Repository:** https://github.com/Cheng-HanHuang/pr_diffusion

## 0. Assignment, what you must NOT do
You are a specialist **static candidate/physics forensic executor**, *not* an alternative Track A scientific planner or Track B's basin-aware method developer. We already have 128 complete candidate trajectories from A1.2. Your job is to leverage **those exact saved 128 terminal outputs** to measure clean-free score behavior, physical ambiguity and failure mechanisms with **zero additional GPU inference**. The researcher operates PAC manually; you cannot access it, and your first return must be a bounded safe PAC inventory + source/test plan, **not** an assertion of successful execution.

**No new diffusion trajectories, CUDA, pretrained model inference/denoising, observation generation, FFHQ IDs beyond the frozen eight, broad OOD data search, tuning on GT, B24 protected confirmation305 access of any sort (including membership checks), Track B algorithm changes, B21–B26 edits, or overwrite of existing A1.2 artifacts.** A successful A1.3 cannot itself establish generalization or justify a large GPU campaign.

**Required reading order:** 
1. `AGENTS.md`; `docs/planning/metaplanner/2026-10-15_AUTONOMY_AMENDMENT.md`; original/shared charter; **live** [coordination issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41).
2. New scientific decision `docs/planning/track_a/2026-10-20_A13_SCIENTIFIC_DECISION.md`; `docs/planning/track_a/2026-10-16_ORACLE_FIRST_PUBLICATION_REFRAME.md`; initial Track A planner return. 
3. Complete predecessor A1 report `docs/track_a/executors/a1-oracle-budget/A1_PLANNER_HANDOFF_RESULTS_PAC_INDEX_20261020.md`, machine index `A1_PAC_ARTIFACT_INDEX_20261020.json`, detailed `A1_2_GPU128_VERIFIED_SCIENTIFIC_ANALYSIS_20261020.md` **on A1 branch** `codex/prdiff-track-a-a1-oracle-budget-20261016` doc HEAD `3d119ba25959f79c47edadd20aa819b6715cd934`. Read full source/controller `scripts/track_a/a1_2_k4_exposed_b22.py` and its source checksum evidence at **A1 execution commit** `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd`. You inherit from *planner* commit, not from A1: use `git show` or GitHub references to read the A1 branch, **do not cherry-pick/merge it wholesale**.
4. A0 verified B22 archived candidate audit `docs/track_a/executors/a0-native-selector-audit/A0_B22_RETROSPECTIVE_RESULTS_2026-10-16.md` on A0 branch at `ecc66b890b63123c0445a13d33277e09f6ab4ab5`. Read B22 SITCOM and NP worker source and DAPS/SITCOM operator implementations pinned by A1 only as source references.
5. Optional OOD novelty context: `https://arxiv.org/abs/2410.11730` (Hu et al. 2024 mismatched diffusion prior inversion); `https://proceedings.mlr.press/v235/li24bj.html` (DiffFPR phase-ambiguity baseline); prior-mismatch is NOT new by itself. A1.3 is not commissioned for an OOD pilot.

## 1. Frozen cohort, source and exact PAC file access
The ONLY allowed scientific inputs are the completed **8 previously exposed B22 images** and their A1.2 frozen tensors/measurements/candidate outputs, plus their exact provenance files:
`69073, 68539, 67948, 60140, 67293, 67174, 61101, 66631`.
Frozen list SHA256 `faa2e292e915690dc8dd7d477591bb47a12a5f5977fd1c9648849e25c2d09f32`.

Root: `ROOT=/egr/research-pac/huang248` (NOT /home). 
**READ-ONLY PREDECESSOR:**
`A1=$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016`
`RUN=$A1/a1_2_exposed_b22_k4_v1`
`INPUT=$A1/a1_2_exposed_b22_inputs_v1`
`$RUN/RUN_IDENTITY.json`
`$RUN/A1_2_COMPLETE.json`
`$RUN/A1_2_OFFLINE_ORACLE_SUMMARY.json`
`$INPUT/A1_2_INPUT_FREEZE_COMPLETE.json`
`$INPUT/<ID>/ground_truth_tensor.pt`
`$RUN/<ID>/IMAGE_CANDIDATES_COMPLETE.json`
`$RUN/<ID>/<METHOD>_rep<R>_COMPLETE.json` for `METHOD=DAPS,SITCOM,NP_LF,NP_S2`, `R=0..3`.
Read the **per-candidate terminal path stored in COMPLETE**, not guessed path when different. A1 report notes:
- `$RUN/<ID>/daps/rep<R>/daps_results/rep<R>/samples/00000_run0000.png`
- `$RUN/<ID>/sitcom/rep<R>/sitcom1/reconstruction.pt`, `reconstruction.png`, `result.json`
- `$RUN/<ID>/np_lf/rep<R>/np1/reconstruction.pt`, `reconstruction.png`, `result.json`
- `$RUN/<ID>/np_s2/rep<R>/np1/reconstruction.pt`, `reconstruction.png`, `result.json`
- per-candidate `cuda_timing.json`, `run.log`, method config/input manifest.

Input measurement identity is the original B21 saved signed `ffhq<ID>_phase_noise005_meas5401.pt` (exact path/sha from A1 input freeze) and **not regenerated**. Use only the eight logged paths, no scan of B21 images/other cohorts. Approx shape `float32 [1,3,384,384]`, original image model-range GT `[-1,1]`; canonical common outcome saved PNG8 RGB raw PSNR, 180° PSNR diagnostic. Operator oversampling 2 and sigma_y 0.05, checkpoint SHA256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`, but **do not load the pretrained checkpoint**.

Provenance audit:
- Source of A1 GPU job `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd`; method native external DAPS `e7a77d094167084faed19b599b96673b7bb11447`, SITCOM `275ab67efbd8146bffca20155171ba6be1169c09`, DiffFPR `a45ffe58f18fed8a63d3446600424e2b08733524`; A1.2 metadata ZIP SHA256 `e82ad0c0314b4cff0fed82db656500307cd82f6927f42c7bc667f94878d7e9fc`.
- Use `A1_2_COMPLETE`, per-candidate digest/seed/raw observation SHA checks. Missing/changed GT, fewer/more than 128 distinct candidate/terminal IDs, nonfinite values, differing measurement identity -> **STOP** and report, no silent partial-case filtering.
- A1.2 reports **128 unique terminal content digests**, 32/method and 8/image, all PASS. These are predecessor claims; user PAC audit must verify existence and hash for **all 128 saved terminals** before declaring A1.3 success. DAPS saved PNG was not independently rehashed by A1 report because it was omitted from metadata ZIP; verifying its recorded bytes on PAC is A1.3's proper role.

**New executor worktree RESERVED, must check absence before clone/worktree:**
`/egr/research-pac/huang248/pr_diffusion_track_a_a13_candidate_audit`
**New A1.3 outputs RESERVED, check nonexistence before creation:**
`/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a13_candidate_audit_20261020/`
Within this tree, unique non-overwrite stage IDs `a13_0_inventory_v1/`, `a13_1_cpu_forensics_v1/`; never write `A1` or `RUN` or historical bXX outputs. PAC interpreter explicitly `$ROOT/conda-envs/prdiff_ffhq/bin/python`; can use `torch` **CPU only** for saved tensors/forward FFT. Never assume system Python. No writing to `/home`, no terminal-exiting flags (`set -e`, `set -u`, `set -o pipefail`, `exit`, `logout`), no broad `find` or recursive `du`, no source installs or destructive reset.

## 2. Stage A1.3.0: safe inspection & pure-Python/CPU unit-test gate (FIRST)
Design and commit `scripts/track_a/a13_inventory.py`, `scripts/track_a/a13_physics.py`, `scripts/track_a/a13_analysis.py`, `scripts/track_a/test_a13_*.py`, `configs/track_a/a13_frozen_analysis_v1.json`, and `scripts/track_a/a13_run_inventory_v1.sh` and `a13_run_cpu_forensics_v1.sh` as appropriate. Implement scripts with explicit `--stage inventory|analyze` and input `--run-root`, `--input-root`, `--out`; forbid arbitrary globbing outside approved roots. Python imports must not import a diffusion UNet, checkpoint or `torch.cuda` execution. Set `CUDA_VISIBLE_DEVICES=""` and assert CPU-only after import, do not probe/free GPU memory by executing CUDA.

**First user PAC command** should be an explicit, short read-only path/branch/candidate-file inspection and status; no newly generated output until the user verifies A1.3 exclusive output nonexistence/source identity. Show source/status of A1.3 checkout, A1 predecessor root/3 JSON receipt keys, 8 ID cardinality, a handful of expected terminal paths and current available environment; do not open actual pixels or tensors in preliminary inventory. Preserve terminal, no implicit one-shot runs, no change to A1 checkout. Return smallest useful concise stdout; STOP if directories occupied by unexpected work.

**Engineering tests required before reading all 128:**
- synthetic FFT forward operator from A1 exact spec: channel-wise `x_model [-1,1]` to `x_phys=(x+1)/2`, pad64 to 384, Fourier magnitude fft-centered per **actual native source**, operator scaling/normalization tested against native/source-provided deterministic witness, NOT merely assumed. Test shape, dtype, padding, FFT normalization and orientation with identity/synthetic impulse and known magnitude. If exact native op cannot be numerically reconciled on CPU, status `OPERATOR_NOT_RECONCILED` -> **STOP before residual analysis**.
- known residual for perfect nonnegative noiseless measurement approximately zero; sensitivity to manually constructed signed negative measurement target; `y_plus=max(y_raw,0)` versus raw negative handling and non-negativity of `A(x)`.
- GT float `[-1,1]` to PNG8 saved candidate numerical conversion and RGB PSNR formula (and reference MAX peak consistent with A1). A1.2 summary raw oracle K1,K2,K4 per method and NP mixed8 must replay exactly within declared 1e-4 dB numeric tolerance (allow operator score tolerance independent from PSNR); if disagreement -> stop.
- 180° image transform on a **real-valued synthetic object**: test `||A(x)-A(rot180(x))||` and channel-wise; exactness depends on source FFT centering/pad. Test discrete conjugate-reflection/translation distinctions; never silently call simple flip an exact invariance without an operator audit.
- synthetic generator of multiple candidate PSNR and scores that proves min/max direction (A0 previously caught a genuine minimization bug), stable-tie rule, NaN rejection, oracle cannot be used by a GT-free selector, and no GT in feature extractor.

## 3. Stage A1.3.1: candidate-by-candidate static output analysis (NO diffusion inference)
Once stage0 tests and PAC integrity checks PASS, read **only the specified 128 A1.2 saved reconstructions**, eight A1 frozen GT tensors and eight saved observations. Compute the following data dictionary per (image, family, rep):
- immutable ID, candidate source/result digest, method, config LF vs S2, exact seed32, completion/elapsed time, source head and original output path (recorded only, no images in GitHub).
- canonical `psnr_raw_rgb8_db` (replay), `psnr_rot180_rgb8_db` (auxiliary), `delta_rot180_minus_raw_db`; optionally per-channel diagnostics; no transform selected during inference.
- signed Fourier magnitude fidelity `r_signed = ||A(x)-y_raw||_2/(||y_raw||_2+epsilon)`, `r_clipped = ||A(x)-clamp_min(y_raw,0)||_2/(||clamp_min(y_raw,0)||_2+epsilon)`; also unnormalized sum squared loss, per-element RMSE, residual per channel, and negative-measurement count/fraction. Denominator/epsilon/count, shape, scaling, Fourier sign/centering, sample normalization are **written into schema**. A1 frozen NP uses clipped observation during scoring/project, DAPS/SITCOM use signed. **Do not treat r_signed and r_clipped as equivalent or claim native likelihood equality**; if two methods use different preprocessing, that is a methodological condition to discuss.
- Native terminal score: `DAPS exact terminal signed loss` (compute only as a clearly labeled *new external terminal diagnostic*, not 'historical native K4'); `SITCOM stored correction_norm` only if native run actually recorded an equivalent B22 step160 score; otherwise N/A; `NP final post-projection mean LF` only if archived A1 NP1 reports this on comparable independent roots; otherwise N/A. Never relabel within-trajectory NP 5-proposal scores as a cross-root confidence measure without source evidence.
- Source-backed LF-magnitude scoring against raw/clipped measurement if computable entirely from saved output/known observation: document exact Fourier crop/radius and error normalization; if score definition uncertain, omit.
- Per-image K1/2/4 within-family and fixed interleaved NP mixed4/8: GT-oracle raw PSNR/Good25/Good30/bad20, clean-free PSNR selected by **predeclared** score if available, selector regret distribution, oracle-selected vs score-selected rep, per-candidate full wall cost and if available CUDA event elapsed *not GPU utilization*.
- Candidate proximity/difference in output space as descriptive diversity, but **do not** claim topological basin transitions or local minima from a final image cluster alone. No perturbation or trajectory reruns, no claiming an intermediate diffusion path unless intermediate states were actually archived.

**No train/test split illusion:** these eight images are EXPOSED, so compute correlations/scatterplots and counterexamples descriptively. You may predeclare physical diagnostics and compare them, but cannot tune weights/thresholds to these GT outcomes and call them prospective GT-free policies. Scores requiring supervised regression on the eight GT images are explicitly `UNSUPPORTED/RETROSPECTIVE_GT_FITTED` and not a success.

## 4. Taxonomy and rigor
Raw RGB PSNR is **always primary**. No GT-based rotation 'fix' is executable from a single Fourier magnitude when symmetry is exact; GT-orientation-aware PSNR is *only auxiliary diagnostics*.

Propose descriptive tags with pre-written conditions and **continuous** residual/PSNR plot, do not tune thresholds on cases:
- `RAW_GOOD25` if `PSNR_raw>=25`.
- `ROTATION_EQUIVALENT_CANDIDATE` when `PSNR_raw<25`, `PSNR_rot180>=25`, and the **source-validated** Fourier operator shows the relevant symmetry; otherwise `POSSIBLE_ROTATION_CASE` without asserting true equivalence.
- `OTHER_MEASUREMENT_CONSISTENT_LOW_RAW` when raw+rotated both <25 and signed residual is at or below a separately calibrated **synthetic known-noise reference range**, with threshold locked *before* reading real residuals. If no independent reference/noise calibration, use descriptive rank/quantile, mark class inconclusive.
- `POOR_MEASUREMENT_FIT` when both raw and rotation PSNR low and signed residual demonstrably exceeds the reference range; otherwise `UNRESOLVED_LOW_RAW`.
- Distinguish a noisy signed magnitude's unavoidable residual on negative entries from algorithm failure. The native candidate can legitimately be near feasible clipped magnitude but not minimize squared raw signed target to zero.

Report confusion matrices per family; special cases **68539 DAPS rep0**, **67293 DAPS rep1–3**, **67174 NP LF K1–3**, and original B22 SITCOM selector failure **60140** as *illustrative separate, frozen cohorts*; only A1.2 candidate IDs should be quantitatively audited, do not mix original B22 trajectory IDs into A1.2.

## 5. Selection experiment: frozen and clearly tagged
Primary A1.3 question: can already recorded physically motivated clean-free terminal scores reliably identify high-quality candidates **within a method**, not can we create a new basin-finding algorithm?
- Frozen score methods: raw signed terminal full-`A` residual, clipped full-`A` residual, optionally genuine B22/A1 native score (only verified), optionally verified LF radius/rule; evaluate **stable min, rep index tie**.
- Compare each score's GT-selected quality against a GT-oracle of the **exact same candidate set**, K1/2/4; GT oracle non-deployable. Report mean regret, max regret, Good25/30 rescues/harms, raw PSNR scatter, score/PSNR Spearman with caveat n=8, sample-level rank disagreement, and charged cost of computing the offline score (CPU wall+FFTs) *separately* from candidate-generation work. If score costs additional FFT(s), **charge** them in executable costs. No score hyperparameter selection based on which produced high GT PSNR.
- Distinguish **within-method candidate restart selection** from **cross-family method selection**, the latter has incompatible priors/source objectives; no cross-family router advertised.
- A1.2 GT oracle is still valid offline candidate-generation study even if GT-free results negative. If physical magnitude is invariant under a 180° inversion and prior log probability is unavailable, a GT-free score cannot generally resolve correct orientation; state impossibility/ambiguity limitation carefully, not as observed impossibility of all non-GT methods.
- If native completion selector stat not recorded or comparison across independent trajectories is unjustified, mark `N/A`, use explicitly named *new terminal residual diagnostic*. Negative results are valuable scientific evidence.
- Supplementary `SCORE_LEAKAGE_AUDIT.md`: confirm every selected index and score uses only (saved measurement, saved candidates, frozen non-GT scores, source config), **not** GT/PSNR/failure labels; GT loaded only in post-selection evaluation branch.

## 6. Outputs, checkpoints and quality-control
After a valid user-run CPU stage, output **new** `a13_1_cpu_forensics_v1` files under A1.3 namespace, with exact source SHA and manifest/digest:
- `A13_SOURCE_AND_INPUT_AUDIT.json`, `A13_SOURCE_AND_INPUT_AUDIT.md`: stage0 checks, all 128 full terminal byte digests and eight measurement/GT receipts, source-operator equivalence test, N/A register.
- `A13_CANDIDATE_METRICS.csv`: 128 rows with scores/PSNR/cost, no unlisted images or raw data.
- `A13_K_POLICY_ROWS.csv` and `A13_SUMMARY.csv`: per-image, method, budget, score type, GT oracle vs frozen GT-free selected, repeatable tables.
- `A13_FAILURE_TAXONOMY.csv`, `A13_SCORE_RANKING.csv`, `A13_EXAMPLE_CASES.md`: classifications and counterexamples; no cherry-picked evidence to hide negative results.
- `A13_ORACLE_REPLAY_CHECKS.json` verifying all historical A1.2 primary oracles/per-image prefixes; synthetic unit tests and full stdout/stderr.
- `A13_UNSUPPORTED_CELLS.md`, `A13_SCORE_LEAKAGE_AUDIT.md`, `A13_CPU_RESOURCE.json` with elapsed CPU wall/cores and physical inference=0; standalone source/checksum ledger.
- Optional concise max ~16-image contact sheet using existing saved PNG8 outputs with explicit method/rep/raw PSNR labels, only for already exposed eight, if useful; **avoid automatically uploading GT tensors or reconstruction bytes**. Main ZIP should be compact **metadata-only**, sha256 printed and verified on PAC; user returns it for independent scientific review.
- Final commit `docs/track_a/executors/a13-candidate-audit/A13_EXECUTOR_RETURN.md` or `A13_STOP_REPORT.md`, source SHA, rows/tests, numerical results, limitations, no held-out claims, paths and zip checksum, next-stage GO/STOP. Post final return link to issue #41 and inform Track A lead.

**STOP if** missing/checksum-drifted any saved candidate, operator mismatch, GT/original A1.2 metric replay fails, CPU script attempts GPU or model import, old A1 output would be modified, or any request for B24 confirmation305 membership arises. Do not substitute missing data, silently drop worst candidates, re-run diffusion, automatically fit a selector, or ask user to run unsafe wide scans.

## 7. Governance and the next larger stage
A1.3 does **not** authorize a distribution-shift GPU pilot. Prior mismatch novelty is an open research question: Hu et al. (2024) and broader OOD diffusion inverse literature make generic OOD stress testing incremental unless it leads to distinctive phase-retrieval reliability insight. The planner—not this executor—will decide A1.4 after seeing A1.3. If user favors parallel desk research, a separate literature-only subexecutor may be commissioned, without PAC access.

Track B independently owns new basin transitions/diversification. You may classify frozen terminals as 'measurement-consistent alternate solution' descriptively but cannot alter candidate generation, create phase transitions, or implement Track B geometry experiments. All code/docs remain in `scripts/track_a/a13_*`, `configs/track_a/a13_*`, `docs/track_a/executors/a13-candidate-audit/`. Do NOT fast-forward the dedicated historical A1 source worktree to your branch.

Shared #41 record is mandatory before user-run CPU stage (EXECUTOR status), and separately if a GPU run ever proposed (not in scope). PAC physical GPU3 from A1.2 is **RELEASED**. Research lead returns signed scientific conclusions from this executor; no automated background execution is possible.

## 8. Your FIRST reply in the new executor chat
Confirm you read the pinned full A1 report, artifact index, A1.2 128-candidate audit, this handoff, scientific decision and live ledger. Explain source/metric distinctions succinctly. Before any actual CPU analysis, prepare commit of **safe limited read-only PAC inventory** and CPU FFT/selector tests, with **one explicit user-executed command block** that never exits the interactive shell on nonzero; do not assume paths exist. Wait for returned PAC inventory evidence before declaring source/data PASS. After it passes, prepare a separate analysis command/ZIP with nonoverwrite output and CPU-only guards. No GPU/PAC commands are being run by this handoff.
