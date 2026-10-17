# A1 executor handoff — paper-style best-of-K and complete compute cost, prospective-preparation
Date: 2026-10-16
Role: dedicated Track A **A1 oracle-cost/prospective-protocol engineering executor**
GitHub repository: https://github.com/Cheng-HanHuang/pr_diffusion
**ONLY branch:** `codex/prdiff-track-a-a1-oracle-budget-20261016`
**Source parent commit:** `997195350f353e07aab58ce12693e9cdde80b52a`
**Initial stage:** scientific design, provenance/compute audit, small test code + metadata-only analyses. **Zero GPUs and no actual new measurements at handoff.** Only the user can run a clearly disclosed PAC stage.

## 1. First read, methodological correction and restrictions
Read `AGENTS.md`, `docs/planning/metaplanner/2026-10-15_AUTONOMY_AMENDMENT.md`, the live [coordination issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41), `docs/planning/track_a/2026-10-16_ORACLE_FIRST_PUBLICATION_REFRAME.md`, the initial Track A return, and:
- A0 complete report `docs/track_a/executors/a0-native-selector-audit/A0_B22_RETROSPECTIVE_RESULTS_2026-10-16.md` on its own branch at `ecc66b890b63123c0445a13d33277e09f6ab4ab5`.
- Machine receipt `docs/track_a/executors/a0-native-selector-audit/A0_B22_RESULTS_RECEIPT.json`.
- `docs/b21/b21_11_fresh2_final_benchmark.md`, `docs/b22/b22_3_scientific_closeout.md`, `docs/b24/B24_3_ZERO_GPU_CLOSEOUT_RESULT.md`, `docs/b25/B25_FINAL_PLANNER_RETURN.md`; B26 completed return read-only.
- DAPS paper arXiv HTML v3 §4.1 `https://arxiv.org/html/2407.01521v3`; SITCOM paper HTML v2 Appendix G `https://arxiv.org/html/2410.04479v2`. Both report PR best-of-four **independent** runs; SITCOM paper's Table 1 PR cost is one-run runtime although quality is best-of-four.

**Corrected mission:** primary scientific endpoint **offline GT-PSNR best-of-K candidate generation**, reliability tail, and *total charged compute*, exactly as a paper-style evaluation (but always marked non-deployable). Native clean-free selected results and selector regret are **secondary, equally honest** views, not publication gates. This explicit distinction is essential. NEVER assert that best-of-K is an executable algorithm.

Do not edit/call/clean/source-patch B19–B26 historical branches, B24 original outputs, A0 branch, or Track B. No B24 confirmation305 registry, payload, paths, IDs or data access. B22/B21 final100 and B24 DEV80 are **exposed**, not pristine independent validation; B24's historical SITCOM source is still unreconciled, so do not silently use it as an externally validated fresh baseline.

## 2. A1.0 first deliverable: fully specified prospective scientific design, **NO PAC GPU**
Before executing new image runs:
1. Use verified B22 A0 report + optionally user-returned **archived metadata only** (not reconstruction tensors/GT recomputation) to complete an **oracle-best-of-K** comparative evidence table: Good25, mean, median, q05, minimum, Bad20, count-and-compute axes, paired image-bootstrap 95% CIs and McNemar for *oracle* NP8 vs DAPS2, NP8 vs SITCOM4 and K4 S2 vs K4 SITCOM, where columns actually exist. Absent fields -> **N/A** with exact data requirement. Avoid repeating A0 unless needed to compute a clearly documented missing statistic. No policy tuning on B22 using GT.
2. Audit native source/checkpoint/preprocessing **for the future prospective run**, not B24 dirty-state repair. Keep B22 verified SITCOM-ODE implementation and native DAPS pinned, preserve signed raw observations for DAPS/SITCOM and NP's historical clipped observation explicitly as a method identity; don't silently change NP. Freeze orientation/range/8bit conventions to one preregistered score for all methods.
3. Design paper-style independent K4 comparison:
   - Native DAPS **4 independent restarts** (does not exist in B22);
   - Native SITCOM-ODE **4 independent restarts** (B22 four are sequential RNG, not independently seeded);
   - NP LF-only4, S2-only4, and optionally a frozen **mixed 2LF+2S2** (the 4LF+4S2 NP8 portfolio is a different 8-candidate algorithm; no retrospective GT-tuned allocation);
   - K1 and K2 intermediate prefixes with consistent seed treatment;
   - optional SITCOM-ODE K8 vs NP8 to probe ~400s total candidate budget, and DAPS2/4 where actual candidate costs allow. Avoid declaring unsupported K as an observed result.
4. Report **two independent comparison axes**, run-matched K and **charged multi-run GPU time**. Quantify GPU-active/worker-wall/model load/concurrency, native UNet forwards/backwards and operator/FFT where support exists; no fabricated FLOP equivalence or “best-of-four at 1-run cost”. Illustrative B22 per-image *candidate reconstruction sums*: DAPS2 312.64s, SITCOM4 196.07s, NP S2-only4 206.46s, NP mixed8 411.78s. These are **not** full calibrated hardware-neutral costs. Preflight measured actual cost on new run and update budget. Don't use wall-time projections as results.
5. Before real-image cohort decisions, check issue #41 *live*. Prepare a **new, disjoint development cohort** and seed namespace, clearly distinct from untouched final cohort. Prior FFHQ dataset split availability must be verified without opening prohibited B24 confirmation registry. No user-independent final-test cohort selection by GT quality; no reusing B22 or B24 DEV80 as fresh. Track B ownership and contested compute priority must be honored.
6. Prepare a **bounded development smoke** proposal only (e.g. 8–12 disjoint images, frozen small experiment matrix) including expected physical-GPU IDs from preflight, per-GPU maximum memory and wall ceilings, total GPU-hours, source/model SHA, exact commands the user runs, failure receipts, rollback, resume, per-candidate PSNR after run and predeclared stop. The executor is not permitted to start a GPU run itself. Large final prospective run still requires joint cohort governance.
7. Predeclare a realistic effect/precision threshold, primary oracle Good25 plus paired mean and q05/Bad20, and source-bounded novelty claim. Never post-select which comparison is primary after GT exposure or celebrate +1/100 as a robust population improvement.

## 3. Current verified facts (do not conflate)
- B22 DAPS2 oracle: **92/100**, mean **29.427249 dB**, 312.64 s/image. Missing DAPS4.
- B22 SITCOM4 sequential oracle: **95/100**, mean **26.666889 dB**, 196.07 s/image.
- B22 NP4 S2-only oracle: **94/100**, mean **unknown in committed summary**, 206.461 s/image, native clean-free 94/100 retrospective.
- B22 NP mixed8 (LF4+S2-4) oracle **96/100**, mean **29.407266 dB**, 411.783 s/image; native95/100.
- These 100 images and observations match; **method K and costs do not**. NP mixed8 +1 vs SITCOM4 oracle, +4 vs DAPS2 oracle, mean NP8 vs DAPS2 -0.019983 dB; neither numerical contrast alone establishes prospective superiority. DAPS4 and SITCOM4 independent are crucial.
- For the GT-oracle 100-image panel, NP8's >=96 Good25 places rank5/rank6 PSNR >=25; DAPS2's 92 puts rank5/rank6 <25, so fifth-percentile threshold ordering is directionally identified, but exact q05 magnitude, paired uncertainty and min are **not** proven from the committed summary.
- The historical NP signed-measurement correction B26 had Good25 53/80 vs original H 55/80 on exposed DEV80: negative outcome, do not revive.
- Cross-family GT-best-of-14 99/100 is a *diagnostic bound*, no deployable cross-method ensemble. All 14 fail image65003.

## 4. Source/output namespace, staging and receipts
Only write code in `scripts/track_a/a1_*`, `configs/track_a/a1_*` and docs in `docs/track_a/executors/a1-oracle-budget/` or `docs/planning/track_a/` if requested. No main/Track B/A0 edits. Branch commits must be pinned before the user performs ANY PAC stage.

**Reserved, must check not in use**: PAC worktree `/egr/research-pac/huang248/pr_diffusion_track_a_a1_oracle_budget`; output root `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/`. No PAC directories presumed to exist. Use environments solely under `/egr/research-pac/huang248/conda-envs`; do not use `/home`. Never use `set -e`, `set -u`, `set -o pipefail`, `exit`, `logout`, broad `pkill`, destructive Git, broad recursive storage scan. User executes all PAC commands and returns actual receipts.

Produce first **A1_STUDY_PREREGISTRATION.md** with exact sources, candidate/seed matrix, primary estimands, costs, source gate, sample/GT exposure gate, a bounded smoke/STOP budget and constraints. Any simulator/tests initially synthetic and CPU only, or source reading. Before actual run log [DATA RESERVATION] and [RUN RESERVATION] with source commit, accepted cohort, GPU IDs/time/memory/stop in issue #41, then user chooses to execute disclosed commands. No hidden action.

Final **A1_EXECUTOR_RETURN.md** after a staged result, or **A1_STOP_REPORT.md** if prereg fails; commit to GitHub and register in #41.

**First reply in dedicated executor chat:** read/report protocol and literature as described, draft A1 preregistration before any GPU commands or runs. Do not request confirmation305; do not claim a held-out cohort has already been verified. Critically evaluate whether this program has enough novelty for smaller-venue publication even if best-of-K wins at equal compute.
