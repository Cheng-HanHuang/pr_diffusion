# Track R · R0-N executor handoff: letter evidence package

**From:** planner (Claude), 2026-10-23
**To:** the executor that ran R0 (no PAC access; Cheng runs every PAC block)

**Binding documents:**
- decision: `docs/planning/track_r/2026-10-23_R0_PLANNER_DECISION.md`;
- analysis config: `configs/track_r/r0n_letter_analysis_v1.json` (FROZEN v1);
- letter plan: `docs/planning/track_r/LETTER_PLAN.md`.

**Amendment:** also read [`R0N_AMENDMENT_1_20261023.md`](R0N_AMENDMENT_1_20261023.md). It adds the N0 checksum list, the N1h training-membership check and N1b/N1g reproduction targets.

**Unchanged rules:** the R0 frozen spec and every R0 rule in `docs/track_r/executors/r0/00_EXECUTOR_HANDOFF.md` §5–§8 still apply: data rules, cost accounting, PAC command style, pitfalls.

---

## 0. Planner's view for Cheng (snapshot)

- **Results.** R0 closed the method question.
  - Exact-orbit errors are rare (7.1% of single-run failures; FAST 3.6%).
  - The prior is not a good general failure detector; the free signed residual is far better for DAPS and FAST (AUROC ≈ 0.985).
  - But the prior fixes orientation almost perfectly: 84/85 orbit errors fixed (S_mid 85/85), 0/2,040 good outputs harmed. UnivFD fixes 62/85 and harms 10.8% of good outputs.
  - After residual selection plus repair, the deployable result is within 0–2 images of the GT oracle everywhere; what remains are generation failures.
  - Failure modes are solver-specific: DAPS and FAST mostly don't fit y; SITCOM's wrong outputs fit y.
- **Plan.** This package (CPU only, plus one optional tiny GPU baseline) produces the letter's evidence. The planner then drafts the letter, and the project winds down.
- **Publication image.** "What best-of-K hides in diffusion phase retrieval: failure anatomy and exact-symmetry repair." An IEEE SPL letter (4 + 1 pages) with:
  - anatomy across four families;
  - the posterior-ratio = prior-ratio proposition with a paired estimator;
  - repair against UnivFD and CLIP zero-shot;
  - deployable versus oracle tables;
  - recommendations.

  Odds about 30–40% at SPL; EUSIPCO/ICIP 2027 as fallback.

---

## 1. Scope and limits

**Authorized:**
- CPU-only re-analysis of immutable R0 artifacts.
- Optional N2: CLIP zero-shot orientation baseline, at most 0.25 GPU-hours, or run on CPU.
- N3: a bounded ImageNet asset inventory.

**Not authorized:**
- new reconstructions or measurements;
- any R1 work;
- changing R0 v1 definitions or editing R0.x outputs;
- any access to B24 confirmation305;
- merging into `main`.

**Stop and escalate on:** replay mismatch (kill criterion K3); a cohort id overlap; missing per-draw losses; a GPU projection above 0.25 hours.

**Branch:**
1. Create `codex/prdiff-track-r-r0n-<YYYYMMDD>` from `codex/prdiff-track-r-r0-20261021` @ `c8eb608`.
2. Run `git merge --no-ff origin/claude/prdiff-track-r-planner-20261021` to bring in the planner docs. This should be conflict-free, because the planner-owned files are unchanged on the executor branch.
3. Record both SHAs in every return.

**PAC worktree:** reuse `/egr/research-pac/huang248/pr_diffusion_track_r`; detach to each pushed SHA, clean tree.
**Interpreter:** `prdiff_track_r`.
**Outputs:** `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0n_<stage>_<UTCSTAMP>/`.

## 2. Tasks

### N0 — commit the R0.8 tables (CPU; no PAC needed if Cheng gives you the ZIP)

1. From `R0_8_PLANNER_RETURN.zip` (SHA-256 `13c008f3…efab`), commit the sanitized aggregate files under `docs/track_r/executors/r0/returns/R0_20261023/`:
   - `R0_7_ANALYSIS.json`, `R0_7_REPORT.md`;
   - `R0_8_P4_MATRIX.csv`;
   - the R0.8 audit JSON;
   - the path-free casebook;
   - `SHA256SUMS.txt`.
2. No GT paths and no per-run GT JSONL.
3. Verify each file's SHA against the ZIP manifest.

### N1 — letter tables (one CPU block on PAC; read-only inputs)

**Script:** `scripts/track_r/r0n_1_letter_tables_cpu.py`, plus a wrapper.

**Inputs:**
- R0.7 evaluated rows;
- R0.4/R0.6 per-candidate score receipts (via the paths enumerated in the R0.7 preparation audit);
- R0.5 FAST receipts.

**Start-up checks:**
- Assert `CUDA_VISIBLE_DEVICES=""`.
- Assert the forbidden-token rule.
- Before any new analysis, replay the R0.7 headline numbers: 3,240 rows; label counts 2,040/85/324/791; repair 84/85 and 0/2,040; the 7 representative P4 Good25 rows in the return §4. Any mismatch → STOP (K3).

**Cohorts:** as defined in config `cohorts`. Assert B22_random ∩ DEV80_enriched = ∅.

Produce:

- **N1a — anatomy by cohort × family.** Counts of GOOD, ORBIT (global / partial), CONSISTENT and INCONSISTENT; single-run failure rate; orbit share of failures with an image-cluster CI.
- **N1b — post-selection decomposition.** For every frozen (source, family, K) policy table:
  - E0 failures split into:
    - orbit errors fixed by E1;
    - orbit errors not fixed;
    - selection misses, i.e. a non-orbit E0 pick while some candidate among the first K has orbit-best PSNR ≥ 25;
    - generation failures, i.e. no candidate reaches 25 even after an orbit change.
  - Also O_raw, O_orbit, E0, E1, E2 Good25 / mean / q05, plus O_raw − E1.
  - Report by cohort where a source spans both (S5).
- **N1c — residual distributions.** Per family × cohort × label: n; median, q10 and q90 of ρ_signed and ρ_clipped; fraction of GOOD runs with ρ_signed < 0.05. Explain SITCOM's AUROC of 0.68 with these numbers only.
- **N1d — repair cost–accuracy grid.** Exactly per config `n1d_cost_accuracy`:
  - 60 configurations;
  - dev/eval split by cohort;
  - pre-registered selection rule applied on dev only;
  - the selected configuration evaluated on eval;
  - the paired-variance factor.

  Everything other than the selected-configuration evaluation is labeled EXPLORATORY.
- **N1e — costs.**
  - FAST per root: CUDA-event sampling-only and wall seconds (R0.5 receipts).
  - Scoring wall seconds per candidate (score receipts).
  - Projected scoring seconds for the N1d-selected configuration: measured forwards per second × forwards.
  - Historical per-run DAPS/SITCOM/NP cost only where already recorded; otherwise N/A.
- **N1f — conditional detection, all families** (EXPLORATORY). AUROC of prior S, S_img, p_fake, ρ_signed and ρ_clipped for measurement-consistent failures against post-repair successes, for each family including SITCOM, NP, NP_LF and NP_S2. Give n.
- **N1g — letter Table 2.** For each family at its maximal K per source: O_raw, O_orbit, E0, E1 (Good25, mean, q05) for the B22 cohort and the DEV80 cohort separately.

**Outputs:**
- `R0N_LETTER_EVIDENCE.json` (every number with n and CI);
- one CSV per table;
- draft figures as PDF and PNG:
  - Fig. 1: anatomy stacked bars;
  - Fig. 2: success versus forwards, with harm marked;
  - Fig. 3: residual violins per family.
- a numbers-only `R0N_REPORT.md`.

Commit the compact outputs under `docs/track_r/executors/r0n/returns/R0N_<UTCSTAMP>/`.

### N2 — CLIP zero-shot orientation baseline (optional, recommended; ≤0.25 GPU-h or CPU)

**Script:** `scripts/track_r/r0n_2_clip_orientation.py`.

**Rules:**
- Load CLIP ViT-L/14 from the local `ViT-L-14.pt` through the UnivFD repo's `models/clip` module. Verify the SHA.
- Use exactly the prompts, preprocessing, score and decision in config `n2_clip_zero_shot`. **No prompt tuning.**
- The scoring step reads only candidate PNGs (2 encodings per candidate: identity and global rotation). No GT.

**Then, on CPU:** join the decisions with the R0.7 labels and report:
- success on GLOBAL orbit errors;
- harm on GOOD;
- the partial orbit error left unfixed;

each for CLIP, prior global-only, prior S_mid global-only and UnivFD global-only. Include image-cluster CIs.

**GPU block:** one GPU, `nvidia-smi` check first, `nohup` with PID, status and stop. Expected runtime is a few minutes.

### N3 — ImageNet asset inventory (CPU, bounded)

Run exactly the bounded checks in config `n3_imagenet_inventory`. Record presence and sizes only.

## 3. Return (`R0N_EXECUTOR_RETURN.md`)

1. **Status for Cheng** — at most 8 plain-language lines.
2. **Identity** — branch, SHAs and interpreter.
3. **Replay check (K3)** — PASS or FAIL, with numbers.
4. **Tables N1a–N1g and N2** — numbers only, each with n and CI.
5. **Kill-criterion inputs**, computed mechanically:
   - **K1:** post-selection fixes on the B22 cohort, per family.
   - **K2:** CLIP versus prior global-only on GLOBAL success and GOOD harm, plus cost per candidate.
6. **Anomalies and deviations.**
7. **Artifacts** — PAC roots, committed outputs and SHAs.
8. **Executor opinion** — optional and clearly separated.

The planner then rules on K1–K2 and starts the letter draft.
