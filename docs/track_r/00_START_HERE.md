# Track R — start here

**Updated:** 2026-11-06 by the planner (Claude).
- After PC-DAPS (arXiv 2610.31592): see the positioning note (document 5).
- After the N2 CLIP result: K2 triggered, so the paper is an audit (decision §10).
- **2026-10-25:** a top-venue attempt (CVPR 2027) with theory plus extended experiments. EUSIPCO 2027 and SPL are the fallbacks (document 1).
- **2026-10-26:** Cheng's decisions, recorded in the update box of document 1:
  - **GO XP, aggressively.** `XP_PREREG_v1_1.md` supersedes v1: tiers A/B/C, 400 GPU-h cap;
  - CVPR;
  - the theory executor starts now (handoff revision B; program v1.1; literature digest);
  - the A8 label strings may be committed (amendment 2 revision D.1).
- **2026-10-26, later:**
  - **XP-E is cancelled.** DAPS `test-ffhq` overlaps the locked holdout by one id.
  - **Hard rule:** never load any bundled FFHQ set (`test-ffhq`, `demo-ffhq`, DAPS FFHQ archives). Never merge `main` or the data-governance audit branch into Track R branches.
  - TGD has been read in full, and the research pass on phase-retrieval theory is digested. This produced prereg amendment A1, theory program v1.2 and handoffs B.1.

- **2026-11-06, morning (planner): third AI review (Astra, weak reject) answered; XP-NF proposed (prereg A26).**
  - **Review:** found real overclaims. These are fixed in the main text, still 8 pages:
    - all six forecast tests now carry interval scores (race 30 vs S2 182; loses tests 1 and 4);
    - Prop. 1 is called a decomposition;
    - stopping claims are scoped to fresh runs of known images;
    - H3 is stated as registered on three panels only;
    - κ notation; noise identity as motivation; "beats the oracle" replaced by 80% recovery of the orbit-oracle gap.
  - **New planner checks (C-R8, C-V4, C-V5):** DAPS stopping needs τ_S within about 1% (at −2% it never stops; at +2% it keeps +0.7 to +1.9); labeled, label-free and cross-cohort thresholds all keep +2.6 to +5.0.
  - **Decision for Cheng:** XP-NF, the frozen protocol for DAPS-2K/4K on 100 new FFHQ validation images (ceiling 50 GPU-h). Needs GO, a holdout-disjoint manifest that only Cheng can check, and the order vs XP-K.
  - **Triage:** [`reviews/AI_REVIEW_3_TRIAGE_20261106.md`](reviews/AI_REVIEW_3_TRIAGE_20261106.md). **Notes, round 23:** [`executors/xp/23_PLANNER_NOTES_20261106_AM.md`](executors/xp/23_PLANNER_NOTES_20261106_AM.md), [`executors/theory/23_PLANNER_GUIDANCE_20261106_AM.md`](executors/theory/23_PLANNER_GUIDANCE_20261106_AM.md).
- **2026-11-06 (planner): two AI reviews answered (prereg A25).**
  - **Reviews:** R1 (ICLR scale, 7.1, accept) and R2 (CVPR, positive). Texts and triage are in [`reviews/`](reviews/AI_REVIEWS_20261105_TRIAGE.md). Every item is classified as writing, (1) out of scope, or (2) more experiments.
  - **Main text (still 8 pages):** threshold robustness, run exchangeability, compute overhead, label-free τ robustness, the likelihood argument for consistent-but-wrong runs, orientation scorers, cohort B / ImageNet validation images, other operators, the tail-competition reading of the race, and related work (FlowSGS, DDRM-PR, I2I-PR, DPnP, adaptive test-time compute; "FlowFPR" could not be found).
  - **New planner checks on committed data (C-V1 to C-V3):** θ sensitivity (SITCOM, PC-DAPS); label-free τ vs pilot size; no run-order dependence (p 0.17–0.69).
  - **Supplement:** 24 pages, new section "Sensitivity checks and scope".
  - **Queued after CP4 (A25):** X1 per-run metrics export, X2 compute table, X3 band residuals (executor); P1–P4 (planner).
  - **Notes, round 22:** [`executors/xp/22_PLANNER_NOTES_20261106.md`](executors/xp/22_PLANNER_NOTES_20261106.md) and [`executors/theory/22_PLANNER_GUIDANCE_20261106.md`](executors/theory/22_PLANNER_GUIDANCE_20261106.md).
- **2026-11-05 (planner): writing pass and defect sweep of the main text (Cheng's request).**
  - **Main text:** short sentences; strengths first; unsolved images framed by the ill-posedness of phase retrieval; SITCOM's consistent-but-wrong runs outside the scope of selection. Still 8 pages (references on pages 8–9); abstract 250 words.
  - **Defects fixed:** references completed (pages added where they exist); notation explicit (ground truth x⋆, noise η = σn, orbit index b, race ceiling with partial orbits tied to w_O, the tighter-prior variant); internal names removed (bag, first image set, new images, weaker-prior, clean GPU); "points" vs "forecasts" disambiguated.
  - **New planner checks (ledger):** the DAPS-4K protocol panels used the label-free threshold (C-G9); PC-DAPS ImageNet gap 2.0 points, all orientation (C-E1).
  - **Supplement:** 22 pages; new section on the selection protocols of recent work; planner-owned sections renamed; theory cleans S1–S3 by Nov 12.
  - **Clean review build:** `pdflatex -jobname=main_clean "\def\cleancopy{}\input{main}"` (claim tags hidden).
  - **Notes, round 21:** [`executors/xp/21_PLANNER_NOTES_20261105.md`](executors/xp/21_PLANNER_NOTES_20261105.md) and [`executors/theory/21_PLANNER_GUIDANCE_20261105.md`](executors/theory/21_PLANNER_GUIDANCE_20261105.md).
- **2026-11-04, 22:12 (planner): G7 complete and accepted (A24).**
  - **Results:** every orientation test on the 280 ground-truth images. S_mid keeps 95/100 ImageNet upright, 2-forward 93, CLIP 92, UnivFD 89; all keep all 180 FFHQ except UnivFD (172). S2 has a new table.
  - **Theory's S2 wording** for the equivariance-defect sentences is adopted.
  - **Remaining:** XP-K (gated) and CP4 (Nov 15). The main-paper cut stays closed.
  - **Notes, round 20:** [`executors/xp/20_PLANNER_NOTES_20261104_PM.md`](executors/xp/20_PLANNER_NOTES_20261104_PM.md) and [`executors/theory/20_PLANNER_GUIDANCE_20261104_PM.md`](executors/theory/20_PLANNER_GUIDANCE_20261104_PM.md).
- **2026-11-04, 00:16 (planner): Tab. 3 E1 and G7 C2/CLIP accepted (A23). The main-paper data cut is met.**
  - **Tab. 3:** without the ground truth, DAPS-4K's E1 with four runs beats the published per-metric ground-truth oracle on every metric on ImageNet (23.10 dB / 0.534 / 0.325 vs 22.28 / 0.518 / 0.340). This is in Sec. 6.
  - **C2:** the ImageNet denoiser is nearly equivariant to the global flip (relative defect 0.015–0.052, against 0.10–0.31 on FFHQ). This is in S2, with the N2 prompts and ground-truth CLIP counts.
  - **Theory:** S3 `ecea698` integrated; the main-text wording for the prior test is adopted.
  - **Remaining, supplement only:** G7 prior/UnivFD on ground-truth images (ceiling 10 GPU-h), XP-K with three gates (30 GPU-h), CP4 by Nov 15.
  - **Notes, round 19:** [`executors/xp/19_PLANNER_NOTES_20261104.md`](executors/xp/19_PLANNER_NOTES_20261104.md) and [`executors/theory/19_PLANNER_GUIDANCE_20261104.md`](executors/theory/19_PLANNER_GUIDANCE_20261104.md).
- **2026-11-03, 17:07 (planner): XP-C3 and XP-I2 complete, reproduced and in the paper (A22).**
  - **XP-C3:** 13,600 candidates in 9.75 GPU-occupancy hours.
    - On the new images, the prior's own orientation test beats generic CLIP by 5 points (K = 16: +5.3 [+1.1, +9.8]).
    - On the first image set the two tests are equal within 1.5 points.
    - The controller stop was an engineering incident (late signal); all panels are complete.
  - **XP-I2:** PC-DAPS on ImageNet, 30.75% → 53% GOOD by selecting among the same particles ([+14.8, +30.0]). Now in the abstract.
  - **Paper:** still 8 pages after trims. Supplement 21 pages, with S3 from theory `2619eec` and two stale S2 sentences resolved.
  - **Notes, round 18:** [`executors/xp/18_PLANNER_NOTES_20261103_PM.md`](executors/xp/18_PLANNER_NOTES_20261103_PM.md) and [`executors/theory/18_PLANNER_GUIDANCE_20261103_PM.md`](executors/theory/18_PLANNER_GUIDANCE_20261103_PM.md).
- **2026-11-03, 12:34 (planner): XP-C3 compute approved (A21).**
  - Ceiling 56 GPU-h; GPUs 0–3; up to 8 workers per GPU; batching at fixed t, with an equivalence test fixed in advance; 8 GiB of free memory kept per GPU; GPU-occupancy-hour accounting; skip rule so G5 can run when XP-N cannot.
  - Main text if returned by Nov 6, 12:00 ET; hard stop Nov 15.
  - XP-I2 scoring requested for Nov 5, 12:00 ET.
  - **Nov 4 outputs reproduced** (A9.2(d) exact; 120/120 comparisons). Test 1's E1 points: race 3/21, S2 21/21; selection-only check shows the miss is generation. Tab. 2 row 1 is now 9/42 vs 36/42.
  - **Fig. 3** built: one unsolved image per dataset. Supplement S6 shows all 15 (20 pages). On FFHQ, 5 of 7 unsolved images are near misses.
  - **Notes, round 17:** [`executors/xp/17_PLANNER_NOTES_20261103.md`](executors/xp/17_PLANNER_NOTES_20261103.md) and [`executors/theory/17_PLANNER_GUIDANCE_20261103.md`](executors/theory/17_PLANNER_GUIDANCE_20261103.md).
- **2026-11-02, 16:50 (planner): XP-C3 defined (A20: S_mid global2, ceiling 8 GPU-h); theory round 15 integrated (supplement 17 pages).**
  - The Nov 4 CPU items are finished on PAC; publication is requested.
  - **Notes, round 16:** [`executors/xp/16_PLANNER_NOTES_20261102_PM.md`](executors/xp/16_PLANNER_NOTES_20261102_PM.md) and [`executors/theory/16_PLANNER_GUIDANCE_20261102_PM.md`](executors/theory/16_PLANNER_GUIDANCE_20261102_PM.md).
- **2026-11-02, 03:01 (planner): XP-N complete and reproduced; A19.**
  - **Test 6 (new images):** race 10/10 vs S2 0/10; coverage overforecast again.
  - **Protocol:** N-H1a **fails** (−1.5 at N_max = 16); N-H1b holds (H1 fails at 64, −3.1, as registered).
  - **Paper:** updated, still 8 pages; supplement 16 pages.
  - **Next:** XP-C3, extended to the XP-N candidates, then XP-I2. No new selector before submission.
  - **Notes, round 15:** [`executors/xp/15_PLANNER_NOTES_20261102.md`](executors/xp/15_PLANNER_NOTES_20261102.md) and [`executors/theory/15_PLANNER_GUIDANCE_20261102.md`](executors/theory/15_PLANNER_GUIDANCE_20261102.md).
- **2026-11-01, 19:15 (planner): v5 registered (bit-identical); XP-N cap 28 GPU-h; A18.**
  - **v5** (XP `55ba86b`) was pushed before any held-run scoring; the planner rerun is bit-identical. Runs 4–63 are unsealed.
  - **XP-N** stopped at 72/90 chunks on a reservation artifact. The cap is raised to 28 GPU-h. An incomplete-panel rule is declared. CP-N part 1 (n = 16) is due now and by Nov 5; part 2 (n = 64 and v5) by Nov 8.
  - **XP-I** is reproduced in full (E1 PSNR/SSIM/LPIPS and intervals).
  - **Notes, round 14:** [`executors/xp/14_PLANNER_NOTES_20261101.md`](executors/xp/14_PLANNER_NOTES_20261101.md) and [`executors/theory/14_PLANNER_GUIDANCE_20261101.md`](executors/theory/14_PLANNER_GUIDANCE_20261101.md).
- **2026-10-31, 20:45 (planner): v5 code registered; A17; XP-N generation may start.**
  - **Theory round 12:** v5 code frozen before any XP-N run (`81003cc`, receipt `3ad5f2d`); XP-N pre-score registered (`a1ea26b`); S1–S3 edits adopted.
  - **A17:** the XP executor runs v5 from the frozen commit and registers it by push before scoring runs ≥ 4. Planner native QA of v5 is in `planner_analysis/A17_v5_qa/`.
  - **Notes, round 13:** [`executors/xp/13_PLANNER_NOTES_20261031_NIGHT.md`](executors/xp/13_PLANNER_NOTES_20261031_NIGHT.md) and [`executors/theory/13_PLANNER_GUIDANCE_20261031_NIGHT.md`](executors/theory/13_PLANNER_GUIDANCE_20261031_NIGHT.md).
- **2026-10-31, 14:24 (planner): CP3 accepted (reproduction exact); A16.**
  - **A16:** post-CP3 exploratory analyses (`planner_analysis/A16_cp3_and_exploratory/`); XP-N image-disjoint ImageNet panel, conditional on Cheng's GO, with hypotheses and planner forecasts registered at `21babef`; XP-C3 prior orientation test.
  - **Paper:** Table 1 complete with gain intervals; selection-first framing; supplement 15 pages with planner S4.
  - **Notes, round 12:** [`executors/xp/12_PLANNER_NOTES_20261031_PM.md`](executors/xp/12_PLANNER_NOTES_20261031_PM.md) and [`executors/theory/12_PLANNER_GUIDANCE_20261031_PM.md`](executors/theory/12_PLANNER_GUIDANCE_20261031_PM.md).
- **2026-10-31, 01:20 (planner): CP2 accepted (H1 fails at N_max 32/64 on ImageNet; v4 selection 18/18, coverage 0/6); CP3 option 1 (XP-I descriptive); A15.**
  - **Paper:** claims rebuilt around "selection predictable, coverage not"; five-test table and calibration figure.
  - **Supplement:** first build `docs/paper/track_r_cvpr27/supp/` (S1–S3 from theory `061c618`).
  - **Notes, round 11:** [`executors/xp/11_PLANNER_NOTES_20261031.md`](executors/xp/11_PLANNER_NOTES_20261031.md) and [`executors/theory/11_PLANNER_GUIDANCE_20261031.md`](executors/theory/11_PLANNER_GUIDANCE_20261031.md).
- **2026-10-29, 21:45 (planner): CP1 accepted (H2 passes on DAPS, +3.4 points); A14.**
  - **Protocol held-out:** H1 3/3; H2 pass (as the theory pre-score note forecast); H3 fail ×2; H4 pass. F8 is reproduced exactly.
  - **A14:** G2 production-mode pilot, G2 ceiling 25 GPU-h; PC-DAPS clone by Cheng (G3 to Nov 4).
  - **Notes, round 10:** [`executors/xp/10_PLANNER_NOTES_20261029_NIGHT.md`](executors/xp/10_PLANNER_NOTES_20261029_NIGHT.md) and [`executors/theory/10_PLANNER_GUIDANCE_20261029_NIGHT.md`](executors/theory/10_PLANNER_GUIDANCE_20261029_NIGHT.md).
- **2026-10-29, 16:05 (planner): autopilot (A13), the first held-out results, v4 registered.**
  - **D1-P2 under the protocol:** H1 pass, H3 fail, H4 pass (planner reproduction bit-identical). **B16 vs v1:** race 18/18 selection points, S2 2/18. B16 under the protocol: H1 pass (planner run).
  - **Prereg A13:** the executor runs on autopilot with checkpoints CP1–CP4 only. **v4 registered** (theory `149331c`, SHA `b0ff25bc…`); XP-D4 unsealed.
  - **Notes, round 9:** [`executors/xp/09_PLANNER_NOTES_20261029_EVE.md`](executors/xp/09_PLANNER_NOTES_20261029_EVE.md) and [`executors/theory/09_PLANNER_GUIDANCE_20261029_EVE.md`](executors/theory/09_PLANNER_GUIDANCE_20261029_EVE.md).
- **2026-10-29, 12:35 (planner): GPU campaign A12 and theory round 7.**
  - **F8 complete** (920/920). Theory round 7 accepted: Proposition 1 is now in Sec. 4, the exact evaluator matches the frozen one within 3e-4, and a counterexample narrows the mechanism claim.
  - **Prereg A12:** a 150 GPU-h campaign inside the unchanged 400 cap. Stages: FAST ImageNet to 32/64 (sealed for an optional theory v4), XP-G, PC-DAPS (conditional), DAPS-4K to 8 runs (FFHQ and ImageNet), clean timing, XP-C.
  - **Notes, round 8:** [`executors/xp/08_PLANNER_NOTES_20261029_PM.md`](executors/xp/08_PLANNER_NOTES_20261029_PM.md) and [`executors/theory/08_PLANNER_GUIDANCE_20261029_PM.md`](executors/theory/08_PLANNER_GUIDANCE_20261029_PM.md).
- **2026-10-29, 03:55 (planner): direction (a) and protocol v1.**
  - **Cheng chose (a):** the paper is a GT-free best-of-K protocol, plus "selection is predictable, coverage is not". Symmetry theorems go to the supplement (prereg **A11**).
  - **Protocol v1 frozen** at `c598176`: [`protocol/GTFREE_PROTOCOL_v1.md`](protocol/GTFREE_PROTOCOL_v1.md), with code and dev results. Registered in the predictions registry under "Frozen protocols". Held-out panels: D1-P2, B16, F8 (H1–H4).
  - **Earlier, 02:45:** v3 registered (theory `e71026e`, planner `356ec2a`); B16/F8 unsealed. D1-P2 result: E1 race 6/6, S2 0/6; oracle overforecast. A10.2 extension: the DAPS lane may run until Nov 1, 12:00 ET.
  - **Paper restructured** (`docs/paper/track_r_cvpr27/`); the plan's ninth update has the revised Nov 8 go/no-go.
  - **Notes, round 7:** [`executors/theory/07_PLANNER_GUIDANCE_20261029.md`](executors/theory/07_PLANNER_GUIDANCE_20261029.md) and [`executors/xp/07_PLANNER_NOTES_20261029.md`](executors/xp/07_PLANNER_NOTES_20261029.md).
- **2026-10-28, 15:30 (planner):**
  - **`v2 E1` frozen and registered** (theory `9cb57f5`, planner `7d1882c`). **D1 phase 2 is unblinded.** B16/F8 stay sealed (A9.4).
  - **FAST decomposition:** the miss is generation. Six of the 7 never-GOOD images are INCONSISTENT in all 64 runs. Given the realized classes, the race fits selection within 1.4 points (race_tau1 within 0.3), while S2 falls 4–8 points short.
  - **Prereg A10:**
    - co-tenancy on all four GPUs for generation, clean GPUs only for timing-measured work (Cheng);
    - the B16 campaign continues through Oct 30, 12:00 ET (70 GPU-h cap with the DAPS lane);
    - D1 phase 2 is compared with v2 and v2 E1.
  - **Notes, round 6:** [`executors/theory/06_PLANNER_GUIDANCE_20261028_PM.md`](executors/theory/06_PLANNER_GUIDANCE_20261028_PM.md) and [`executors/xp/06_PLANNER_NOTES_20261028_PM.md`](executors/xp/06_PLANNER_NOTES_20261028_PM.md).
- **2026-10-28, 06:00 (planner):**
  - **First prospective result:** FAST-DIPS FFHQ against v1 (XP `07d5460`). The race forecast holds on cohort A and fails on cohort B. The oracle is overforecast in both: 7 images never produce a GOOD run in 64, against 2.2 expected. By the frozen rule, S2 scores better on E0 (15 vs 6 of 21), from offsetting errors. v1 is not refitted.
  - **v2 E1 addendum:** the theory code has an anchor error (E1 +0.14–0.34). The patch is in `executors/theory/patches/`; it is frozen after the theory executor applies it. D1 phase 2 stays sealed until then, or Oct 29, 12:01 ET at the latest.
  - **Prereg A9:** exploratory decomposition; release rules; an optional v3 for B16/F8 (B16/F8 outcomes sealed until v3, its decline, or Oct 31, 12:00 ET); the GPU campaign B16 + replay → F8.
  - **Notes, round 5:** [`executors/theory/05_PLANNER_GUIDANCE_20261028_AM.md`](executors/theory/05_PLANNER_GUIDANCE_20261028_AM.md) and [`executors/xp/05_PLANNER_NOTES_20261028_AM.md`](executors/xp/05_PLANNER_NOTES_20261028_AM.md).
- **2026-10-28, 03:30 (planner):**
  - **TH1 v1, its per-image companion and v2 (ImageNet FAST) are frozen, pushed and registered** (theory `8ef5f94`, `5782149`, `340dc5d`). The planner ran the theory executor's production code. **The FFHQ K > 4 embargo is lifted.** D1 phase 2 stays embargoed until a v2 E1 addendum, or Oct 29, 12:01 ET at the latest.
  - **Executor:** D2/D3 K = 4 aggregates published. D2's per-metric oracle (22.3 dB) is below DAPS Table 1 (25.78) but above PC-DAPS's independent DAPS-4K rerun (20.98). Comparability checks run under A8.6.
  - **Prereg A8:** registration; the D1-P2 embargo and the E1 addendum; E1_CLIP as a per-candidate rule; the comparison rule; lineage-gate exclusions; the D2 comparability checks; GPU order.
  - **Notes, round 4:** [`executors/theory/04_PLANNER_GUIDANCE_20261028.md`](executors/theory/04_PLANNER_GUIDANCE_20261028.md) and [`executors/xp/04_PLANNER_NOTES_20261028.md`](executors/xp/04_PLANNER_NOTES_20261028.md).
- **2026-10-27, 17:15 (planner):**
  - **Executor:** A8b and the D1-P1 strings are published (v2 due Oct 29, 12:01 ET). D1-P2 and D3 are generated. E′ stopped at 45/720 because labmates' jobs took the GPUs; resume as audited.
  - **Theory:** mech v1.1 registered. The planner ran the A6 forecaster: the validator fails without a control-variate fix and passes with it.
  - **Prereg A7:** co-tenancy for generation stages, the B16 lineage gate, F8 seeds.
  - **Notes, round 3:** [`executors/theory/03_PLANNER_GUIDANCE_20261027_EVE.md`](executors/theory/03_PLANNER_GUIDANCE_20261027_EVE.md) and [`executors/xp/03_PLANNER_NOTES_20261027_EVE.md`](executors/xp/03_PLANNER_NOTES_20261027_EVE.md).
- **2026-10-27, 12:30 (planner):**
  - **Overnight generation certified:** XP-A16, XP-A64, XP-D1 phase 1, XP-D2 (about 39 GPU-h in total).
  - **TH2 mechanism forecasts v1 registered.**
  - **Prereg A6:**
    - A4.4 = 400 × 5, to be confirmed by replay;
    - XP-E′ eligible;
    - evaluation panels;
    - predictive-interval rule;
    - the `docs/track_r/executors/xp/theory_inputs/` directory.
  - **Notes, round 2:** [`executors/theory/02_PLANNER_GUIDANCE_20261027_PM.md`](executors/theory/02_PLANNER_GUIDANCE_20261027_PM.md) and [`executors/xp/02_PLANNER_NOTES_20261027_PM.md`](executors/xp/02_PLANNER_NOTES_20261027_PM.md).
- **2026-10-27, 00:00 (planner):**
  - **NP dropped** from the paper (Cheng, 2026-10-26 23:01). Paper counts are now on DAPS, SITCOM and FAST-DIPS only (1,960 runs; 540 image–solver pairs).
  - **Prereg amendment A5:**
    - XP-I (PC-DAPS) becomes CRITICAL-if-ready, and XP-I2 adds ImageNet;
    - new XP-K: a parameter-only calibration of each solver's data term to σ;
    - new analysis A5.5 (the noise floor) and the A8b residual strings;
    - A7 is now required;
    - the XP-F8 replay gate;
    - v1 freezes both the S2 reference and a rank-calibrated forecast.
  - **New facts from A8 and N1c:**
    - S2 is rejected: within an image, GOOD beats CONSISTENT in 200/205 pairs;
    - DAPS and SITCOM fit the noise: every GOOD run has ρ < σ.
  - **Notes:** [`executors/theory/01_PLANNER_GUIDANCE_20261027.md`](executors/theory/01_PLANNER_GUIDANCE_20261027.md) and [`executors/xp/01_PLANNER_NOTES_20261027.md`](executors/xp/01_PLANNER_NOTES_20261027.md).
  - **Freeze:** v1 target Oct 29, 18:00 ET; hard deadline Oct 31.

**Track R** is reliability per compute for diffusion-prior Fourier phase retrieval.

**Status:**
- R0 and R0-N are complete. The method-paper line is **stopped**: no R1, no R2.
- K2 removed diffusion-prior orientation repair as a contribution.
- **The deliverable is an audit paper with a selection theory.**
  - Attempt: CVPR 2027 (enrollment Nov 10, paper Nov 16, planner go/no-go Nov 8).
  - Fallbacks: SPL, or EUSIPCO 2027 (Feb 6).
- After that, the project winds down, except a possible separate TH2 theory study.

**Relation to earlier tracks:** Track R superseded the parked Track A (NP method paper). Track B is paused.

**Roles:**
- Planner: Claude.
- PAC-side executor: a GPT/Codex session (no PAC access itself).
- **Theory executor:** a separate chat or agent (no PAC).
- PAC operator: Cheng.

## Current stages

| Stage | What | Status |
|---|---|---|
| **R0-N.2** | Amendment 2 **revision D.1**, CPU: N1h → **A8 per-image strings (by Oct 28)** → ladder (vs O_orbit) → A7 export with SSIM/LPIPS → crosswalk | **AUTHORIZED, urgent; return by 2026-10-29** |
| **XP** | `XP_PREREG_v1_1.md` (+ amendments A1–A7; A2 = DAPS-4K for the published-protocol replication; A5 = XP-I promoted, XP-K, A5.5, A8b, F8 gate; A6 = A4.4 ruling, XP-E′, panels, interval rule, theory inputs; A7 = co-tenancy, B16 lineage gate, F8 seeds, v2 deadline):<br>• **Tier A (Nov 2):** XP-0/T; FAST FFHQ n=16; XP-C; XP-C2 equivariance defect; FAST ImageNet n=16<br>• **Tier B (Nov 6):** FAST n=64; SITCOM n=16 on 180 ids; DAPS ImageNet n=4 on 100; SITCOM ImageNet; DAPS FFHQ n=8<br>• **Tier C (Nov 20):** SITCOM n=32, DPS, PC-DAPS, noise robustness, 512-canvas control, FAST ImageNet n=64, conditional XP-E′<br>• XP-E cancelled | **GO given (2026-10-26); FROZEN v1.1 + A1–A7; two lanes (CRITICAL / BACKGROUND); cap 400 GPU-h.** A16, A64, D1-P1, D2 generated (about 39 GPU-h). Planner notes 2026-10-27 (two rounds) |
| **TH1** | Selection theory for the paper; predictions v1 (S2 reference + rank-calibrated race) — target **Oct 29 18:00 ET**, hard **Oct 31** (embargo on K > 4 outcomes until registered); note due Nov 6 | **IN PROGRESS** (interim `12e0ff3` accepted; guidance 2026-10-27) |
| **TH2** | Noise-floor theory (over-weighted data terms; within-image residual races; orientation leak); optional mechanism forecasts for XP-K/XP-G by Oct 31 | AUTHORIZED, refocused 2026-10-27 |

## Documents (read in this order)

1. **Top-venue plan** (2026-10-25; read the 2026-10-26 update box first): [`docs/planning/track_r/2026-10-25_TOP_VENUE_PLAN.md`](../planning/track_r/2026-10-25_TOP_VENUE_PLAN.md)
2. **Experiments:** [`docs/planning/track_r/XP_PREREG_v1_1.md`](../planning/track_r/XP_PREREG_v1_1.md) (v1 is superseded; amendments A1–A5: lanes, NP dropped, SITCOM settings, DiffFPR; A5 = PC-DAPS promoted, XP-K, noise floor), with handoff revision C (autopilot) [`executors/xp/00_XP_EXECUTOR_HANDOFF.md`](executors/xp/00_XP_EXECUTOR_HANDOFF.md) and the planner notes [`executors/xp/01_PLANNER_NOTES_20261027.md`](executors/xp/01_PLANNER_NOTES_20261027.md)
3. **Theory:**
   - program v1.2 [`docs/planning/track_r/THEORY_PROGRAM_v1.md`](../planning/track_r/THEORY_PROGRAM_v1.md);
   - literature digest [`2026-10-26_THEORY_LIT_DIGEST.md`](../planning/track_r/2026-10-26_THEORY_LIT_DIGEST.md);
   - kickoff prompts: [`executors/xp/KICKOFF_PROMPT.md`](executors/xp/KICKOFF_PROMPT.md), [`executors/theory/KICKOFF_PROMPT.md`](executors/theory/KICKOFF_PROMPT.md);
   - handoff revision B.1 [`executors/theory/00_THEORY_EXECUTOR_HANDOFF.md`](executors/theory/00_THEORY_EXECUTOR_HANDOFF.md) and the planner guidance [`executors/theory/01_PLANNER_GUIDANCE_20261027.md`](executors/theory/01_PLANNER_GUIDANCE_20261027.md);
   - predictions registry [`theory/PREDICTIONS_REGISTRY.md`](theory/PREDICTIONS_REGISTRY.md).
4. **R0 decision and rulings** (§§1–10; §10 is K2): [`docs/planning/track_r/2026-10-23_R0_PLANNER_DECISION.md`](../planning/track_r/2026-10-23_R0_PLANNER_DECISION.md)
5. **PC-DAPS positioning** (read its K2 update box first): [`docs/planning/track_r/2026-10-24_PC_DAPS_POSITIONING.md`](../planning/track_r/2026-10-24_PC_DAPS_POSITIONING.md)
6. **Fallback paper plan** (EUSIPCO/SPL condensed): [`docs/planning/track_r/LETTER_PLAN.md`](../planning/track_r/LETTER_PLAN.md). Extension options: [`2026-10-24_EXTENSION_OPTIONS.md`](../planning/track_r/2026-10-24_EXTENSION_OPTIONS.md)
7. **R0-N executor handoff and amendments 1–2:** [`executors/r0n/00_EXECUTOR_HANDOFF.md`](executors/r0n/00_EXECUTOR_HANDOFF.md)
8. **R0 record:**
   - executor return on `codex/prdiff-track-r-r0-20261021`;
   - R0 spec [`docs/planning/track_r/R0_EXPERIMENT_SPEC.md`](../planning/track_r/R0_EXPERIMENT_SPEC.md);
   - R0 handoff [`executors/r0/00_EXECUTOR_HANDOFF.md`](executors/r0/00_EXECUTOR_HANDOFF.md).
9. **Paper draft:** [`docs/paper/track_r_cvpr27/`](../paper/track_r_cvpr27/README.md). Read `CLAIMS_LEDGER.md` before editing numbers.
10. **Background:**
   - roadmap (historical) [`2026-10-21_TRACK_R_PLANNER_ROADMAP.md`](../planning/track_r/2026-10-21_TRACK_R_PLANNER_ROADMAP.md);
   - literature basis [`2026-10-21_LITERATURE_BASIS.md`](../planning/track_r/2026-10-21_LITERATURE_BASIS.md).

## Stage ledger

| Stage | Content | Status | Who |
|---|---|---|---|
| R0.0–R0.8 | Inventory → frozen analysis and audit (3.43 GPU-hours) | COMPLETE, accepted | executor and Cheng |
| R0-N | N0 tables, N1 analyses, N2 CLIP baseline, N3 inventory | COMPLETE, accepted (N1 `5af47db`; N2 `33a34bd`). K1 not triggered, **K2 TRIGGERED**, K3 PASS | executor and Cheng |
| R0-N.2 | Amendment 2 revision D.1 (CPU) | NEXT, urgent (A8 Oct 28; return Oct 29) | executor and Cheng |
| XP | `XP_PREREG_v1_1` (GPU, ≤ 400 GPU-hours, tiers A/B/C) | **GO given 2026-10-26** | executor and Cheng |
| TH1 / TH2 | Theory program v1.2 | AUTHORIZED, starting | theory executor |
| Paper | CVPR 2027 attempt (go/no-go Nov 8) → SPL / EUSIPCO fallback. Draft in `docs/paper/track_r_cvpr27/` (claims ledger, figures plan) | revised Oct 27: NP-free counts, E1_CLIP table, S2 rejection, noise floor, theory corrections (builds; ≈7.5 pp. of text with placeholders) | planner drafts, Cheng and coauthors edit |
| N4 | Clean-image probe (withdrawn) | — | — |
| X5/X6 | PC-DAPS audit and noise robustness: now XP-I and XP-G (Tier C, supplement) | in XP v1.1 | executor and Cheng |
| R1 / R2 | Pipeline and prospective evaluation | CANCELLED | — |
| T1 | √h hard-selection theory | parked | — |

## Branches

| Role | Branch |
|---|---|
| Planner | `claude/prdiff-track-r-planner-20261021` |
| Executor, R0 (finished) | `codex/prdiff-track-r-r0-20261021` @ `c8eb608` |
| Executor, R0-N / R0-N.2 | `codex/prdiff-track-r-r0n-20261023` (N2 return @ `33a34bd`) |
| Executor, XP | `codex/prdiff-track-r-xp-<YYYYMMDD>`, from `33a34bd`, merged with the planner branch |
| Theory executor | `theory/prdiff-track-r-th-<YYYYMMDD>`, from the planner branch; writes only `docs/track_r/theory/` and `scripts/track_r/theory/` |
| Historical, read-only | Track A parked; Track B TB0 STOP |

## PAC namespaces (verified in R0.0)

| Item | Path |
|---|---|
| Checkout | `/egr/research-pac/huang248/pr_diffusion_track_r` |
| Outputs | `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/` |
| Logs | `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/logs/` |
| Caches | `/egr/research-pac/huang248/outputs/.cache/track_r/` — never `/home` |
| Env | `/egr/research-pac/huang248/conda-envs/prdiff_track_r` |
| FAST env | `/egr/research-pac/huang248/conda-envs/fast_dips_a15_blackwell` |

## Governance (unchanged)

- No agent launches GPU jobs; Cheng runs every PAC block.
- Runs only from pushed, clean commits.
- Historical outputs are read-only; nothing is overwritten.
- Bounded inventory only; no recursive scans of large trees.
- Safe shell style: no `set -e`, `exit` or broad `pkill`; long jobs use `nohup` with PID, status and stop.
- Full compute accounting.
- Raw-oriented RGB8 PSNR is primary.
- No GT in deployable steps.
- **B24 confirmation305 is locked.** Because DAPS `test-ffhq` contains one of its images, **no bundled FFHQ set is ever loaded** in Track R.
