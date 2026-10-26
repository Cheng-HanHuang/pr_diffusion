# Claims ledger: Track R CVPR 2027 draft

Every number or factual statement in the draft carries a `\claim{ID}` tag that points here.

**Status values:**
- **VERIFIED:** checked by the planner against the named source.
- **PENDING-DATA:** waits for an executor return.
- **PENDING-PROOF:** waits for the theory executor.
- **DECISION:** needs Cheng.

A claim may appear in the submission only if it is VERIFIED, or if it is proven or measured by Nov 14.

| ID | Claim (short) | Source | Status |
|---|---|---|---|
| C-P1 | Protocols. DPS, DAPS, SITCOM and FAST-DIPS report best-of-4 for PR. The DAPS and FAST-DIPS reference code pick the best run **per metric** against GT. TGD selects by lowest measurement error. PC-DAPS baselines are GT best-of-4 | DAPS arXiv v3 Sec. 4.1 ("best result out of four independent samples"); DAPS code `e7a77d0` `eval.py` 66–84; FAST-DIPS `a9758b38` `misc.py` 665–668; TGD App. F.4 and Tab. 2; PC-DAPS full read (2026-10-24); SITCOM "best results from 4 independent runs" (earlier read) | VERIFIED for DAPS, FAST-DIPS, TGD, PC-DAPS. **SITCOM and DPS criteria not stated** (ask Cheng for SITCOM). DiffFPR protocol unknown |
| C-S1 | The operator equals the DPS/DAPS PR operator (pad 64, ortho FFT, σ = 0.05) | R0 spec l. 59–60; TGD App. F.3; DAPS `configs/task/phase_retrieval.yaml` | VERIFIED |
| C-S2 | The noise exceeds the signal at most high frequencies | Planner check on generic natural images: 79–93% of magnitudes < σ, 92–98% in the outer half. To be measured on our GT (XP-C, A3) | PENDING-DATA (plausible) |
| C-S3 | The standard canvas violates Hayes's support condition (2/3 > 1/2) | Barnett et al. §1.1 (verified on ar5iv); arithmetic | VERIFIED |
| C-S4 | τ (95th percentile of ρ over GOOD) was frozen before the analyses | R0 spec (FROZEN v1); R0.7 | VERIFIED |
| C-T1 | Lemma: A(R_s x) = A(x) | TH1.1(a) | PENDING-PROOF (route known) |
| C-T2 | Measurement-based selection and likelihood-weighted particle methods are orientation-blind, with no assumptions | TH1.1 Cor. 1, 1′ | PENDING-PROOF (elementary) |
| C-T3 | Every non-network component of the solvers is G-equivariant | TH1.1 Cor. 4, family-by-family check | PENDING-PROOF |
| C-T4 | TV bound by the equivariance defect | TH1.1 Cor. 4(b) | PENDING-PROOF (route: conjugate chain + Pinsker + KL chain rule) |
| C-T5 | Closed-form K-curves under S1–S3; equivalence with accept-first | TH1.2 | PENDING-PROOF |
| C-T6 | The gap never shrinks with K and saturates at (c + o_p)/s | TH1.2 item 3 (planner algebra checked) | PENDING-PROOF (write-up) |
| C-T7 | Order-statistic U-statistics are exact and unbiased | TH1.3 | PENDING-PROOF |
| C-A1 | FFHQ: 85 of 3,240 runs are orbit errors, 7.1% of failures (FAST 3.6%) | R0 K3 totals 2,040/85/324/791 | VERIFIED for the R0 run set. **DECISION:** the run set includes the in-house NP family and small sources S3/S4. Recount for the paper's solver set |
| C-A2 | CLIP fixes 84/84 global orbit errors with 0/2,040 harm at 8.5 ms per candidate; S_mid 84/84; UnivFD 71/84 with 108/2,040 harm | N2 return @ `33a34bd` (comparison JSON SHA `322c3ef0…0233`) | VERIFIED. The exact prompts are still to be quoted from the N2 record |
| C-A3 | GT-free recovery of 93–97% of the orbit-oracle gain (public solvers); within 2 images in Good25 | Planner arithmetic from R0.8 P4 (top-venue plan §2) | VERIFIED with the prior-based E1. **E1_CLIP version: PENDING-DATA (R0-N.2 A4.3)** |
| C-A4 | Of 114 failed residual picks: 15 fixed, 93 generation failures, 5 selection misses, 1 uncorrected | R0-N N1b @ `5af47db` | VERIFIED for the R0 run set; recount for the solver set |
| C-A5 | 57% of SITCOM's failed runs are CONSISTENT | R0-N N1a | VERIFIED |

**Open decisions for Cheng:**
1. **NP.** The in-house DiffFPR-based family: drop it, or keep it in the supplement as an in-house variant?
2. **SITCOM.** Its selection criterion over 4 runs (C-P1), and its ImageNet PR settings, if defined (Cheng co-authored SITCOM).
3. **Title.**
