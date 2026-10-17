# Track A lead scientific update — best-of-K is a legitimate primary research endpoint
Date: 2026-10-16
Status: **A0 B22 VERIFIED, SCIENTIFIC GO TO PREPARE A1 PROSPECTIVE COST-RELIABILITY STUDY; NO GPU/PAC RUN INITIATED**
Authoritative A0 executor: `codex/prdiff-track-a-a0-native-selector-audit`, final commit `ecc66b890b63123c0445a13d33277e09f6ab4ab5`.
Report: https://github.com/Cheng-HanHuang/pr_diffusion/blob/codex/prdiff-track-a-a0-native-selector-audit/docs/track_a/executors/a0-native-selector-audit/A0_B22_RETROSPECTIVE_RESULTS_2026-10-16.md
Receipt: https://github.com/Cheng-HanHuang/pr_diffusion/blob/codex/prdiff-track-a-a0-native-selector-audit/docs/track_a/executors/a0-native-selector-audit/A0_B22_RESULTS_RECEIPT.json

## 1. Explicit correction to Track A's previous overrestriction

**Do not make clean-free native selection a prerequisite to publication viability.** DAPS explicitly reports phase-retrieval results as the best of **four independent samples**, and the SITCOM paper explicitly follows this convention for **all phase-retrieval baselines**; SITCOM Appendix G states that Table 1's run-time is **one-run runtime** even when reconstruction quality comes from the best of four. Authoritative papers:
- DAPS, CVPR 2025, arXiv HTML v3, §4.1, `https://arxiv.org/html/2407.01521v3`: “reporting the best result out of four independent samples”.
- SITCOM, ICML 2025, arXiv HTML v2, Appendix G, `https://arxiv.org/html/2410.04479v2`: “best results from 4 independent runs” and “run-time is reported for one run”.

**Primary publication question:** at a defined sampling/compute budget, does NP produce a more reliable **set of candidate reconstructions** (PSNR-oracle best-of-K quality, low tail/catastrophic failures, success-vs-cost curve) than strong native DAPS and SITCOM-ODE? This is a legitimate *paper-style benchmark* even if optimal candidate identification requires GT, but must be called `offline GT oracle`, not an executable selector, and its **entire K-run work must be charged** when making cost-efficiency claims. Published single-run runtime is a reporting convention, not the cost to attain best-of-four.

**Secondary distinct question:** can clean-free NP native selection preserve best-of-K candidate advantages in a deployed reconstruction? Do not conflate posterior distribution fidelity, GT-oracle success, and executable selection, all distinct.

A smaller-venue paper can be defensible on substantial, robust, controlled candidate-generation reliability even without claiming better automatic terminal selection. It still needs a nontrivial method/contribution and fresh scientific evidence; merely repeating exposed B22 with a different reporting perspective is insufficient.

## 2. Verified B22 retrospective facts and what they do NOT prove

| Portfolio | K | Mean GT-oracle PSNR | Oracle Good25 | Native Good25 | Mean candidate reconstruction time per image |
|---|---:|---:|---:|---:|---:|
| DAPS Fresh2 | 2 | 29.427249 | 92/100 | 92/100 | 312.64 s |
| SITCOM-ODE (sequential stream) | 4 | 26.666889 | 95/100 | 93/100 | 196.070948 s |
| NP LF4+S2-4 | 8 | 29.407266 | 96/100 | 95/100 | 411.782967 s |
| NP S2-only | 4 | **Not yet extracted in a committed fixed summary** | 94/100 | 94/100 (retrospective replay) | 206.461 s |

**Retrospective directional promise:**
- NP mixed8 vs DAPS2: **+4 Good25 successes**, mean oracle PSNR differs by only `-0.019983 dB`. At n=100, 96 successful images forces empirical rank-5/6 PSNR >=25 dB for NP8; 92 successful images forces rank-5/6 PSNR <25 for DAPS2. Thus NP8 vs DAPS2 has a threshold-relevant fifth-percentile ordering; do not invent exact q05 or min dB before extracting the actual percentiles.
- NP mixed8 vs SITCOM4: **+1 Good25 success**, mean oracle PSNR **+2.740377 dB**. The success-rate difference is tiny at n=100, and pipeline cost is over double SITCOM4.
- **Essential counterweight:** K-matched NP S2-only4 **oracle Good25 94/100**, while SITCOM4 **oracle Good25 95/100**. At roughly similar recorded candidate reconstruction time (~206 vs ~196 s), current B22 does **not** show NP better Good25 candidate availability than SITCOM. We still need the paired mean, q05 and other metrics for S2K4; cannot claim they favor NP without data. Its retrospectively scored native Good25 94 vs SITCOM native93 is a *different*, non-oracle claim, only +1 on this exposed set.
- No DAPS4 samples exist in B22. DAPS paper's primary PR result uses four independent runs; DAPS2 is **not** a publication-standard K4 DAPS control.
- NP8 is two score configurations (LF4 + S2-4), not eight iid retries of one fixed NP policy. SITCOM B22 four samples are sequential-state RNG, not four independent seeds; for a genuine paper-style K4 comparison, independent SITCOM4 matters.
- Selection regret: DAPS2 0.129 dB mean, SITCOM4 0.225, NP8 0.138. Native/oracle Good25 DAPS 92/92, SITCOM 93/95, NP 95/96; these figures inform deployability, not candidate quality ranking.
- NP8 oracle success 96/100 vs SITCOM4 95/100 vs DAPS2 92/100 is **not** significance evidence. Compute paired Good25 rescue/harm + CI from same-image candidate rows, and mean/quantile CIs; with 1 or 4 image difference a large uncertainty is plausible. No claims from B24 source-unreconciled cohort; B24 confirmation305 is permanently excluded.

## 3. Precise estimands, to state in method and paper

For image/measurement pair `i`, native method `m`, **frozen and declared** candidate policy `p`, and budget `K`, define
`Q*_{i,m,p}(K) = max_{1<=j<=K} PSNR(x_{i,m,p,j}, x_i^GT)`,
`S*_{m,p}(K) = (1/n)sum_i 1{Q*_{i,m,p}(K) >=25 dB}`,
and `R_{m,p}(K)=mean_i Q*_{i,m,p}(K)` plus q05, min and bad20. These are **offline candidate-set/reliability measures**, with GT used ONLY in evaluation. For mixed NP8, `p` is the two-configuration portfolio, not a homogeneous sampler; if K varying across mixed cases, freeze the allocation (e.g. 2 LF+2 S2 at K4, 4 LF+4 S2 at K8) before testing.

Separately define deployable `Q^native_{i,m,p}(K)` using historically or prospectively fixed **measurement-only/native** selection; score only within the correct family. Report GT-oracle/native gap; no cross-method forced 'min measurement residual' rule as a substitute for DAPS/SITCOM's actual policies. If no native selector exists at K4, show N/A. Report PSNR, SSIM and LPIPS on PSNR-chosen candidates if PSNR is the single declared primary; avoid secretly selecting a different candidate per metric.

**Two fairness axes:**
(a) **run-matched K** compares candidate-generation differences (restarts and seeds, no claims of equivalent compute); 
(b) **compute-matched GPU-seconds / predeclared wall budget** compares what candidate sets can be afforded (count complete runs and source-specific model/FFT/backprop work); do not show oracle K4 image quality with 1-run cost. Publish the trade-off surface `K vs total realized candidate seconds vs oracle Good25, mean, q05, Bad20`.

**Do not invent a unique physically meaningful 'first successful run time'**, since GT-only detection would again be an oracle stopping policy. One can report offline diagnostic time-to-first-good candidate but never label it executable time-to-success.

## 4. Editorial hypotheses and falsifiers

**H1 (main, oracle candidate generation):** NP shows a favorable **success/mean/low-tail vs total compute Pareto trade-off** against native DAPS and SITCOM-ODE under matched measurements, seeds and metric representation, even when all three are evaluated by the published GT best-of-K convention.

**H2 (tail protection):** At comparable full candidate compute, NP reduces catastrophic below-20 dB reconstructions or raises q05 without degrading mean PSNR unacceptably. Failure events must be defined for the candidate-set max, NOT just selected output.

**H3 (diminishing returns/portfolio diversity):** Mixing LF and S2 candidates might improve tail after K4, but an 8-vs-4 success improvement of 1 on an exposed panel is not evidence. Compare NP LF-only, S2-only and mixed predeclared *candidate-generator* groups with strict equal total K and full compute accounting. No post-hoc selecting K mix or tuning on exposed B22.

**H4 (deployment secondary):** native GT-free policy may lose individual successes despite good candidates. Only claim operative tail improvement when actual fixed selection is tested. The A0 catastrophic SITCOM 60140 and NP 65269 are useful examples but not a fresh validation set.

**NO-GO for strong oracle-first performance paper** if newly collected independent/run+cost matched DAPS/SITCOM (especially DAPS4 and SITCOM4 independent/8) dominate NP on the relevant failure/mean/budget trade-off; if positive results occur only by running more expensive NP8 or by favorable K/config mixtures selected after GT exposure; if effects are too small/uncertain to warrant a reliability claim; or if no eligible new cohort/source integrity/fair operation accounting is available. A reproducibility/negative benchmark remains optional but should not be mislabeled algorithmic advance.

## 5. Decision: A1 preparation GO; prospective pilot conditional, no heavy campaign yet

**A1:** create a separate planning/implementation executor (no historical source edits) to (i) extract exact B22 **oracle** q05/min/Bad20 and paired K-curves from existing committed metadata or *bounded read-only archived A0 tables* only, (ii) prepare source-verified DAPS4 / independently seeded SITCOM4+8 / NP4+8 candidate generation comparison and a frozen 8–12 image *development smoke*, (iii) confirm FFHQ image/measurement cohort eligibility and independent seed namespaces through [issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41), (iv) establish actual GPU-id/time/memory envelope and stop gates, (v) never launch GPUs except exact disclosed user-run commands after source/code preflight and ledger reservation. Any untouched prospective **final** cohort requires distinct allocation and a separate frozen protocol before results; B24 DEV80, B22/B21 exposed and the locked B24 305 cannot be used as fresh confirmation.

Illustrative **not-yet-approved** budget points from B22 archived *candidate reconstruction sums* (hardware/version and sharing may differ in new runs):
- Around 200 GPU-s/image: SITCOM4 ~196; NP S2-only4 ~206; DAPS1 ~156.
- Around 400 GPU-s/image: NP mixed8 ~412; SITCOM8 would be ~392 **only as a projection, not an observed B22 result**; DAPS2 ~313. Do not interpolate pseudo trajectories or call these equal costs.
- Matched K4: fresh DAPS4 estimated 4×~156=~625 s/image, SITCOM4~196 and NP4~206; huge compute disparity is a finding to display, not suppress. Final predictions subject to exact new measurements.

Next significant machine effort should be a **bounded stage-0 metadata/reproducibility plan and stage-1 independent-data pilot** before a larger test. Track B has GPU priority when resources conflict; Track A can proceed independently in its own namespace and must register actual runs and data cohorts in #41. No GPU, PAC command, cohort reservation or new scientific experiment is implied by this document.

## 6. Publication claim wording

Defensible possible abstract-level direction **if prospective H1/H2 passes**: “On challenging oversampled Fourier phase retrieval, guided noise-candidate evaluation generates candidate portfolios with a favorable lower-tail reconstruction-quality/compute trade-off compared to established diffusion posterior samplers. We systematically separate oracle best-of-K reporting, native deployable selection and complete multi-run cost, revealing how restart count, candidate diversity and selection affect catastrophic failure rates.”

Not defensible today: “NP8 is a superior equal-compute posterior sampler”, “NP achieves the same PSNR as published DAPS4”, “NP beats SITCOM at K4 Good25 in GT-best-of-K”, “a GT best-of-K paper score is operationally attainable without GT”, or “95–96/100 shows significant advantage”. The historical B22 result gives rationale to investigate, not a final outcome.

**Source/data firewall:** A0 B22 report & receipt verified, B24 SITCOM effective historical patch still source-uncertain, B24 confirmation305 inaccessible, all B19–B26 frozen and Track B separate.
