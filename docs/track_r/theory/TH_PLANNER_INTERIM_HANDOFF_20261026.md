# Track R theory interim handoff to planner — October 26, 2026

**INTERIM, NOT THE OCTOBER 31 PREDICTION FREEZE.**
**Branch:** theory/prdiff-track-r-th-20261026. Audience: Cheng and the Track R planner. The theory executor remains on a separate workstream. No PAC access, no ground-truth image file, no confirmation305, and **no embargoed XP K>4 return opened**. Only the planner may edit docs/track_r/theory/PREDICTIONS_REGISTRY.md.

## Immediate recommendation

**REPORT THIS INTERIM STATUS NOW, AND CONTINUE TH1/TH2 IN PARALLEL WITH A8.** There are important source-level and mathematical corrections that affect the existing planner-authored Sections 4 and 5. Reporting does not require an A8 file and does not lift the predictions embargo.

## 1. What is already known versus what Track R can genuinely establish

| Primary predecessor | What the predecessor already does | What Track R can distinctly establish, subject to new evidence |
|---|---|---|
| Classical Fourier phase retrieval (Hayes, Fienup) and DAPS CVPR 2025 Appendix F.2 | Twin/trivial-associate ambiguities; DAPS already **shows eight measurement-identical reconstructions** | Exact symmetry factorization **for the fixed noisy 256→384 three-channel operator**; orientation versus non-orbit failure frequencies; effects of learned-prior bias across native solvers |
| PC-DAPS, Kocabas et al. arXiv:2610.31592 (October 22, 2026) | Prior-inconsistent, measurement-consistent modes; a 2-atom toy and a corrective particle method; criticism of GT-best-of-K | **Not a new sampler:** controlled, common-y/ids, multiple published solvers, failure taxonomy, deployable GT-free selection ceilings and frozen scaling forecasts |
| TGD/A-TGD, Makris et al. arXiv:2605.03712 (2026) | The **same Fourier PR operator**, DAPS residual-selected best-of-4 and early-particle pruning; explicitly flags DAPS's GT per-metric oracles | Quantify group-blind measurement selection versus prior-assisted orientation correction; cross-solver failure/metric decomposition; prospectively predicted K-scaling |
| Stroebl, Kapoor & Narayanan, ICLR 2026 | Imperfect-verifier saturation and easy/hard mixture | Physically grounded **GOOD/global-ORBIT/partial-ORBIT/CONSISTENT/INCONSISTENT** verifier classes; exact subgroup blindness, GT-free correction and preregistered tests of the S1–S3 approximation |
| Yano, Ishibashi & Oyamada, arXiv:2610.19362 (October 14, 2026) | General best-of-N **correct-vs-incorrect score tail** competition and possible decline | Test whether rare low-*Fourier-residual* incorrect reconstructions explain K-scaling of the three solvers. The extreme-value theory itself is not new here |
| Dandapanthula & Boffi (2026), GeoDiff/equivariant generative diffusion, TDS (NeurIPS 2023) | Guidance mode limitations, symmetry-preserving kernels, ideal particle convergence to a model posterior | Joint connection to the **particular Fourier orbit, non-equivariant learned prior, practical solver dynamics and actual output-law asymmetry**, conditional on code and trajectory assumptions |
| Stein/SURE, classical discrepancy principle and Gauss–Radau quadrature | Measurement-noise-fitting optimism and finite-count mixture nonidentification | Transparent physics-specific interpretation of empirical CONSISTENT decoys and honest statements on **the limits of K>4 forecasting from four calibration runs** |

**Bottom line: the new result is the integrated empirical-and-theoretical characterization with a prospectively falsifiable forecasting protocol. No isolated algebraic identity or residual selection rule should be introduced as a new general method or theorem.** If XP evidence is weak, the target venue/story must be adjusted rather than inflating novelty.

## 2. New positive scientific distinction: exact ambiguity vs noisy consistency

We now prove a clean distinction that can make the failure taxonomy much more informative.

(1) For exact orbit members, A(gx)=A(x) for **every** measurement; no likelihood-only verifier can prefer an orientation.

(2) For any two *fixed, noise-independent* candidates with delta=||A(nu)-A(x)||>0 under Gaussian observations Y=A(x)+sigma Z, the optimal equal-prior likelihood/residual discrimination succeeds with probability Phi(delta/(2sigma)), while the KL between their measurement laws is delta²/(2sigma²). Both candidates' **normalized** residuals can be near sigma when m is large yet discrimination can be very strong. For m=442368, sigma=.05, and a hypothetical delta=5sigma=.25, the root expected squared residual at the alternative is 0.05000141 versus 0.05 at the truth, while paired ranking is correct about 99.38% of the time.

(3) Actual solver outputs depend on the observed noisy y. A local linearized Gaussian inverse model shows **noise fitting** can systematically lower the in-sample residual while moving AWAY from the clean forward signal. For Jacobian J of rank r and least-squares projector P onto range(J):
  - observed fit improvement equals the clean-signal fitted displacement norm squared exactly: ||y-a||²-||y-(a+sigma PZ)||²=||sigma PZ||²;
  - E in-sample residual squared = sigma²(m-r), while E squared fit displacement = sigma²r;
  - for a ridge smoother H, replicated-noise optimism = 2sigma² tr(H).
This is classical statistics, **not a novelty claim**. The Track R-specific implication is that empirical CONSISTENT does not mean exact second Fourier preimage, close posterior support, or low statistical KL. A solver can land at a noise-adapted low-residual bad reconstruction. Whether this explains the observed methods is **unverified**, and can be tested later against class-conditioned residual ranks, without introducing another solver or changing registered decisions.

Proof and limitations: docs/track_r/theory/TH2_NOISE_ADAPTIVE_CONSISTENCY_20261026.md. Synthetic test: scripts/track_r/theory/check_th2_noise_adaptive.py.

## 3. Status of mathematical proofs

- **TH1 symmetry/posterior:** fixed padded per-channel 180-degree orbit invariance and posterior ratio (with correct domain/zero-density caveats): PROVED; classical group symmetry applied to the fixed operator.
- **Equivariant kernels:** invariant initial law plus equivariant transitions gives invariant marginals; with eight free distinct orbit members and disjoint matched neighborhoods each has conditional 1/8 mass: PROVED under the assumptions; inherited general invariance principle.
- **Gaussian pathwise TV:** TV(L,g#L) ≤ sqrt(sum E||m_i(X_i)-gm_i(g^-1 X_i)||²/(4s_i²)): PROVED for strictly positive **state-independent** Gaussian transition scales; not a general native-solver certificate.
- **Posterior-versus-solver symmetry:** TV(Pi,g#Pi) ≤ 2TV(Pi,L)+TV(L,g#L), PROVED elementary triangle inequality. **Non-equivariance may be necessary to approximate a prior that favors upright orientation**, rather than intrinsically pathological.
- **TH1 S1–S3 verifier curves and monotone gaps:** PROVED with explicit assumptions and zero-mass edge cases; specialize Stroebl's existing general theory. Without S2, we proved an exact class-conditioned residual-race formula and explicit **nonmonotonic** counterexamples.
- **TH1 unbiased subset estimators, per-metric oracle:** PROVED elementary rank combinatorics; new part is protocol/evidence, not the estimator.
- **TH1 finite-run identifiability:** PROVED classical truncated-moment obstruction. An explicit three-component Gauss–Radau latent-success mixture is **indistinguishable from Uniform[0,1] under every possible four-run count**, yet differs at K=64 by **9.57 percentage points** in oracle success and by 11.11pp as K→infinity. **Forecast intervals remain model-conditional**, not identification bounds.
- **TH2:** zero-flip Fourier pair, simple two-point selector mechanics and matched-total-innovation double-well contrast PROVED for toy models. These have precedents (zero flips, Langevin/Kramers, PC-DAPS toy) and currently **do not establish a native-solver ordering or standalone theorem novelty**.

## 4. Corrections the planner should make to paper Sections 4–5 before further drafting

1. **Do not write** posterior p(gx|y)/p(x|y)=pi(gx)/pi(x) without specifying where ratios are defined; cross-product formula avoids zero priors.
2. Uniform 1/8 is conditional on equivariant output law, distinct orbit members and disjoint orbit neighborhoods. Real FFHQ learned denoiser is expected to break the symmetry; that non-equivariance can reflect the learned prior rather than a “defect” needing removal.
3. **The existing draft asserts every solver is a Gaussian Markov chain and all ingredients except the denoiser are equivariant. That is too strong** without source-specific handling of deterministic terminal maps, optimizer states, NaN branches, tie-breaking, and step scheduling. The TV theorem cannot be numerically verified by XP-C2 measurements of the denoiser on noised clean images without connecting those probes to the trajectory expectation and Jacobian defects.
4. Only likelihood-only SMC potentials/residual pruning are G-blind. Full TDS importance ratios also depend on proposal/reverse-model factors, and must not be called universally invariant.
5. Monotone O_orbit-E1 and O_raw-E0 gaps require **S1–S3**, although the theorem holds for fixed orientation error rates including h>0. The infinite-K orbit gap equals 1-a/s only if q>0; it is zero if q=0. Likewise for the raw gap and p=0.
6. Under global-only repair, missed passing mass is **partial-ORBIT + CONSISTENT**, not merely CONSISTENT; no passing run u^K is not the same event as no orbit-correct run (1-q)^K.
7. Keep only the **three publication solver families** in all paper denominators and all 2026 R0 headline counts. Current symmetry draft's 3240/84/2040 values came from the earlier 4-family aggregate including NP and must be recomputed.
8. Fitter fixed a material beta-moment reversal (posterior E[u^K], not E[(1-u)^K]), and pooled forecasts now preserve fixed B22:DEV80 weights 100:80. DAPS/B22 historical P4 K=4 has n_images=0 (missing), **not** success rate zero. Full end-to-end fitter remains unexecuted against absent A8.
9. **Native source finding:** FAST-DIPS pixel @ a9758b38 returns x0hat rather than final measurement-corrected x0y in last step; DAPS @ e7a77d0 uses a re-noising branch whose condition is always true inside the loop; SITCOM-ODE @ 275ab67 has inner stochastic SGD/Langevin rather than the separate lightweight local implementation. **Ask executor to confirm actual pinned wrappers; don't silently change the frozen method.**

## 5. Recommendations and blocker policy

- **Do report this handoff now.** It improves the planner's in-progress writing and explains how to position against especially PC-DAPS and TGD. This is not the TH1_PREDICTIONS_v1 freeze.
- **A8 check as of this turn:** allowed directory docs/track_r/executors/r0n/returns/ on codex/prdiff-track-r-r0n-20261023 lists R0N_20261024T061111Z and R0N_N2_RECOVERY_20261024T073405Z only. No R0N2 A8 per-image counts return is visible there. We deliberately did not access any XP directory.
- **Priorities:** protect A8 K<=4, XP-A/C/C2, ImageNet FAST and K-scaling under the existing critical lane. Ask XP executor to verify source-code terminal semantics/config at the pinned commits, without changing any preregistered benchmark.
- **Near-competitor:** PC-DAPS is direct contemporary prior art, currently designated background XP-I. If an inexpensive same-y/same-image PC-DAPS comparison can be available for the main paper **without holding up the critical lane**, it would substantially strengthen reviewer confidence. Else acknowledge it prominently and treat subsequent XP-I as supplemental/fallback evidence.
- **October 31 freeze:** Once authorized K<=4 A8 counts appear, validate true schema (run labels, orientation member, e0 rank order, original decisions), test S1–S2 where possible, fit only allowed data, and commit the v1 JSON with file SHA-256, git commit and UTC freeze timestamp for planner registration. **No XP K>4 data before registration.** If A8 is absent by the fallback threshold, aggregate P4 cannot identify the five-class per-image mixture, so freeze an explicitly adjusted identifiable/weakly identified P4-based model, with candid caveats.
- **TH2 continuation:** prioritize physically grounded consistency and solver-faithful nonlinear mechanisms, not more small “proofs” of already known OU, zero-flip, projection, or best-of-N theory.

## 6. The recommended contribution paragraph (conditional, for paper drafting)

> The noisy Fourier-magnitude inverse problem combines exact observation symmetries, learned-prior orientation preference, solver-induced candidate mode occupancy, and imperfect measurement-based verification. Existing diffusion methods and GT-oracle best-of-K reporting mix these effects. We disentangle them on common measurements and images for three published solvers, measure which failures can be corrected without ground truth, and test a calibrated five-class selection model by freezing K>4 predictions before the additional runs are revealed. Crucially, we distinguish truly unidentifiable orbit representatives from non-orbit reconstructions that merely fit the realized noisy measurements.

**Remaining novelty gate:** the new matched cross-solver and cross-domain evidence, plus the prospective K forecasts, must be strong enough to validate that characterization. The report does **not** assert those experiments have already succeeded.

**Important files:** TH1_WORKING_PROOFS_20261026.md, TH1_POSTERIOR_SOLVER_SYMMETRY_BRIDGE_20261026.md, TH1_SOLVER_EQUIVARIANCE_CHECKS.md, TH1_2_SELECTION_PROOFS.tex, TH1_SELECTION_TAILS_20261026.md, TH1_5_RADAU_NONIDENTIFIABILITY_20261026.md, TH2_NOISE_ADAPTIVE_CONSISTENCY_20261026.md, and TH_PROGRESS_20261026.md (all relative to docs/track_r/theory on the theory branch). Corresponding synthetic verification scripts reside under scripts/track_r/theory.

**Primary papers:** DAPS (CVPR 2025), SITCOM (ICML 2025), FAST-DIPS (ICLR 2026), Stroebl et al. (ICLR 2026), TGD (arXiv:2605.03712; NeurIPS 2026), PC-DAPS (arXiv:2610.31592), Yano et al. (arXiv:2610.19362), Dandapanthula & Boffi (arXiv:2606.02884), TDS (NeurIPS 2023), Stein/SURE and classical phase retrieval.
