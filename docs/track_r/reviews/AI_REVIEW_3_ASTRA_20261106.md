**Overall assessment: weak reject in its current form, with a plausible path to acceptance after targeted revision.** The paper addresses a worthwhile evaluation problem and contains useful empirical findings. My main reservations concern the strength of the predictive claims, the evidence for generalization, and discrepancies between the mathematical statements and the conclusions drawn from them.

I evaluated the manuscript independently of your previous research preferences. This review covers the eight-page main text and references; I have not reviewed the supplement, code, raw outputs, or registration records.

Source: TrackR_CVPR27_main_20261106.pdf

## 1. Summary

The paper studies how to choose among multiple diffusion-based Fourier phase-retrieval reconstructions when ground truth is unavailable. It separates three problems:

1. **Selection:** whether a usable reconstruction exists among the generated candidates, and whether measurement residual can identify it.
2. **Orientation:** whether a selected reconstruction belongs to the correct Fourier-magnitude symmetry orbit but has the wrong orientation.
3. **Coverage:** whether the solver generates any sufficiently accurate reconstruction at all.

The proposed protocol generates independent runs until two meet a solver-specific residual threshold, or a maximum budget is reached. It selects the lowest-residual reconstruction, applies a zero-shot orientation test, and reports consistency and orientation flags.

Experiments cover DAPS, SITCOM, FAST-DIPS, and PC-DAPS on FFHQ and ImageNet. The paper reports that residual selection with orientation correction nearly matches an orbit-aware oracle on FFHQ, that selecting PC-DAPS particles improves over reporting their average quality, and that adaptive stopping improves accuracy at matched mean compute for two DAPS configurations.

A separate contribution models selection through a class-weighted residual race, evaluates forecasts prospectively, and examines why a small number of runs cannot reliably identify images with very low success probabilities.

**The most convincing contribution is the empirical decomposition of selection, orientation, and candidate coverage. The least established contribution is the broad claim that selection is predictably modeled and that the paper characterizes exactly when stopping helps.**

## 2. Strengths

**S1. The evaluation problem is real and practically relevant.**

A per-metric ground-truth oracle does not produce a single deployable reconstruction. Quantifying the gap between that reporting convention and an actual test-time decision rule is worthwhile.

The premise is supported by the public literature: DAPS explicitly reports best-of-four phase-retrieval results, and TGD explicitly distinguishes ground-truth metric selection from measurement-error selection. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Improving_Diffusion_Inverse_Problem_Solving_with_Decoupled_Noise_Annealing_CVPR_2025_paper.pdf?utm_source=chatgpt.com)

**S2. The separation of failure mechanisms is useful.**

Distinguishing incorrect orientation, incorrect but measurement-consistent outputs, and failure to generate any acceptable candidate makes the experiments more informative than a single PSNR table. These mechanisms require different remedies.

The finding that a residual can be an effective **within-image ranking score** despite being a poor **absolute correctness certificate** is particularly useful.

**S3. Several results have immediate practical value.**

The PC-DAPS result is compelling as a practical intervention: selecting among already generated particles improves the reported quality without additional diffusion sampling. The reported improvements from 79.0% to 84.4% on FFHQ and from 30.8% to 53% on ImageNet are substantial enough to warrant attention.

Likewise, the positive matched-compute results for DAPS-2K and DAPS-4K are meaningful if the evaluation and calibration procedures are confirmed.

**S4. The paper reports inconvenient results.**

It acknowledges that:

- stopping can lose at larger budgets;
- the generic ImageNet orientation scorer is imperfect;
- the forecasting model misses candidate coverage;
- most held-out panels reuse development images;
- part of FFHQ cohort B may overlap denoiser training data.

This transparency improves the credibility of the empirical study.

**S5. Prospective evaluation and exact subset averaging are good methodological choices.**

Freezing forecasts before scoring additional runs is stronger than fitting a curve retrospectively. Exact averaging over subsets also avoids unnecessary Monte Carlo error when estimating finite-budget policies from stored runs.

These are methodological strengths, although they do not resolve the generalization and calibration issues below.

## 3. Weaknesses

### W1. The predictive headline is substantially more favorable than the complete result

**Locations:** Abstract; Section 5; Table 2.

The abstract emphasizes 67/69 covered forecasts in five of six tests. The excluded test is also the largest and a severe failure:

| Selected-run forecasts | Race model | Exchangeable reference |
|---|---:|---:|
| Test 1: FAST-DIPS, FFHQ | 9/42 | 36/42 |
| Remaining five tests | 67/69 | 17/69 |
| All six tests | 76/111 | 53/111 |

The paper correctly states that forecasts within a test are correlated. These totals therefore should not be interpreted as independent Bernoulli trials. Nevertheless, they show why the abstract gives an incomplete impression of reliability.

The proposed explanation—that the reference performs better through cancellation of generation and selection errors—is plausible. But the deployed forecast must predict the final outcome, regardless of which internal component is responsible for failure.

The later experiment that conditions on all 64 realized run classes is a useful diagnostic. It is **not equivalent to forecasting from four observed runs**, because the eventual class composition supplies information unavailable at forecast time.

**Required revision:** distinguish clearly between:

- end-to-end prospective forecasting;
- conditional modeling of selection given candidate composition;
- retrospective diagnosis of model failure.

Report all six tests prominently, together with interval widths, prediction errors, and an interval score that penalizes both excessive width and missed outcomes. Coverage alone is insufficient.

### W2. The strongest positive stopping results are not validated on genuinely new images

**Locations:** Sections 3–4; Table 1; limitations paragraph.

Fresh seeds on previously studied images provide valid evidence about future stochastic runs **conditional on those images**. They offer weaker evidence that the protocol transfers to new images.

This distinction matters because image difficulty is central to the paper, and development choices include the residual threshold, stopping target, orientation prompts, and uncertainty rule.

The image-disjoint ImageNet′ panels are valuable, but their stopping gains are negative:

- −1.5 points at \(N_{\max}=16\);
- −3.1 points at \(N_{\max}=64\).

Thus, the strongest positive stopping claims come from DAPS on previously studied FFHQ images, while the new-image tests do not establish a positive stopping benefit.

This does not invalidate the DAPS findings. It limits their scope.

**Most valuable additional experiment:** evaluate the frozen DAPS-2K and DAPS-4K protocols on a new FFHQ validation cohort, with no image overlap with development or calibration.

### W3. Proposition 1 does not establish the advertised characterization of when stopping helps

**Locations:** Lines 96–117, 341–359, and 641–645.

Proposition 1 expresses expected compute and success using \(s\), \(H_1^+\), \(H_2^+\), and \(H_N^-\). Under an i.i.d. run model, the decomposition is reasonable; the expected stopping-time expression agrees with the truncated negative-binomial calculation.

However, the proposition does not compare the adaptive policy with a fixed-budget policy at equal population-average compute. In particular, it does not establish a necessary-and-sufficient condition for improvement.

The quantities \(H_j^+\) already contain much of the difficult selection behavior. Expressing success in terms of them is an exact decomposition, not yet an explanation of the sign of the matched-compute gain.

Consequently, these statements are too strong:

- “We show exactly when stopping helps.”
- “Where failed runs also fit the measurement, stopping cannot help.”

Some failed runs passing a threshold does not, by itself, imply that adaptive allocation cannot help. The result depends on their prevalence, residual ordering, image heterogeneity, and the marginal benefit of more candidates.

**Required revision:** either establish a precise comparison theorem with explicit assumptions or describe this as an empirical pattern supported by the tested configurations.

### W4. “H3 fails wherever it was tested” appears inconsistent with Table 1

**Locations:** Lines 390–391, 419–421; Table 1.

H3 states that at least half of FLAG\(_C\) images are unsolved. Table 1 contains:

| Panel | FLAG\(_C\) images | Unsolved among them | Fraction |
|---|---:|---:|---:|
| DAPS-4K, FFHQ | 5 | 3 | 60% |
| DAPS-4K, ImageNet | 1 | 1 | 100% |
| FAST-DIPS, ImageNet′, \(N_{\max}=16\) | 4 | 3 | 75% |

These entries satisfy the stated numerical criterion.

Possibly “fails” means that a confidence-bound criterion was not met, or H3 was registered only for a restricted subset of panels. Neither qualification is clear in the main text.

There is a related issue with H1. A point estimate above −1 point does not establish statistical noninferiority when its confidence interval extends substantially below −1. For example, SITCOM’s −0.4 ± 2.0 is compatible with a loss greater than the stated tolerance.

**Required revision:** distinguish “the point estimate meets the criterion,” “the evidence supports the hypothesis,” and “the result is inconclusive.” Reconcile the H3 statement with the table.

### W5. The central model is not specified sufficiently in the main text

**Location:** Section 5.

The race model is a headline contribution, but the main text does not provide enough information to reconstruct its forecasts. Missing or unclear elements include:

- the finite-\(K\) selection formula under the race;
- the likelihood used to fit class weights;
- the Dirichlet hierarchy and hyperparameter estimation;
- the sources of uncertainty included in predictive intervals;
- the handling of images with few informative mixed-class runs;
- whether orientation-error rates depend on residual rank.

There is also an important notation and modeling distinction. In Proposition 1,

\[
s=\Pr(\rho\leq\tau_S).
\]

In Section 5,

\[
s=p+o_g+o_p+c,
\]

which is the probability of belonging to any class except INCONSISTENT.

These are not automatically equal. GOOD is defined by PSNR, and \(\tau\) is the 95th percentile of GOOD residuals, so some GOOD runs exceed the threshold. ORBIT runs need not all pass either.

Equation (2) may be valid under its idealized class assumptions, but identifying its \(s\) with the operational stopping probability requires an additional approximation or assumption.

**Required revision:** use distinct notation for threshold-passing probability and class-membership probability, quantify their discrepancy, and explain how the forecasting model connects to the stopping calculation.

### W6. Noise-energy cancellation does not establish reliable residual ranking

**Location:** Lines 287–306.

The identity

\[
m(\rho_i^2-\rho_j^2)
=
\|e_i\|^2-\|e_j\|^2
-
2\langle\eta,e_i-e_j\rangle
\]

is correct. But canceling the common noise-energy term does not establish that the remaining cross term is small or harmless.

Crucially, \(e_i\) and \(e_j\) depend on reconstructions generated from the noisy measurement. They therefore generally depend on \(\eta\). A concentration argument treating these errors as fixed independently of the noise would not apply without further justification.

The reported 200/205 ordering result is useful empirical evidence. The identity alone is not a theoretical explanation for that level of reliability.

Also, the canceled random quantity is \(\|\eta\|^2\); \(m\sigma^2\) is its expectation.

**Required revision:** present the algebra as motivation, give the construction of the reported pairs and their image-level dependence, and avoid suggesting Gaussian noise alone guarantees good ranking.

### W7. The novelty must be stated more narrowly and demonstrated more directly

The constituent ideas have substantial precedent:

- residual-based selection;
- recognition that ground-truth best-of-\(K\) is unavailable at deployment;
- recognition that measurement consistency cannot resolve prior-dependent ambiguities;
- adaptive sampling;
- class-dependent ranking models.

TGD explicitly implements measurement-error selection, while PC-DAPS explicitly identifies best-of-\(K\) evaluation as masking prior-inconsistent reconstructions. [arxiv.org](https://arxiv.org/html/2605.03712v1?utm_source=chatgpt.com)

The manuscript acknowledges these connections. Its strongest novelty is therefore likely the **systematic audit, quantitative decomposition, and prospective analysis**, rather than any one component of the protocol.

An empirical contribution can be sufficient. But it then needs especially strong validation and comparisons. The main text provides limited evidence against other simple deployment choices, such as residual-plus-prior scoring or alternative stopping rules.

**Required revision:** state exactly what is newly established beyond prior observations. Include a compact comparison of several practical selectors and stopping rules under the same candidate pool and compute accounting.

### W8. Calibration accuracy is assessed indirectly, and the compute accounting needs clarification

**Locations:** Lines 309–325; Table 1.

The label-free threshold is reported to lie within 2% of the labeled threshold. That does not establish similar policy behavior. If residuals concentrate tightly near the cutoff, a 2% change may substantially affect:

- the probability of passing;
- the stopping time;
- the selected output;
- the FLAG\(_C\) rate.

The displayed development sensitivity range, \(\tau(1\pm0.003)\), is narrower than the reported label-free discrepancy.

Moreover, the label-free rule uses a four-run pilot. Table 1 reports subsequent mean budgets below four runs for several configurations. This is legitimate if calibration uses a separate reusable population, but the main text should make the accounting explicit.

**Required revision:** identify the pilot images, whether their runs are reused, and whether calibration compute is excluded or amortized. Compare labeled and label-free calibration by resulting accuracy and compute, not just threshold proximity.

### W9. The DAPS reproduction gap remains materially unresolved

**Location:** Lines 549–559.

The manuscript reports 22.3 dB under the reproduced DAPS best-of-four protocol, compared with 25.78 dB in the original paper. The original table indeed reports 25.78 dB for ImageNet phase retrieval. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Improving_Diffusion_Inverse_Problem_Solving_with_Decoupled_Noise_Annealing_CVPR_2025_paper.pdf?utm_source=chatgpt.com)

A roughly 3.5 dB discrepancy is large for a paper auditing evaluation practice. Reporting another independent rerun with a lower value is informative, but does not identify the cause.

The within-pool comparison between \(E_1\) and the reproduced oracle remains useful. What requires restraint is the language around “beating the published oracle”: the reported 23.1 dB beats the oracle **on this reproduction’s candidate pool**, not the original paper’s 25.78 dB result.

**Required revision:** provide a concise reproduction audit covering checkpoint, image identities, preprocessing, measurement generation, solver settings, and metric implementation. Use wording that clearly separates the published selection procedure from the originally published numerical result.

### W10. “Beating the oracle” compares different permitted output sets

The raw oracle selects from

\[
\{\hat x_1,\ldots,\hat x_K\},
\]

whereas \(E_1\) can output a rotated candidate. Thus, beating the raw oracle is possible because the proposed method has a larger action set.

The paper explains this, so the comparison is not mathematically incorrect. Nevertheless, it receives more rhetorical emphasis than its substantive meaning warrants.

The informative comparison is with an oracle allowed the same transformations. On ImageNet, gaps to the orbit oracle remain appreciable for FAST-DIPS.

**Required revision:** emphasize recovery of the orbit-oracle gap and present the raw-oracle comparison as evidence that legacy reporting overlooks a cheap postprocessing step.

### W11. The uncertainty flag is a relative ranking rule, not a calibrated confidence estimate

**Location:** Lines 279–281, 332–336, and H4.

FLAG\(_O\) marks the lowest 30% of margins among evaluated outputs. This is a batch-relative rule: it flags a fixed fraction regardless of whether the batch is easy or difficult, and an individual output’s flag can change when unrelated outputs are added.

The reported increase in accuracy after excluding flagged outputs demonstrates selective ranking value. It does not establish calibrated confidence or a fixed deployment threshold.

**Required revision:** report a risk–coverage curve, retained sample counts, comparison with residual-only confidence, and transfer of a frozen absolute margin threshold to a new cohort. Alternatively, explicitly frame the flag as a batch-level triage mechanism.

### W12. “Unsolved” sometimes shifts from a finite-budget observation to a stronger conclusion

**Locations:** Abstract; Sections 5–7.

The empirically justified statement is:

> No candidate satisfying the selected quality criterion was found by this solver within the evaluated budget.

That does not establish that the image is intrinsically unrecoverable or that the solver’s success probability is zero.

The paper understands this distinction: Section 7 describes the observed zero-success share as an upper bound on an unsolved population component. However, statements such as “no selection can recover such images” should consistently specify **selection from the generated candidate pool**.

The inference that “generation is not predictable” is also too broad. The experiments establish that the tested generation model is poorly calibrated in important settings. They do not establish that better generation models cannot predict the relevant probabilities.

Finally, the four-run moment argument concerns information contained in class-count observations under the mixture formulation. It does not rule out useful predictions from image features, residual trajectories, or other side information.

## 4. Limitations

These are scope restrictions rather than necessarily fatal weaknesses.

- **Narrow measurement setting.** The study uses one simulated Fourier-magnitude operator, mainly one noise level, and two natural-image datasets. Its conclusions should not automatically extend to physical diffraction measurements or other noise models.

- **Limited image diversity and independence.** Repeated sampling produces many reconstructions but does not increase the number of independently sampled images. Cohort B’s screening and possible training overlap further complicate population-level interpretation.

- **Threshold-defined success.** “GOOD” uses dataset-specific PSNR cutoffs. The manuscript reports some sensitivity checks, but the headline conclusions would be stronger with broader threshold sweeps and continuous quality metrics.

- **An auxiliary learned prior determines orientation.** CLIP supplies orientation preferences beyond the diffusion solver’s measurement model. This is legitimate, but success on predominantly upright benchmarks does not establish correct orientation recovery for arbitrary acquisitions.

- **Restricted symmetry handling.** The operational correction considers only global rotation, while the oracle considers eight channelwise rotations. Other ambiguities and the assumed uniqueness regime are not fully resolved.

- **Average compute is not latency.** Equal mean runs is a reasonable within-solver efficiency measure, but it does not characterize tail latency, batching, or the distribution of resources across images.

- **Single-output reconstruction differs from posterior sampling.** Selecting the most promising particle may improve a point-estimate metric while changing the distribution of returned samples. The recommendation should be scoped to applications seeking one reconstruction.

## 5. Questions for the authors

1. **What information was available at each freeze point?** Please provide a timeline separating protocol design, calibration, forecast fitting, generation, and scoring, with image overlap identified.

2. **Can the positive DAPS stopping gains be reproduced on entirely new FFHQ images?** This is the most important missing generalization test.

3. **How is H3 reconciled with the Table 1 entries that meet its stated criterion?** Was success defined using point estimates, confidence bounds, or another registered rule?

4. **What exactly does Proposition 1 establish about matched-compute superiority?** Is a separate comparison result proved in the supplement?

5. **How do the two definitions of \(s\) relate?** What fraction of GOOD and ORBIT runs fail the operational threshold, and how is that handled in forecasts?

6. **How do the race and reference models compare across all six tests using prediction error, interval width, and a proper interval score?**

7. **What changes if the generation model allows a distinct near-zero-success component or a more flexible mixing distribution?** Does the claimed separation between predictable selection and unpredictable generation remain?

8. **What is the complete calibration cost?** Are the four-run pilot images disjoint from evaluation, and are those runs counted or amortized?

9. **What explains the DAPS reproduction discrepancy?** Which parts of the original experimental configuration have been verified exactly?

10. **How robust are selection and stopping to threshold perturbations of the size actually produced by label-free calibration?**

11. **Does FLAG\(_O\) outperform residual-based confidence, and does a frozen orientation-margin threshold transfer to new images?**

12. **What do representative qualitative cases look like?** A main-text panel showing successful residual selection, successful rotation correction, a confidently wrong correction, and a measurement-consistent failure would make the analysis easier to assess.

## 6. Rating

**Provisional recommendation: Weak Reject.**

**Illustrative score: 4/10**, using a scale where 5–6 is the borderline region. This is not a claim about the official CVPR 2027 review scale.

**Confidence: 3/5 — moderate.** The main-text evidence supports the concerns above, but the supplement may resolve several specification and statistical questions.

| Criterion | Assessment |
|---|---|
| Importance | Strong practical motivation |
| Originality | Moderate; strongest as an empirical and methodological audit |
| Technical soundness | Plausible core calculations, but broader conclusions exceed what is established |
| Experimental evidence | Substantial repeated sampling; limited new-image confirmation of positive gains |
| Clarity | Generally readable, but several claims need more precise scope |
| Reproducibility | Cannot yet be assessed from the main text alone |

**What would most change my recommendation:** new-image confirmation of the positive DAPS results; a complete and balanced forecasting evaluation; reconciliation of the hypothesis-testing statements; and a precise account of calibration, model assumptions, and matched-compute comparison.

I would not reject this paper simply because the proposed procedure is simple. The present acceptance barrier is whether the evidence supports the paper’s strongest conclusions.