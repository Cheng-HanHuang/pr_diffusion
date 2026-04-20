# Phase 10+ second-host literature review and recommendation

## Scope and decision target

This note reviews candidate diffusion inverse-problem hosts and selects a **second host** for Phase 10+ that fits the current project constraints:

- keep the current DDPM face-prior stack,
- avoid reusing DPS/ReSample as the host algorithm family,
- preserve a clean plug-and-play path for the Phase-5-style masked projection mechanism,
- and prefer publicly available source code with practical implementation paths.

---

## Current project facts that constrain host choice

From the current internal plans and results:

1. The core gain appears to come from **late low-frequency projection enforcement**, while masked score-only is weak.
2. The project already has robust code and sweeps around Noise Picking + SITCOM and wants a second host to test mechanism portability.
3. Phase 10 asks for a source-backed host and explicitly warns against inventing a hand-crafted new solver.

These constraints favor hosts with an explicit data-consistency/projection step that can be swapped with low-frequency masked projection.

---

## Candidate host families from literature

### A. DiffPIR (useful reference, but not PR-specific)

**Paper/repo signal**

- DiffPIR is a plug-and-play diffusion restoration method with open-source implementation.
- Its reported benchmarks are super-resolution, deblurring, and inpainting (not Fourier phase retrieval), so it should be treated as a transferable host design pattern rather than PR-native evidence.
- The official repository states it integrates plug-and-play iterations into diffusion sampling and supports practical inference workflows.
- The method is written in an iterative denoise + data-subproblem form, which is structurally aligned with “insert late masked projection” experiments.

**Why it matches Phase 10+**

- It is a distinct host family from DPS/SITCOM-style posterior-gradient guidance.
- It naturally exposes a data-fidelity/prox step where low-frequency Fourier magnitude projection can be injected.
- It can be implemented as a single-trajectory baseline first (mirroring the current one-candidate idea), then expanded to variants.

**Risk notes**

- Hyperparameters are known to require tuning (especially balancing data fidelity and denoiser terms).
- Existing off-the-shelf wrappers often assume linear physics; phase retrieval is nonlinear, so a custom data-fidelity/prox operator is required.

### B. DiffFPR (new primary recommendation)

**Paper/repo signal**

- DiffFPR is specifically designed for oversampled Fourier phase retrieval and is validated on that task.
- The ICML 2024 PMLR page explicitly states both the phase-retrieval focus and public code availability.

**Why it matches Phase 10+ better than DiffPIR**

- It is already phase-retrieval-native, reducing task mismatch risk.
- It integrates a diffusion prior with an iterative Fourier PR engine, which is exactly where late low-frequency masking variants can be tested.
- It remains distinct from DPS/ReSample family while still being source-backed and practical.

### C. DDRM / DDRM-derived options (do not choose as the main second host)

**Paper signal**

- DDRM was introduced for **linear** inverse problems and typically relies on linear-operator structure.
- DDRM-PR extends the idea to nonlinear Fourier phase retrieval and is relevant scientifically.

**Why not first choice now**

- Base DDRM assumptions are less aligned with the current nonlinear magnitude-only pipeline.
- Publicly available, production-ready phase-retrieval DDRM code is less straightforward to integrate than DiffFPR-style hosts.
- This is better treated as a later comparison track once the Phase 10 mechanism transfer question is answered.

### D. MCG-Diff and other posterior-sampling variants (defer)

- MCG-Diff has available code but focuses on Bayesian **linear** inverse problems.
- It is useful as an additional baseline family later, but not the best immediate host for nonlinear phase retrieval mechanism transfer with low engineering risk.

---

## Final recommendation

Use **DiffFPR-style host** as the official “second host” for Phase 10 and beyond, with DiffPIR treated as a secondary fallback reference implementation pattern.

Concretely:

1. Implement a `difffpr_host` runner in-repo using current UNet + scheduler plus a Fourier phase-retrieval iterative engine (RAAR-style update as in DiffFPR).
2. Keep a single-trajectory base variant first (to parallel current Phase-10 logic).
3. Add three projection variants:
   - no lowfreq masked projection,
   - lowfreq projection always-on,
   - lowfreq projection late (`start=400`, `r=0.5`), plus `r=0.2` check.
4. Reuse current run-level/image-level CSV format so Phase 10/11 remain comparable with existing analyses.

This preserves a clean scientific story:

- if late low-frequency projection helps a **phase-retrieval-native** host too, Phase-5 mechanism portability is strongly supported;
- if gains fail to transfer, then mechanism may be host-specific and should be framed accordingly.

---

## Minimal implementation order

1. **Pilot implementation** (`validation_10`, 5 seeds):
   - `difffpr-base`
   - `difffpr-lateproj-0.5`
   - `difffpr-alwaysproj-0.5`
2. Add `difffpr-lateproj-0.2` and compare against current NP canonical.
3. Run full `validation_25` with 10 seeds.
4. Promote best variant into Phase 11 forcing-type comparison and Phase 12 cross-host confirmation.

---

## Sources consulted

- DiffPIR paper (arXiv): https://arxiv.org/abs/2305.08995
- DiffPIR official code: https://github.com/yuanzhi-zhu/DiffPIR
- DiffFPR paper (ICML 2024, PMLR): https://proceedings.mlr.press/v235/li24bj.html
- DiffFPR official code: https://github.com/Chilie/DiffFPR
- DDRM paper (arXiv): https://arxiv.org/abs/2201.11793
- DDRM official code: https://github.com/bahjat-kawar/ddrm
- DDRM-PR paper (arXiv): https://arxiv.org/abs/2501.03030
- DPS paper (arXiv, for family distinction): https://arxiv.org/abs/2209.14687
- MCG-Diff official repo: https://github.com/gabrielvc/mcg_diff
- DeepInverse DiffPIR docs (algorithm form and practical notes): https://deepinv.github.io/deepinv/_modules/deepinv/sampling/diffusion.html

---

## Addendum: DiffPIR nonlinearity and RED-diff comparison (2026-04-18)

### Is DiffPIR non-linear just because it includes inpainting?

No. Inpainting is usually modeled as a **linear masking operator** (`y = Mx`), so inpainting by itself is not evidence of nonlinear-operator support.

That said, DiffPIR's optimization form is written with a generic forward operator `H(x)`, so a nonlinear data subproblem is possible in principle if you implement a solver for that subproblem.

### RED-diff as a comparison baseline

RED-diff is a strong and practical comparison candidate for your setting.

Reasons:

1. The RED-diff repo explicitly includes both linear and nonlinear degradations and lists a `phase_retrieval` option in its degradation presets.
2. RED-diff is already structured as inverse-problem optimization with a diffusion prior, which makes plug-and-play style insertion and measurement-enforcement ablations straightforward.
3. RED-diff can be used as a **comparison host** even if DiffFPR remains the primary phase-retrieval-native second-host recommendation.

### Updated practical ranking for Phase 10+

1. **Primary second host:** DiffFPR-style (phase-retrieval-native).
2. **Primary comparison baseline:** RED-diff (`phase_retrieval` degradation).
3. **Fallback transferable pattern:** DiffPIR-style host (requires custom nonlinear data-step implementation for PR).

### Suggested experiment block (small pilot)

On `validation_10`, 5 seeds:

- `difffpr-base`
- `difffpr-lateproj-0.5`
- `reddiff-phase-retrieval` (default)
- `reddiff-phase-retrieval + late lowfreq projection` (if easy to inject)

If RED-diff integration is quick, this gives a stronger external comparison than using only in-repo host variants.
