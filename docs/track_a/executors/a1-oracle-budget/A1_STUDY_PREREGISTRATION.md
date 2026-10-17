# A1 study preregistration — independent oracle best-of-K and total charged compute

**Version:** `A1.0-DESIGN-FROZEN-20261017`, date 2026-10-17. **Owner:** continuing Track A executor, incorporating completed A0. **Study state:** `DESIGN_ONLY / NO_PAC_OR_GPU`. This preregistration governs proposed A1 development work; it does not certify source preflight, reserve any data, or authorize any actual-image run.

**Source documents:** [Track A oracle-first plan](../../../../docs/planning/track_a/2026-10-16_ORACLE_FIRST_PUBLICATION_REFRAME.md), [A1 handoff](00_EXECUTOR_HANDOFF.md), [A0 return on its frozen branch](https://github.com/Cheng-HanHuang/pr_diffusion/blob/ecc66b890b63123c0445a13d33277e09f6ab4ab5/docs/track_a/executors/a0-native-selector-audit/A0_B22_RETROSPECTIVE_RESULTS_2026-10-16.md), [A1 B22 source-backed evidence](A1_B22_ORACLE_EVIDENCE.md), shared [issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41), the [DAPS paper §4.1](https://arxiv.org/html/2407.01521v3) and [SITCOM paper Appendix G](https://arxiv.org/html/2410.04479v2).

## 1. Scientific question, distinct from executable selection

For frozen image/measurement (i), method/policy (m), ordered independent seed stream (r=1,\ldots,K), define the *offline GT-oracle* candidate-set score

\[
Q^*_{i,m}(K)=\max_{1\le r\le K}\operatorname{PSNR}_{\rm raw,RGB}(\widehat x_{i,m,r},x_i),\qquad
S^*_{m}(K)=\frac1n\sum_i\mathbf 1\{Q^*_{i,m}(K)\ge25\}.
\]

Also report mean, median, fifth percentile (`q05`, interpolation rule `linear`), minimum and fraction with (Q^* <20) dB. **GT is used only to evaluate the candidate set after the run**; never used for restarter selection, execution, early stopping, interpolation, method-specific configuration choice or measurement generation.

The **native clean-free** selected candidate (where a documented method-native cross-run selector exists) is a secondary, separately tagged quantity; if no native K policy exists, report N/A rather than inventing cross-method measurement-error ranking. PSNR is the single designated candidate selection metric for all GT oracle rows. SSIM and LPIPS, when computed, are evaluated on **the PSNR-chosen candidate**, not independent metric-specific oracle winners. Rotation-180-aware PSNR is auxiliary and cannot replace raw orientation PSNR.

## 2. Frozen candidate generators and comparisons

**Native problem:** 256×256 FFHQ RGB; Fourier phase retrieval magnitude, oversampling 2.0, observation noise `sigma_y=0.05`; one *identical saved noisy measurement* shared across methods for each image, verified byte/content SHA before each arm. Raw signed observations are used by DAPS/SITCOM; NP's historical `clamp_min_zero_in_memory` is preserved as an explicit NP method identity, NOT silently repaired. Primary scoring maps float `[-1,1]` outputs to RGB `[0,1]` after clamping and evaluates PSNR on the *unquantized* float raw orientation for **all arms**; never compare B22 float with B24 saved-PNG 8-bit PSNR.

Frozen intended study arms (all fixed prior to seeing new GT scores):

| Arm | K | Candidate semantics | Status/gate |
|---|---:|---|---|
| DAPS-independent4 | 4 | 4 separately seeded native DAPS full fresh starts | **Essential new**; B22 only had DAPS2 |
| SITCOM-ODE-independent4 | 4 | 4 *independently initialized* native SITCOM starts with full RNG reset or process isolation | **Essential new**; B22 sequential RNG K4 is not equivalent |
| NP-LF4 | 4 | 4 independent LF candidate-generation trajectories; B22 LF config `score_mode=lf`, `lambda=0` | Essential ablation |
| NP-S2-4 | 4 | 4 independent S2 preprojection-regularized trajectories; B22 S2 `score_mode=prev_l2`, `lambda=0.01` preprojection only | **Frozen primary NP4 candidate arm** |
| NP-mixed8 | 8 | fixed 4 LF + 4 S2, disjoint seed namespaces, no GT tuning of allocation | Secondary K8 portfolio |
| SITCOM-ODE-independent8 | 8 | 8 independently seeded native SITCOM trajectories | Optional **separate predeclared extension**, NOT assumed observed |
| NP-mixed4 | 4 | fixed 2 LF + 2 S2, total 4 candidate evaluations | Optional ablation, non-primary; cannot retrospectively optimize allocation |

For all homogeneous arms report nested prefixes `K=1,2,4` from the **predeclared candidate ordering**; `K=8` only when eight actual independent attempts were executed. For mixed4 use 2LF+2S2 (order LF0,S20,LF1,S21), for mixed8 use cyclic balanced order (LF0,S20,LF1,S21,LF2,S22,LF3,S23), both fixed **before** first real-image run. Never interpret NP mixed8 as eight i.i.d. repetitions of one sampler. No computational credit for unused candidates in a prefix row; a full K8 output incurs its full eight-run cost.

**Two independent axes:**
(A) *Run matched*: DAPS4 vs independent SITCOM4 vs NP-LF4 vs NP-S2-4 (NP-mixed4 optional) at exactly four complete candidates per image.
(B) *Compute matched*: methods' **completed** candidate prefixes at a common charged per-image budget, using a prospective cost ledger, not inferred from number of runs. A source/arm incapable of completing a candidate within budget is reported explicitly (zero available), not silently dropped. No interpolation between K points and no oracle stopping-time claims.

## 3. RNG independence, native implementation and source freeze

A1 candidate seeds (not measurement seeds) are generated by domain-separated SHA-256: `seed63(tag,image_id,k) = int.from_bytes(SHA256(UTF8("PRDIFF-A1-V1|"+tag+"|"+image_id+"|"+str(k)))[0:8],"big") & ((1<<63)-1)`. `tag` is one of `DAPS`, `SITCOM`, `NP_LF`, `NP_S2`. `k` is the zero-based replicate 0–7; different methods/configs use distinct tags. A stable seed mapping is computed and frozen once image IDs are reserved. An NP mixed portfolio **reuses** the exact independently run LF and S2 candidates for the same image; **do not rerun and re-charge a duplicate candidate** merely to draw a mixed policy curve. All simulation RNGs (Python, NumPy, CPU torch, CUDA torch, method scheduler and generator) must be seeded/reinitialized per candidate as applicable. For SITCOM, full process isolation is preferred if the sampler retains state after an RNG reset; verify actual restart independence before quality work. Do not equate "different numeric seeds" with verified independent reconstruction streams without a deterministic replay test.

New measurement seeds are generated with a **different** `A1_MEAS_V1` domain and frozen *once per image*, not per method. Pixel inputs, operator, noise distribution, shape, noise normalization, model checkpoint and raw observation content hash must match across all native pipelines. Freeze DAPS `phase-retrieval` native solver schedule, SITCOM B22 `anneal_steps=200; diff_steps=5; lgvd_steps=100; tau=0.01; selector_step=160`, and NP B22 `num_steps=1000; soft=5; hard=1; proj_start=300; proj_radius=0.2; LF/S2 settings` unless a source-to-source discrepancy requires **STOP + protocol revision before data**. No GT-derived hyperparameter changes.

**Source identities still require independent PAC preflight**: A0 validated B22 native SITCOM HEAD `275ab67efbd8146bffca20155171ba6be1169c09` and its one-line posterior mkdir patch with native checkpoint content SHA `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`; historical NP DiffFPR HEAD `a45ffe58f18fed8a63d3446600424e2b08733524`; historical B24 DAPS HEAD `e7a77d094167084faed19b599b96673b7bb11447` with dirty tracked source diff hash `fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69`. None of these alone certifies a **new** native DAPS4 implementation or a fresh run. Need current HEAD/full/source-only tracked diffs, file hashes, model SHA, native operator/normalization comparison, environment versions and tiny no-GPU launch dry-run before running a real image. Preserve all dirty native PAC checkouts without reset/patch; if necessary make a separate controlled clean copy after user reviews the difference. **Never use B24 full-diff mismatch as evidence of B22 run invalidity or invent a B24 verification pass.**

## 4. Cohort / GT exposure firewall

- B21/B22 final100 and B24 DEV80 are fully **exposed**; may be used to set these hypotheses and write archival diagnostics, **not** as untouched prospective evaluation.
- B24 confirmation305 is **completely excluded** (no registry, path, file, metadata, ID, GT, reconstructed tensors, hashed membership checks or commands referring to it).
- Before new real-image data, inspect **only eligible allowed** manifests for the publicly available FFHQ image IDs, already-exposed B21/B22 and B24 DEV80 membership, and live cross-track reservations in issue #41. Do not assume there is a fresh disjoint pool before this targeted check.
- Development cohort: propose eight images, using a deterministic *predeclared hashed image-ID ranking* after excluding exposed panels/other reserved IDs, subject to user's acceptance. Do **not** select images based on original GT appearance, algorithm failures, B24 locked subsets, any PSNR or learned difficulty.
- Untouched final cohort: **not allocated**; requires a separate source/inventory and issue #41 joint-cohort approval, disjoint IDs and independent seed namespace. Provisional target `n>=200` eligible independent test images to improve precision, **not** authorization or proof of feasibility. Revisit precision/resources **before any final GT exposure**, not after results are seen.
- Register `[DATA RESERVATION]` with split/eligible ranking recipe, IDs only when allowed, GT rule, image/measurement seed hashes and output root; register `[RUN RESERVATION]` with actual physical GPU IDs, process/memory/time costs and frozen source commit immediately prior to launch. Track B has priority if GPU demand conflicts.

## 5. Required preregistered reporting and uncertainty

**Primary K4 reliability contrast:** NP-S2-4 minus **independent SITCOM-ODE4** on the exact same images and raw measurement. Report paired Good25 difference in percentage points, exact two-sided McNemar, 95% paired-image bootstrap interval, mean PSNR difference and q05 difference (with 95% paired-image bootstrap intervals), Bad20 delta and observed full compute. No GT-driven switching of primary to NP LF4 or NP mixed8. Report mean improvement and tail reliability **as separate objectives**, not a single victory score.

**Primary strong reliability claim gate:** only if newly held-out primary contrast has Good25 difference **at least +5 percentage points** with a paired 95% bootstrap lower bound **above zero**, *and* q05 point estimate nonnegative, and a supporting cost-normalized result at a transparently comparable charged budget. This is a deliberately stringent prospective reliability claim, not a prediction of success. If only mean PSNR improves with no verified tail advantage, report *mean-quality trade-off* rather than a lower-tail reliability gain.

**Secondary gates:** DAPS4/NP-S2-4 run-matched metrics; DP/NP/SITCOM compute-frontier; NP LF vs S2 vs fixed mixed4 at K4; NP mixed8 vs optionally independent SITCOM8 at comparable charged budget; clean-free native selector regret/Good25. Multiplicity: declare H1 K4 vs SITCOM as the only confirmatory reliability contrast; all other hypothesis and venue viability judgments exploratory unless a separately frozen confirmatory protocol is approved before final data. For q05 confidence interval, use paired image resampling, with explicit instability caveat and sensitivity to individual hard cases. Report binomial exact CI for each Good25 rate as a supplementary measure. Publish complete per-image paired outcomes; compute-matching cannot drop hard images or failed arms.

**Engineering development pilot gate:** 8 exposed-to-development but formerly untouched images; DO NOT use pilot PSNR to modify alpha/seed/config allocation, objective or primary choice. Pilot is an execution/provenance/memory/cost gate. If poor source compatibility or severe performance outlier occurs, stop/record and make a dated new protocol before touching final data. A larger final test cannot re-use these images as held-out.

## 6. Stage boundaries, GPU caps and non-destructive execution

**A1.0 (NOW):** documents, literature/source audit, read-only targeted PAC *inventory request*, deterministic seed/split script and CPU-only synthetic validation. No new measurement/no model inference/no FFT/no GPU. No unauthorized directory changes.

**A1.1 (NOT YET APPROVED TO RUN):** after source + split + ledger checks, an 8-image *development engineering smoke* with 4 DAPS + 4 independent SITCOM + 4 NP LF + 4 NP S2 trajectories per image, totaling **128** unique full trajectories; retrospective NP mixed8 is an **offline recombination of those same 8 NP candidates**, not a fifth run arm. Historical per-image candidate-time illustration ~625(DAPS4 projected)+196(SITCOM4 historical sequential)+205(NP LF4)+206(NP S2-4)=~1232 candidate reconstruction seconds; across 8 images ~2.74 candidate GPU-hours **before** model load, process overhead and hardware effects. This is a *projection*, not a run result. Pilot maximum **6 cumulative allocated GPU-hours across all physical devices** and **32 GiB VRAM process peak per GPU** unless a stricter observed hardware gate is set; zero hidden retries/extra K8 SITCOM. Exact per-candidate and per-GPU wall ceilings, available devices and failure management are to be determined in the dedicated A1.0 PAC preflight **before any launch command is proposed**. Every pilot candidate must have source/measurement/method/seed/output SHA, separate RNG identity, fixed-representation PSNR after generation, monotone prefix oracle check, no overwrites, and an explicit STOP on any missing identity. Retain GPU telemetry receipts, terminal failures and stopped attempts. In all cases the user alone invokes a pinned safe PAC command.

**A1.2 (LATER):** any larger prospective study requires issue #41 cohort + machine reservation and Track A/Track B governance; no implicit move from a successful pilot to a full run. **Track B has GPU priority** when hardware/time conflict. No script may auto-start full-scale experimentation after smoke PASS.

Reserved (unverified) A1 PAC paths only:
- worktree: `/egr/research-pac/huang248/pr_diffusion_track_a_a1_oracle_budget`;
- output: `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/`;
- conda envs strictly under `/egr/research-pac/huang248/conda-envs`.

No editing source under B19–B26, A0, Track B, or native external repositories. No `set -e`, `set -u`, `set -o pipefail`, `exit`, destructive git or broad storage search in user shell. On STOP, preserve new A1-specific logs/partial outputs and return to interactive prompt. GitHub branch and commit pinned in the receipt before user-runnable stage.

## 7. Publication go/no-go independent of historical overinterpretation

The small-venue paper will be viable only with (a) a true method/contribution grounded in NP candidate-generation design rather than merely a renamed benchmark, (b) meaningful **prospective** oracle best-of-K reliability/cost trade-off against strong independent DAPS/SITCOM native baselines, (c) transparent sample/measurement/source/RNG/compute accounting, and (d) honest uncertainty. A negative result at K4 or cost matched is not repaired by post hoc adding eight NP trajectories, switching to favorable GT scores, reusing DEV80/final100 or renaming a heuristic selector an oracle. New stage results warrant a fresh committed `A1_EXECUTOR_RETURN.md` or `A1_STOP_REPORT.md`.

**A1.0 hard-stop conditions before any PAC GPU:** missing native source/operator provenance; cannot demonstrate independent SITCOM/DAPS seeds; no disjoint eligible development cohort; conflicting reserved GPUs; no comparable float-metric pipeline; cannot account full run work; first full-image result requires reading excluded B24 confirmation; or any leak from GT into candidate generation. No silent fallback.
