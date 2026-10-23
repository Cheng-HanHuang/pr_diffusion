# Track R — R0 complete: frozen scientific return to planner (2026-10-23)

**FROM:** R0 executor (scientific audit and implementation)  
**TO:** Track R scientific planner (Claude) / project metaplanner  
**STATUS:** R0 completed, independently replayed; **scientific planner review required before R1**  
**Decision source:** Frozen R0 v1 (2026-10-21), unchanged  
**Source scope:** exposed B22, B24 DEV80, A1.2/A1.5 development only; **no B24 confirmation305 access**.

## 0. Evidence and precise provenance

User-uploaded, independently inspected `R0_8_PLANNER_RETURN.zip`:
SHA256 `13c008f3bca59b7d8d717ec3215fa1f783ca453b31b9001886c84633b513efab`.
On PAC:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_8_overnight_20261023T082909Z/R0_8_PLANNER_RETURN.zip`.

The operator verified the ZIP SHA and `ZipFile.testzip() is None`. The
receipt is `R0_8_PLANNER_HANDOFF_READY`; R0.8 independently replayed
3,240 candidate labels, P1/P2 denominators, all 37 P4 tables and pooled
DAPS/FAST P3 detector AUROCs. The printed `SystemExit(0)` traceback
and shell RC=8 were due solely to an entrypoint that caught
`sys.exit(0)` with `except BaseException`. The output ZIP and receipt
were already written; this bug was subsequently patched and documented
at `docs/track_r/executors/r0/R0_8_COMPLETE_EXIT_CODE_CORRECTION_20261023.md`.
**No rerun is required to correct the exit status.**

R0.7 complete:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_7_cpu_20261023T080434Z`;
executor `c6dbd6089579c79e9e6fd4a9dd20fc402bc1185b`.
Both CPU stages RC=0; status `R0_7_ANALYSIS_COMPLETE`;
3,240/3,240 cases, **no missing GOOD-based tau**, **zero recorded
PSNR-replay anomalies**. The original R0.2 optional measurement SHA
was blank for 1,400 of 2,520 historical rows; the R0.7 strict join
rehashed signed-y files against mandatory frozen R0.4 hashes.

R0.4: 2,520/2,520 historical candidates scored;
R0.5: 720/720 newly generated native FAST roots;
R0.6: 720/720 newly scored FAST roots.
**Aggregate PAC GPU allocation = 3.4309563972961366 hours / 16 hours.**
No further GPU operations in R0.7/R0.8. All R0 scientific conclusions
are retrospective, using exposed/development data, not prospective
confirmation. Full numerical files, policies, CIs and secondary
diagnostics reside in the above ZIP (`R0_7_ANALYSIS.json`,
`R0_7_REPORT.md`, `R0_8_P4_MATRIX.csv` etc.). This memo is a
faithful high-level return, not a replacement for those files.

**Grounding note:** The package contains aggregate/scientific tables
and exploratory examples, not the entire R0.7 per-run GT JSONL or
private offline GT-path catalog. The independent per-run audit is
confirmed by the PAC receipt, not reproduced on this host from the
aggregated ZIP.

## 1. P1 — prevalence: primary D-material gate NOT met

| Family | Single runs | Raw failures (PSNR<25) | Exact-orbit failures | Share among failures | Image-bootstrap 95% CI |
|:--|--:|--:|--:|--:|:--|
| DAPS pooled S1+S2+S3 | 552 | 198 | 26 | **13.13%** | [7.02%,20.10%] |
| FAST pooled S4+S5 | 752 | 308 | 11 | **3.57%** | [1.34%,6.39%] |
| SITCOM pooled | 752 | 314 | 25 | 7.96% | [4.26%,12.28%] |
| NP pooled S1+S2 | 1,120 | 359 | 22 | 6.13% | [2.93%,10.15%] |
| NP_LF A1.2 | 32 | 11 | 0 | 0% | [0%,0%] |
| NP_S2 A1.2 | 32 | 10 | 1 | 10% | [0%,50%] |

Pooled families are defined by the frozen evaluator and intentionally
keep NP_LF/NP_S2 distinct from the older NP family. Across **all**
3,240 raw single-run results: 2,040 GOOD, 85 ORBIT_ERROR,
324 NONORBIT_CONSISTENT and 791 NONORBIT_INCONSISTENT.
Thus raw failures = 1,200, with just **85/1,200=7.08%**
exact-orbit-correctable failures. Most failures cannot be fixed
by any of the eight exact-orbit changes.

The frozen **D-material threshold is >=20% for FAST or DAPS**;
neither point estimate reaches it. DAPS upper CI crosses 20%, so
avoid claiming the population value is conclusively below 20%;
FAST has a much tighter upper CI well below 20%.
S1 B22 DAPS alone was 7/35=20.0%, but the primary pooled DAPS
gate is 26/198=13.13%, not the source-specific S1 result.

Nearly all orbit errors are global 180°: DAPS 25/26,
FAST 11/11, SITCOM 25/25, historical NP 22/22;
one DAPS case is a partial per-channel orbit error.
This is an additional scientific limitation on a broad
"general orbit repair" narrative.

## 2. P2 — orbit repair: frozen D-repair PASS by a wide margin

| Repair selector | Corrected ORBIT_ERROR | Harm on previously GOOD |
|:--|--:|--:|
| **Diffusion prior S (primary)** | **84/85 = 98.82%** | **0/2,040 = 0%** |
| `S_mid` (registered secondary, half the timesteps) | **85/85 = 100%** | **0/2,040 = 0%** |
| UnivFD argmin-p_fake orbit | 62/85 = 72.94% | 220/2,040 = 10.78% |

Primary prior-by-family: DAPS **26/26**, FAST **11/11**,
SITCOM **25/25**, historical NP **21/22**, NP_LF **0/0**
(undefined), NP_S2 **1/1**. Prior harm on GOOD is 0 for
all six family strata. Every one of these numbers is defined
using canonical RGB8 PSNR and frozen Good25.

The frozen **D-repair PASS** is success >=90% and harm <=1%,
so both criteria pass. This is a compelling *technical*
ability to disambiguate the orientation of a face in place,
although no claim of novelty has been established.
UnivFD is substantially worse at the **orbit-choice** task
despite sometimes being better at the **failure-detection** task;
the two tasks must not be conflated.

Important qualification: `0/2,040` is an **observed event
count**, not a zero-risk guarantee. The image-bootstrap CI is
degenerate [0,0] when no sampled observation has an event.
Moreover, the P2 `harm` definition specifically asks whether
a previously GOOD case drops below 25 dB. Some already-bad
reconstructions become appreciably worse, e.g. the exploratory
S1 B22 NP `67520` record with raw 19.82 dB and repaired
10.90 dB. Thus "safe on GOOD" does **not** imply monotone
PSNR improvement on every candidate.

`S_mid` uses the four central registered timesteps instead
of eight (128 rather than 256 UNet image forwards per eight-orbit
candidate, with four noise draws). Its observed 85/85 is a
**secondary retrospective** result; no prospective runtime,
stability or novelty claims follow automatically.

## 3. P3 — detection: full-family D-detect NOT met

Outcome is **post-prior-repair PSNR<25**, and higher detector
scores predict failure.

| Family | n | Remaining failures | Prior S AUROC | S_img AUROC | UnivFD AUROC | Signed rho AUROC | Clipped rho AUROC |
|:--|--:|--:|--:|--:|--:|--:|--:|
| DAPS pooled | 552 | 172 | 0.8551 | 0.7538 | 0.9105 | **0.9860** | 0.8789 |
| FAST pooled | 752 | 297 | 0.8326 | 0.7481 | 0.9286 | **0.9838** | 0.9147 |
| SITCOM pooled | 752 | 289 | 0.8527 | 0.7780 | 0.8949 | 0.6802 | 0.8740 |
| historical NP pooled | 1,120 | 338 | 0.9420 | 0.7524 | 0.9491 | 0.9356 | 0.8297 |

Frozen D-detect compares **max(AUROC(S), AUROC(S_img))
minus AUROC(UnivFD)**, requiring >= +0.03 for *both*
FAST and DAPS to be superior (>= -0.02 for both to be
non-inferior). Instead:

- DAPS best-prior delta **-0.05535 AUROC**.
- FAST best-prior delta **-0.09594 AUROC**.
- Both are below the frozen **non-inferiority** threshold,
  not just the superiority threshold.
- The *signed measurement residual* is particularly strong
  (0.986 DAPS, 0.984 FAST), so the prior is not presently a
  compelling replacement for simple GT-free detection.

Pre-registered failure-subtype secondary/diagnostic AUROCs
suggest a possible *conditional* niche, not a passing gate:

| Family / raw subtype | n remaining failures | Prior S | UnivFD | Signed rho |
|:--|--:|--:|--:|--:|
| FAST nonorbit-consistent | 23 | **0.9033** | 0.6744 | 0.8183 |
| DAPS nonorbit-consistent | 18 | **0.7646** | 0.6452 | 0.9101 |
| FAST nonorbit-inconsistent | 274 | 0.8267 | 0.9499 | **0.9977** |
| DAPS nonorbit-inconsistent | 154 | 0.8657 | 0.9415 | **0.9949** |

The frozen evaluator's subtype procedure compares surviving
failures of the named **raw** subtype against all
post-repair successes. Because the "consistent" subtype is
itself thresholded using signed rho, these conditional
comparisons have selection/conditioning effects; they
are **not** substitute full-family AUROCs. The FAST subset
has only 23 positive failures, and DAPS 18. A restricted
"measurement-consistent but wrong" detector deserves
planner consideration only as a *new, clearly labeled R1
hypothesis* with independently frozen protocol.

## 4. P4 — practical first-K policies: E1 adds modest value; E2 has no reliable advantage

Definition:
E0=min signed rho among first K (raw);
E1=repair E0;
E2=repair all and choose minimum prior S under the frozen
rho-feasibility threshold (fallback min rho);
E3=analogous with UnivFD p_fake;
O_raw/O_orbit are GT oracle ceilings.

Selected representative **Good25 rates**, expressed as
percent of images, with matched K and native runs:

| Source/family | K (images) | E0 | E1 | E2 | E3 | O_orbit |
|:--|:--|--:|--:|--:|--:|--:|
| S1 B22 DAPS | 2 (100) | 92% | 95% | 95% | 95% | 95% |
| S1 B22 SITCOM | 4 (100) | 89% | 94% | **96%** | 95% | 96% |
| S1 B22 NP | 8 (100) | 94% | 94% | 94% | 94% | 96% |
| S2 DEV80 DAPS | 4 (80) | 82.5% | **86.25%** | 85% | 85% | 86.25% |
| S2 DEV80 SITCOM | 4 (80) | 68.75% | **72.5%** | 71.25% | 67.5% | 72.5% |
| S2 DEV80 NP | 4 (80) | 70% | 70% | 71.25% | 71.25% | 71.25% |
| **S5 FAST native** | **4 (180)** | **85.56%** | **86.11%** | **86.11%** | **85.56%** | **86.67%** |

The *actual policy gain* is small for native FAST:
K4 E1−E0 mean **+0.1288 dB**, 95% image-bootstrap CI
[0.000, 0.386] and Good25 **+0.56 percentage points**,
95% CI [0, 1.67 pp].
E2−E1 FAST K4 mean **-0.1223 dB**, CI
[-0.193, -0.063] (no Good25 benefit).
S5 FAST K2 E2−E1 is also negative:
-0.0610 dB, CI [-0.114, -0.023].

Other families sometimes gain more from simple repair:
S1 SITCOM K4 E1−E0 mean **+1.076 dB** (CI
[+0.342,+1.983]), Good25 **+5 pp**
(bootstrap CI [+1,+10 pp]; exact McNemar p=0.0625).
S2 SITCOM K4 E1−E0 mean **+1.053 dB**
(CI [+0.260,+2.018]) and Good25 **+3.75 pp**
(CI [0,+8.75 pp]); S2 DAPS K4 E1−E0
**+0.795 dB** (CI [0,+1.872]), Good25 +3.75 pp
(CI [0,+8.75 pp]). These are cohort-specific outcomes;
the full 37 tables and paired tests should be reviewed,
and exploratory multiple comparisons not overinterpreted.

E2 versus E3 may show occasional advantages (e.g.
S2 SITCOM K4 E2−E3 mean +0.627 dB), but its
Good25 +3.75 pp CI [-1.25,+10 pp] includes zero.
There is no robust broad evidence for a substantially better
new multi-start prior policy than simple signed-rho
selection followed by repair.

S1 B22 DAPS supplies only **two** recorded runs per image:
the K4 table has zero eligible images. Do not treat an
unavailable K4 result as a failure of the algorithm.
S3 and S4 eight-image exposed subsets are small and reused
within larger panels; conclusions need not generalize.

## 5. Runtime, novelty and cost implications

- Frozen GT-free score evaluation: **256 UNet image forwards**
  plus UnivFD/measurement residuals per candidate, mean
  observed score wall ~**2.755 seconds**.
- Native S5 FAST generation: ~**3.335 seconds**
  mean wall per root on the PAC setup.
- Scoring every FAST candidate therefore adds roughly
  **83%** of native generation wall per candidate
  (2.755/3.335), assuming sequential per-candidate
  wall-time comparability. Treat as a context-specific
  approximation, not a controlled paired end-to-end
  speed benchmark.
- A simpler residual detector already nearly saturates
  AUROC for the primary FAST/DAPS overall failure task.
- Global 180° rotation dominates the exact-orbit failures.
  Enumerating the eight known symmetry members is not
  itself novel. A paper would need clear theoretical,
  low-cost, practical or prospective advantage versus
  existing orientation/face-prior heuristics, and against
  FAST-DIPS and FA-FPR as appropriate.
- Exposed/development reuse and selection cannot license
  performance/generalization claims on locked B24 confirmation305.

## 6. D-gate adjudication request (executor does NOT authorize R1)

**Frozen numerical inputs:**

| D gate | Required | Observed | Mechanical result |
|:--|:--|:--|:--|
| D-material | FAST or DAPS orbit share >=20% of raw failures | FAST 3.57%; DAPS 13.13% | **NOT MET** |
| D-repair PASS | pooled correction >=90%, GOOD harm <=1% | 98.82%; 0% observed | **MET** |
| D-repair FAIL | correction <75% or harm >3% | neither | **NOT MET** |
| D-detect superior | ΔAUROC(S or S_img − UnivFD) >=+0.03 both FAST & DAPS | -0.09594 FAST, -0.05535 DAPS | **NOT MET** |
| D-detect noninferior | ΔAUROC >=-0.02 both | below -0.02 both | **NOT MET** |

The frozen planner decision table does not specify an automatic
path for *D-material fail + D-repair pass + D-detect not
superior*. Therefore `PLANNER_GREY_ZONE_REVIEW` is a
correct **logic branch**, not evidence that all results are
ambiguous. Primary scientific evidence strongly disfavors the
original `repair + prior-consistent detection` publication
positioning on FAST.

**Executor assessment / recommendation to planner:**
Do **not** authorize the previously sketched broad FAST R1
pipeline merely from the excellent 84/85 repair accuracy.
The target bottleneck is rare in FAST; its global
detection quality is worse than UnivFD and signed rho;
and the K4 FAST practical return is +0.56 pp Good25
from E1 while E2 hurts mean PSNR. Absent a sharper
novelty/reliability-per-compute argument, **park Track R
as a full method-paper route**, or archive as a rigorous
small failure-anatomy/repair study. Planner has final say.

Two clearly *non-primary, prospective* hypotheses, only
if the planner sees novelty worth a **short, bounded R1
feasibility check** (not an auto-launched GPU campaign):

1. **Conditional rescue of measurement-consistent failures:**
   on the small subgroup that passes the signed-rho
   feasibility test but is still wrong, the prior shows a
   possible supplementary detection signal. The subset
   definition is threshold-dependent and its positive
   count small, so require held-out protocol and suitable
   conditional baselines. Any new decision rule must be
   pre-specified in a distinct R1 amendment, not retrofitted.
2. **Cheap selective orientation repair:** `S_mid`
   retrospectively corrected 85/85 exact-orbit errors
   with zero Good25 harm, using half the timesteps.
   Investigate only if a clearly novel, cheaper
   uncertainty-triggered repair (perhaps gated by signed
   residual, confidence, or orientation diagnostics) can
   outperform simpler orientation heuristics on
   *reliability per compute*. Such a study must compare
   against the trivial global flip/uprightness baseline.

Neither hypothesis restores the failed primary gate in R0;
both remain exploratory and require a newly frozen protocol.

## 7. Examples and follow-up questions

Explicit `EXPLORATORY` representative cases from the
path-free casebook:

- `S1_b22|DAPS|68539|fresh2` raw ~6.36 dB,
  prior repaired ~31.89 dB, optimal orbit ~31.89 dB;
  180° global orientation `111`.
- `S1_b22|NP|64022|lf_seed101` raw ~5.02,
  repaired ~31.22 dB.
- `S1_b22|NP|67520|s2_preproj_lam001_seed100`
  raw ~19.82 → repaired ~10.90 dB:
  large negative correction on an *already-bad*
  run, excluded from the P2 Good25-harm numerator.
- `S1_b22|NP|62739|lf_seed103` raw and repaired
  ~9.87 dB while the best orbit reaches ~25.10 dB:
  prior selects identity `000` instead of GT `111`.

**Questions for scientific planner:**
1. Given the original method-paper claim, does the
   D-material/D-detect failure justify parking Track R now?
2. Is any conditional rescue or half-grid prior orientation
   estimator sufficiently distinct from uprightness
   heuristics, residual detection and FA-FPR to merit a
   small prospective R1?
3. If not, should we preserve this as a bounded
   negative-result/failure-anatomy note and release
   remaining project resources to other directions?
4. What frozen R1 protocol, outcome, competitor and hard
   compute cap would be required **before** any additional
   experiments?

## 8. Next operational state

**R0 scientific closeout submitted for planner consideration.
No R1 runs scheduled or authorized.**

All root result receipts, source manifests and R0.7 reports remain
immutable on PAC. Executor branch holds the code and this report.
The user-uploaded R0.8 ZIP (SHA noted above) contains the
complete P1–P4 numeric tables, P4 matrix, 10,000
image-cluster bootstrap summaries, and secondary diagnostics.
The only requested handoff is planner adjudication.

**Do not read/list/hash/release B24 confirmation305 until a
separate explicit R2 approval.**
