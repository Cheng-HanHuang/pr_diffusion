# Track A A0 — B22 historical native-selector vs best-of-K audit (VERIFIED RETROSPECTIVE)

**Date:** 2026-10-16. **Executor:** Track A A0 archival candidate auditor. **Status:** `B22_METADATA_REPLAY_PASS`; **B24:** `SOURCE_UNCERTAIN / NOT ANALYZED`.

The user ran the dedicated PAC Track A A0 branch at commit `2b167dd110a7a5ab3104ab16b36e7eebcfd1f8cf` and returned `A0_B22_METADATA_REPLAY_V3_RETURN.zip`. This report is based on the **actual returned ZIP**, not projected measurements. I opened the original ZIP, checked its SHA-256, inspected all available CSV/JSON records, replayed winner selection independently from candidate rows, recomputed selector regrets, the frozen three Good25 checks, image-bootstrap intervals, and the flagships' cost ledger. It is **entirely archival metadata analysis**, not new diffusion inference.

## 1. Provenance and validated completeness

| Evidence | Observed result |
|---|---|
| Returned ZIP SHA-256 | `e0641c4c78e05489d8df53dd42587995b2af03460403b4ac3b22928c4b20d242` — **exact match** to PAC terminal |
| Test log | 12/12 synthetic tests PASS; SHA `4c7b6199997ba2acfc9010e46e42dabfe7097e52a9039cce199024d7c00d48db` |
| PAC replay | `A0_B22_REPLAY_RC=0`, `A0_FINAL_STATUS=0`, `A0_V3_RUNNER_RC=0` |
| Candidate metadata | **1,400** rows = 100 × (DAPS 2 + SITCOM 4 + NP 8); **100 unique images** in each family, all finite PSNR; zero duplicate candidate IDs |
| Policy metadata | **3,600** rows = 36 policy/K cells × 100 images; **12** predeclared paired comparisons |
| Measurement identity | Every image has one identical *raw* measurement content SHA across all 14 candidates; per-candidate B22 file/content/GT/source identities checked by fail-closed loader |
| Native/oracle identity replay | Every archived SITCOM-1, SITCOM-4S, SITCOM-oracle4, NP-1, NP-8-RS and NP-oracle8 selected candidate identity and PSNR reproduced from individual candidate metadata; B21 DAPS exact signed-loss rule replayed |
| Historical Good25 replay | DAPS Fresh2 92/100; SITCOM-4S 93/100; NP-8-RS 95/100, all **PASS** |
| Source | B22 SITCOM pinned HEAD and exact one-line changed source (only trailing blank diff context differs); native checkpoint content hash independently checked. B24 frozen **full** tracked diff still differs; do not transfer B22 finding to B24 |

Artifact files live under `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a0_native_selector_audit_20261015/A0_B22_METADATA_REPLAY_V3/`. The ZIP contains candidate rows, policy rows, summaries, paired analyses, historical replay receipts, unsupported-cell register and source/audit evidence. Machine-checkable CSV SHA-256:

- `A0_CANDIDATE_ROWS.csv`: `d7bd56446139e4d85f9f08f4e78b7ac1a2b17daf8bfc47890f696bb1c8448d8c`
- `A0_POLICY_ROWS.csv`: `6b58c10e72f59a00f6fee38d63a5cbb3635dedaf0e05d8630429806052200ceb`
- `A0_METRIC_SUMMARY.csv`: `5a80f56c70a417d73477669bd49f5bb614025b1bca3af1590eab93fb28d159e5`
- `A0_PAIRED.csv`: `db6f16f9048801b726e8e2d5db4faa9578280d7babf4f28ee054e8c499e5b113`
- `A0_REPLAY_CHECKS.json`: `04bedc50d1ca3d782093a1286250e7682bd558da15a318796eabf67eef38c411`

## 2. Frozen B22 native selector versus *same-candidate* offline GT oracle

All PSNR values use B21/B22 **float-tensor, raw RGB orientation**; Good25 = raw PSNR at least 25 dB. The offline oracle sees the ground truth and **is not deployable**. The native score is calculated without GT from each method's historically available trajectory or terminal metadata.

| Family and candidate policy | Native Good25 | Same-candidate GT oracle Good25 | Native mean PSNR | GT oracle mean PSNR | Native PSNR-best on images | Mean selector regret |
|---|---:|---:|---:|---:|---:|---:|
| **DAPS Fresh2 K2**: accept rep1 iff signed loss1 < loss0 − 0.7 | **92/100** | 92/100 | 29.299 dB | 29.427 dB | 59/100 | **0.129 dB** |
| **SITCOM-4S K4**: min step160 correction norm over sequential RNG stream | **93/100** | **95/100** | 26.442 dB | 26.667 dB | 39/100 | **0.225 dB** |
| **NP-8-RS K8**: min postprojection LF MSE over 4 LF + 4 S2 trajectories | **95/100** | **96/100** | 29.269 dB | 29.407 dB | 26/100 | **0.138 dB** |

The selector-regret estimates are *PSNR(oracle) – PSNR(native)*; therefore they are nonnegative imagewise by construction. Independently replayed paired-image bootstrap 95% intervals (10,000 draws, seed 5401) for **mean regret**: DAPS **[0.049, 0.230] dB**, SITCOM **[0.036, 0.569] dB**, NP **[0.066, 0.231] dB**. These intervals characterize these fixed historical candidate portfolios, not the expected performance of a new clean-free GT predictor. Near-zero aggregate regret may conceal large failure outliers.

**Threshold outcomes:** DAPS oracle creates **0** additional Good25 successes over Fresh2; SITCOM oracle rescues **2** (`60140`, `64518`); NP oracle rescues **1** (`65269`). Because the oracle comparison is post-hoc, exact McNemar p-values for the Good25 selector gaps are unconvincing at 100 images (SITCOM 0.5, NP 1.0). DAPS mean PSNR regret exists despite zero Good25 ceiling advantage.

## 3. Candidate-level failure diagnosis — concrete falsifiable cases

| Image | Family | Native PSNR | GT-oracle PSNR | Regret | Implication |
|---|---|---:|---:|---:|---|
| `60140` | SITCOM-4S | 11.265 | 27.007 | **15.742 dB** | Single catastrophic selector miss across saved four trajectories; GT oracle flips Good25 |
| `64518` | SITCOM-4S | 24.797 | 25.007 | 0.209 dB | Small regret, crosses exact Good25 threshold |
| `65269` | NP-8-RS | 24.369 | 27.750 | **3.382 dB** | Non-GT native LF selector chooses subthreshold candidate; GT oracle rescues |
| `68539` | DAPS Fresh2 | 6.360 | 9.641 | 3.282 dB | Meaningful PSNR miss, but neither candidate achieves Good25 |
| `65553` | DAPS Fresh2 | 16.207 | 18.704 | 2.497 dB | PSNR miss without Good25 rescue |
| `65003` | Cross-family | all below 25 | all below 25 | — | Still failed by all 14 candidates, even with GT selection |

**Cross-family offline bound:** Taking the best of the three **already-selected native policies** achieves **99/100 Good25**, and taking the best across all 14 candidate reconstructions **also achieves 99/100**. Both are GT-only diagnostics; **no clean-free cross-family router was evaluated**, and no source establishes that such an oracle could be implemented prospectively.

## 4. K ladders and heterogeneous NP portfolios

Only historically documented runtime native policies are labeled **NATIVE**; arbitrary K prefixes without an actually documented native selector are not promoted to deployed algorithms. NP K2/K3/K4 prefix comparisons below are retrospective partial-portfolio *clean-free score replays* of archived trajectories, **not previously deployed B22 policies**. All GT-oracle K values are offline ceilings.

| Portfolio | K | Clean-free score-selection Good25 | Offline oracle Good25 | Interpretation |
|---|---:|---:|---:|---|
| DAPS | 1 | 80/100 | 80/100 | Historical Fresh1 |
| DAPS | 2 | 92/100 | 92/100 | Historical Fresh2 |
| SITCOM sequential | 1 | 71/100 | 71/100 | Historical SITCOM-1 |
| SITCOM sequential | 2, 3 | *unavailable native policy* | 86/100, 92/100 | GT upper bounds only |
| SITCOM sequential | 4 | 93/100 | 95/100 | Historical SITCOM-4S |
| NP **LF-only** | 2 | 86/100 | 86/100 | Retrospective score-selection prefix |
| NP **LF-only** | 4 | **93/100** | 94/100 | 4 homogeneous LF trajectories |
| NP **S2-only** | 4 | **94/100** | 94/100 | 4 homogeneous preprojection-regularized trajectories |
| NP **mixed LF+S2** | 8 | **95/100** | **96/100** | Historical 8-trajectory policy |

The four **S2-only** candidates (seeds 100–103) score **94/100** under the same frozen NP clean-free LF terminal selection criterion, compared with **95/100** for the mixed eight-candidate portfolio. They require roughly half of the archived NP reconstruction seconds, but S2-only K4 is **not** a historically deployed or prospectively validated standalone algorithm. This is a potentially valuable **prospective cost-efficiency hypothesis**, not established superiority. The paired NP8 vs S2-only4 Good25 discordance is just **2 rescues versus 1 harm** (exact two-sided McNemar p=1); infer no reliable difference from 100 images.

## 5. Runtime / compute fairness: not matched merely because K is equal

| Portfolio | Mean *sum of recorded candidate reconstruction wall seconds* per image |
|---|---:|
| DAPS Fresh2 K2 (B21 per-arm wall) | 312.64 s |
| SITCOM-4S K4 (B22 per-trajectory reconstruction) | 196.07 s |
| NP LF-only K4 (B22 reconstruction) | 205.32 s |
| NP S2-only K4 (B22 reconstruction) | 206.46 s |
| NP mixed K8 (B22 reconstruction) | 411.78 s |

These are archive-specific *candidate-time sums*, not consistently measured complete end-to-end costs or FLOP-equivalent work. B22 reports model-load sharing and distinct native infrastructures; DAPS uses B21 wall times. SITCOM-4S uses **one sequential RNG stream**; NP8 consists of **two different algorithm configurations** (4 LF + 4 S2). No pair between DAPS2, SITCOM4S and NP8 is compute matched. They are all paired on image/measurement IDs, but not on wall/FLOP budgets or independent restart distributions.

The PAC 12-pair table reproduces historic flagship deltas: DAPS2 minus NP8 **+0.029 dB** (bootstrap 95% CI **[−1.092,+1.038]**, paired Good25 DAPS rescues 4, harms 7, exact McNemar p=0.5488); DAPS2 minus SITCOM4S **+2.856 dB** (95% CI **[+1.754,+3.834]**, Good25 5/6 rescues/harms); SITCOM4S minus NP8 **−2.827 dB** (CI **[−3.499,−2.248]**, Good25 3/5 rescues/harms). These are descriptive across unequally expensive and structurally different pipelines, **not** evidence of a method's intrinsic superiority at equal compute.

## 6. Scientific inference and recommendation to Track A lead

1. **Selector performance is not the dominant Good25 bottleneck for every family.** DAPS's PSNR selector regret does **not** lose any Good25 successes; SITCOM loses 2/100 and NP8 loses 1/100 versus their saved-candidate PSNR oracles. Do not claim that replacing native selection with a GT-guided portfolio universally changes the headline.
2. **A few selector misses are scientifically nontrivial.** SITCOM image 60140 exposes a 15.74-dB within-portfolio miss, NP image 65269 a 3.38-dB miss. If proposing a deployable improvement, target robust clean-free discrimination and predeclare threshold/tail metrics, not just tiny average PSNR regret.
3. **Candidate availability is a separate limiting factor.** Even the offline best among all 14 candidates does not rescue image 65003. A new proposal needs to distinguish *more capable generation* from *better selection among the same candidates*.
4. **NP S2-only K4 is a plausible cheap ablation** but retrospectively evaluated on an already-exposed 100-image cohort. Prospectively test if scientific lead wants; do not retrofit as a proven eight-to-four efficiency gain.
5. **Do not equate B22 and B24.** B24 DEV80 uses a selected, stratified population and a different canonical 8-bit-vs-float reconstruction metric. Its historical frozen SITCOM full tracked source identity remains unresolved and no B24 results are newly claimed by A0.

**A0 executor decision:** B22 archival/selection assignment completed; hand off findings to Track A planner for independent choice of prospective benchmark or new executor. **Do not independently launch GPU/P2, remove archived outputs, or access B24 confirmation305.** The delivered file hashes make this audit reproducible in the existing dedicated PAC output namespace. Cross-track issue #41 is the coordination ledger.
