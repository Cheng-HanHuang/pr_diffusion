# R0-N amendment 2, revision D (planner, 2026-10-25): the R0-N.2 CPU block

**Revision D (2026-10-25) is additive to revision C and makes R0-N.2 urgent**, because of the top-venue attempt (`docs/planning/track_r/2026-10-25_TOP_VENUE_PLAN.md`).
- **New §A8:** committable per-image label counts, which feed the theory executor's frozen predictions (TH1.5).
- **New §A4.3 requirement:** report every ladder rung against **O_orbit** as well as O_raw. The recovered share must use O_orbit as the oracle.
- **Return target: 2026-10-29.** Priority order: B2 (N1h) → A8 → A4.3 → A7 → A4.1–A4.2 → A3.
- This block can run in parallel with XP-0 (CPU) from the XP handoff.

**Revision C (2026-10-24 late morning) is additive to revision B.** It adds §A7, a planner observation export plus SSIM/LPIPS on CPU, for extensions X1/X2 in `docs/planning/track_r/2026-10-24_EXTENSION_OPTIONS.md`. Nothing in revision B changes. If R0-N.2 is already running, finish it and run §A7 as a small add-on in the same environment.

**Revision B (2026-10-24 morning) supersedes revision A.** It follows the N2 recovery and the K2 ruling (decision §10: K2 TRIGGERED; the paper becomes an audit).
- **Removed:**
  - A's B3 (N2 on CPU), now done by the N2 GPU recovery;
  - A's B4 (N1d miss identity), no longer needed;
  - A's A5 (N3b);
  - N4.
- **Promoted to required:** the mean-of-K ladder, plus a new **E1_CLIP** policy.

**Inputs already in hand.**
- R0-N N1 return: `R0N_20261024T061111Z` @ `5af47db`.
- N2 recovery: `R0N_N2_RECOVERY_20261024T073405Z` @ `081fc75`/`33a34bd`, comparison SHA `322c3ef0…0233`.

**Scope.**
- CPU only (`CUDA_VISIBLE_DEVICES=""`).
- Changes no frozen R0 or R0-N v1 definition, and reuses the frozen N2 CLIP decisions as stored. **No new CLIP or UNet inference.**
- **Excluded:** GPU work, new reconstructions or measurements, and any access to confirmation305.
- Write to a new `r0n2_<stage>_<UTCSTAMP>/` root. Existing R0-N outputs are immutable.

**Target date:** return by 2026-11-07. Items not returned by then are dropped from the paper.

## B1 — branch (required)

1. Merge `origin/claude/prdiff-track-r-planner-20261021` (≥ the commit that adds this revision) into `codex/prdiff-track-r-r0n-20261023`, or into a new `codex/prdiff-track-r-r0n2-<YYYYMMDD>` from `33a34bd`. The planner's dry run of this merge is conflict-free.
2. Record both SHAs in the return.

## B2 — N1h training-membership lookup (required; owed from amendment 1)

1. Run amendment 1 §A2: look up the FFHQ `category` field by image id only, for B22 and DEV80, in `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/ffhq-dataset-v2.json`. Do not enumerate image trees.
2. Report the counts per cohort.
3. Record any documentation of `ffhq_10m.pt`'s training set from targeted greps of `external/dps` and `external/DiffFPR`, or write "UNKNOWN".
4. Split by category:
   - single-run Good25 rates per family;
   - repair success and harm (CLIP and S_mid, global-only).

## A4 — N1i ladder (required) and outcome crosswalk (recommended)

### A4.3 Protocol ladder (REQUIRED)

For each (source, family), and per cohort where a source spans both (S5), at K = 1 and at maximal K, report Good25 / mean / q05 for each rung:

| Rung | Definition |
|---|---|
| **mean-of-K** | Per image, the mean PSNR over its K runs; then the mean over images. Also give the per-run Good25 rate, which is the single-run failure rate |
| **E0, E1, E2, O_raw, O_orbit** | Existing frozen policies, reproduced. E1 is the prior-based (8-member S) repair |
| **E1_CLIP** (new, required) | Take the E0 pick, then apply the **stored frozen N2 CLIP global decision** for that candidate: choose 111 iff u(R₁₁₁x) > u(x); otherwise keep 000. Evaluate the chosen member's raw PSNR |
| **E1_2F** (optional) | As E1_CLIP, with the N1d-selected `single_300\|J1\|global2` decision computed from the stored draw losses |
| **E_P** (optional, EXPLORATORY) | Take the argmin over the K candidates of the stored identity-member S (ties → lowest run index), then apply E1's repair |

**Also report, per stratum:**
- **GT-free share of the K-run gain** = (E1_CLIP − first run) / (O_raw − first run), and the same with E1;
- the planner-check value: E1_CLIP − E1 (expected near 0);
- for E1_CLIP, the max-K post-selection decomposition, i.e. N1b's fixed / selection miss / uncorrected / generation-failure counts.

**Planner reference values** (R0.8 P4 / N1g; use them to sanity-check the ladder, not as targets):

| Stratum | First run, K=1 E0 (dB) | O_raw at max K (dB) | E1 at max K (dB) |
|---|---|---|---|
| B22 DAPS | 27.23 | 29.42 (K=2) | 30.07 |
| B22 SITCOM | 22.97 | 26.66 (K=4) | 26.64 |
| B22 NP | 25.58 | 29.40 (K=8) | 29.12 |
| DEV80 DAPS | 23.29 | 28.83 | 28.82 |
| DEV80 SITCOM | 19.59 | 24.12 | 24.70 |
| DEV80 NP | 22.32 | 26.07 | 25.86 |
| FAST (S5, both cohorts) | 23.81 | 29.00 | 28.99 |

### A4.1–A4.2 PC-DAPS outcome crosswalk (RECOMMENDED)

**Why.** PC-DAPS (arXiv 2610.31592; code `github.com/voilalab/PC-DAPS` @ `59b2f04`, MIT) reports 14.5% of DAPS runs on FFHQ as upside-down or color-inverted, using an orientation-only label. We report 3.5% (B22) and 5.3% (DEV80) of DAPS runs as repairable orbit errors.

**Inputs.** Use the same candidate images and GT tensors as the R0.7 PSNR_raw replay, per source. Record the representation each source uses (float after clamp, or RGB8). GT is used for evaluation labels only.

**Label rule** (the exact rule of `pr_outcomes.py` @ `59b2f04`; reimplement it, or copy it with the MIT notice):
1. **Map to [0, 1].** v·0.5 + 0.5, then clamp to [0, 1], for both x and g.
2. **Per-channel PSNR.** 10·log10(1 / max(MSE_c, 1e-10)), with the MSE over H×W for channel c.
3. **Shift search.** For r ∈ {g, rot180(g)}, take the maximum over the 9 circular shifts (dy, dx) ∈ {−1, 0, 1}² (`torch.roll` on the spatial dims).
4. **Channel assignment.** upright_c ⇔ best(g) > best(rot) + 1.0 dB; rotated_c ⇔ best(rot) > best(g) + 1.0 dB.
5. **Run label.**
   - faithful: all channels upright;
   - flipped: not faithful, and every channel assigned;
   - failed: otherwise.
6. **Unrecoverable image:** none of its K runs is faithful.

**Unit test (synthetic, CPU).** A smooth non-symmetric test image should be labeled faithful; its rot180, flipped; a version with only channel G rotated, flipped; uniform noise, failed.

**Tables (DESCRIPTIVE):**
1. Per-run faithful / flipped / failed rates per family × cohort, with n and image-cluster CIs.
2. Cross-tab of PC-DAPS label × R0.7 label (GOOD, ORBIT-global, ORBIT-partial, CONSISTENT, INCONSISTENT).
3. Unrecoverable rate at maximal K, next to 1 − O_orbit Good25.
4. The labels of the E0 and E1_CLIP outputs at maximal K.

### A4.4 DAPS configuration record (required, one line per source)

From the historical receipts of S1, S2 and S3, record:
- annealing steps;
- ODE steps;
- NFE per run.

The planner reads the launch defaults as 400 × 5 = 2,000 NFE per run; confirm or correct from receipts.

## A3 — image identities (optional)

List the image ids behind the max-K post-selection fixes, misses, uncorrected errors and generation failures: 15 / 5 / 1 / 93 under E1. Repeat under E1_CLIP if the counts differ. These are for example figures only.

## A7 — planner observation export and evaluation metrics (recommended; revision C)

**Purpose.** Give the planner one compact, path-free per-candidate table, so the exploratory analyses for X1/X2 run on the planner side without further PAC iterations.

**Delivery.**
- Deliver it as a ZIP that Cheng uploads to the planner, like the R0.8 ZIP.
- Do **not** commit the per-candidate table to git, consistent with the rule against per-run GT records.
- Commit only the script, the schema and the ZIP's SHA-256.

**A7.1 Per-candidate table** (`R0N2_OBSERVATION_TABLE.csv`, 3,240 rows; no paths, no tensors). One row per candidate:

| Column group | Columns |
|---|---|
| Identity | `image_id`, `cohort`, `source`, `family`, `run_index` (the frozen K order), `ffhq_category` (from B2) |
| R0.7 labels and PSNR | `label` (R0.7); `psnr_raw`; `psnr_member_<s>` for the 8 members s ∈ {000,…,111} |
| Residuals | `rho_signed`, `rho_clipped` |
| Prior and other scores | `S_<s>` and `S_mid_<s>` for the 8 members (stored values); `loss_t300_j0_000` and `loss_t300_j0_111` (the stored per-draw losses behind the 2-forward rule); `univfd_pfake_000`, `univfd_pfake_111` |
| CLIP | `clip_u_000`, `clip_u_111` and the frozen `clip_choice`, from the stored N2 scores |
| Other | `hf_energy` (R0.7 secondary); `native_cost_seconds` where recorded |
| **New evaluation metrics** | `ssim_raw`, `lpips_raw`, `ssim_111`, `lpips_111` (the global-rotated member) |

**Metric definitions for the new columns.** They match the DAPS/FAST-DIPS evaluators: piq SSIM with `data_range=1.0`, and piq `LPIPS(replace_pooling=True)`, on (x·0.5 + 0.5).clamp(0, 1) against GT.
- CPU only. LPIPS's pretrained network is an evaluation metric, allowed here **for evaluation only**.
- Record the piq version and the LPIPS weight hash.

**A7.2 Per-image table** (`R0N2_IMAGE_FEATURES.csv`, 180 rows). Per image:
- **GT-free measurement features**, computed from y only:
  - total energy Σy²;
  - the fraction of Σy² beyond radii 16, 32, 64, 96 from the canvas center (384² canvas);
  - the radial log-power slope fitted on radii 8–128;
  - the fraction of entries with |y| < σ;
  - the signed-y negative fraction.
- **GT-derived descriptive features** (analysis only; never used for selection): GT high-frequency energy, GT total variation, GT mean saturation, GT luminance standard deviation.

**A7.3 Contact sheets** (`contact_*.png`, ≤ 12 sheets, 256-px tiles, ≤ 20 MB total). Each row shows GT | E0 pick | E1_CLIP pick, with the PSNR printed:
1. all max-K generation failures;
2. 40 SITCOM measurement-consistent failures, chosen by a fixed rule: the 40 lowest `rho_signed` among SITCOM CONSISTENT candidates, ties broken by id;
3. all orbit errors.

FFHQ images only. No confirmation305.

**A7.4 Compute-accounting table** (`R0N2_TIMING_SOURCES.csv`). Per source and family:
- GPU model;
- timing definition (CUDA event, sampling-only wall, or full process wall);
- batch size;
- whether the measured time includes model load.

The planner needs this to put all four solvers on one compute axis (X3).

**Rules.**
- CPU only (`CUDA_VISIBLE_DEVICES=""`).
- Read-only inputs. New output root `r0n2_obs_<UTCSTAMP>/`.
- Stop if the LPIPS projection exceeds 2 CPU-hours. Then deliver everything except the LPIPS columns.

## A8 — per-image label counts (required; revision D; **committed to git**)

**Purpose.** The theory executor fits the selection model to per-image outcome counts. Counts are aggregates: no paths, no PSNR values, no images. They may be committed.

**File.** `docs/track_r/executors/r0n/returns/R0N2_<UTCSTAMP>/R0N2_PER_IMAGE_COUNTS.csv`. One row per (image_id, cohort, source, family). Columns:

| Column | Content |
|---|---|
| `n_runs` | Number of runs |
| `n_GOOD` | GOOD runs |
| `n_ORBIT_global`, `n_ORBIT_partial` | Orbit errors, global and partial |
| `n_CONSISTENT`, `n_INCONSISTENT` | Non-orbit failures, by residual class |
| `run_labels` | The labels in frozen run order, as a compact string such as `GGOCI…` (G = GOOD, O = ORBIT, C = CONSISTENT, I = INCONSISTENT) |
| `e0_rank_labels` | The labels sorted by ascending ρ_signed, same alphabet |
| `clip_fix_ok` | For ORBIT runs only: did the frozen N2 CLIP decision fix it? Encoded 1/0 in run order, empty for other runs |
| `ffhq_category` | From B2 |

`e0_rank_labels` lets the theory executor test assumptions S1/S2 without any per-run residual values.

**Data-rule note.** The label strings are derived from GT per run, but contain no paths, metric values or images. The planner judges them compatible with the "no per-run GT records" rule. **If Cheng vetoes**, commit only the count columns, and put `run_labels`, `e0_rank_labels` and `clip_fix_ok` in the A7 planner ZIP.

**Checks.**
- The row sums reproduce the K3 totals: 2,040 / 85 / 324 / 791.
- `run_labels` reproduces N1b's maximal-K decomposition.

## Not authorized here

- Any new inference: CLIP, UNet or UnivFD. The only exception is the A7 LPIPS evaluation metric, on CPU.
- N3b and N4 (withdrawn).
- GPU extensions X3–X6. They need a planner pre-registration and Cheng's go.
- R1 or R2 work.
- Access to confirmation305.

## Return (`R0N2_EXECUTOR_RETURN.md`)

1. **Status for Cheng** — at most 8 lines.
2. **Identity** — branch, SHAs and interpreter.
3. **Sections:** B2 (N1h), A8 (per-image counts: path and SHA), A4.3 (ladder, with E1_CLIP; both O_raw and O_orbit), A4.4, A4.1–A4.2 (crosswalk), A3, A7 (the ZIP's SHA-256 and a row and column count; the ZIP itself goes to Cheng for the planner).
4. **Anomalies.**
5. **Artifacts** — PAC root, committed CSV/JSON files and SHA256SUMS.
