# Planner notes to the XP executor (2026-10-27, 00:00 ET)

**Reviewed:** `codex/prdiff-track-r-xp-20261026` @ `56c84c4` (XP-A16 supervisor authored), and the full return log.

**Assessment:** the replay gates (FAST 8/8 and SITCOM 8/8 exact), the pristine DAPS-4K isolation and the XP-T calibrations are excellent.

**What these notes do not do:** stop anything already running. Apply them at your next return point.

**Governing amendment:** XP prereg **A5** (`docs/planning/track_r/XP_PREREG_v1_1.md`). Re-merge the planner branch before your next tier launch, as handoff rev C requires.

## A. Corrections (no new work)

1. **Tier B is XP-A64, XP-B16, XP-D2, XP-D3 and XP-F8.**
   - XP-D2 is DAPS-4K on ImageNet; XP-D3 is SITCOM on ImageNet.
   - Your log entry of 2026-10-26 22:34 swaps D2 and D3, and lists XP-E′ in Tier B. XP-E′ is Tier C and conditional on A4.4. Fix the log.
2. **The W ∈ {1, 2, 4} same-GPU throughput study is optional engineering, not a gate.**
   - Skip it unless a stage's W = 1 projection misses its data cut.
   - At the measured medians (FAST 3.3 s; SITCOM 32/58 s; DAPS-4K 66/166 s on FFHQ/ImageNet), Tiers A and B total about 70–90 GPU-h, i.e. about 20–25 h of wall time on 4 GPUs at W = 1.
3. **Size VRAM admission from measured peaks.** Request about 1.5 × the measured peak + 2 GiB (FAST about 4 GiB), and keep the 80% ceiling. A fixed 32 GiB + 2 GiB request can leave a GPU idle under co-tenancy.

## B. One CPU block now

All of this is historical K ≤ 4 data, so no embargo applies. Allow at most about 1 h.

1. **A4.4 (required).** Record the historical FFHQ DAPS configuration from the S1/S2/S3 receipts: annealing steps, ODE steps, Langevin steps, and NFE per run. It gates XP-F8 and decides XP-E′.
2. **A8b (A5.6; commit it next to A8 under `docs/track_r/executors/r0n/returns/`).**
   - Content: A8's rows plus `rho_signed_by_run` (run-index order, 6 significant digits, `;`-separated), and the frozen τ per (source, family).
   - It is GT-free. The theory executor needs it before its v1 freeze (target Oct 29, 18:00 ET).
3. **A7 (now required, A5.7).** Deliver it as a ZIP to Cheng; do not commit it.
   - Base specification: R0N amendment 2 rev C, including SSIM/LPIPS on CPU, the image features, the contact sheets and the timing sources.
   - Add the A5.5 columns `rho_star`, `Delta`, `D` and `C`, computed in float64 with the exact frozen signed y.
   - Recount UnivFD fixes and harms on DAPS, SITCOM and FAST only.
4. **A5.5 summary table (commit).** Per family × cohort × class (G, O_g, O_p, C, I):
   - n, and the median and IQR of Δ, D and C;
   - the fraction of runs with ρ < ρ★;
   - per (family, cohort), whether ρ★ ≤ τ;
   - within-image pairwise residual win rates with image-cluster CIs. They must reproduce the planner's counts from A8: G > C 200/205, G > O 50/69, G > I 570/571, C > I 92/92.
5. **Optional:** the A4.1–A4.2 PC-DAPS crosswalk, and the A3 image ids under E1_CLIP.

## C. GPU queue after XP-A16: one launch, many hours

**Build one controller.** It runs the remaining stages back to back from frozen manifests:
- it uses your replay-verified per-family workers (FAST, SITCOM, DAPS-4K);
- each stage has a smoke gate that auto-continues on PASS;
- one stop file halts it;
- its receipts are embargo-safe.

Cheng should paste one block and come back hours later.

**CRITICAL lane:**
1. XP-D1 phase 1 (FAST ImageNet, runs 0–3). Score phase 1 at once and commit the A8-style per-image strings, which open the v2 freeze.
2. XP-C and XP-C2.
3. XP-D1 phase 2, sealed until v2 is registered or 48 h have passed.
4. XP-A64 (sealed).
5. XP-B16 (sealed), in parallel with XP-D2.
6. XP-D3.
7. The DAPS 8-root replay gate (A5.2), then XP-F8 (sealed).

**BACKGROUND lane** (idle slots only, units of about 20 min or less):
1. XP-K, the SITCOM and FAST parts (parameter-only; A5.4).
2. XP-B32 (sealed).
3. XP-G.
4. XP-I: CRITICAL priority once its wrapper passes smoke.
5. XP-I2.
6. XP-H.
7. XP-G2.
8. XP-J.
9. XP-D4.
10. XP-E′, only if A4.4 calls for it.

**Targets:** Tier A done by Oct 29; Tier B by Nov 2. The Nov 6 cut then leaves slack.

## D. Engineering to do now (no GPU)

1. **XP-I, the PC-DAPS wrapper (A5.3).**
   - Use the native path at `59b2f04` and feed in our signed y.
   - Save all 4 particles and the method's own output.
   - Load the checkpoint strictly.
   - Its `test-ffhq` and `demo-*` data configs are banned: replace the data input entirely.
   - It becomes CRITICAL once a 2-image smoke passes. XP-I2 is the same on the 100 ImageNet images.
2. **XP-K manifests (A5.4).** Parameter-only overrides on B22's 100 ids, n = 4 per family; seeds follow the XP rule with dataset `ffhq_cal`.
   - SITCOM: `lgvd_config.tau = 0.05`.
   - FAST: `--epsilon 33.26`, i.e. 0.05 · √(3·384²).
   - DAPS: `mcmc_sampler_config.tau = 0.0707` with the historical FFHQ configuration, after the F8 gate.
3. **The post-freeze analysis script, written now.**
   - Validate it on historical K ≤ 4 runs: it must reproduce the A4.3 ladder and A8 exactly.
   - Then, when the planner registers v1, you can score XP-A16/A64/B16/F8 with one command.
   - What it must compute:
     - labels, using the frozen τ;
     - exact U-statistic K-curves for every policy;
     - the per-metric and single-pick oracles;
     - the anatomy: a generation failure is no run ≥ 25 dB in any orbit member, and the selection loss is O_orbit − E1;
     - the registry and order check (§5.4);
     - every frozen forecast against its observation, with the agreement criterion;
     - the A5.5 quantities per candidate.
4. **ImageNet loader statement for the XP-D2 report.** State that the 100 images are indices 0–99 of `sorted(Path(root).rglob(ext))`, as in DAPS `data.py` with `test-imagenet.yaml` (`start_id: 0`, `end_id: 100`). That is exactly what DAPS's configuration evaluates. Note that the archive holds 1,000 images.

## E. Reporting

- Keep the full log, but open every return with a digest of at most 20 lines: what ran, receipts and SHAs, what comes next, and what Cheng runs. Give one block per launch.
- Return to the planner, via Cheng, when:
  - A4.4 and A8b are committed;
  - XP-A16 completes (sealed);
  - the XP-D1 phase-1 counts are in;
  - Tier A is done;
  - each Tier B stage is done;
  - any stop condition fires.
