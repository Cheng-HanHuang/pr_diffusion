# Theory predictions registry (planner-maintained)

**Purpose.** This file proves that each predictions file was frozen before the outcomes it predicts were analyzed (`XP_PREREG_v1_1.md` §5).

**Who writes it.** Only the planner, on `claude/prdiff-track-r-planner-20261021`, as soon as the theory executor reports a freeze.

**How the XP analysis uses it.** For each entry, the analysis checks three things:
1. the entry exists;
2. `git show <commit>:<path> | sha256sum` matches the SHA-256 below;
3. the commit time precedes the first XP commit that contains the matching outcomes.

| Freeze | Theory branch | Commit | Path | SHA-256 | Commit time (UTC) | Seen by planner (UTC) | Covers |
|---|---|---|---|---|---|---|---|
| v1 | — | — | `docs/track_r/theory/TH1_PREDICTIONS_v1.json` | — | — | — | FFHQ: FAST (S5), SITCOM (B22+DEV80), DAPS (B22+DEV80), on the A6.4 panels; target Oct 28–29, hard deadline **Oct 31** |
| v2 (optional) | — | — | `docs/track_r/theory/TH1_PREDICTIONS_v2.json` | — | — | — | ImageNet FAST (panel n = 16); due 48 h after the XP-D1 phase-1 strings land in `docs/track_r/executors/xp/theory_inputs/` (A6.6) |
| mech v1 | `theory/prdiff-track-r-th-20261026` | `225885c297d9a02f9934e65aa1b17e1c45a2d732` | `docs/track_r/theory/TH2_MECH_PREDICTIONS_v1.json` | `65364b2f0be7e34f7681dfb9a6b40e1edf00ba2ea816b16d56961c724ff38436` (planner-verified with `git show` at the commit and at head `58418ed`) | 2026-10-27T04:21:21Z (on GitHub by the push event of 2026-10-27T04:23:20Z, head `c190b5b`) | 2026-10-27T15:35Z (approx.; first `git show` by the planner) | XP-K and XP-G (prereg A5.4): directional forecasts only (DAPS and SITCOM GOOD-run ρ/σ rises toward 1 under calibration); FAST and XP-G abstain. Also within-image class order on new native replicates: G beats C in > 75% of pairs; DAPS and FAST favor G over O more than SITCOM. **REGISTERED** |
| mech v1.1 | `theory/prdiff-track-r-th-20261026` | `5abad48dfeb759550c0954f286f73c0ac2eeeb58` | `docs/track_r/theory/TH2_MECH_PREDICTIONS_v1_1.json` | `11262301fc00bb0c9483639719564c7f2e04bd8fa0fc146cef2c8477324b758d` (planner-verified at the commit and at head `1e3e8d9`) | 2026-10-27T15:56:47Z (on GitHub, confirmed through the REST API at 21:02Z; no A5.5 value existed then) | 2026-10-27T20:55Z (approx.) | A5.5 noise-floor decomposition for GOOD runs.<br>• DAPS: D/r ≈ 1.02, C/r ≈ 1, Δ/r ≈ −0.98.<br>• SITCOM: D/r ≈ 1.04, C/r ≈ 1, Δ/r ≈ −0.96.<br>• FAST: Δ ≥ 0 if the noise coupling is negligible.<br>• Orientation-leak directions.<br>Note: the signs of Δ and C for DAPS and SITCOM GOOD runs follow already from N1c (ρ < σ ⇒ Δ < 0 ⇒ C > D/2). The informative forecasts are the ratios D/C and Δ/C, and FAST's sign. **REGISTERED** |

**Embargo status:**
- FFHQ K > 4 outcomes (XP-A, XP-B, XP-F; B22 DAPS K > 2): **EMBARGOED** until v1 is registered.
- ImageNet FAST outcomes with run index ≥ 4: **EMBARGOED** until v2 is registered, or until 48 h after the phase-1 counts if no v2 is made.
- XP-K and XP-G outcomes: mechanism forecasts v1 are **registered** (2026-10-27), so these outcomes may be shared with the theory executor once they exist.
- The mech file's class-order forecasts concern new native FFHQ runs, which stay under the v1 embargo. Registering the mech file does **not** lift it.
- A5.5 (historical noise-floor decomposition): may be shared with the theory executor now that mech v1.1 is registered.
- **v2 clock:** the XP-D1 phase-1 strings landed at `41d4121` (2026-10-27 12:01:51 ET), so v2 is due by **2026-10-29 12:01 ET** (prereg A7.4).

**Freeze contents (XP prereg A5.9).** v1 keeps the S1–S3 curves as the exchangeable-ranks reference and adds a rank-calibrated forecast. The A8 ranks reject S2 at K ≤ 4.
