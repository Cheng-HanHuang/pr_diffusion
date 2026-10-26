# Theory predictions registry (planner-maintained)

**Purpose.** This file proves that each predictions file was frozen before the outcomes it predicts were analyzed (`XP_PREREG_v1_1.md` §5).

**Who writes it.** Only the planner, on `claude/prdiff-track-r-planner-20261021`, as soon as the theory executor reports a freeze.

**How the XP analysis uses it.** For each entry, the analysis checks three things:
1. the entry exists;
2. `git show <commit>:<path> | sha256sum` matches the SHA-256 below;
3. the commit time precedes the first XP commit that contains the matching outcomes.

| Freeze | Theory branch | Commit | Path | SHA-256 | Commit time (UTC) | Seen by planner (UTC) | Covers |
|---|---|---|---|---|---|---|---|
| v1 | — | — | `docs/track_r/theory/TH1_PREDICTIONS_v1.json` | — | — | — | FFHQ: FAST (S5), SITCOM (B22+DEV80), DAPS (B22+DEV80); due **Oct 31** |
| v2 (optional) | — | — | `docs/track_r/theory/TH1_PREDICTIONS_v2.json` | — | — | — | ImageNet FAST; due 48 h after the XP-D1 phase-1 counts |

**Embargo status:**
- FFHQ K > 4 outcomes (XP-A, XP-B, XP-F; B22 DAPS K > 2): **EMBARGOED** until v1 is registered.
- ImageNet FAST outcomes with run index ≥ 4: **EMBARGOED** until v2 is registered, or until 48 h after the phase-1 counts if no v2 is made.
