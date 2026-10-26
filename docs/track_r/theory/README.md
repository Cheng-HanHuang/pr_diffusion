# Track R theory workspace

This folder is written by the theory executor, on `theory/prdiff-track-r-*` branches.
Its handoff is `docs/track_r/executors/theory/00_THEORY_EXECUTOR_HANDOFF.md` (revision B, 2026-10-26).

**Exception:** `PREDICTIONS_REGISTRY.md` is written only by the planner, on the planner branch.

## Expected files

- `TH1_NOTE.tex` and `TH1_NOTE.pdf`: statements and proofs (Nov 6).
- `TH1_PREDICTIONS_v1.json`: frozen FFHQ predictions (**Oct 31, hard**). Must be registered before any FFHQ K > 4 outcome is analyzed.
- `TH1_PREDICTIONS_v2.json` (optional): frozen ImageNet FAST predictions, within 48 h of the XP-D1 phase-1 counts.
- `TH_RETURN_<date>.md`: returns to the planner.
- `TH2_FEASIBILITY_MEMO.md`, due 2026-11-13.
