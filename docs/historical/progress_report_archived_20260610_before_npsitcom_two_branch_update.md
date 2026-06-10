# Archived progress report: pre NP-SITCOM two-branch update

Archived: 2026-06-10

This file preserves the active project state immediately before the June 2026 NP-SITCOM reorientation.  The active `docs/progress_report.md` has been rewritten around the two new branches:

- Branch A: NP-SITCOM candidate selection and future per-step defect/OOD controller.
- Branch B: NP-to-SITCOM sigma-space handoff and continuation diagnostics.

The previous active progress report was the May 23 multi-lambda NP selector validation report.  Its main conclusions were:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
configs      = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
selection    = mean post-projection winner LF-MSE vs noisy observation
```

Key May 23 findings:

1. Multi-lambda LF/S2 selection was a real improvement over fixed LF/S2.
2. The selector was usually near-oracle over the available NP candidate pool.
3. The full FFHQ-25 run with seeds `100,101,102,103` passed with raw mean about `29.36 dB`, min about `27.08 dB`, and no image below `25 dB`.
4. Later two-seed validations still failed on recurring hard images such as `00005`, `00013`, and `00028`.
5. Therefore, the limiting factor was candidate generation and seed diversity, not only final selector error.
6. This showed that multi-run NP selection was useful but still not a single-run, always-successful phase-retrieval solver.

The exact old file content is preserved in Git history before this archive/update commit.  This archive is kept as a navigation snapshot so the active docs can focus on the current NP-SITCOM direction.
