# Archived progress report: May 12 FFHQ NP/SITCOM state

Archived: 2026-05-23

This file records that the previous active progress report was superseded by the selector-focused report written after the LF/S2 selector and tie-break validation experiments.

The previous active file was:

```text
docs/progress_report.md
```

Its content is preserved in Git history immediately before the selector report replacement.  The active pre-replacement blob was:

```text
path: docs/progress_report.md
blob sha: 39fa4dd4ce3fd72d20bdb06e549e6983ce3aca04
last fetched title: Progress report: FFHQ phase retrieval, tuned Noise Picking, and SITCOM-ODE comparison
updated date in file: 2026-05-12
```

The archived report covered the FFHQ-25 guided NP tuning phase before the LF/S2 selector work, including:

- practical NP setting `score_radius=0.6`, `proj_radius=0.2`, `proj_start=300`, `soft=5`, `hard=1`;
- top-8 confirmation and projection-radius ablations;
- SITCOM-ODE comparison at `sigma_y=0.05`;
- noise-level sweep against SITCOM-ODE;
- candidate-count ablations;
- score-mode experiments S1--S4;
- focused S2 lambda sweep;
- conclusion that small low-frequency projection is important, broad projection is harmful, and the simple LF score is fragile.

The new active report keeps those conclusions as background and extends the project state through the selector, tie-break, and validation experiments.
