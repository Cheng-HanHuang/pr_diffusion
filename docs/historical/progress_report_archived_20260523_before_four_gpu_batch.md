# Archived progress report: LF/S2 selector validation before four-GPU batch

Archived: 2026-05-23

This file marks the previous active report before replacing it with the four-GPU batch report.

Previous active file:

```text
docs/progress_report.md
```

Pre-replacement blob:

```text
blob sha: 512dec7f033a0c1ccbca4f6481433aa327b3e2ba
title: Progress report: FFHQ phase retrieval, LF/S2 selector, and failure analysis
updated date in file: 2026-05-23
```

That report covered the trajectory from the May 12 NP/SITCOM study through:

- scheduled/static S2 and memory-bank near-term experiments;
- diagnostic selector tracing;
- lightweight LF/S2 selector;
- seed tie-break postprocessor;
- successful seeds `100,101` tie-break result;
- failed validation on seeds `102,103` caused by missing good LF/S2 candidates for `00005` and `00014`.

The new active report extends this state with the four-GPU next-batch experiments:

- full four-seed LF/S2 selector control;
- focused S2 projection-start diagnostic;
- focused S2 lambda diagnostic;
- focused memory fallback diagnostic.

The exact previous full text is preserved in Git history through the blob SHA above and the commit immediately before the active report replacement.
