# Archived progress report: four-GPU batch before full-25 multi-lambda validation

Archived: 2026-05-23

This file marks the active progress report before replacing it with the report that includes the focused and full-25 multi-lambda selector validation results.

Previous active file:

```text
docs/progress_report.md
```

Pre-replacement blob:

```text
blob sha: 3620bf15cd6f61f1da05252b564d24cf143ffb3c
title: Progress report: four-GPU LF/S2 selector, lambda selection, and memory fallback study
updated date in file: 2026-05-23
```

That report covered:

- four-seed fixed LF/S2 selector candidate-availability control;
- projection-start diagnostics;
- S2 lambda diagnostics;
- memory hard2 fallback diagnostics;
- the conclusion that multi-lambda selection was the next main direction.

The new active report extends this state with:

- focused multi-lambda selector passing on the hard/guard subset;
- full FFHQ-25 multi-lambda selector passing with seeds `102,103`;
- score-radius `0.4` and projection-start `200` ablations failing;
- a revised plan focused on validation across seed pairs, four-seed multi-lambda, and single-seed/adaptive selection analysis.
