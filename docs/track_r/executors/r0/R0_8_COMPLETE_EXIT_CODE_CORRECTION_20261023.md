# R0.8 overnight completed; success exit reported incorrectly — 2026-10-23

## Operator-returned PAC facts

Run root:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_8_overnight_20261023T082909Z`

Actual observed stdout:
```
R08_OVERNIGHT_AUDIT_PASS 3240 P4_TABLES 37
R08_PLANNER_GATE PLANNER_GREY_ZONE_REVIEW
R08_PLANNER_RETURN_ZIP /egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_8_overnight_20261023T082909Z/R0_8_PLANNER_RETURN.zip
R08_FINAL_STATUS R0_8_PLANNER_HANDOFF_READY
STOP_R08_OVERNIGHT SystemExit(0)
Traceback ... try:sys.exit(main()) ... SystemExit: 0
```

The independent research audits and package creation had already completed,
and the success receipt was written, **before** a Python entry-point bug
intercepted a normal `sys.exit(0)` with `except BaseException`.
This produced an incorrect final shell exit code 8. It is **not evidence
of a scientific or input-audit failure**.

## Narrow engineering repair

The entry-point exception handler now catches `Exception`, rather than
`BaseException`; `sys.exit(rc)` runs *outside* that handler.
Regression tests exercise a successful `main() -> 0` process and a
true `RuntimeError` process and check expected codes 0 and 8.

The original run's output and Git HEAD should be retained unmodified.
**Do not rerun R0.8 solely to fix the shell exit code**.

The operator should independently verify the existing
`R0_8_COMPLETE_RECEIPT.json`, ZIP SHA256 and ZIP integrity,
then attach `R0_8_PLANNER_RETURN.zip` for scientific analysis.

`PLANNER_GREY_ZONE_REVIEW` is a *provisional* R1 decision branch from
the frozen gates; its numerical evidence lives in
`R0_8_PROVISIONAL_DECISIONS.json` and the accompanying
`R0_7_ANALYSIS.json`. Do not infer the numerical gate values from
the status alone or claim a planner-reviewed conclusion.

No new GPU scoring, R1 experiment, access to B24 confirmation305, or
change to the frozen R0 definition is authorized by this repair.
