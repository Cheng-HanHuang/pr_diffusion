# B23.0 checkpoint report

Status: **REVISE_B23_0_CORRECTIVE_RERUN_REQUIRED**
GPU work performed: **NO**

The return capsule `B23_0_return_20260731T174838Z.tar.gz` and evidence commit
`0d35656b360b4b0d04a28812079f18de8a03a9af` are preserved but invalid as a B23.0 PASS.

The capsule's `STDERR_TAIL.txt` records two `ModuleNotFoundError: No module named 'prdiffusion'`
errors. The unit-test process returned nonzero because it ran outside the repository without an
explicit import path. The original grouped shell runner then executed later successful commands and
lost that nonzero status, so it emitted and published a false PASS.

The correction binds the PAC interpreter to the repository, records every prerequisite return code,
stops after the first failure, and makes the publisher independently reject any non-PASS step
ledger. The earlier evidence is not deleted or rewritten.

Failures: `PAC_IMPORT_CONTEXT`, `FAIL_OPEN_GROUPED_SHELL_STATUS`, `FALSE_PASS_PUBLICATION`

B23.1 and B23.2 remain unauthorized. A new timestamped zero-GPU capsule and post-run evidence commit
must pass review before any `PLANNER_RETURN` recommends B23.1 consideration.
