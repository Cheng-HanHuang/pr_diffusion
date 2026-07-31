# B23.0 correction ledger

This ledger is append-only provenance for partial or invalid B23.0 returns. An entry here is never
silently deleted, rewritten as a success, or used to authorize GPU work.

## Bootstrap Git-identity stop

- Observed before `20260731T174838Z`.
- The clean PAC worktree lacked local `user.name` and `user.email`; the bootstrap returned code 5.
- The stop was non-scientific and performed no GPU work. The worktree-local identity was later set
  to `Cheng-HanHuang <156343199+Cheng-HanHuang@users.noreply.github.com>`.

## Invalid return `20260731T174838Z`

- Pre-run commit: `78b32319dd49719c0ffa7bdcc0b358b5d13317cd`.
- Preserved evidence commit: `0d35656b360b4b0d04a28812079f18de8a03a9af`.
- Archive SHA-256: `57552c104153aa4c7a6e7cb0da901fc5b5726124753ace82c5a04eef29d1cadd`.
- `STDERR_TAIL.txt` shows `test_contracts` and `test_protocol` failing import with
  `ModuleNotFoundError: No module named 'prdiffusion'`; only five discovered tests completed.
- Root cause 1: the PAC test invocation did not bind the execution repository as its import root.
- Root cause 2: a grouped shell sequence retained only its last command status, allowing later
  successful steps to mask the failed tests.
- Disposition: invalid as a B23.0 PASS and preserved as failure evidence. It must not support a
  `PLANNER_RETURN`, planner sign-off, or B23.1 authorization.
- GPU work performed: no.

## Corrective gate

A valid replacement must have a later timestamp and pre-run commit, run all four zero-GPU steps
from the repository with the explicit DAPS Python, and include `ZERO_GPU_STEP_RESULTS.tsv` with this
exact ordered PASS set: unit tests, repository validation, B23.1 dry renderer, PAC evidence
collection. Packaging and publication must stop if any row is absent or nonzero.
