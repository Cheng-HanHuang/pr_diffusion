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

## Planner revision at reviewed head `5096dc02d9a6ecd6a8615d25f026433adf660e5e`

- Planner verdict: `REVISE_BEFORE_SIGNOFF`.
- Accepted evidence: corrected fail-closed PAC execution, ancestry, zero-GPU status, 21 tests,
  model/environment/source freeze, and preservation of the invalid evidence commit.
- Required bounded correction: evidence-derived replay PASS rules; compute/FRE/RNG reconciliation
  and recomputation; conservative unresolved exposure and image-level future disjointness; bounded
  untracked-source classification/hashing; current start-file/provenance records; and always-on full
  validation of every schema assertion used by B23.
- Scope: B23.0 correction only. GPU work, B23.1, registry population, B23.2, Track B, and Track C
  remain unauthorized.
- GPU work performed for this planner review/correction: no.

## Planner contract-closeout revision at reviewed head `718998098cde1d8051e29e8d50d1285a04ca6ee9`

- Planner verdict: `REVISE_BEFORE_SIGN_OFF`.
- Accepted and preserved: correct ancestry/draft PR state, pre-run commit `4cf026d8`, evidence commit
  `241551d9`, zero-GPU 62/62 PAC tests, capsule `20260811T031927Z`, exposure/source/schema/compute
  corrections, and all earlier correction history.
- Required bounded closeout: tolerance-qualified replay must identify a pre-wrapper frozen envelope
  and per-metric numerical floors, mechanically remain inside envelope plus floor, and carry a
  complete or justified-unavailable deterministic audit; executed calibrated work must have
  strictly positive GPU-active/wall timing and a non-placeholder timer method; negative tests must
  reject the supplied out-of-envelope and zero-GPU/`NOT_RUN` counterexamples.
- No numerical tolerance or calibration result may be invented in B23.0.
- Scope: B23.0 contract closeout only. GPU work, B23.1, registry population, B23.2, large panels,
  Track B, and Track C remain unauthorized.
- GPU work performed for this planner review/correction: no.
