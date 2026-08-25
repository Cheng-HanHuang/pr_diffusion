# B23 protocol and gates

## Precedence

The accepted final plan and PAC return protocol control this work:

- `docs/planning/01_B23_EXECUTOR_START_HERE.md`
- `docs/planning/2026-07-30_b23_final_research_plan.md`
- `docs/planning/2026-07-30_b23_supersession_ledger.md`
- `docs/planning/02_B23_PAC_EXECUTION_AND_RETURN_PROTOCOL.md`

The superseded July 29 plan and July 30 amendment are historical context only; their schedule
suggestions are not combined with the final plan.

## Authorization ledger

| Unit | State | Permitted now |
|---|---|---|
| B23.0 repository/protocol freeze | authorized | CPU-only validation, inventory, schemas, tests, manifests, evidence packaging and scoped push |
| B23.1 native replay/donor audit | not authorized | code/runbook preparation and non-executable dry rendering only |
| B23.2 schedules/screens | not authorized | empty preregistration template only |
| Large panels, Track B, Track C | not authorized | none |

No experiment launcher may run in B23.0. Importing PyTorch for a synthetic CPU serialization test
is permitted only with `CUDA_VISIBLE_DEVICES` empty and must leave CUDA uninitialized.

## Immutable repository gates

- planning branch: `codex/post-b22-reliability-plan`
- operational handoff: `d1119e37fa688ac07f48ffc87ce19b13dbfb1c27`
- accepted plan: `ed4f46e8f116648eda76d387388d762d7cb8f3d7`
- frozen B22 base: `ba78c06e0c5eac0c915263e4faed0b262d5e917a`
- execution branch: `codex/b23-execution`

The execution head must descend from the operational handoff. The planning PR and protected branch
must not be merged, retargeted, rebased, squashed, force-pushed, or deleted. The dirty historical
checkout and `b19_solver_integration` are read-only evidence.

## B23.0 gate checklist

1. Remote and ancestry identities match the immutable SHAs.
2. The clean PAC execution worktree is on `codex/b23-execution` and matches the remote head.
3. Historical, DAPS, SITCOM, NP/SITCOM, and DiffFPR commits and tracked diff hashes match.
4. `daps`, `prdiff_ffhq`, and `sitcom_ode_bw` are probed through their explicit PAC Pythons.
5. Model and SITCOM checkpoint identities match the frozen SHA-256.
6. Operator, oversampling, noise, clipping, normalization, and seed policies are explicit.
7. The exposure manifest covers every bounded evidence source; uncertainty becomes exposure.
8. Future split and smoke registries remain empty.
9. Four distinct native state types are retained; no universal tensor semantics are assumed.
10. Raw compute, RNG, branch, candidate, memory, and timing ledgers validate.
11. FRE arithmetic tests pass, while numeric atomic weights remain absent until measured.
12. Replay schemas, complete-or-justified-unavailable deterministic-flag audit, pre-wrapper
    tolerance-freeze identity/floors, envelope-bound negative tests, seed vectors, and CPU
    serialization tests pass.
13. Executed calibrated work has positive GPU-active/wall timing and a non-placeholder timer method.
14. Dry rendering emits zero executable GPU commands and all B23.1/B23.2 authorization flags are false.

Every item appears in the B23.0 `GATE_DECISION.json` and final `PLANNER_RETURN`.

## Immediate stops

Stop and preserve the partial capsule if any of these occurs:

- a remote head, ancestry, source SHA, patch hash, checkpoint hash, or environment identity differs;
- a critical parent source exists only as an unresolved dirty change;
- model/operator/measurement/preprocessing/seed identity cannot be recovered;
- a proposed new split cannot be proven disjoint from pre-B23 exposure;
- a B23.0 subprocess initializes CUDA or reaches a GPU;
- an operation, RNG draw, retry, branch, or terminal candidate would be hidden;
- a tolerance-qualified wrapper lacks a pre-wrapper freeze identity, exceeds any frozen envelope
  plus floor, or has an incomplete and unjustified deterministic-flag audit;
- executed calibrated work reports zero GPU-active/wall time or a placeholder timer method;
- a schema/config/manifest/test is incomplete, duplicated, corrupt, or non-finite;
- progress would require a schedule candidate or scientific amendment.

Do not repair a mismatch by changing the parent. Report observed and expected identities verbatim.

## Repository/PAC split

Commit text, JSON, CSV, schemas, hashes, concise diagnostics, and the explicitly approved small
capsule. Keep checkpoints, datasets, measurements, full logs, raw tensors, and reconstruction panels
under `/egr/research-pac/huang248/outputs/pr_diffusion/b23`. PAC-only claim evidence must have an
absolute path, role, byte count where meaningful, retention note, and SHA-256 where the target is a
file.

Branch-stack recommendation: leave planning PR #36 unchanged; maintain one draft execution PR from
`codex/b23-execution` against `codex/post-b22-reliability-plan`; do not perform history surgery.
