# B24 start here

## Current status and authorization boundary

B24 remains isolated from B23 and descends from signed-off B23.1 final head `27505e6328157ac9296c95dc5e611cbeef80de98`. B23 cross-family H0 failed: NP-1 and SITCOM-1 remain `BASELINE-ONLY` across family boundaries and no NP/SITCOM cross-family adapter qualified. B24 does not reinterpret that result.

**B24.0 PASS.** PRE_B24 exposure freeze has 333 rows with SHA-256 `d475c9c29b4f6ab2839ae21f4b19e33a52fa46f2fd7f0a6a7c5fff491e4b3068`.

**B24.1 PASS.** Four-independent-process serial/concurrent terminal equivalence was established for DAPS and SITCOM.

**B24.2 baseline screening COMPLETE.** The cumulative fixed screen contains 7424 realized rows with final census:

- A = 6925
- B = 107
- C = 307
- D = 85

Final screen manifest file SHA-256: `b516c8154cbbb790d8a3592b86736bb0d4bd47d0833d85ecf3d6a9d710e950ba`.

The primary hash-ranked balanced cohort is frozen as ABC300 = 100 A + 100 B + 100 C, CSV SHA-256 `4599c2a8c1f4a5922640e0c26d2c1efce7f1996d75dcabbff2e9a1c4b427cbce`.

The secondary severity diagnostic C1 is frozen as the 100 class-C cases with lowest pinned-SITCOM best-of-four PSNR, CSV SHA-256 `9c04994dbd91f6a5bb04280736e4dea346c337508a89f30ae8553a42867576b6`. C1 does not redefine class C and is not an unbiased primary benchmark.

**B24.3 image roles are frozen.** The primary 385-image method panel is split before method execution into:

- A100: 20 development / 80 confirmation
- B100: 20 development / 80 confirmation
- C100: 20 development / 80 confirmation
- D85: 20 development / 65 confirmation

Development total = 80. Confirmation total = 305. Pilot16 is 4 A + 4 B + 4 C + 4 D. Pilot16 CSV SHA-256: `124d3759e4fd540d2e870618dde59ff73d02cbb798d773a785296da5b140e98a`.

### Current GPU authorization

The user/planner explicitly authorized B24.3 implementation plus:

1. exactly one full-method Pilot16 engineering/calibration image on its prospectively frozen **new** development measurement;
2. if and only if that one-image run passes integrity/resource gates, the already-frozen Pilot16 may run;
3. the calibration image is part of Pilot16 and must be reused, not silently discarded or regenerated.

Not authorized: the remaining 64 development images, confirmation measurement materialization, confirmation execution, C1-only extra development exposure, or arm/checkpoint search beyond the frozen specification.

Read:

- `configs/b24/b24_3_method_dev_spec.json`
- `docs/b24/B24_3_METHOD_SPEC_AND_PILOT.md`
- `docs/b24/B24_3_GPU_AUTHORIZATION.md`

before B24.3 execution.

## Scientific question

The next question is whether changing **how native NP retains and allocates proposals** improves recovery beyond independent NP populations at comparable compute.

The frozen B24.3 arms are:

- NP-1 native single-trajectory control;
- four independent NP-1 trajectories (primary compute-matched population control);
- NP_EPP: early population pruning + saved-compute reallocation;
- NP_EPP_RANDOM_PRUNE: same compute and branch schedule, random/hash pruning control;
- NP_EPP_NO_REALLOCATION: measurement-based pruning without spending saved compute;
- NP_DPS: delayed proposal selection, the preferred main-method hypothesis.

Historical NP-8-RS remains two scoring configurations × four seeds and will be used in the later development comparison, not silently redefined as eight identical NP runs.

## Runtime information contract

Ground truth must never decide proposal retention, pruning, branch allocation, survival, routing, stopping, or executable terminal selection.

Runtime early pruning uses the frozen measurement-only trailing-32 low-frequency MSE score. Executable terminal selection uses the frozen post-projection measurement statistic. Ground truth may be used only for offline reporting/oracle diagnostics after runtime decisions are fixed.

Every scientific run reports proposal/UNet counts, NP-1-equivalent work, GPU-active/wall time, terminal count, live-branch events, and memory.

## Resource contract

The global B24 process/group hard ceiling remains **52,452 MiB**. The old baseline 10,240-MiB admission gate is not automatically reused for NP branching.

The one-image B24.3 smoke launches under the conservative pre-existing 52,096-MiB free-memory gate, measures actual NP branching memory, and derives a pilot-only admission gate from the observed B24 process peak plus a 4096-MiB reserve. It never evicts or kills other jobs and uses an explicit physical GPU binding.

## Repository identities

- branch: `codex/b24-bestof4-failure-sweep`
- required ancestry point: `27505e6328157ac9296c95dc5e611cbeef80de98`
- draft PR: #38, base `codex/b23-execution`
- PAC worktree: `/egr/research-pac/huang248/pr_diffusion_b24`
- PAC output root: `/egr/research-pac/huang248/outputs/pr_diffusion/b24`

Do not modify PR #37 or the B23 output root. Never merge, rebase, squash, retarget, force-push, or rewrite B24 history.
