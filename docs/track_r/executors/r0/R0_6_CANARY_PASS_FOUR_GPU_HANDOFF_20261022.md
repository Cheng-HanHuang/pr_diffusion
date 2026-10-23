# Track R — R0.6 scoring canary PASS; four-GPU scoring continuation handoff
Date: 2026-10-22 America/Detroit / 2026-10-23 UTC

## Received PAC outcome

- Source head: `bc62488aa088ae681cbedd4264fa3a099bed82bb`.
- CPU test suite: **53/53 PASS**.
- CPU source and native FAST integrity audit: **720/720 PASS**.
- R0.6 canary root:
  `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_6_canary_20261023T020016Z`
- Canary status: `R0_6_EIGHT_CANDIDATE_CANARY_PASS`.
- CPU preparation and 8-candidate GPU scorer: RC=0 each.
- 8/8 prior, UnivFD and both-residual receipts created, no GT in GPU worker.
- Canary GPU allocation: **29.908057685010135 seconds**.
- Pre-canary known R0 compute: **2.84538 aggregate GPU-hours**.
- Post-canary known R0 compute: **2.8537 aggregate GPU-hours**.
- Immutable 720-candidate FAST score task list:
  SHA256 `f268e29b482da10aee31f79de43a6bf19d460f4203f7813c46dbee83b0591596`.

This is an execution-validity milestone, not a scientific efficacy result.
No reconstructed GT comparisons, thresholds, policy claims or prospective
findings are licensed by the canary.

## R0.6 4-GPU continuation implementation

`scripts/track_r/r0_6_parallel_autopilot.py`

The controller uses only the original R0.6 canary's GT-free scoring task
list, which the worker accepts across Git revisions ONLY with its exact
path, producing commit and SHA. Nothing rewrites or recomputes the
original eight scored candidates or the 720 generated native FAST roots.

The task interval [8,720) is divided into exactly **12** disjoint
contiguous chunks: eleven×64 and one×8. Each chunk runs on one of four
explicitly idle physical Blackwell GPUs, always with prior batch 64.
A 12-shard campaign must complete all 712 new candidates plus preserve
and account for the eight already scored canary candidates.

Every shard receives the exact saved candidate PNG, signed measurement,
and SHA; no GT or historical PSNR fields. Workers use the same prior
timesteps, noise coupling, 256 UNet forwards and score as R0.4;
UnivFD's eight-orbit batch probabilities remain authoritative with
the singleton/repeated-batch diagnostic; signed and clipped residual
scores stay frozen. The controller revalidates each result receipt,
including original source identity, no GT access, repaired-versus-orbit
classifier score, repaired-versus-prior score, and exact-orbit residual
numerics. It checks all shard candidates again before its PASS state.

The controller maintains a single atomic ledger and exclusive
controller lock. Individual owned GPU child PID and immutable attempt
logs are recorded. An active child contributes GPU seconds until it
terminates; four simultaneously running children contribute 4 GPU
seconds per elapsed second. R0.6's engineering budget is at most
**4 aggregate GPUh**. The controller also subtracts the exact
previous R0.5 known cumulative GPU seconds plus R0.6's 29.908s
canary from the unchanged **16 aggregate GPUh R0** ceiling.
Each 64-candidate chunk has a 900s maximum execution window.

Failure or exhaustion prevents new shard assignments. Only owned
child processes may be signaled by the controller; successful
results and failed-shard partial receipts remain immutable.
A restart must use the SAME executor SHA, output root and task SHA;
previous GPU allocation is charged on resume, with no active PIDs
and all completed output integrity validated. An interrupted chunk
can only resume on its original physical GPU.

Final expected result:
`R0_6_ALL_720_FAST_SCORES_COMPLETE_4GPU`
with `completed_scores=720`, `len(completed)=12`,
`active_attempts=[]`, and total R0 GPU allocation under 16h.

**All scripts are committed, but CPU regression tests and the actual
four-GPU job have not yet been executed on PAC.** Do not assert PASS
until the user's returned receipts show it.

## Next: R0.7 offline scientific analysis

After R0.6 completion, merge R0.4's exact 2520 scored-candidate
lineage with R0.6's 720 scored FAST roots and evaluate
all 3240 exposed/development candidates on CPU.

Use frozen GT RGB8, labels GOOD and ORBIT_ERROR, P1–P4, image-cluster
10k bootstrap, paired policy comparisons E0–E3 and oracle ceilings.
Use original R0.4 `p_fake_orbit[s_star]` as the authoritative
repaired UnivFD score for consistent analysis across original and
recovery heads. Set and verify residual thresholds using only GOOD
candidates of each source/family, never invent missing selector
statistics. Planner alone adjudicates D-material/D-repair/D-detect.

B24 confirmation305 must remain completely inaccessible through R1;
no reading, listing, hashing, querying membership, or scanning
protected paths. R0 remains retrospective and exposed-data only.
