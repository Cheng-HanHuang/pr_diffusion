# Track R R0-N — complete N0–N3 scientific return and K2 adjudication request

**Date:** 2026-10-24 (America/Detroit)  
**From:** Track R executor  
**To:** Track R scientific planner `claude/prdiff-track-r-planner-20261021` and metaplanner  
**Scientific scope:** Frozen R0-N evaluation-letter evidence; **not a new method, no R1**.

## 1. State, immutable sources and receipts

R0-N N0, N1, optional N2, and N3 are **complete**. The immutable
N0–N1–N3 primary return remains at
`docs/track_r/executors/r0n/returns/R0N_20261024T061111Z/`,
GitHub commit `5af47dbb4f6dba5ba5aa140b9b30677039b31e39`.

The supplemental, complete N2 return was committed and pushed
from PAC at
`docs/track_r/executors/r0n/returns/R0N_N2_RECOVERY_20261024T073405Z/`,
commit `081fc753a29fb61742ffe8553f30794f2eea3459`.

N2 PAC files:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0n_n2_recovery_20261024T073405Z/`.
ZIP `R0N_N2_RECOVERY_RETURN.zip` SHA-256
`47a0b6fcc67828be48ab8721cb7ac2a583d6a079d1c3704fd141f210bb7eb6f4`.
Scoring manifest SHA-256
`7be734de4cd2e6f395add521717506be9260c37708f8902e248c1848b68b705b`
reused 3,240 original, losslessly round-tripped, GT-free PNG tasks.
No reconstruction or new measurements were generated.

N2 execution diagnostics:
- `107 passed` CPU regression tests prior to final N2 retry.
- GPU physical device 0 admitted at 97,250 MiB free, zero reported used.
- `R0N_N2_CLIP_GPU_PASS 3240`, instrumented GPU score wall
  **27.588 s**; GPU-stage complete subprocess wall **29.758 s**.
- `R0N_N2_CLIP_OFFLINE_EVAL_PASS` on all 3,240 frozen candidates.
- Including the original 6.83-second failed N2 GPU invocation,
  conservative summed GPU-stage subprocess wall **36.588 s**,
  well below the frozen 900-second / 0.25-hour ceiling.
  An intervening controller argument bug launched no GPU child.
- Published N2 `comparison_sha256`:
  `322c3ef02c1f8b509b1048e9ac9f3896bb86988a76afb8f62407e0a528db0233`.
- Invariants: external UnivFD/CLIP repo pinned, CLIP checkpoint SHA
  pinned, six prompt strings/preprocessing/scoring fixed, no GT or
  measurement path in GPU manifest; offline metric-only R0.7 PSNR join.
  The UnivFD bundled CLIP code's modified transformer returns
  `(intermediate_features, final_tensor)`; a local wrapper reproduced
  its original encode-text operations using that final tensor.
  See `docs/track_r/executors/r0n/R0N_N2_CLIP_TEXT_TUPLE_RECOVERY_20261024.md`.

The prior partial N0–N3 executor return mentioned N2 unavailable;
**this new, independently verified N2 supplement supersedes
only that old N2-status field**. Do not interpret the stale
partial return as a currently missing CLIP comparison.

## 2. N2 frozen K2 comparison

**Outcome:** globally oriented exact-orbit raw failures
(`s_gt == '111'`), not every incorrect reconstruction. The one
partial-channel orbit error is excluded from the global-only
success denominator. GOOD harm means a previously raw Good25
candidate becomes below 25 dB after the global-orientation choice.

| GT-free global-only selector | Corrected global-orbit failures | Harm among raw GOOD |
|:--|--:|--:|
| **CLIP zero-shot**, exact six frozen upright/inverted prompts | **84/84 (100%)** | **0/2,040** |
| Diffusion prior `S`, restricted to `000/111` | 83/84 (98.81%) | 0/2,040 |
| Diffusion prior `S_mid`, restricted to `000/111` | 84/84 (100%) | 0/2,040 |
| UnivFD `p_fake`, restricted to `000/111` | 71/84 (84.52%) | 108/2,040 (5.29%) |

Unstratified image-cluster bootstrap CIs:
- CLIP global repair [1.000,1.000], observed GOOD harm [0,0].
- `S` repair [0.9565,1.000], observed GOOD harm [0,0].
- `S_mid` repair [1.000,1.000], observed GOOD harm [0,0].
- UnivFD repair [0.7101,0.9467], harm [0.0299,0.0790].

**Important inference caveat:** When no event occurs in the observed
sample, the ordinary nonparametric image bootstrap gives a
degenerate [0,0] interval. This does **not** establish zero
population harm or guaranteed perfect repair. The 180 source
images contribute correlated solver/run outcomes, not 3,240
independent subjects.

CLIP's 100% global-orbit repair and zero raw-GOOD harm persisted
in each populated source/family stratum, including 10 global
errors in new native FAST S5, one FAST S4, DAPS, SITCOM and NP.
Strata with zero global-orbit errors have no defined repair rate;
do not score those as 100%.

**N2 raw flip counts:** CLIP chose global 180 for 319/3,240;
prior S 215; S_mid 342; UnivFD 520. The higher flip count
does not by itself mean more Good25 harm because many decisions
concern candidates that were already raw-bad; Good25 harm is not
a full PSNR-monotonicity claim.

## 3. Cost and the exact scope of the K2 inference

CLIP scoring processed both candidate orientations with batched
CLIP image forwards (32 reconstructions / 64 images per batch):
**27.588 seconds total** instrumented GPU stage,
**8.515 ms per candidate**, including the model-load/scoring stage's
wall-time definition. Full child-process wall: 29.758 seconds.

The full `S` orientation score was originally measured as part
of a 256-image-forward per-candidate GT-free scorer, averaging
**2.7554 seconds** per candidate on its native setup; this total
also includes its previously bundled detector/residual work.
`S_mid` computes a subset of those saved draws and was **not**
timed independently.

N1 selected from the 60 configurations, based **only** on
DEV80, `single_300|J1|global2` (two UNet image forwards).
Retrospective outcomes:
- DEV80: 35/36 exact-orbit errors (includes one partial
  error inaccessible to `global2`), 0/617 Good25 harm.
- B22 random: 49/49 exact-orbit errors and 0/1,423 Good25 harm.
- N1's **0.02153 s per candidate** for that configuration
  is **only a forward-count extrapolation** from the original
  full-score mean; there is no directly timed two-forward
  implementation in R0-N.

Therefore **CLIP's measured 0.008515 s/candidate is cheaper
than N1's 0.02153 s/candidate projection under those workloads,
but this is *not* a controlled paired speed experiment.**
It would be incorrect to claim a definitive 2.5× runtime
advantage over an optimized two-forward implementation.
Against the originally expensive full-prior scorer, CLIP is
clearly much lighter in the observed experiment, but its
setup, batching and work done differ.

## 4. Frozen scientific letter kill criteria

**K3 — replay:** PASS. N1 independently replayed all 3,240 R0.7
candidates, the K3 2,040/85/324/791 labels, 84/85 primary
prior repairs and 0/2,040 GOOD harm, all seven registered
P4 Good25 witnesses, and the SHA-pinned 3,240 individual
score receipts. Final N1 numerical evidence and figures
were committed without changing source protocols.

**K1 — repair survives random cohort:** NOT KILLED.
At B22 deployable selection, E1 repaired DAPS **3/8** E0
failures at K2 and SITCOM **5/11** at K4;
NP at K8 **0/6**. This supports a *limited* orientation
failure-anatomy contribution, not a new solver.
The random cohort's failures are heavily generation-limited.

**K2 — CLIP >= prior at lower cost:** **Strongly implicated;
planner adjudication required.**
The frozen CLIP baseline achieved **84/84** global repairs and
**0/2,040** harm, versus full prior **83/84** and the same
observed harm. CLIP also ties half-grid prior on those two
metrics and is inexpensive. N1's cheap `single_300`
global-only rule has 49/49 B22, 35/35 globally repairable
DEV80 errors, likewise zero raw-GOOD harm; CLIP does not
establish an accuracy advantage over that cheap variant.
The precise "lower cost" criterion relative to *that*
two-forward variant should be interpreted with the timing
caveat in §3, not silently assumed.

If the planner rules K2 triggered, the preregistered
consequence is **remove the diffusion-prior repair as a
distinct methodological contribution**. The letter,
if written, should honestly frame global-orientation
resolution as a low-cost generic uprightness problem and
center its contribution on:
1. failure anatomy of diffusion PR after deployable,
   GT-free best-of-K selection;
2. signed residual versus image quality by solver family;
3. limitations of GT best-of-K oracle reporting;
4. measurement-invariant orbit and practical orientation
   corrections using CLIP or a generic orientation test.

This does **not** resurrect R1, authorize new trials,
or imply novel CLIP orientation technology.

**K4 — submission date:** the planner's 2026-12-11
time box remains unchanged.

## 5. Final executor recommendation and required planner action

R0-N is **scientifically complete under its approved scope**.
No new GPU study is requested. Please issue the frozen K2
ruling, and either:
- reframe the four-page SPL evaluation letter around
  oracle-vs-deployable failure anatomy and generic
  orientation repair, with explicit limitations, **or**
- park the letter if its residual contribution is
  insufficient relative to related literature.

Novelty is **not** established merely by achieving 84/84
or reducing to a two-forward criterion; CLIP's result
provides direct negative evidence for claiming a uniquely
diffusion-prior orientation contribution. The executor
does not initiate further experiments or choose publication.

**Preservation requirements:** no R1/R2, no further
reconstruction or measurement experiment, no
confirmation305 listing or access. All original N0–N3
GitHub returns and immutable PAC receipts remain in place.
