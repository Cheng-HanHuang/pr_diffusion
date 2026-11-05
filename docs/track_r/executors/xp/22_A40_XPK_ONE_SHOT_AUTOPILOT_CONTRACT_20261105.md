# A40 — XP-K gated one-launch autopilot contract

**Planner:** A24 / round 20, `aafaf7f35e9d9d0beafa459259ca10b7fda4e8f5`. **Deadline:** CP4 Nov 15, 2026, 12:00 ET. **Scientific scope:** A23.5b XP-K only. **Main paper cut:** closed.

## User's operational request

Replace repeated per-gate PAC shell invocations with **one pinned, resumable controller**. It should perform and verify the next stage itself, promote only scientifically verified gates, safely admit co-tenant GPUs, score and publish completed panels, and finish CP4 without asking for routine launches. **A science gate that fails is a STOP/ASK, not a retry with loosened tolerance or changed method.**

The *single-launch controller is not yet runnable*: required replay/generation adapters must be finished, pinned and CPU-tested before it is offered for PAC launch. A39 provides only two config-leaf screens; A40 adds prospective native source-binding/Hydra evidence. Neither substitutes for original-parameter XP-K identity replays.

## Stage machine — execute in the registered order

<pre>
A39_CPU_CONFIG_SCREEN_PASS
  -> A40_NATIVE_SOURCE_BINDING_PROOF (SITCOM, FAST, DAPS)
  -> COMMITTED_ONE_PARAMETER_ADAPTER_DIFF (per family)
  -> B22_ORIGINAL_PARAMETER_2_ROOT_IDENTITY_REPLAY (per family)
  -> [DAPS ONLY] PRIOR_F8_8ROOT_REPLAY_PROOF (historical 400 x 5)
  -> CALIBRATION_1_IMAGE_PILOT (unchanged settings except registered leaf)
  -> CALIBRATION_100_IMAGE_4_ROOT_FAMILY_PANEL
  -> FROZEN_OUTCOME_SCORING (G/O/C/I, residual/sigma, rho_star, E1_CLIP, O_orbit)
  -> REGISTERED_MECH_V1_AND_V1_1_COMPARISON
  -> CP4_FINAL_PROJECT_WIDE_LEDGER_AND_NOT_RUN_INVENTORY
  -> IMMUTABLE_EXACT_GIT_COMMIT_PUBLICATION
</pre>

**Native registered scalars:** SITCOM Langevin `tau=0.01→0.05`, Euler measurement-gradient step *unchanged*; FAST-DIPS projection `epsilon=0.05→33.26`; DAPS historical FFHQ Langevin `tau=0.01→sqrt(2)*0.05`, original 400 annealing × 5 ODE (not DAPS-4K 400 × 10).

**Gate I (source diff).** The same calibrated adapter must run in historical original-parameter mode and calibrated one-parameter mode. CPU diff must verify the one changed native bound leaf and identical code, model, measurement/operator, noise, seeds, stopping rules, preprocessing, and Euler step. Native `sampler.py` in SITCOM, `main.py` FAST, and **patched original** DAPS `sampler.py` + `cores/mcmc.py` + resolved Hydra config are verified by Git SHA; do not silently substitute pristine DAPS-4K. A39 JSON leaf diff alone never certifies this.

**Gate II (identity replay).** From the exact future calibrated worker in **original-parameter mode**, reproduce **two registered historical B22 roots per family** with the frozen original measurement and historical seed/RNG rules. Compare immutable candidates with prior B22 source on GPU-independent CPU scorer. The ordinary A4 tolerance is *raw PSNR difference ≤ 0.1 dB and identical R0 label*; report bitwise PNG matching separately. **Do not relax this tolerance for co-tenancy or change it after observing results.** If the gate fails, record all data and STOP to ask the planner. Never adapt an alternate source after seeing an identity failure without renewed preregistration.

**Gate III (DAPS).** Must independently verify the existing registered XP-F8 historical 8/8 pixel-exact receipt *before* running calibrated DAPS. Historical XP-F8 success alone is NOT the XP-K identity replay; both are mandatory.

**GPU policy.** Under A24, co-tenant-safe XP-K replays and calibrated runs are allowed. A quiet GPU is preferable for the identity replays, not a hard clean-GPU gate. Read VRAM/foreign PIDs before admitting (hard ≤80% memory). Never signal foreign processes; cancel only own groups on real safety errors. Clean-GPU-only timing is unrelated and remains parked. If no safe GPU exists, wait with a bounded resource interval and persist `WAIT_SAFE_GPU` (no false science failure). Any replay mismatch means STOP/ASK regardless of GPU cleanliness.

**Budget and time.** Separate cumulative XP-K physical GPU occupancy ≤30 h, project cumulative ≤400 h (last accepted G7 247.80191737622505 h is only a checkpoint, not proof no other projects ran since), include failed pilots, replays, and overlapping worker union time. CP4 hard cutoff 2026-11-15 12:00 America/Detroit. A stage can advance only if the *measured* projected remaining budget can finish the **whole registered image panel**, otherwise stop and list not-run portions; don't silently reduce to selected images.

**Reproducibility.** Original parameter two-root replay and calibrated seeds `dataset=ffhq_cal` use frozen measurement identities and distinct declared seed domains. Preserve historical RNG stream for replay; new calibration seeds must be collision-audited against historical roots. Keep GPU manifests GT-free; score candidates and comparisons only on CPU/offline with frozen thresholds. Store each result atomically with SHA and native provenance. On restart, verify and reuse all exact receipts, never overwrite or double-charge. Publication pushes the **same exact local commit** until remotely verified, with no new commits for network retry.

**Terminal statuses:**
- `XP_K_CP4_ALL_THREE_FAMILIES_PUBLISHED` only when all three calibrated panels, forecast comparisons, full ledger and unrun inventory pass.
- `WAIT_SAFE_GPU` means resource blocked; scientific gates stay pending, not failed.
- `STOP_ASK_A23_5B_GATE` means a native/source/identity/F8 gate failed; planner guidance mandatory.
- `STOP_BUDGET_OR_DEADLINE_WITH_RECEIPTS` means incomplete production, never report a full panel.
- `ENGINEERING_NOT_READY_NO_GPU` means a required audited adapter is missing from the pinned checkout.

## Provenance already verified

- A39 PAC CPU tests 6/6 PASS and config-only screens PASS: `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_a39_xpk_cpu_20261105T070342Z`; **native gate still pending**.
- DAPS earlier F8 8/8 pixel-exact replay is documented by `docs/track_r/executors/xp/returns/XP_F8_LEGACY_DAPS_8ROOT_REPLAY_EXACT_PASS_20261029.md`; not an XP-K replay.
- G7 already **planner-accepted**; use four-method closeout `4d1bd609f70428c767a95a51b37f8c3a7a5d0c71`; no G7 or A21.10 reruns.
- The A40 CPU evidence script `scripts/track_r/xp_a40_xpk_native_binding_cpu.py` checks native SITCOM and FAST *actual* use of the changed parameter and DAPS historical Hydra resolved one-leaf change. **It has not yet run on PAC and does not certify a new calibrated adapter or GPU replay**.

The next implementation task is a single two-mode native worker per family (`--historical-replay` / `--calibrated`), a strict CPU replay comparator, a shared resource/budget controller, then the registered scoring and CP4 publisher. **Do not supply a purported all-in-one PAC launch command until all those components exist and pass CPU tests.**
