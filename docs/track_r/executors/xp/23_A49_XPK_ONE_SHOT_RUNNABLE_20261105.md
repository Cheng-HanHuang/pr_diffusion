# A49 — XP-K one-shot gated autopilot is runnable

**Scope:** A24 / A23.5b XP-K only. **Deadline:** CP4 2026-11-15 12:00 America/Detroit. **Main paper cut:** closed.

A49 completes the engineering promised by A40. The remaining XP-K campaign is now owned by one pinned, resumable PAC controller. The operator launches it once; routine gate promotion, GPU admission, pilot calibration, full generation, CPU scoring, ledger closeout, and CP4 publication are controller responsibilities.

## One automatic sequence

```text
pinned branch + CPU regression
 -> A40 native source binding
 -> committed one-parameter adapter proof
 -> frozen GT-free 100 x 4 manifest
 -> SITCOM / FAST / DAPS original-parameter two-root identity replays
 -> CPU A4 gate: |raw PSNR delta| <= 0.1 dB AND identical frozen R0 label
 -> independent historical DAPS F8 8/8 pixel-exact lineage receipt
 -> one calibrated image x 4 roots per family
 -> measured full-panel budget projection
 -> remaining 99 images x 4 roots per family
 -> frozen CPU scoring and v1/v1.1 forecast comparison
 -> independent A45 full receipt audit
 -> CP4 project-wide ledger
 -> one immutable result commit, retried unchanged until remote verification
```

No successful gate requires a new PAC command.

## GPU policy

The controller uses at most two physical GPUs concurrently by default. Identity replays and one-image pilots prefer GPUs in order **1, 0, 2, 3**, so currently quiet GPUs 1 and 0 are used first when safely admissible.

Preference is not an exemption from safety. Immediately before every launch the controller re-reads GPU memory/utilization/process telemetry. A worker is admitted only when its conservative 28 GiB reservation keeps projected use at or below 80% of physical VRAM and leaves at least 4 GiB free. A resident but inactive local-LLM allocation on GPU 0 is therefore harmless if this hard memory test still passes; otherwise the controller waits or uses another admissible GPU. It never signals foreign processes.

Resource contention persists `WAIT_SAFE_GPU` and retries. It is not converted into a scientific failure.

## Scientific stops

The original-parameter replay gate is fixed at raw-PSNR difference <= 0.1 dB plus identical frozen R0 label. Any mismatch records the evidence and stops as `STOP_ASK_A23_5B_GATE`. The controller never changes the tolerance, solver, seed rule, measurement, schedule, selector, or calibrated parameter after seeing the replay.

Calibrated generation cannot start before the corresponding replay receipt passes. DAPS additionally requires the already registered historical F8 8/8 exact receipt.

## Budget and completeness

The durable event journal records START before a worker is spawned and END after it exits. GPU time is charged by union of active intervals on each physical GPU; unfinished workers remain conservatively charged on restart.

Default launch accounting starts from the accepted G7 project checkpoint `247.80191737622505` GPU-h and zero prior XP-K A49 GPU-h. The operator may override these two values only when a newer accepted ledger exists.

After the three one-image pilots, A49 projects the cost of all remaining 99 images per family from measured native occupied time. The full registered panels are admitted only when the projection fits both the XP-K 30 GPU-h cap and project 400 GPU-h cap. It never silently reduces the image panel.

## Resumption and publication

The launch wrapper records `LAST_XPK_A49_RUN.txt`. Re-running the same pinned wrapper resumes that run instead of creating another one, unless the prior run already has the terminal publication receipt.

The source branch must remain at the exact launch SHA. If it moves while the campaign is running, publication fails closed rather than mixing engineering revisions with observed XP-K results. When all gates pass, A49 creates one results commit and pushes that same commit until the remote branch verifies it.

Expected successful terminal status:

`XP_K_CP4_ALL_THREE_FAMILIES_PUBLISHED`

Other terminal states retain all receipts rather than claiming a full panel.

## Operator surface

The only intended operator action is the single A49 launch wrapper:

`scripts/track_r/xp_a49_xpk_one_shot_launch.sh <exact-pinned-commit>`

The wrapper itself verifies the remote branch SHA, detaches the PAC worktree at that exact commit, records the run/log/PID pointers, and starts the controller under `nohup`.
