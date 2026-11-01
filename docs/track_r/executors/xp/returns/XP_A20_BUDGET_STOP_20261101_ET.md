# XP-N A20 budget-reservation terminal stop — 2026-11-01 ET

## Evidence from PAC operator (do not infer extra run outcomes)

- Frozen A17 / original XP-N cohort: 100 new ImageNet images, 64 independent FAST roots each, 6,400 roots / 90 chunks. Immutable native scientific settings, manifests, and seeds unchanged.
- Frozen v5 prospective forecast **already registered and pushed before held-run scoring** on `codex/prdiff-track-r-xp-n-v5-20261101T194525Z` commit `55ba86b71cafbd8eb066c777ac141823d9cd9a1d`; sealed forecast SHA-256 `a09f553292947f4cf9c0f95294a9afd205938f0dc18dd75f4d6ff655916a3ee3`.
- PAC state from `xp_a20_xpn_recovery_*/XP_N_RECOVERY_STATE.json`: `STOP_RECOVERY_20H_OR_PROJECT_CAP`, `done=72/90`, `active=[]`, `failures=[]`, `charged_new_gpu_h=18.2625`, `original_charged_gpu_h=0.493733`, total `18.7562/20` GPU-hours.
- Completed chunks by frozen phase order: 20/20 pilot (400 roots), 20/20 main (1,200 roots), 32/50 late (3,072 roots). Accepted full chunks: **4,672 / 6,400**. The 18 remaining late chunks contain 1,728 roots; additional individually receipted partial roots, if any, must be audited before inclusion. **No claim of complete n=64 panel**.
- At last PAC GPU screenshot, only foreign jobs occupied GPU 0; GPUs 1, 2, and 3 showed approximately 0.622 GiB allocated each and 0% utilization. This is NOT a GPU-admission failure; the supervisor had terminally stopped on its budget reservation.
- `scripts/track_r/xp_a20_xpn_recover_gpu.py` prior to any changes computes:
  ```python
  def used_h():
      return prior+state["charged_new_gpu_h"]+sum(
          (time.monotonic()-v["started"])/3600 for v in active.values())
  def safe_budget(extra=0):
      committed=used_h()+(len(active)+extra)*RESERVATION_PER_NEW_WORKER_H
      return committed<=STAGE_CAP_H and (
          project+committed-prior<=PROJECT_CAP_H)
  # ... STOP on "not safe_budget(1)" even with live jobs, then kill_ours()
  ```
  `RESERVATION_PER_NEW_WORKER_H=0.25`, stage cap 20.0. With four active jobs and next-one reservation, 18.7562+5*0.25=20.0062 GPU-hours. Thus the terminal stop is **consistent with the reservation gate**, with 1.2438 GPU-hours unused. Exact admission-event occupancy was not captured in the submitted snapshot. The 400 GPU-hour project cap was nowhere near binding.
- Recent measured late throughput: 61 -> 72 completed chunks, 1,056 additional roots while charged GPU-hours advanced 15.3691 -> 18.7562 (+3.3871h). At this conditional throughput the 18 remaining late chunks would take approximately `3.3871*18/11=5.54` GPU-hours, *if future performance is comparable*. This is a planning estimate, not a measured forecast. Extra budget above the remaining 1.2438h would be approximately 4.30h, plus a co-tenancy margin.

## Scientific/preregistered interpretation

A17 v5 fit/check/seal and GitHub publication succeeded. Held scoring may legally proceed after registry confirmation, but only on completed validated panels. The 100-image `n=16` panel is already generated; a full 100-image `n=64` panel is not. Do not treat 64 full late-phase image panels as representative of all 100 without a predeclared missingness analysis. No forecasts or seed/configuration may be changed after seeing held data.

## Decision requested from planner — no silent cap increase

The frozen XP-N **20 GPU-hour stage cap** must remain binding until explicit planner/user authorization. Request approval for **a separately accounted cap extension to 26 GPU-hours** (additional six above initial stage limit), with the existing 400 GPU-hour project cap unchanged. This is enough to cover the rough 5.54h late-phase extrapolation from the 18.756h stopping point with additional margin (~7.24h remaining if approved). If declined, retain all 72 completed chunks and report the n=16 complete panel and the visibly truncated n=64 panel with transparent limitations.

Do **not** restart the original A18/A20 controllers, remove or edit receipts, modify historical GPU metadata, or submit workers on available GPUs without a new ledger-reconciled, bounded continuation controller. A future A21/A22 continuation should reuse exact original native execution HEAD and frozen manifests; reconcile every A20 attempt cost and completed/partial root hash; use the correct GT-free v5 seal; and fix the reservation design so no unnecessary early terminal at a valid active-worker budget. It must be tested on PAC, and hard caps must still be enforced.

## Operator status

This report reflects latest status user supplied on 2026-11-01 and analysis of committed A20 source; it does not claim independent read access to PAC or newly executed tests.
