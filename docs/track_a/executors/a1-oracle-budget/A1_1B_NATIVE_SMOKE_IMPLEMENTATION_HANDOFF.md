# A1.1b implementation and gated PAC deployment handoff

**Date:** 2026-10-19. **Owner:** continuing Track A executor. **Status:** `CPU_PLAN_READY_GPU_UNRESERVED`.

## Verified predecessor

A1.1a **correct** return ZIP verified SHA-256 `a2f61731638602f38563502c0c86e3d0896fa9a95f8fbf6fbb7937c507bc290f`, 3/3 CPU unit tests passed. Native procedural observation SHA `992a33efa9bfb8e81d97ed61cf28369263ae1f11aeae19273c53973c2470acdb`, signed float32 shape (1,3,384,384), fixed native DAPS operator with sigma=0.05, oversample=2; synthetic RGB model-range GT content SHA `6eda5926b1cc56e23cdfed5fd70b743bdfe61067c955f778900683a42a6a1123`. No FFHQ or B24 confirmation305 data. Details in `A1_1A_VERIFIED_RETURN.md`.

## A1.1b objective

**Exactly 4 full native candidates on the SAME frozen procedural observation:** DAPS1, SITCOM1, NP-LF1 and NP-S2(preproj lambda 0.01)1. No model/algorithm tuning, no K4/8 extension, no implicit reruns. No paper reliability claim possible on a single procedural image.

### A1-owned implementation (never invokes B24 cohorts/runs)

- `scripts/track_a/a1_1b_native_one_each.py`: coordinator/one-each subprocess launcher and common offline evaluator.
- `scripts/track_a/test_a1_1b_native_one_each.py`: synthetic seed, NP config, input identity, nonexec planning tests.
- `scripts/track_a/a1_1b_run_cpu_preflight.sh`: **NEXT USER STEP**, CPU source/probe/checkpoint SHA preflight and synthetic tests; never imports native models, launches solvers, or reserves GPU.
- `scripts/track_a/a1_1b_run_gpu_smoke.sh`: **NOT YET AUTHORIZED**, sequential GPU native one-each stage, requires explicit `A1_RUN_RESERVATION_ACK` and selected physical GPU3; no automatic run after CPU preflight.

The A1 controller calls functions from frozen B24.1 **source** `scripts/b24/run_b24_1_method_smoke.py` for compatibility manifests and per-process child invocation only. It does **not** execute a B24 cohort, access historical B24 output or modify a historical B24 working checkout.

DAPS runs source HEAD `e7a77d094167084faed19b599b96673b7bb11447`, original B21 start-seed override, native anneal400/diff5, fresh process. DAPS requires temporary custom `+data` Hydra registration in **external native DAPS** checkout, with a unique A1 prefix `a1-1b-procedural` and a symlink to the A1 procedural PNG; only these two newly created scratch paths are unlinked and `rmdir`'d in `finally`. Hydra run output explicitly redirected to A1 output. No modifications to tracked DAPS source or B19–B26 historical outputs.

SITCOM uses frozen B22.1 one-trajectory runner and B24.1 single native process helper, anneal200/diff5/lgvd100, independently seeded `seed32_numpy("SITCOM","11111",0)`. That specific native entrypoint does load GT for **offline** metric calculation and `sampler.get_start(gt)`, but A1.0b source audit showed native `get_start(ref)` reads only shape/device/dtype via `torch.randn_like(ref)`, **not numerical GT**. This is an integration smoke using a procedural input, not an independently demonstrated stronger GT-firewall adapter. GT is never used for candidate allocation or selection.

NP uses exact B22.1 native single-candidate runner with frozen configuration `configs/b22/b22_1_smoke.json`, independently allocated NP LF and S2 seeds and one new process each. S2 overrides only `score_mode=prev_l2`, `score_reg_lambda=0.01`, `score_reg_lambda_schedule=pre_projection_only` as the frozen B22.2 config, 1000 steps and soft5/hard1 settings unchanged. NP explicitly clamps negative measurement entries **in memory** per historical method convention. All methods verify raw measurement content SHA before execution where native wrappers support it; parent hashes raw .pt input before and after all execution.

Primary reported offline metric, for **pipeline diagnostics only**: raw orientation RGB PSNR of **the canonical saved 8-bit PNG of every method** against quantized 8-bit procedural GT; likewise report rot180 ambiguity-aware PSNR. No inappropriate DAPS PNG vs NP/SITCOM unquantized float comparison. Report source/input/timing/peak reserved memory/terminal SHA for each.

**Resource ceiling:** one process at a time, user-selected physical GPU3 pinned UUID `GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a`, initial used VRAM <=1 GiB, minimum free 52,096 MiB, hard observed device-used 52,452 MiB (approx 51.2 GiB); per-candidate wall max 1500 s, aggregate wall max 3600 s. The controller terminates **its own process group** on failure or limit; does not kill other users' work. The GPU command is **not run or approved by this document**.

## CPU PAC instructions (run only after verifying latest commit SHA)

```bash
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
# Pin to the specific source HEAD supplied in the executor message.
A1_EXPECTED_HEAD="<PINNED_NEW_COMMIT_SHA>" \
  bash "$REPO/scripts/track_a/a1_1b_run_cpu_preflight.sh"
```

Returns `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/A1_1B_CPU_PLAN_RETURN_V1.zip` and its SHA. No GPU, no GT image bytes in ZIP. On STOP preserve logs and report, do not rerun over existing output.

## Strict sequencing

1. User runs A1.1b CPU preflight and returns terminal/ZIP; executor independently verifies PASS and audits any missing/brittle source assumptions.
2. Coordinator confirms a discrete `[RUN RESERVATION]` in issue #41, with physical device UUID, exact time window, max VRAM/time/compute, and Track B GPU priority respected. Merely observing an idle GPU does not confirm a reservation.
3. User runs the explicit A1 GPU wrapper **once**, records logs, returns `A1_1B_GPU_NATIVE_RETURN_V1.zip`. No automatic retry. If one method fails, diagnose it first, preserving prior outputs.
4. Only if all four native smoke methods, common measurement and saved-8-bit metric checks pass, propose longer independent K4 and compare GPU cost (with separate reservation). Source-disjoint FFHQ DEV eligibility must still be resolved; the procedural source cannot support face-domain reliability claims, and protected B24 confirmation305 membership/data must not be queried.
