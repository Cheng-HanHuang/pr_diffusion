# A23 co-tenant-safe work policy — Nov 4, 2026

**Governing science:** round 19 A23, planner `6f5ee7f`, `19_PLANNER_NOTES_20261104.md` and prereg A23.5. **Human scheduling clarification:** The earlier "no idle GPUs" instruction does **not** mean all remaining experiments need clean or exclusive GPUs. Cheng now requests that tasks which can execute safely under GPU co-tenancy proceed, leaving only capacity-/isolation-dependent tasks parked. This note supersedes the **GPU hold policy only** in `23_A23_CPU_ONLY_PARKING_20261104.md`, but does **not** supersede any science/ledger gates or authorize eviction of other users.

## Three execution classes

| Class | Stages | Admission / status |
|---|---|---|
| **CPU-only: proceed immediately** | G7 exact model/input-source audit and frozen manifests; UnivFD source/checkpoint and CPU execution if feasible; XP-K frozen one-parameter native code/config diff and replay harness; CP4 aggregation shell, not-run inventory and ledger readout; tests/documentation/publish | No CUDA context; must never mark GPU replay PASS |
| **Co-tenant eligible: proceed when RAM/VRAM and source gates pass** | G7 S_mid global2 (first), then S_mid all8, then 2-forward; UnivFD model scoring if GPU unavoidable; XP-K two-root identity-replay gates with original parameters; after successful gates, calibrated generation one family at a time | A clean GPU is **not** required by A23. Run on an eligible GPU only after a fresh PAC GPU inventory, per-process model-load VRAM reservation, source freeze, live-project ledger reconciliation and projected cost. Start W1, ramp only on demonstrated safe memory and throughput/cost. Never kill or throttle foreign processes. |
| **Prefer exclusive/quiet GPU; defer if unavailable** | Sensitive exact-bitwise identity replay if contention obstructs deterministic comparison; timing or throughput characterization needing interpretable wall-clock numbers; heavy XP-K stage where co-tenancy makes the 30 GPU-h ceiling infeasible; any run with insufficient free VRAM | "Prefer" does not mean an intrinsic prereg clean-GPU requirement. If replay at original parameter fails bitwise/A4 tolerance, **stop and ask**; never weaken tolerance, change algorithm, substitute settings or simply blame contention. |

There is **no generally required clean-GPU stage in A23**. Two historical-root identity replays require source-equivalent deterministic execution, not necessarily exclusive GPU ownership. Their result is `PENDING` until a real GPU execution passes; do not claim PASS from static inspections. The user is not authorizing an unbounded run on already saturated cards: a physically busy GPU with safe memory and noninterference may still be eligible; if none has adequate capacity, GPU parts remain queued.

## Frozen science gates

- **G7:** exact same 280 published ground-truth images and GT-only scoring; output is upright-retention fraction. Frozen order **S_mid global2 (exact XP-C3 A20.3) → S_mid all8 → 2-forward → UnivFD**; A21 single-t batching/equivalence. G7 total cap `10.0` physical GPU-occupancy h including prior `2.6265298946173345`; **maximum additional G7 budget `7.3734701053826655` h** before fresh ledger changes. Estimate cost after W1 pilot before distributing workers; no 10-hour reset. Pilot may run co-tenant after source and RAM/VRAM gates, never before.
- **XP-K:** 100 B22 FFHQ images ×4 roots each family; exactly one changed solver parameter versus original B22 native source: SITCOM Langevin tau 0.05 with Euler measurement step untouched; FAST-DIPS projection epsilon 33.26; historical DAPS FFHQ tau sqrt(2)*0.05, with DAPS F8 replay first. **Prerequisite gates:** (i) committed *native-source* one-parameter diff; (ii) original-parameter identity replay of two historical B22 roots per family bitwise or A4 tolerance; (iii) DAPS F8 gate. Gate failure → STOP and ask under A13.4, no substitute parameter or setting. XP-K cap `30` physical GPU-h; consider GPU co-tenancy only when the budget projection fits.
- **Global project ceiling:** `400` physical GPU-occupancy h; last published project ledger `244.45709476042322`. Validate current PAC ledger before *any* prospective GPU allocation. No clean/timing/matched-compute claim on co-tenant executions. No new selector/protocol v2. Main-paper cut met; remaining data supplement only.
- **Optional A21.10:** remains excluded.

## Autopilot architecture and boundary

1. Continue source-code/CPU engineering via GitHub, with tests and frozen manifests now; the existing scheduled automation cannot directly use PAC shell or GPUs.
2. Prospectively build one bounded resource-admitting PAC controller. When user invokes it, it should check actual GPU inventory (including in-flight model loads and foreign jobs), apply W1 pilot, estimate physical GPU-occupancy costs, and then schedule the next scientifically authorized G7 or XP-K arm, **skipping cards/arms that cannot fit**. Do not run any GPU code by guessing spare capacity.
3. Keep per-image atomic receipts, automatic CPU-only readout and independent publication; never repeat an already completed image or re-create commits on a push timeout. At prereg gate failure, stop the affected science arm and report it; continue independent CPU work. Preserve all source/ledger hashes.
4. If no GPU is safely admissible, leave the GPU queue parked and complete CPU-only deliverables. Checkpoint to planner only with a genuine result, approved freeze, or blocking scientific issue. CP4 due November 15 noon ET reports any parts not run accurately.

**This is a scheduling policy and engineering authorization, not evidence that a GPU was idle, that a native worker was run, or that an XP-K/G7 pending gate passed.**
