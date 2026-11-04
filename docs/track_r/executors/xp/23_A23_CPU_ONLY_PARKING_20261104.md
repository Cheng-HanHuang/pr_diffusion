# Track R XP — A23 CPU-only autopilot / GPU queue parked (2026-11-04 01:04 ET)

**Governing:** planner branch `claude/prdiff-track-r-planner-20261021` @ `6f5ee7f`, `docs/track_r/executors/xp/19_PLANNER_NOTES_20261104.md`, prereg A23.5 in `docs/planning/track_r/XP_PREREG_v1_1.md`. **Human resource constraint:** Cheng reported **no idle GPUs** and requested GPU tasks **skipped/saved for later**; until an explicit resource-unpause instruction, do not start GPU jobs, smokes, identity replays, native GPU preflights, or background GPU polling/launching. Do not evict labmates. This pause supersedes A23's general work-conserving admission permission. Continue CPU-only GitHub engineering, tests, source audits, and report preparation autonomously. A scheduled automation is configured for this CPU-only queue.

## Accepted, complete (no retries, no duplicate GPU)

| Stage | Return commit | Science status | Audit |
|---|---|---|---|
| XP-C3 | `91fe7a39`, `a6c6a64`, `591a28fb` | 13,600 prior decisions, accepted earlier | Source frozen; original controller teardown incident preserved; A21.10 optional replicate NOT run |
| XP-I2 | `1eae9bf` | Accepted earlier | ImageNet PC-DAPS vs native |
| Table 3 E1 G1 ImageNet | `df6e8cd5c62f506343f2ff8fe6d493c4d032ea68` | A23 **accepted** | 3 aggregate rows + 300 per-image rows, 100/100 for DAPS K4/K8; SITCOM internally consistent; no matched compute |
| G7 C2 + CLIP GT-only | `9090a50a4d4ffe922bdb43c5ee9e877c059a7dd1` | A23 **accepted**, supplement S2 | C2 280/280 images, 10,080 equivariance terms, 36 summaries; CLIP 280 GT-only images; **not entire G7** |
| Frozen N2 face prompts | `a42e14e`, historical source `b1637f7c` | A23 **accepted**, quoted in S2 | Exactly 3 upright and 3 inverted strings |

**Main-paper data cut met**, per A23. Further registered results are **supplement only**, except correction of error. Do not propose new selector, protocol v2, or matched-compute claim.

## GPU budget ledger and strict "not run" inventory

- Published A22 G7 C2 stage: `2.6265298946173345` GPU-occupancy h; published cumulative project ledger `244.45709476042322` GPU-h; no GPU allocated now.
- Project hard cap `400` GPU-h; arithmetic unspent project headroom `155.54290523957678` GPU-h **before any newer charges or reservations**. Before unpause or launch, reconcile live ledger and any other controllers (these figures are a frozen published checkpoint, not a fresh PAC accounting query).
- A23 total G7 cap `10` GPU-h, of which `2.6265298946173345` is already used: **remaining at most `7.3734701053826655` GPU-occupancy h** across all new G7 diagnostics. Do not assume full 10 additional GPU-h.
- XP-K cap `30` GPU-h; global project cap 400, tail cap 60 if still binding. No XP-K candidate has been generated.
- Optional A21.10 cross-GPU replicate remains **NOT RUN; user excluded**. Historical A21 Git publication timeouts were recovered, not new scientific time.
- CP4 due **2026-11-15 12:00 ET**. If the GPU pause still applies by then, publish a truthful CP4 with `NOT RUN — GPU HOLD` entries, not fabricated data.

## Autonomous CPU-only engineering queue (priorities)

### 1. G7 remaining 280-image GT-only orientation diagnostics, *supplement only*

Use exactly the **same** 280 GT images frozen by C2/CLIP (180 FFHQ incl B22/DEV80, 100 ImageNet), source SHA and GT preprocessing unchanged; use GT only to assess whether the selector keeps upright input. No new inverse-problem reconstruction. Run order after the user explicitly unpauses GPU:

1. **`S_mid global2`**, byte-for-byte/mathematically equivalent to prereg XP-C3 A20.3. Four timesteps `300,400,500,600` × four frozen draws `j=0,1,2,3` × two global orientations = 32 example evaluations/image; frozen threshold `margin > 10^{-7}` chooses `111`, otherwise `000`.
2. **`S_mid all8`**, same frozen 8-orbit test and allowed estimator; no leakage from GT scoring.
3. **`2-forward`**, identical to the frozen XP-C prereg row (not a redesigned proxy).
4. **`UnivFD`**, pinned original vision model/checkpoint and GT-only orientation test.

Follow A21 single-t batching and B8/B32 numerical equivalence logic; use multiple workers only when measured physical GPU memory/cost admission permits and no foreign GPU jobs are displaced. The rest of G7 plus published C2 **must stay under total 10 GPU-h**. While parked, do source-to-source derivation, per-image ID/source SHA contract, CPU-only mock/pure tests, exact input receipt and cost projection; mark every unrun arm `GPU_ON_HOLD`.

### 2. XP-K one-parameter calibration, *supplement only*

A5.4 and A23 demand **three explicit gates before any new generation**:

- **Gate (i), committed one-parameter diff:** exact historical per-family solver configuration and code-to-parameter binding audited, with **one and only one** native parameter changed: SITCOM Langevin `tau=0.05` (do not change Euler measurement step), FAST-DIPS projection `epsilon=33.26`, historical FFHQ DAPS `tau=√2×0.05≈0.07071067811865475`. Do NOT reuse ImageNet DAPS-4K tau or modify two settings. Build a machine-checkable one-leaf config diff **and** native source binding proof; pure JSON diff alone is not complete approval.
- **Gate (ii), identity replay:** adapter at **original** historical parameter must reproduce 2 historical **B22** roots per family (bitwise or A4 tolerance). It needs GPU and is **PENDING GPU UNPAUSE**, never claim PASS from code/tests alone. Replay exact measurement SHA, seed, checkpoint, image and precision flags.
- **Gate (iii), DAPS ordering:** DAPS arm cannot start until original F8/A5.2 replay gate has documented PASS. If a gate fails, stop/ask under A13.4 and do not substitute a setting.

Future generation: original 100 B22 FFHQ image IDs ×4 roots ×3 families, dataset seed namespace `ffhq_cal`; evaluate GT-only offline after generation, class rates, GOOD25, ρ/σ med/IQR, ρ<ρ★, E1_CLIP and orbit oracle; compare registered mech v1/v1.1 forecasts. Stage budget 30 GPU-h; project 400 GPU-h. While parked, write exact adapters and frozen source/seed manifests, unit tests, replay plans, no native generation.

### 3. CP4 report shell, *supplement only*

Prepare a deterministic CPU-only report with tabs for XP-K vs mech v1/v1.1, G7 GT orientation rates, full project GPU ledger with physical occupancy union and prior charges, and explicit **NOT RUN** inventory. Report `PENDING_GPU_RESOURCE_HOLD` for future experiment slots until validated receipts arrive. Do not import data from locked holdouts; do not fill theoretical predictions post hoc.

## Checkpoint/communication

- Accepted A23 main-paper results are **already in the planner's hands**; avoid spamming Issue #46 with duplicate A22 returns.
- Post this parked queue/prospective code-only engineering summary once. Return to planner at meaningful CPU gate ready, identity replay PASS/FAIL, new genuine G7 result, or CP4 checkpoint; otherwise run independently and silently.
- No PAC GPU launch command should be given while human GPU hold applies. When user says GPUs can be used again, first re-check A23/latest planner branch and PAC GPU occupancy/ledger; freeze source and source-code SHA **before** launching.
