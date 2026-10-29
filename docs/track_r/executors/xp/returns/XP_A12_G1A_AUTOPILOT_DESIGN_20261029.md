# A12 G1a FAST ImageNet autonomous campaign: launch contract (Oct 29)

**Governing:** planner A12, `08_PLANNER_NOTES_20261029_PM.md` at `65ba723`. The 400 GPU-hour project cap is unchanged. The former A10 70-GPU-hour limit is superseded by A12's 150-GPU-hour campaign cap. **G1a stage limit: 25 occupied GPU-hours**, deadline November 3, 12:00 ET. D4 run indices >=16 remain sealed until theory v4 registered/declined or Nov 2 noon ET.

## Frozen CPU preparation (completed on PAC)

- `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_a12_g1a_prepare_20261029T170316Z`
- Two historical replay roots (distinct ImageNet IDs, rep 4), manifest SHA256 `2ad7a8de865814d3ea21e51cc2b8b60dc5054324768048478006f71c434fde43`.
- New FAST ImageNet roots 16–31 (100 × 16 = 1600), manifest SHA256 `acb38367dcda4e1bd657e03b5f365be65cad725413b4965c6e42ab86d241cfd0`.
- Same frozen ImageNet measurement bytes, original `XP|FAST|imagenet|<id>|<run>` SHA seed domain, ImageNet DDPM, original FAST-DIPS source/config. No GT or D4 outcomes opened.

## One-command autopilot

`scripts/track_r/xp_a12_g1a_launch.sh launch HEAD PREP_DIR VERIFIED_LEDGER_JSON`.

After CPU tests, the launcher starts an isolated controller in tmux. The controller checks the verified cumulative **project** ledger from XP-0, inputs and historical reference receipts, replays both roots using untouched FAST logic, requires **pixel equality** to original D1 roots, and then schedules 100 batches of 16 new runs. Four workers maximum, one per GPU. Co-tenancy is allowed under the 80% VRAM hard ceiling; only controller-owned PIDs may be signalled. Each successful, failed or interrupted attempt is charged by wall time for its occupied GPU. All D4 labels, GT and PSNR remain sealed. A failure does NOT automatically relax the replay gate or admission caps.

- Stop file: `RUN/STOP_G1A` — prevents new admissions and drains active workers.
- Logs: `RUN/logs/`, controller's `XP_A12_G1A_EVENTS.jsonl`.
- Durable state: `RUN/XP_A12_G1A_CONTROLLER_STATE.json`.
- Per-attempt charges: `RUN/cost/XP_A12_G1A_COST_*.json`.
- First A12 ledger receipt: `RUN/XP_A12_FIRST_LEDGER_RECEIPT.json`.
- Replay gate: `RUN/XP_A12_G1A_REPLAY_GATE.json`.
- Final: `RUN/XP_A12_G1A_FINAL_STATUS.json`.

**Current blocker:** a source-backed **whole-project** historical cost ledger, required for the first A12 receipt. The inventory script `xp_a12_project_ledger_inventory_cpu.py` discovers candidate historical cost receipts without double counting; its output is *not* itself a verified ledger. The launch command refuses GPU admission until reconciled. Do not fill project totals by using planner's approximate 123 GPU-hours.

**Caution:** G1a autopilot implementation committed, but its end-to-end launch is untested on PAC until the tests, inventory and ledger gate are completed.

**Independent work:** D1-P2 first held-out protocol v1 succeeded: H1 PASS, H3 FAIL, H4 PASS; result JSON primary SHA256 `8dc1f5bad10e041b253addfee2a74868bc280c30822c4d289b427ba8480e11d3`. Its strings and per-candidate G1/G2 margins should be published to `xp/returns` and `theory_inputs` immediately for theory-v4 fitting.

## Oct 29 PM: A10 failed B16 gate 104836 conservative repair

Original B16 gate `xp_a10_b16_gate_20261028T104836Z` persisted a RUNNING state with one active PID (1057433) even though the lock is released, that PID no longer exists, and its native worker log subsequently records all four roots PASS. Recorded finished-worker time is 100.83540434669703 sec; exact process-exit time for PID 1057433 is unavailable. Original SHA256 state `2d8cf00361688852aae2216115adc9491203970dffbac8a2565ba2facc975527`; events `1da54503b4b94f65103e98e943cd02ee796015033368fdd2227ae1d2567a7ad0`.

The CPU-only `xp_a12_legacy_gate_upper_charge_cpu.py` creates a **non-exact, conservative budget upper-bound**: it bills the recorded 100.835 seconds **plus the entire time from PID 1057433's recorded launch to a new PID-absent audit**. Exact actual use is known only to lie between the log's final PASS mtime and that audit. Its charge may exceed actual use by many hours. This is intentionally safe for the 400-GPU-hour project cap, **not suitable for reporting exact GPU utilization or runtime in the paper**. The reconciler requires this signed-off source receipt and declares the project total to be a `CONSERVATIVE_PROJECT_CAP_BILLING_UPPER_BOUND_NOT_EXACT_MEASURED_TOTAL`. Never change or rewrite the original B16 state.

Before launching G1a run the full CPU suite and then the correction and `xp_a12_project_ledger_reconcile_cpu.py`. Any missing source or cross-check stops before GPU admission.
