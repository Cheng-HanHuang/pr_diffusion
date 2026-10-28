# XP-A11 / A9.6 B16 recovery milestone — 2026-10-28, 14:14 EDT

**Administrative and execution-status return only.** No B16 run-index ≥4 outcome, GT-based metric, class count, reconstruction, or per-image performance is in this document. The A9.4 B16/F8 outcome embargo remains in force.

## Verified milestone snapshot (from PAC terminal; not a live cluster query)

- B22 SITCOM historical 400/400 root replay: `XP_A10_B16_GATE_PASS`, zero failures; accepted historical gate is CPU validated and its replay input SHA is `9249b12d5e694e3ba06e19570eaffa35128f360663ed0b114b60851d61dcd423`.
- Production input frozen at `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_a11_b16_prod_20261028T113034Z`, manifest SHA `778ce4fa7d5ef9299e2530bbb32e8a28b3512e5abe97f0569bb3ec74e00247a9`, original GPU-worker HEAD `079228b3ebe55dba2ac7bc7ae2c8187113290409`.
- Complete native sampling and reconstruction receipts for four B22 image groups (indices 58–61; 48 new roots) initially rejected solely by an incorrectly global ±50% smoke timing gate. Verified per-root SHA, measurement identity, GT-free flags, and required native steps. **No reconstruction/label mismatch observed** for those four groups; no new-run labels were evaluated.
- Corrected recovery controller HEAD `44434501295aa97a65417c6298cb3c3e92c44294`. Five synthetic tests PASS, controller SHA and runner SHA checked independently; original worker code and manifest remain pinned.
- PAC recovery receipt: 66/1,060 B16 tasks and **748/2,160** new roots accepted; B32 **0/2,880**. Both B16 smokes PASS; four GPU devices observed in use shortly after restart. Controller session `prdiff_b16_recover2_20261028T181357Z`; original 12-hour deadline `2026-10-28T22:57:37Z` (18:57:37 EDT).
- At earlier diagnostic 14:00 EDT, stored accounting: 3.187 GPU-h replay + 7.723 GPU-h production = **10.910/48 GPU-h**, with 37.090 GPU-h unspent. Snapshot at ~14:14 EDT: ~4.72 wall-hours remained. Counts/hours **will change** after this report.

## Runtime integrity and A7.1 policy

- Native FFHQ SITCOM sampling increased from ~22 s to ~77.5 s in later B22 groups across multiple GPUs. This is **not** a demonstrated co-tenant-induced slowdown. The old controller captured contamination booleans only in successful completion events; its four misclassified groups have no preserved completed-job contamination classification. These historical timing measurements must be reported **UNVERIFIED**, not CLEAN or necessarily DIRTY.
- The preregistered paper compute frontier (§6) uses clean **XP-T batch-1** times exclusively; large-run throughput/timestamps are operational/accounting data and must not replace clean scientific timing evidence.
- A7.1 permits co-tenancy for generation given VRAM and one Track R worker/GPU, but **also caps Track R at two GPUs when foreign jobs are present**. The current controller can use four concurrently and does **not enforce this global A7.1 cap**. Whether actual temporal co-tenancy occurred remains unverified. Flag this as a **potential preregistration deviation requiring planner decision/amendment**, not as accepted A7.1 compliance. No foreign process was signaled or stopped.
- Do not alter completed receipts, GT/outcomes, seed lineages, or frozen timing baseline. Recommend future read-only per-GPU occupancy/PID timestamp logging for contamination auditing. Do not infer dirty status from CUDA-event duration alone.

## Science and execution priorities

- Continue **B16 critical** with four eligible GPUs (actual occupancy may vary); worker must maintain the immutable B22 seed-43 post-model sequence and DEV80 original independent seeds. Controller currently prioritizes B16 and only admits B32 after 2,160/2,160 valid B16 roots and completion receipt.
- B16 remaining as of this snapshot: 1,412 prospective roots, including 38 B22 image groups and 956 DEV80 independent roots, so there are ~1,564 native samplings including the 152 required B22 prefix recomputations. At 77 s/sampling, approximately 33.5 aggregate GPU-h are needed (8.4 h on four GPUs) before overhead; at 22 s, ~9.6 GPU-h (2.4 h). **The 18:57 EDT deadline may stop B16 incomplete.**
- **F8 historical DAPS 400×5 eight-root replay and subsequent 920 F8 roots remain NOT LAUNCHED**, a critical-path gap for the Nov 6 Tier B data cut. No F8 production should bypass its own replay and seed gate.
- The already published `XP_A9_EXCHANGEABILITY_20261028T100727Z/A9_FAST_EXCHANGEABILITY.json` marks A9.2(a) post-hoc exchangeability diagnostics complete; do not repeat or refit v1 based on its result. Check further A9.2(b–d), D1-P2 CLIP, XP-E′/A8.6, and the DAPS/F8 gate with the planner.
- Relevant release clocks: D1-P2 outcomes after the v2 E1 addendum or **Oct 29, 12:01 EDT**; B16/F8 outcomes after v3 registered/declined or **Oct 31, 12:00 EDT**. Data cuts: **Nov 2** Tier A, **Nov 6** Tier B, **Nov 8** paper go/no-go, **Nov 16** CVPR main deadline, **Nov 23** supplement deadline.

## Requested planner decisions

1. Acknowledge 400/400 B16 lineage gate PASS and the clean recovery of 48 previously timing-rejected yet fully SHA-verified roots. Keep B16/F8 outcome embargo intact.
2. Decide whether to ratify **up to four** concurrent Track R GPU workers under **measured** co-tenancy or enforce the original A7.1 maximum of two whenever foreign processes are present. The user desires four-GPU work conservation; the existing A7.1 cap is explicit.
3. Prioritize an F8 historical replay/generation lane and the outstanding CPU work. If B16 misses the current 18:57 EDT deadline, authorize a separately accounted continuation (no silent clock extension) to reach Nov 6.
