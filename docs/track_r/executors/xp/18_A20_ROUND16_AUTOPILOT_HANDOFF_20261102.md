# Round 16 A20 XP execution handoff — 2026-11-02 PM

**Planner:** a6ea8cc; authoritative: docs/track_r/executors/xp/16_PLANNER_NOTES_20261102_PM.md and prereg A20.

## Recorded outcomes / not yet claimed
- Nov4 PAC logs: 7/7 FAST never-GOOD preview panels; 8/8 XP-D4 never-orbit preview panels; 180/180 FAST FFHQ face-CLIP images, 11,520 decisions; CSV SHA256 3595676cd2f7686e747d2659d25a77a6a4a2e34edaa7261a8b7f1fbf53f542e3. These are process logs, not yet GitHub-published PNG/CSV data.
- A24 XP-I2 detached on source head 81ae297cc444c5573909ecdc1f72b3416be5517b, PID 3899225 when launched. Its completion/status is UNKNOWN here. DO NOT check out/change that worktree or launch another supervisor.
- A20 corrected the *planner's* conflation of the 2-forward prior with S_mid. The sole A20 XP-C3 test is S_mid global2. The earlier A13.4 clarification gate a27ef76 is resolved.

## Ready engineering, not yet PAC-executed
- \`scripts/track_r/xp_a25_nov4_publish_cpu.py\`: independently SHA-audits the Nov4 outputs and A9 source roots; regenerates **15 full 256x256 pixel panels** (reference, min-residual, GT-orbit oracle) with PSNR annotations, no IDs drawn; writes PNG-to-id maps/SHAs and XP-D4 SHA-selected id rule. Renames original 11,520 face-CLIP rows to the planner's exact \`f_choice,f_margin,f_chosen_good\` fields, creates theory_inputs copy; computes full exact K-grid E1/U-statistic curves and registered v1 E1 coverage (all 120 FAST E1 prediction rows). GT only affects figure/oracle/quality evaluation.
- \`scripts/track_r/xp_a25_xpc3_source_freeze_cpu.py\`: builds SHA-checked 13,600 source rows for XP-N (6400), G5 (800), XP-D4 (6400), from immutable native root receipts and previously scored label strings. No candidate model tensor decoded and no new prior experiment performed. Includes all source file hashes, D1 ImageNet model SHA a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1 and exact UIDs.
- \`scripts/track_r/xp_a25_xpc3_mid_gpu.py\`: sole S_mid global2 with t=300,400,500,600 and j=0..3. Reuses R0 prior_seed (uid,t,j), frozen_alpha_schedule and rotated paired Gaussian noise; 16 batched-2 calls = 32 model examples/candidate. Converts native Tweedie output to epsilon prediction using the exact frozen schedule. Discrete t-conditioning in the native VP preconditioner is explicitly mapped to the frozen integer timestep, avoiding continuous-sigma indexing drift. Margin L000-L111, choice 111 iff margin>1e-7; no GT.
- \`scripts/track_r/xp_a25_xpc3_freeze_and_launch.sh\`: writes SHA code freeze with source SHA and original prior implementation hash, **commits and pushes** both actual manifest and code freeze before any GPU scoring; otherwise no GPU launch. Then detached controller waits on the shared XP-I2 project accounting lock.
- \`scripts/track_r/xp_a25_xpc3_autopilot.py\`: once XP-I2 terminal has been accepted, single admissible dirty GPU, 80% VRAM cap, no Track R co-tenant, 8 GPU-h stage / 400 project cap, one completed XP-N image smoke first, cost projection and complete panels in XP-N -> G5 -> XP-D4 order. Every process wall-second (including failures) charged. Terminal output includes source/code SHA, per-image SHA receipts and project ledger.
- \`scripts/track_r/xp_a25_xpc3_offline_final_cpu.py\`: only after 100/100 images of one panel, computes PSNR20 chosen-GOOD using ImageNet GT, but does not alter GT-free choice; exactly \`image_id,run,label,prior_choice,prior_margin,prior_chosen_good\`.
- \`scripts/track_r/xp_a25_xpc3_publish_cpu.py\`: only after GPU terminal and successful offline panel receipts, commits and pushes complete panels' CSVs/receipt to returns and theory_inputs (no partial panel, no matched-compute claim). Any GitHub conflict is fail closed.
- \`scripts/track_r/xp_a25_round16_autopilot_launch.sh\`: one PAC invocation from a NEW detached worktree; CPU preflight/tests and publication commit/push first; then full source SHA prep, freeze commit/push and detached GPU controller. It MUST NOT be run from the A24 (XP-I2) or original A25 worktree.

## Remaining
- **No PAC native smoke or scoring has yet been performed by this code**. All candidate scoring remains prospective until after actual manifest+code SHA registration. Run static/behavioral tests in PAC and observe real logs; do not invent status.
- Validate Nov4 published outputs/forecast comparison with planner A20.2 once pushed.
- XP-I2 offline GT-isolated (a)/(b) and Table3 E1 for D2/D3 due Nov6, XP-K, G7 remaining work and CP4 Nov15.
- The project's original A13.4 stop/ask exceptions (protocol/gov conflict, irreconcilable SHA, budget, permissions) remain binding. No new method, protocol v2 or matched-compute claim.
