# Planner notes to the XP executor, round 7 (2026-10-29, 04:00 ET)

**Responds to:**
- the D1 phase-2 return (`70b290f`);
- the DAPS replay PASS and F8 authorization (`b9987b5`);
- the B16 analysis branch (`9c8a353`).

**Governing amendment:** XP prereg **A11**. Merge the planner branch at its current head into your analysis branches; never into a running production checkout.

## 1. What changed

- **The paper's direction is decided** (Cheng, option (a)). The deliverable is a **ground-truth-free best-of-K protocol**, plus the finding that GT-free selection is predictable while unsolved images are not.
- **Protocol v1 is frozen** at planner `c598176`:

  | File | SHA-256 |
  |---|---|
  | `docs/track_r/protocol/GTFREE_PROTOCOL_v1.md` | `7e9a5fd8b817a97351d07ae371b947688c8430cd5b9268fc99b954a493223919` |
  | `docs/track_r/protocol/gtfree_protocol.py` | `d5f973fe4ef82243d6b687981dd4c193822e30634254853adfe8d483b3920658` |

  Verify both with `git show c598176:<path> | sha256sum` before any run.
- **Registry.** `PREDICTIONS_REGISTRY.md` has a new table, "Frozen protocols", with key `protocol v1`. If `verify_freeze` parses every table row, make it accept or skip that key; do not let it fail.
- **The protocol needs no GPU work.** Everything below is CPU on existing or in-production runs.
- **Accepted from your last returns:**
  - the D1-P2 comparison;
  - the 8/8 pixel-exact DAPS replay (good work on the corrected DEV80 seed);
  - the B16 v1-primary / v3-secondary separation.

## 2. Order of work

1. **B16 against v1 (primary) and v3 (secondary),** with the A9.2 decomposition, as already planned. Commit it to `xp/returns/` first.
2. **D1 phase 2 under the protocol.** All the data exist.
   1. **Strings.** Export `XP_D1P2_16RUN_STRINGS.csv` in the A9 schema:
      - columns `image_id, cohort, labels_by_run, rho_signed_by_run, tau_signed_good_q95, n_G, n_O_g, n_O_p, n_C, n_I`;
      - `labels_by_run` is `;`-joined in run order 0–15, with labels G/O_g/O_p/C/I under the frozen ImageNet labels;
      - `cohort` = `IN`.
   2. **CLIP decisions.** Export `XP_D1P2_16RUN_CLIP_DECISIONS.csv` in the phase-1 schema (`image_id, run, label, g1_choice, g1_margin, g1_chosen_good, g2_choice, g2_margin, g2_chosen_good`), all 1,600 candidates. `g1_margin` is u(R₁₁₁x) − u(x), the convention of the phase-1 file.
   3. **Run, from `c598176`, unchanged** (`P="python3 -I docs/track_r/protocol/gtfree_protocol.py"`):
      ```
      $P --strings S --clip C --tau 0.0500829 --m 2 --nmax 16 --R 4000 --seed 20261029 --out D1P2_m2_n16.json             # PRIMARY (H1, H3, H4)
      $P --strings S --clip C --tau 0.0500829 --m 2 --nmax 12 --R 4000 --seed 20261029 --run-range 4:16 --out D1P2_fresh_m2_n12.json
      $P --strings S --clip C --tau 0.0500829 --m 2 --nmax 16 --R 1000 --seed 20261029 --frontier --out D1P2_frontier.json
      $P --strings S --clip C --tau-rule self_mad3 --m 2 --nmax 16 --R 4000 --seed 20261029 --out D1P2_self.json
      $P --strings S --clip C --clip-good-col g2_chosen_good --clip-margin-col g2_margin --tau 0.0500829 --m 2 --nmax 16 --R 4000 --seed 20261029 --out D1P2_g2.json
      $P --strings S --tau 0.0500829 --m 2 --nmax 16 --R 4000 --seed 20261029 --out D1P2_e0.json                            # no orientation step
      ```
3. **B16 under the protocol.**
   1. **Strings** for all 16 runs of the 180 images (`cohort` = `B22` or `DEV80`).
   2. **N2 face-set CLIP decisions** for all 2,880 candidates, with columns `image_id, run, label, f_choice, f_margin, f_chosen_good`. The margin is u(R₁₁₁x) − u(x).
   3. **Run per cohort**, adding `--clip-good-col f_chosen_good --clip-margin-col f_margin`:
      - `--cohort B22 --tau 0.0448503 --nmax 16`;
      - `--cohort DEV80 --tau 0.0449706 --nmax 16`.

      Then compute the pooled values as image-weighted means of the cohort outputs (100 : 80).
   4. **Secondary runs:** the same with `--frontier`, with `self_mad3`, and without `--clip`.
4. **F8 under the protocol,** once F8 is complete and scored against v1/v3. Same as B16:
   - `--cohort B22 --tau 0.0474649 --nmax 8`;
   - `--cohort DEV80 --tau 0.0475986 --nmax 8`;
   - face-CLIP for all 1,440 candidates.
5. **Hypothesis table.** One table with H1–H4 per spec §4, held or failed, with the numbers. **Commit the protocol outputs to `xp/returns/` before any other analysis of them.** I reproduce each run; a difference above 0.002 is a stop.
6. **Secondary panels (Tier C):**
   - **E′:** DAPS-4K FFHQ, 4 runs, `self_mad3`, face-CLIP.
   - **D2/D3:** ImageNet, 4 runs. `self_mad3` for both; for SITCOM also `--tau 0.0448503` (the FFHQ τ transferred). G1 decisions for all 800 candidates.
   - **FAST FFHQ 64:** face-CLIP decisions for all 11,520 candidates (A9.2(d); E1 against v1), then the protocol with the orientation step.
7. **Unsolved-images figure (new, for the paper).** One PNG for the 7 FAST FFHQ images that never produced a GOOD run in 64 (A9 case list):
   - 7 columns × 3 rows: the ground truth, the lowest-residual run, and the best run under the orbit oracle (rotated if that member wins);
   - PSNR under each image, no ids, 256 px tiles, at most 2 MB;
   - commit it to `xp/returns/` with a receipt.

   Use only the R0 manifests' 180 ids. Do not touch any other FFHQ data.
8. **Still open, after the above:**
   - E′ scoring and the A8.6 checks (GT/y identity, VGG LPIPS, checkpoint SHA, the per-metric best of 4 for cohort A);
   - A5.5;
   - an independent re-read of the full D1-P2 comparison JSON on PAC, with a persisted digest for the paper tables.

## 3. Rules

- **No GT in any deployable step.** The protocol script uses labels only for scoring.
- **Release.** Strings and protocol outputs may be copied to `theory_inputs/` once committed (A11.4). No other change to the release rules.
- **GPU.** F8 continues under A10 (cap 70 GPU-h, window to Nov 1, 12:00 ET). No new GPU stage is authorized by this note.
- **Every return:** a digest of at most 20 lines.
- **Return when:**
  - the B16 v1/v3 comparison is committed;
  - the D1-P2 protocol outputs are committed (this is the first held-out protocol result, so send it at once);
  - the B16 protocol outputs are committed;
  - F8 is complete;
  - any stop condition fires.
