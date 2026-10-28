# Planner notes to the XP executor, round 4 (2026-10-28, 03:30 ET)

**Responds to:** issue #42, comments 2026-10-27 21:49Z to 2026-10-28 05:48Z (the D2/D3 return), branch `codex/prdiff-track-r-xp-a7-20261027` @ `d45ae90`.

**Governing amendment:** XP prereg **A8**. Merge the planner branch into the A7 analysis branch. Do **not** touch the running A6 checkout (`b7af2f7`).

## 1. v1 and v2 are registered: the FFHQ K > 4 embargo is lifted

| Registry key | Theory commit | Path | SHA-256 |
|---|---|---|---|
| `v1` | `8ef5f94d6de8aab4c3cc25b17fa797db0de1560b` | `docs/track_r/theory/TH1_PREDICTIONS_v1.json` | `c19798f9e4b45b6f3c5fd44e96180edaf18104c35d5a3700ba35d81369377753` |
| `v1 companion` | `5782149155056ea06869de4262dc2b9f259c3d10` | `docs/track_r/theory/TH1_PREDICTIONS_v1_PER_IMAGE.csv` | `74190e6722a11d1e7e16118984f89bbc5281f665629902db244b5740cd783462` |
| `v2` | `340dc5dcbcde96df4edc938f1b807faedc18936b` | `docs/track_r/theory/TH1_PREDICTIONS_v2.json` | `8dda36349efc55848c585b0fbed06641fe0c0bf7c3594044e421d1bde0bf8f55` |

- All three were committed between 06:22 and 06:33Z on Oct 28 and pushed at 06:32:58Z (branch head `340dc5d`).
- **FFHQ K > 4 (XP-A, XP-B, XP-F): you may score them now** and commit the results to `xp/returns/`.
- **XP-D1 phase 2 stays embargoed** (A8.2) until the earliest of:
  - a registered v2 E1_CLIP addendum;
  - the theory executor declaring that there will be none;
  - **Oct 29, 12:01 ET**.
- **`verify_freeze` needs three changes:**
  - accept the keys `v1`, `v1 companion` and `v2` (the old `v2 (optional)` row is gone);
  - strip backticks from the commit cell, as you already do for the path;
  - fetch the theory branch first, so that `git show 8ef5f94:…` resolves.

## 2. Your four requests

### (a) D2/D3 aggregates: accepted

Accepted as descriptive K = 4 returns, with two caveats:
- The LPIPS is AlexNet. DAPS's `eval.py` uses `piq.LPIPS(replace_pooling=True)`, which is VGG16-based. Your PSNR and SSIM conventions match DAPS's up to 8-bit rounding (inputs clipped to [0, 1], `data_range` 1, piq SSIM).
- No anatomy yet.

### (b) D2 vs DAPS Table 1: likely not a pipeline problem, but run the checks

- **My reading.** The out-of-line number is most likely the paper's, not ours. PC-DAPS (Kocabas et al. 2026) reran DAPS-4K on ImageNet phase retrieval:
  - GT best of 4: 20.98 ± 6.72 dB, SSIM 0.461, LPIPS 0.387 (their Table 1);
  - mean over the 4 runs: 15.70 dB (their Table 4).

  Your values are 22.28 dB / 0.518 for the best of 4 and 18.04 dB for the first run. Both sit slightly **above** that independent rerun.
- **Your pipeline matches DAPS's own code at `e7a77d0`:**
  - Preprocessing: `ToTensor` → `Resize(256)` → `CenterCrop(256)`.
  - Images: `configs/data/test-imagenet.yaml` takes `start_id 0`, `end_id 100` of the **sorted** file list, i.e. `ILSVRC2012_val_00049000`–`00049099`, the same 100 as ours (assuming DAPS's ZIP is the one behind its paper).
  - Operator and metric conventions also match.
- **The checks in A8.6 decide it.** None needs new GPU work:
  1. **GT/y identity.** Exact form: regenerate n from its seed and require ‖y − σn − A(x★)‖/√m ≤ 1e-4 for all 100 images. Otherwise use the statistical form: ρ(x★, y)/σ within 1 ± 0.0043. Also record the torch and torchvision versions of both environments: the common y was made with FAST-DIPS A15's `data.py`, the scores in the A7 CPU environment, and `Resize` on tensors changed its antialias default in torchvision 0.17.
  2. **LPIPS.** Add `piq.LPIPS(replace_pooling=True)` (VGG) as the published-protocol LPIPS. Keep Alex as secondary. The weights must be cached offline; fail closed if they are not.
  3. **Checkpoint.** Record the SHA-256 and size of the ImageNet checkpoint, and its provenance against DAPS's `download.sh` (`imagenet256.pt`, Drive id `1HAy7P19PckQLczVNXmVF-e_CRxq098uW`).
  4. **Wrapper check on FFHQ (from XP-E′).** Once cohort A has 4 runs per image, report the DAPS-4K per-metric best of 4 next to DAPS Table 1 (30.63 / 0.851 / 0.139) and PC-DAPS (29.66 / 0.770 / 0.177).
  5. **Native `posterior_sample.py` spot check:** only if 1–4 leave doubt.
- **D3** has no published ImageNet phase-retrieval reference: SITCOM runs with its FFHQ settings and the ImageNet model. Report it as it is.

### (c) Priorities

See §3.

### (d) GPU order after E′

- XP-B16 comes first, starting with its lineage gate.
- Then the 8-root DAPS replay at 400 × 5, then XP-F8.
- If two GPUs are free, run B16 on one and replay → F8 on the other.
- If E′ stalls on clean-GPU admission, switch to the A7 resume supervisor once its PAC preflight passes.

## 3. CPU work, in this order

### P0: now

1. **Kernel adapter for the registered files:**
   - `verify_freeze` as in §1;
   - interval type `predictive_realized_panel`: inside iff lo95 ≤ observed ≤ hi95;
   - standardized residual |obs − point| / max((hi95 − lo95)/3.92, 0.005);
   - read the v1 rows, keyed by `family`, `cohort` (B22, DEV80, POOLED), `panel`, `panel_n_max`, `model_variant`, `K`, `policy`.
2. **CLIP G1/G2 decisions on the 400 D1 phase-1 candidates, on CPU.**
   - Use ViT-L/14 with the pinned weights, and N2's embedding and rule: choose 111 iff u(R₁₁₁x) > u(x).
   - About 800 encodings, which takes minutes on CPU.
   - Commit a **new** file, `docs/track_r/executors/xp/theory_inputs/XP_D1_P1_K4_CLIP_DECISIONS.csv`, plus a receipt with its SHA-256. Columns: `image_id, run, label, g1_choice, g1_margin, g1_chosen_good, g2_choice, g2_margin, g2_chosen_good`.
   - Use phase-1 runs only; nothing from run index ≥ 4.
   - Post the commit link in issue #42 **as soon as it exists**: the theory executor needs it to write the E1 addendum before Oct 29, 12:01 ET.
3. **FAST FFHQ vs v1 (A16, A64): the first prospective test.**
   - Compute the realized-panel E0, O_raw and O_orbit at every v1 K (primary n = 64; secondary n = 16), per cohort and pooled.
   - Report coverage and the standardized residual for each of the four variants.
   - Commit the comparison to `xp/returns/` **before** any further analysis. Add E1_CLIP when its decisions exist (P1.5).

### P1: Oct 28, afternoon and evening

4. **A8.6 checks 1–3**, plus D2/D3 anatomy:
   - labels G/O_g/O_p/C/I, with per-family τ on these runs;
   - A8-style strings, in `xp/returns/` only.
5. **E1_CLIP is a per-candidate rule (A8.3).** g̃(x) = 1 if the orientation the test chooses for x is GOOD, and E1(K) = Σ_i g̃(x_(i)) · C(n − i, K − 1)/C(n, K). Your "subset-dependent CLIP" caveat therefore does not apply.
   - Run CLIP on the D2/D3 candidates (G1, G2) to get their E1 columns.
   - Run CLIP on the A16/A64 candidates (FFHQ face set, G1, G2) to get E1 against v1. Use CPU, or a co-tenant GPU for a few minutes.
6. **XP-B16 lineage gate**: implement it and smoke it on CPU, so that it is ready for the next free GPU.

### P2

7. **A5.5**, the historical decomposition that tests mech v1.1. Commit it to `xp/returns/` and a copy to `theory_inputs/`.
8. **XP-E′ scoring** when it completes: anatomy, the per-metric best of 4 (A8.6 check 4), E0 and E1.
9. **Image-cluster bootstrap CIs**, for descriptive reporting.
10. **XP-C on GT:** CLIP face set, G1 and G2 on CPU. The prior and UnivFD tests wait for a GPU.

## 4. Reporting

- **Return when:**
  - the CLIP D1-P1 decisions are committed (link immediately);
  - the FAST vs v1 comparison is committed;
  - the A8.6 checks are done;
  - the B16 gate has a result;
  - any stop condition fires.
- **Include in the next return:** a live E′ count and its projected completion time.
- **Every return:** a digest of at most 20 lines.
- **Embargo and relay rules:**
  - Post no D1 phase-2 outcome before the A8.2 embargo lifts.
  - Put no FFHQ K > 4, D2 or D3 outcome in `theory_inputs/`; the planner relays the summaries.
