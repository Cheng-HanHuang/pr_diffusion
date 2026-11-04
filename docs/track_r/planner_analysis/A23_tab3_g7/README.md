# A23: Tab. 3 E1 check and G7 C2/CLIP read (planner, 2026-11-04)

## Tab. 3: `code/tab3_check.py` → `out/TAB3_CHECK.txt`

**Inputs.**
- XP `df6e8cd`: `returns/XP_A22_TAB3_20261104T040346Z/XP_A22_TAB3_IMAGE_E1_G1.csv` (300 rows) and the aggregate `XP_A22_TAB3_IMAGENET_E1_G1.csv` (`124d6703…`; the returns copy is identical to the `theory_inputs` copy).
- Frozen G5 strings and G1 decisions (CP3, XP `1981dea`; `14201083…`, `ca5d0d7e…`).
- XP-I2 DAPS comparator (XP `1eae9bf`, `a1a94a3b…`).

**Result.**
- Aggregates equal the per-image means, and GOOD20 = (PSNR ≥ 20) in every row.
- DAPS-4K at K = 4 and K = 8: the residual pick among the first K runs, the G1 choice and the GOOD outcome agree with the frozen data for **100/100** images.
- The K = 8 PSNR equals the XP-I2 comparator for 100/100.
- SITCOM can be checked only for internal consistency, because its per-run detail is local to PAC.

## G7: read of XP `9090a50`

**C2 table** `841f2b77…` (36 rows).
- δ̃ is the ratio of means of ‖D(R x_t) − R D(x_t)‖² to ‖D(x_t) − x‖² (see `scripts/track_r/xp_a12_g7_c2_gpu.py` at `9090a50`).
- The values quoted in S2 are copied from it. The bounds of the 95% bootstrap intervals lie within 7% (FFHQ) and 28% (ImageNet) of each value.

**Ground-truth CLIP CSV** `5507d1ce…`, counted by the planner:
- FFHQ: G1, G2 and face keep all 180 images upright;
- ImageNet: G1 and G2 choose 111 for 8 of 100 (the face test is not applied).

**N2 prompts.** `XP_A22_FROZEN_N2_FACE_PROMPTS.json` (XP `a42e14e`) equals `PROMPTS_UP` and `PROMPTS_INV` in `b1637f7c:scripts/track_r/r0n_2_clip_orientation.py`.
