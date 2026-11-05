# A24: G7 ground-truth orientation diagnostics (planner check, 2026-11-04)

**Inputs** (`theory_inputs` copies in each commit):

| Test | Commit | Per-image file |
|---|---|---|
| S_mid global2 | XP `da130b5` | `XP_A23_G7_S_MID_GLOBAL2_GT_PER_IMAGE.csv` |
| S_mid all8 | XP `967e73c` | `XP_A35_G7_S_mid_all8_PER_IMAGE.csv` |
| 2-forward | XP `6d2afbf` | `XP_A35_G7_2_forward_PER_IMAGE.csv` |
| UnivFD | XP `9d379d2` | `XP_A35_G7_UnivFD_PER_IMAGE.csv` |
| CLIP | XP `9090a50` | `XP_A12_G7_C_CLIP_GT_ORIENTATION.csv` |

The aggregates come from the same commits.

**Script:** `code/g7_gt_check.py` → `out/G7_GT_CHECK.txt`.

**Checks:**
- The same 280 images (180 FFHQ, 100 ImageNet), with the same source hashes, appear in every file.
- The S_mid threshold rule holds in every row.
- Each `upright_kept` flag equals (choice == 000).
- The aggregates equal the per-image counts.

**Results:**
- **Upright kept, FFHQ / ImageNet:**

  | Test | FFHQ | ImageNet |
  |---|---|---|
  | G1 | 180 | 92 |
  | G2 | 180 | 92 |
  | S_mid global2 | 180 | 95 |
  | S_mid all8 | 180 | 95 |
  | 2-forward | 180 | 93 |
  | UnivFD | 172 | 89 |

- **S_mid all8 = S_mid global2** on all 280 images.
- **UnivFD's non-upright choices:** 18 global (111) and 1 channel-wise (010).
- **ImageNet overlaps:**
  - G1 ∩ S_mid global2: 2 of 8 and 5;
  - S_mid global2 ∩ 2-forward: 4 of 5 and 7;
  - all four (G1, S_mid global2, 2-forward, UnivFD): 0;
  - at least one: 20.
