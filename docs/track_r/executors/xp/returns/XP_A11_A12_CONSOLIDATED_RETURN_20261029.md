# Track R A11/A12 combined executor return — 2026-10-29
1. **First held-out D1-P2 return** committed/pushed: `5f6b21334668d1a614c011f79de0684335c3ab87` (`codex/prdiff-track-r-xp-a11-d1p2-heldout-20261029`).
2. Authoritative 100 ImageNet images × ordered FAST run 0–15 labels and signed-residual strings: `docs/track_r/executors/xp/theory_inputs/XP_D1P2_16RUN_STRINGS.csv` on that branch.
3. **STRINGS SHA256:** `48169156486de4438e715d97a5a3251a728f52e055f135caf14fbcb02fd33e27`.
4. Authoritative 1,600 candidate G1 (and G2) choices/margins: `docs/track_r/executors/xp/theory_inputs/XP_D1P2_16RUN_CLIP_DECISIONS.csv` on that branch; header `image_id,run,label,g1_choice,g1_margin,g1_chosen_good,g2_choice,g2_margin,g2_chosen_good`.
5. **G1 DECISIONS CSV SHA256:** `bd50ce51eb3eaccfeb5c23f45a53cc19e350852c8365750fffcaeda6e730db5f`.
6. The two files above were verified on GitHub against the published input receipt (`n_images=100`, `n_rows=1600`); no new GPU, GT or CLIP inference for their export. **Theory v4 can consume them immediately.**
7. Six byte-frozen `c598176` protocol-v1 D1 results also committed; primary result JSON SHA256 `8dc1f5bad10e041b253addfee2a74868bc280c30822c4d289b427ba8480e11d3`.
8. D1 primary M2 N16: success `0.514455`; mean runs `3.3592625`; equal-compute fixed-E1 gain `+0.0033265975`; unflagged success `0.6493644`.
9. D1 prereg hypotheses: **H1 PASS, H3 FAIL** (FLAG_C unsolved precision `1/3`, 3 expected flags), **H4 PASS**.
10. **B16 v1-primary/v3-secondary scientific return** committed/pushed: `87984620debb6876eaa018a3d0914bcf4b78d49a` (`codex/prdiff-track-r-xp-a11-b16-return-20261029`).
11. 180 images × 16 SITCOM roots: 2880/2880 GT-scores, 2880/2880 N2 face-CLIP; prospective comparison SHA256 `82f8f09c29393e7515f472e1d7facc58ba9892ca6ce41210d3f20e710b1e02cb`.
12. B16 predictive-interval coverage: race v1 `31/36`, v3 `17/36`; S2 v1 `15/36`, v3 `7/36`; tail v1 `30/36`, v3 `12/36`; 15 never-GOOD and 17 K16 raw E0 selection-loss cases.
13. **A12 G1a complete, sealed generation**: 1600/1600 FAST ImageNet runs 16–31; mandatory 2/2 historical replay pixel-exact PASS; failed attempts 0; all four GPUs released.
14. G1a run `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_a12_g1a_prod_20261029T180624Z`, frozen producer `8b7f257`, executor `c15651b152198d0bbd014c279499b2c5a25814b3`; stage charged `3.572931348127313` GPU-h of cap 25.
15. Full source-backed A12 project ledger SHA256 `3eabf7b978e1341549f72690061d65d2aee1c29046c0749df0e2b821d74a2f8c`; project **conservative billed upper** before G1a `153.581581288` GPU-h; after G1a `157.154512636` GPU-h of cap 400.
16. Of the pre-G1a project charge, 31.314365 GPU-h is a purposely OVER-BILLED upper bound for an unrecorded exact exit time of old failed B16 gate PID 1057433; this is **not** actual measured runtime nor paper timing.
17. A12 campaign charged through G1a: `55.54578541457511` historical A10 + `3.572931348127313` G1a = `59.11871676270242` GPU-h of 150.
18. XP-D4 new-run outcomes remain **sealed** pending theory v4 registration/declination or Nov 2, 12:00 ET. No labels/GT used in G1a generation. G2 next; B32/XP-K/XP-H/XP-G2/XP-J/XP-I2 still unauthorized.
19. **Planner action:** acknowledge accessible D1 100×16 strings + G1 margins for theory v4, reproduce frozen D1/B16 scientific returns, and confirm A12 subsequent stage and scheduling policy. No change to prereg.
