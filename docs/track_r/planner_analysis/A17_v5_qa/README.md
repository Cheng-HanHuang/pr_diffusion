# A17.1: native QA of the frozen v5 code (planner, 2026-10-31)

**Code.** Theory commit `81003cc5a9167def74a04281501b4eae2837f44c`, checked out detached. The five files match `docs/track_r/theory/TH_XPN_V5_CODE_FREEZE_20261031.md` (receipt at `3ad5f2d`, SHA-256 `0208b753…`). Python 3.13.16, NumPy 2.5.3.

| Check | Result |
|---|---|
| `test_th1_imagenet_v5_synthetic.py` (real sibling modules, fabricated non-D1 ids) | `XP_N_V5_SYNTHETIC_PASS 100 20 4`, 7.7 s |
| Production parameters (250 / 60 / 3) on the **released D1 runs 0–3** (`XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv` `c2177880…`, `XP_D1_P1_K4_CLIP_DECISIONS.csv` `0a298918…`) | Contract PASS 100/20/4 in 41 s; oracle-anchor max shift 0.0010 / 0.0016 / 0.0033 / 0.0020 per variant (gate 0.01) |
| Same model as the registered v2 E1 (same images, same four runs) | Race E1 0.578 at K = 8 and 0.599 at K = 16, against v2 E1 0.578 and 0.603 |

**`D1_P1_DRYRUN_NOT_A_FORECAST.json` is not a forecast.** The later runs of these images are known (XP-D4). Against them, the race covers E1 at all five K, E0 at 4 of 5, O_raw at 5 of 5 and O_orbit at 0 of 5 (0.944 forecast against 0.780 observed at K = 64). S2 covers E0 and E1 at none. This is the familiar pattern: selection is predicted, coverage is overforecast. It is recorded here before any XP-N run exists, so that v5's prospective result on new images can be read against it.

**Sibling modules imported by v5** (pinned by the commit, not listed in the receipt), SHA-256 at `81003cc`:
- `audit_a8_rank_race.py`: `dd58cb7f96d47f5374fc56ccbb2a096ceab42bddf808e281f8565d2a14835405`
- `fit_th1_dirichlet.py`: `45c166dccff75a933b374c93ac8609a8500c79fbcfc12ce89326638a1b06962b`
- `fit_th1_prefix_v1.py`: `db20d9a8276ba3325dc1b4a64b71c072006672cd2a94de62a2810acae71c9310`
