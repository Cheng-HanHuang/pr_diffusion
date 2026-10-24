# R0-N executor return to the Track R scientific planner

Source R0 executor c8eb60877fadeb1b9c7be63b0ff389f7681e7107
Planner decision 6686322b919e891b5dd37d14ce5dddfa96a97dcd
Merge commit 1936e79acd4040f67e754276415bdac503b1121c
R0N code b1637f7c27a3caaa03a8b204171fca0d3fccfcd5
N0 original ZIP SHA 13c008f3bca59b7d8d717ec3215fa1f783ca453b31b9001886c84633b513efab
N0 commit b1637f7c27a3caaa03a8b204171fca0d3fccfcd5
GitHub publish: PUSHED

R0 methodology stopped; no R1 and no protected data access.
K3 replay completed: 3240; GOOD 2040; ORBIT 85;
consistent 324; inconsistent 791; repairs 84/85; GOOD harm 0/2040.

## K1 B22 repair counts (planner decides)

| Source | Family | K | n images | Post-selection repaired |
|---|---|---:|---:|---:|
| S1_b22 | DAPS | 1 | 100 | 2 |
| S1_b22 | DAPS | 2 | 100 | 3 |
| S1_b22 | NP | 1 | 100 | 1 |
| S1_b22 | NP | 2 | 100 | 0 |
| S1_b22 | NP | 4 | 100 | 0 |
| S1_b22 | NP | 8 | 100 | 0 |
| S1_b22 | SITCOM | 1 | 100 | 5 |
| S1_b22 | SITCOM | 2 | 100 | 2 |
| S1_b22 | SITCOM | 4 | 100 | 5 |
| S3_a12 | DAPS | 1 | 8 | 1 |
| S3_a12 | DAPS | 2 | 8 | 1 |
| S3_a12 | DAPS | 4 | 8 | 1 |
| S3_a12 | NP_LF | 1 | 8 | 0 |
| S3_a12 | NP_LF | 2 | 8 | 0 |
| S3_a12 | NP_LF | 4 | 8 | 0 |
| S3_a12 | NP_S2 | 1 | 8 | 0 |
| S3_a12 | NP_S2 | 2 | 8 | 0 |
| S3_a12 | NP_S2 | 4 | 8 | 0 |
| S3_a12 | SITCOM | 1 | 8 | 0 |
| S3_a12 | SITCOM | 2 | 8 | 0 |
| S3_a12 | SITCOM | 4 | 8 | 0 |
| S4_a15 | FAST | 1 | 8 | 0 |
| S4_a15 | FAST | 2 | 8 | 0 |
| S4_a15 | FAST | 4 | 8 | 0 |
| S5_r0_fast | FAST | 1 | 100 | 1 |
| S5_r0_fast | FAST | 2 | 100 | 0 |
| S5_r0_fast | FAST | 4 | 100 | 0 |

## N1d dev selection and B22 evaluation

{
  "J": 1,
  "T": [
    300
  ],
  "cohort": "DEV80_enriched",
  "configuration": "single_300|J1|global2",
  "exact": 0.9722222222222222,
  "forwards": 2,
  "harm": 0.0,
  "member_set": "global2",
  "n_candidates": 1280,
  "n_exact": 35,
  "n_good": 617,
  "n_harm": 0,
  "n_orbit_error": 36,
  "n_success": 35,
  "success": 0.9722222222222222
}
{
  "B22": {
    "J": 1,
    "T": [
      300
    ],
    "cohort": "B22_random",
    "configuration": "single_300|J1|global2",
    "exact": 1.0,
    "forwards": 2,
    "harm": 0.0,
    "member_set": "global2",
    "n_candidates": 1960,
    "n_exact": 49,
    "n_good": 1423,
    "n_harm": 0,
    "n_orbit_error": 49,
    "n_success": 49,
    "success": 1.0
  },
  "configuration": "single_300|J1|global2",
  "harm_image_bootstrap": {
    "ci95": [
      0.0,
      0.0
    ],
    "estimate": 0.0,
    "n_images": 100,
    "n_valid_boot": 10000
  },
  "status": "DEV_SELECTED_EVALUATED_ON_B22",
  "success_image_bootstrap": {
    "ci95": [
      1.0,
      1.0
    ],
    "estimate": 1.0,
    "n_images": 100,
    "n_valid_boot": 10000
  }
}

## N2 CLIP comparator and K2 inputs
Status: UNAVAILABLE_GPU
N2 notes: CLIP stage RC 8

## N3 bounded ImageNet asset inventory
Status: PASS
No ImageNet runs authorized.

## Anomalies / deviations
Optional environmental blockers are reported by status,
not interpreted as failures of the scientific baseline.

## PAC roots
N1: /egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0n_n1_20261024T061111Z
N2: unavailable
N3: /egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0n_n3_20261024T061111Z

K1 and K2 rulings reserved for the scientific planner.
No R1 or protected confirmation305 access.
