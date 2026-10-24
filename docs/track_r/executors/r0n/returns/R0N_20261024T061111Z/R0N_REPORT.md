# Track R R0-N N1 — Numbers-only letter evidence

All findings retrospective. B22 random and DEV80 hard-enriched
are disjoint in image ID. No confirmation305 accessed.

## K3 replay

```json
{"harm_good_count": 0, "harm_good_denominator": 2040, "labels": {"GOOD": 2040, "NONORBIT_CONSISTENT": 324, "NONORBIT_INCONSISTENT": 791, "ORBIT_ERROR": 85}, "n": 3240, "p4_witnesses_verified": 7, "repair_orbit_denominator": 85, "repair_orbit_success": 84, "status": "R0N_K3_REPLAY_PASS"}
```

## N1a — cohort × family anatomy

| Cohort | Family | n | GOOD | Orbit | Global | Partial | Consistent | Inconsistent |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B22_random | DAPS | 232 | 184 | 9 | 9 | 0 | 0 | 39 |
| B22_random | FAST | 432 | 289 | 5 | 5 | 0 | 9 | 129 |
| B22_random | NP | 800 | 598 | 20 | 20 | 0 | 62 | 120 |
| B22_random | NP_LF | 32 | 21 | 0 | 0 | 0 | 0 | 11 |
| B22_random | NP_S2 | 32 | 22 | 1 | 1 | 0 | 1 | 8 |
| B22_random | SITCOM | 432 | 309 | 14 | 14 | 0 | 65 | 44 |
| DEV80_enriched | DAPS | 320 | 170 | 17 | 16 | 1 | 18 | 115 |
| DEV80_enriched | FAST | 320 | 155 | 6 | 6 | 0 | 14 | 145 |
| DEV80_enriched | NP | 320 | 163 | 2 | 2 | 0 | 40 | 115 |
| DEV80_enriched | SITCOM | 320 | 129 | 11 | 11 | 0 | 115 | 65 |

## N1a image-cluster orbit-share confidence intervals

| Cohort | Family | Orbit share among raw failures | 95% CI |
|---|---|---:|---|
| B22_random | DAPS | 0.1875 | [0.0556, 0.3333] |
| B22_random | FAST | 0.0350 | [0.0075, 0.0682] |
| B22_random | NP | 0.0990 | [0.0443, 0.1667] |
| B22_random | NP_LF | 0.0000 | [0.0000, 0.0000] |
| B22_random | NP_S2 | 0.1000 | [0.0000, 0.5000] |
| B22_random | SITCOM | 0.1138 | [0.0550, 0.1810] |
| DEV80_enriched | DAPS | 0.1133 | [0.0479, 0.1908] |
| DEV80_enriched | FAST | 0.0364 | [0.0057, 0.0818] |
| DEV80_enriched | NP | 0.0127 | [0.0000, 0.0426] |
| DEV80_enriched | SITCOM | 0.0576 | [0.0123, 0.1150] |

## N1b — post-selection decomposition

| Cohort | Source | Family | K | Images | E0 failed | Fixed by E1 | Orbit uncorrected | Selection miss | Generation failure |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B22_random | S1_b22 | DAPS | 1 | 100 | 20 | 2 | 0 | 0 | 18 |
| B22_random | S1_b22 | DAPS | 2 | 100 | 8 | 3 | 0 | 0 | 5 |
| B22_random | S1_b22 | NP | 1 | 100 | 25 | 1 | 0 | 0 | 24 |
| B22_random | S1_b22 | NP | 2 | 100 | 14 | 0 | 0 | 0 | 14 |
| B22_random | S1_b22 | NP | 4 | 100 | 8 | 0 | 1 | 1 | 6 |
| B22_random | S1_b22 | NP | 8 | 100 | 6 | 0 | 1 | 1 | 4 |
| B22_random | S1_b22 | SITCOM | 1 | 100 | 29 | 5 | 0 | 0 | 24 |
| B22_random | S1_b22 | SITCOM | 2 | 100 | 15 | 2 | 0 | 0 | 13 |
| B22_random | S1_b22 | SITCOM | 4 | 100 | 11 | 5 | 0 | 2 | 4 |
| DEV80_enriched | S2_b24_dev80 | DAPS | 1 | 80 | 33 | 5 | 0 | 0 | 28 |
| DEV80_enriched | S2_b24_dev80 | DAPS | 2 | 80 | 23 | 4 | 0 | 0 | 19 |
| DEV80_enriched | S2_b24_dev80 | DAPS | 4 | 80 | 14 | 3 | 0 | 0 | 11 |
| DEV80_enriched | S2_b24_dev80 | NP | 1 | 80 | 39 | 1 | 0 | 0 | 38 |
| DEV80_enriched | S2_b24_dev80 | NP | 2 | 80 | 32 | 1 | 0 | 0 | 31 |
| DEV80_enriched | S2_b24_dev80 | NP | 4 | 80 | 24 | 0 | 0 | 1 | 23 |
| DEV80_enriched | S2_b24_dev80 | SITCOM | 1 | 80 | 46 | 2 | 0 | 0 | 44 |
| DEV80_enriched | S2_b24_dev80 | SITCOM | 2 | 80 | 37 | 3 | 0 | 0 | 34 |
| DEV80_enriched | S2_b24_dev80 | SITCOM | 4 | 80 | 25 | 3 | 0 | 0 | 22 |
| B22_random | S3_a12 | DAPS | 1 | 8 | 4 | 1 | 0 | 0 | 3 |
| B22_random | S3_a12 | DAPS | 2 | 8 | 1 | 1 | 0 | 0 | 0 |
| B22_random | S3_a12 | DAPS | 4 | 8 | 1 | 1 | 0 | 0 | 0 |
| B22_random | S3_a12 | NP_LF | 1 | 8 | 4 | 0 | 0 | 0 | 4 |
| B22_random | S3_a12 | NP_LF | 2 | 8 | 3 | 0 | 0 | 0 | 3 |
| B22_random | S3_a12 | NP_LF | 4 | 8 | 0 | 0 | 0 | 0 | 0 |
| B22_random | S3_a12 | NP_S2 | 1 | 8 | 4 | 0 | 0 | 0 | 4 |
| B22_random | S3_a12 | NP_S2 | 2 | 8 | 2 | 0 | 0 | 0 | 2 |
| B22_random | S3_a12 | NP_S2 | 4 | 8 | 0 | 0 | 0 | 0 | 0 |
| B22_random | S3_a12 | SITCOM | 1 | 8 | 3 | 0 | 0 | 0 | 3 |
| B22_random | S3_a12 | SITCOM | 2 | 8 | 0 | 0 | 0 | 0 | 0 |
| B22_random | S3_a12 | SITCOM | 4 | 8 | 0 | 0 | 0 | 0 | 0 |
| B22_random | S4_a15 | FAST | 1 | 8 | 2 | 0 | 0 | 0 | 2 |
| B22_random | S4_a15 | FAST | 2 | 8 | 1 | 0 | 0 | 0 | 1 |
| B22_random | S4_a15 | FAST | 4 | 8 | 0 | 0 | 0 | 0 | 0 |
| B22_random | S5_r0_fast | FAST | 1 | 100 | 34 | 1 | 0 | 0 | 33 |
| DEV80_enriched | S5_r0_fast | FAST | 1 | 80 | 43 | 2 | 0 | 0 | 41 |
| B22_random | S5_r0_fast | FAST | 2 | 100 | 15 | 0 | 0 | 0 | 15 |
| DEV80_enriched | S5_r0_fast | FAST | 2 | 80 | 26 | 1 | 0 | 0 | 25 |
| B22_random | S5_r0_fast | FAST | 4 | 100 | 6 | 0 | 0 | 1 | 5 |
| DEV80_enriched | S5_r0_fast | FAST | 4 | 80 | 20 | 1 | 0 | 0 | 19 |

## N1c — signed/clipped residual distributions

Only observed cohort/family/label quantiles are reported;
the relative overlap in SITCOM versus FAST/DAPS is a descriptive
diagnostic, not an invented causal explanation.

| Cohort | Family | Label | n | Signed median [q10,q90] | Clipped median [q10,q90] |
|---|---|---|---:|---|---|
| B22_random | DAPS | GOOD | 184 | 0.04731 [0.04720,0.04744] | 0.03466 [0.03402,0.03569] |
| B22_random | DAPS | NONORBIT_INCONSISTENT | 39 | 0.04791 [0.04750,0.05039] | 0.03578 [0.03510,0.03966] |
| B22_random | DAPS | ORBIT_ERROR | 9 | 0.04745 [0.04727,0.04751] | 0.03518 [0.03419,0.03610] |
| B22_random | FAST | GOOD | 289 | 0.04983 [0.04972,0.05002] | 0.03674 [0.03605,0.03769] |
| B22_random | FAST | NONORBIT_CONSISTENT | 9 | 0.04999 [0.04996,0.05002] | 0.03818 [0.03710,0.03834] |
| B22_random | FAST | NONORBIT_INCONSISTENT | 129 | 0.05097 [0.05032,0.18761] | 0.03886 [0.03715,0.18463] |
| B22_random | FAST | ORBIT_ERROR | 5 | 0.05006 [0.04989,0.05009] | 0.03631 [0.03614,0.03671] |
| B22_random | NP | GOOD | 598 | 0.04960 [0.04935,0.05030] | 0.03821 [0.03705,0.04022] |
| B22_random | NP | NONORBIT_CONSISTENT | 62 | 0.05017 [0.04969,0.05043] | 0.03853 [0.03730,0.03955] |
| B22_random | NP | NONORBIT_INCONSISTENT | 120 | 0.05169 [0.05057,0.05306] | 0.04147 [0.03971,0.04444] |
| B22_random | NP | ORBIT_ERROR | 20 | 0.04956 [0.04940,0.04973] | 0.03825 [0.03682,0.03860] |
| B22_random | NP_LF | GOOD | 21 | 0.04952 [0.04930,0.04995] | 0.03802 [0.03707,0.03929] |
| B22_random | NP_LF | NONORBIT_INCONSISTENT | 11 | 0.05023 [0.05012,0.05076] | 0.03869 [0.03809,0.03973] |
| B22_random | NP_S2 | GOOD | 22 | 0.04952 [0.04930,0.04993] | 0.03797 [0.03704,0.03926] |
| B22_random | NP_S2 | NONORBIT_CONSISTENT | 1 | 0.04960 [0.04960,0.04960] | 0.03698 [0.03698,0.03698] |
| B22_random | NP_S2 | NONORBIT_INCONSISTENT | 8 | 0.05029 [0.05001,0.05258] | 0.03913 [0.03829,0.04089] |
| B22_random | NP_S2 | ORBIT_ERROR | 1 | 0.04969 [0.04969,0.04969] | 0.03864 [0.03864,0.03864] |
| B22_random | SITCOM | GOOD | 309 | 0.04418 [0.04368,0.04468] | 0.02911 [0.02859,0.03005] |
| B22_random | SITCOM | NONORBIT_CONSISTENT | 65 | 0.04434 [0.04313,0.04470] | 0.03008 [0.02937,0.03094] |
| B22_random | SITCOM | NONORBIT_INCONSISTENT | 44 | 0.04578 [0.04507,0.04805] | 0.03198 [0.03001,0.03583] |
| B22_random | SITCOM | ORBIT_ERROR | 14 | 0.04433 [0.04381,0.04508] | 0.02928 [0.02853,0.03057] |
| DEV80_enriched | DAPS | GOOD | 170 | 0.04733 [0.04719,0.04746] | 0.03498 [0.03401,0.03579] |
| DEV80_enriched | DAPS | NONORBIT_CONSISTENT | 18 | 0.04752 [0.04743,0.04758] | 0.03567 [0.03397,0.03663] |
| DEV80_enriched | DAPS | NONORBIT_INCONSISTENT | 115 | 0.04824 [0.04768,0.05044] | 0.03652 [0.03525,0.03951] |
| DEV80_enriched | DAPS | ORBIT_ERROR | 17 | 0.04733 [0.04721,0.04751] | 0.03525 [0.03488,0.03627] |
| DEV80_enriched | FAST | GOOD | 155 | 0.04985 [0.04969,0.05007] | 0.03690 [0.03603,0.03767] |
| DEV80_enriched | FAST | NONORBIT_CONSISTENT | 14 | 0.05000 [0.04987,0.05007] | 0.03781 [0.03723,0.03846] |
| DEV80_enriched | FAST | NONORBIT_INCONSISTENT | 145 | 0.05066 [0.05022,0.13841] | 0.03863 [0.03732,0.13443] |
| DEV80_enriched | FAST | ORBIT_ERROR | 6 | 0.04982 [0.04955,0.05015] | 0.03748 [0.03647,0.03758] |
| DEV80_enriched | NP | GOOD | 163 | 0.04970 [0.04939,0.05031] | 0.03874 [0.03716,0.04013] |
| DEV80_enriched | NP | NONORBIT_CONSISTENT | 40 | 0.05016 [0.04994,0.05033] | 0.03932 [0.03776,0.03981] |
| DEV80_enriched | NP | NONORBIT_INCONSISTENT | 115 | 0.05135 [0.05069,0.05328] | 0.04118 [0.04022,0.04460] |
| DEV80_enriched | NP | ORBIT_ERROR | 2 | 0.04931 [0.04931,0.04931] | 0.03644 [0.03642,0.03645] |
| DEV80_enriched | SITCOM | GOOD | 129 | 0.04418 [0.04345,0.04483] | 0.02935 [0.02857,0.03013] |
| DEV80_enriched | SITCOM | NONORBIT_CONSISTENT | 115 | 0.04416 [0.04340,0.04485] | 0.03026 [0.02937,0.03124] |
| DEV80_enriched | SITCOM | NONORBIT_INCONSISTENT | 65 | 0.04602 [0.04506,0.05152] | 0.03304 [0.02988,0.03918] |
| DEV80_enriched | SITCOM | ORBIT_ERROR | 11 | 0.04402 [0.04395,0.04431] | 0.02953 [0.02899,0.03049] |

## N1d — DEV80-selected repair configuration

```json
{"J": 1, "T": [300], "cohort": "DEV80_enriched", "configuration": "single_300|J1|global2", "exact": 0.9722222222222222, "forwards": 2, "harm": 0.0, "member_set": "global2", "n_candidates": 1280, "n_exact": 35, "n_good": 617, "n_harm": 0, "n_orbit_error": 36, "n_success": 35, "success": 0.9722222222222222}
```

Eval B22 clustered confidence intervals:
```json
{"B22": {"J": 1, "T": [300], "cohort": "B22_random", "configuration": "single_300|J1|global2", "exact": 1.0, "forwards": 2, "harm": 0.0, "member_set": "global2", "n_candidates": 1960, "n_exact": 49, "n_good": 1423, "n_harm": 0, "n_orbit_error": 49, "n_success": 49, "success": 1.0}, "configuration": "single_300|J1|global2", "harm_image_bootstrap": {"ci95": [0.0, 0.0], "estimate": 0.0, "n_images": 100, "n_valid_boot": 10000}, "status": "DEV_SELECTED_EVALUATED_ON_B22", "success_image_bootstrap": {"ci95": [1.0, 1.0], "estimate": 1.0, "n_images": 100, "n_valid_boot": 10000}}
```

## N1e — costs
```json
{"by_source_family": [{"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "DAPS", "n": 200, "native_recorded_cost_seconds": {"mean": 156.32, "median": 160.5, "n": 200}, "score_wall_seconds": {"mean": 2.7550030688080005, "median": 2.7511159714777023, "n": 200}, "source": "S1_b22"}, {"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "NP", "n": 800, "native_recorded_cost_seconds": {"mean": 51.47287087003002, "median": 51.16651175171137, "n": 800}, "score_wall_seconds": {"mean": 2.755545576759614, "median": 2.7523911816533655, "n": 800}, "source": "S1_b22"}, {"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "SITCOM", "n": 400, "native_recorded_cost_seconds": {"mean": 49.01773709750269, "median": 50.44662321638316, "n": 400}, "score_wall_seconds": {"mean": 2.7553245901747143, "median": 2.752298911800608, "n": 400}, "source": "S1_b22"}, {"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "DAPS", "n": 320, "native_recorded_cost_seconds": {"mean": 195.35481286143767, "median": 162.03877207729965, "n": 320}, "score_wall_seconds": {"mean": 2.75593131101632, "median": 2.751891032559797, "n": 320}, "source": "S2_b24_dev80"}, {"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "NP", "n": 320, "native_recorded_cost_seconds": {"mean": 136.75660332890692, "median": 96.5632661851123, "n": 320}, "score_wall_seconds": {"mean": 2.7549603556923103, "median": 2.7518735921476036, "n": 320}, "source": "S2_b24_dev80"}, {"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "SITCOM", "n": 320, "native_recorded_cost_seconds": {"mean": 88.58764015935012, "median": 74.90044487267733, "n": 320}, "score_wall_seconds": {"mean": 2.7550396731254296, "median": 2.7518255240283906, "n": 320}, "source": "S2_b24_dev80"}, {"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "DAPS", "n": 32, "native_recorded_cost_seconds": {"mean": 65.02642535715131, "median": 55.84596803900786, "n": 32}, "score_wall_seconds": {"mean": 2.7529671997763216, "median": 2.7523641628213227, "n": 32}, "source": "S3_a12"}, {"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "NP_LF", "n": 32, "native_recorded_cost_seconds": {"mean": 36.85123689248576, "median": 33.10234971973114, "n": 32}, "score_wall_seconds": {"mean": 2.7601497700234177, "median": 2.753270937362686, "n": 32}, "source": "S3_a12"}, {"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "NP_S2", "n": 32, "native_recorded_cost_seconds": {"mean": 36.79359714286693, "median": 33.1287351467181, "n": 32}, "score_wall_seconds": {"mean": 2.7591019282699563, "median": 2.7524228068068624, "n": 32}, "source": "S3_a12"}, {"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "SITCOM", "n": 32, "native_recorded_cost_seconds": {"mean": 30.620366076545906, "median": 29.972941586980596, "n": 32}, "score_wall_seconds": {"mean": 2.7572201295843115, "median": 2.752486320445314, "n": 32}, "source": "S3_a12"}, {"FAST_CUDA_event_seconds": {"mean": null, "median": null, "n": 0}, "FAST_sampling_wall_seconds": {"mean": null, "median": null, "n": 0}, "family": "FAST", "n": 32, "native_recorded_cost_seconds": {"mean": 3.712786338481237, "median": 3.705860875546932, "n": 32}, "score_wall_seconds": {"mean": 2.7521712640154874, "median": 2.7522849349770695, "n": 32}, "source": "S4_a15"}, {"FAST_CUDA_event_seconds": {"mean": 3.335007605319553, "median": 3.3295971679687497, "n": 720}, "FAST_sampling_wall_seconds": {"mean": 3.335042295233062, "median": 3.329625853570178, "n": 720}, "family": "FAST", "n": 720, "native_recorded_cost_seconds": {"mean": null, "median": null, "n": 0}, "score_wall_seconds": {"mean": 2.7552983066448684, "median": 2.7521436028182507, "n": 720}, "source": "S5_r0_fast"}], "full_score_seconds": {"mean": 2.755398533982656, "median": 2.7521218236070126, "n": 3240}, "projected_selected_forwards": 2, "projected_selected_score_seconds": 0.0215265510467395, "projection_caveat": "throughput-scaled estimate, not a measured N1d GPU benchmark"}
```

## N1f — exploratory conditional detection
```json
[{"auroc": {"S": 0.7646198830409356, "S_img": 0.577485380116959, "UnivFD": 0.6451754385964912, "rho_clipped": 0.7381578947368421, "rho_signed": 0.9100877192982456}, "family": "DAPS", "label": "EXPLORATORY_CONDITIONAL_DETECTION", "n": 398, "n_failure": 18, "n_success": 380}, {"auroc": {"S": 0.9032967032967033, "S_img": 0.6865742952699474, "UnivFD": 0.6744386048733875, "rho_clipped": 0.8801242236024844, "rho_signed": 0.8183468705207836}, "family": "FAST", "label": "EXPLORATORY_CONDITIONAL_DETECTION", "n": 478, "n_failure": 23, "n_success": 455}, {"auroc": {"S": 0.8613660297878742, "S_img": 0.7974524848302492, "UnivFD": 0.9565844240509503, "rho_clipped": 0.5762374003309764, "rho_signed": 0.8194423549470939}, "family": "NP", "label": "EXPLORATORY_CONDITIONAL_DETECTION", "n": 884, "n_failure": 102, "n_success": 782}, {"auroc": {"S": null, "S_img": null, "UnivFD": null, "rho_clipped": null, "rho_signed": null}, "family": "NP_LF", "label": "EXPLORATORY_CONDITIONAL_DETECTION", "n": 21, "n_failure": 0, "n_success": 21}, {"auroc": {"S": 0.8695652173913043, "S_img": 0.17391304347826086, "UnivFD": 1.0, "rho_clipped": 0.0, "rho_signed": 0.6521739130434783}, "family": "NP_S2", "label": "EXPLORATORY_CONDITIONAL_DETECTION", "n": 24, "n_failure": 1, "n_success": 23}, {"auroc": {"S": 0.9021598272138229, "S_img": 0.7420926325893928, "UnivFD": 0.8559155267578594, "rho_clipped": 0.8323494120470363, "rho_signed": 0.49007079433645306}, "family": "SITCOM", "label": "EXPLORATORY_CONDITIONAL_DETECTION", "n": 643, "n_failure": 180, "n_success": 463}]
```

## N1g — max-K deployable versus oracle

Each cell: Good25 fraction / mean PSNR dB / q05 PSNR dB.

| Cohort | Source | Family | K | n | O_raw | O_orbit | E0 | E1 |
|---|---|---|---:|---:|---|---|---|---|
| B22_random | S1_b22 | DAPS | 2 | 100 | 0.9200 / 29.42 / 15.88 | 0.9500 / 30.14 / 24.92 | 0.9200 / 29.30 / 15.88 | 0.9500 / 30.07 / 24.81 |
| B22_random | S1_b22 | NP | 8 | 100 | 0.9600 / 29.40 / 25.47 | 0.9600 / 29.40 / 25.47 | 0.9400 / 29.12 / 24.90 | 0.9400 / 29.12 / 24.90 |
| B22_random | S1_b22 | SITCOM | 4 | 100 | 0.9500 / 26.66 / 24.94 | 0.9600 / 26.87 / 25.97 | 0.8900 / 25.57 / 11.10 | 0.9400 / 26.64 / 24.53 |
| B22_random | S3_a12 | DAPS | 4 | 8 | 1.0000 / 31.52 / 30.45 | 1.0000 / 31.52 / 30.45 | 0.8750 / 28.26 / 14.78 | 1.0000 / 31.45 / 30.45 |
| B22_random | S3_a12 | NP_LF | 4 | 8 | 1.0000 / 30.39 / 29.24 | 1.0000 / 30.39 / 29.24 | 1.0000 / 30.37 / 29.24 | 1.0000 / 30.37 / 29.24 |
| B22_random | S3_a12 | NP_S2 | 4 | 8 | 1.0000 / 30.22 / 28.60 | 1.0000 / 30.22 / 28.60 | 1.0000 / 30.20 / 28.59 | 1.0000 / 30.20 / 28.59 |
| B22_random | S3_a12 | SITCOM | 4 | 8 | 1.0000 / 27.44 / 26.97 | 1.0000 / 27.44 / 26.97 | 1.0000 / 27.42 / 26.97 | 1.0000 / 27.42 / 26.97 |
| B22_random | S4_a15 | FAST | 4 | 8 | 1.0000 / 31.53 / 29.07 | 1.0000 / 31.53 / 29.07 | 1.0000 / 31.48 / 29.02 | 1.0000 / 31.48 / 29.02 |
| B22_random | S5_r0_fast | FAST | 4 | 100 | 0.9500 / 30.42 / 25.88 | 0.9500 / 30.42 / 25.88 | 0.9400 / 30.30 / 24.77 | 0.9400 / 30.30 / 24.77 |
| DEV80_enriched | S2_b24_dev80 | DAPS | 4 | 80 | 0.8500 / 28.83 / 18.97 | 0.8625 / 29.03 / 20.07 | 0.8250 / 28.03 / 10.62 | 0.8625 / 28.82 / 18.97 |
| DEV80_enriched | S2_b24_dev80 | NP | 4 | 80 | 0.7125 / 26.07 / 13.25 | 0.7125 / 26.29 / 14.51 | 0.7000 / 25.72 / 11.16 | 0.7000 / 25.86 / 11.98 |
| DEV80_enriched | S2_b24_dev80 | SITCOM | 4 | 80 | 0.6875 / 24.12 / 10.90 | 0.7250 / 25.06 / 15.57 | 0.6875 / 23.65 / 8.41 | 0.7250 / 24.70 / 14.60 |
| DEV80_enriched | S5_r0_fast | FAST | 4 | 80 | 0.7500 / 27.22 / 13.45 | 0.7625 / 27.64 / 13.97 | 0.7500 / 27.05 / 11.36 | 0.7625 / 27.34 / 13.14 |

## Figures
{"fig1": "same frozen stacked label fractions", "fig2": "same frozen cost-success/Good25 harm data", "fig3": "actual raw residual violin from frozen candidates", "formats": ["PNG", "PDF"], "note": "kernel density smoothing affects display only, not numeric evidence", "renderer": "Pillow CPU fallback (matplotlib unavailable)"}

## Handoff
K1 is the B22 post-selection orbit_fixed count per family (N1b).
K2 awaits optional N2 frozen CLIP zero-shot baseline.
No letter kill-criterion decision is made by the executor.
N1d's unselected 60-config grid is EXPLORATORY except for the
pre-registered DEV80 selection and B22 evaluation.
