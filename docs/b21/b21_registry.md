# B21+ Method/Policy Registry

Maintained per `docs/b21/b21_master_plan.md` §8. Executor agents append rows and update statuses only per the promotion ladder; every change cites a report file. Do not delete rows — supersede them.

Status values: `idea | specified | smoke-passed | replay-supported | frozen | fresh-validated | adopted | rejected | superseded`.

| id | kind | spec ref | tuned on | validated on | status | key metric | date | report |
|---|---|---|---|---|---|---|---|---|
| DET_P5c50-75keep2 | prefix detector policy | docs/b19/b19_20_*.md | FFHQ25 replay | FFHQ25 fresh (B19.18B), FFHQ100 image-level only after B21.0 G0 fail | fresh-validated | bad25 0/75 @ 3.125 FRE (FFHQ25); B19.20 reclassified as n=100 image-level diagnostic, not n=1000 independent measurement cases | 2026-07-08 | docs/b21/b21_0_measurement_integrity_audit.md |
| DET_P6c100keep2_lhc | prefix detector policy | docs/b19/b19_prefix_policy_findings.md | FFHQ25 replay | B19.16A/B/D | fresh-validated | bad25 0/25 x3 @ 4.0 FRE | 2026-06-23 | b19_prefix_policy_findings |
| SEL_exact_final | final selector | DAPS operator loss | — | many | adopted (default, known rot180-blind) | final-exact failures 13–22/panel, but B19.20 panel is image-level only after B21.0 | 2026-07-08 | docs/b21/b21_0_measurement_integrity_audit.md |
| LF_v1 | guidance patch | docs/b21/patches/daps_b20_lf_guidance.patch; docs/b21/b21_1_lf_patch_capture.md | 00046 (B20.11) | P-HARD4 (B20.12A) | fresh-validated | held-out any-good +13.3 pp in 3-arm portfolio; exact intervention is early measurement-domain LF Fourier-magnitude blend | 2026-07-08 | docs/b21/b21_1_lf_patch_capture.md |
| SCHED_portfolio | schedule portfolio | docs/b19/b20_7_to_b20_9_*.md | 00046 seed bank | — | replay-supported (oracle-style) | 8/8 oracle on tested bank | 2026-06-30 | b20_7_to_b20_9 |
| SEL_v2 | final selector (prior-aware) | b21_experiment_runbooks.md §B21.2 | B19.16A/B/D replay (planned) | B19.20 candidates image-level after G0 fail (planned) | specified | GF: remove >= 80% final-exact failures, median delta >= -0.05 dB | 2026-07-07 | pending |
| CONT_patch | DAPS continuation mechanism | b21_experiment_runbooks.md §B21.3 | — | reproducibility check (planned) | specified | continuation-from-0 reproduces baseline | 2026-07-07 | pending |
| RPOLICY_pilot | branch-vs-fresh reallocation | b21_experiment_runbooks.md §B21.3 | — | 3-image paired panel (planned) | specified | GR: +5 pp any-good or McNemar p<0.05 | 2026-07-07 | pending |
| LF_v2_gs / LF_v2_gated | guidance v2 | b21_experiment_runbooks.md §B21.4 | 00046 seeds 6200–6299 (planned) | P-HARD4 (on GG pass) | specified | GG: rescues >= 0.8x, harms <= 0.4x of lf050 | 2026-07-07 | pending |
| WARM_hio | warm start | b21_experiment_runbooks.md §B21.5 | — | 5-image paired panel (planned) | specified | GW: +10 pp pooled at <= 0.7x cost | 2026-07-07 | pending |
| FORENSICS_attractors | analysis | b21_experiment_runbooks.md §B21.6 | — | — | specified | GX summary table | 2026-07-07 | pending |
