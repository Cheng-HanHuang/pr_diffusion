# R0-N execution preparation: merged planner and N0/N1 CPU implementation

Date: 2026-10-23 EDT.

## Source / merge provenance

New branch `codex/prdiff-track-r-r0n-20261023` was created from
`c8eb60877fadeb1b9c7be63b0ff389f7681e7107` (accepted R0
scientific executor return) and merged with planner commit
`6686322b919e891b5dd37d14ce5dddfa96a97dcd`.

Actual two-parent merge commit:
`1936e79acd4040f67e754276415bdac503b1121c`.
The merge tree changed only the planner's 7 intended paths; the
previously frozen R0 v1 files and other executor files remain intact.

Planner decision:
`R0_ACCEPTED / TRACK_R_METHOD_PAPER_STOP / NO_R1 /
R0N_LETTER_EVIDENCE_AUTHORIZED / PROJECT_WIND_DOWN_AFTER_LETTER`.

R0-N is an **evaluation-letter package**, not a new solver method.
No new reconstructions or measurements, no R1, no B24 protected
confirmation305 access. N2 CLIP baseline is optional with at most
0.25 aggregate GPUh. N3 asset inventory is bounded and CPU-only.

## N0 — original R0.8 aggregate tables

User-verified R0.8 ZIP available on PAC:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_8_overnight_20261023T082909Z/R0_8_PLANNER_RETURN.zip`.

SHA256:
`13c008f3bca59b7d8d717ec3215fa1f783ca453b31b9001886c84633b513efab`.

`scripts/track_r/r0n_0_import_tables_cpu.py` checks the whole
ZIP SHA and seven per-member SHAs, rejects binary or per-run GT or
candidate paths, then stages these exact bytes under
`docs/track_r/executors/r0/returns/R0_20261023/`:

- R0_7_ANALYSIS.json
- R0_7_REPORT.md
- R0_8_P4_MATRIX.csv
- R0_8_PROVISIONAL_DECISIONS.json
- R0_8_READINESS.json
- R0_8_EXPLORATORY_CASEBOOK.json
- R0_8_RESEARCH_AUDIT.md
- SHA256SUMS.txt

The importer intentionally omits the offline GT catalog, evaluated
per-run JSONL and all individual GT/candidate paths. PAC user must
commit/push the seven aggregate files because the exact ZIP resides
on PAC. The importer is an integrity-controlled transfer, not a
recomputation. The N0 worktree must be clean BEFORE importing.

## N1 — CPU-only letter tables

`prdiffusion/track_r/r0n_letter.py` holds pure scientific
definitions, `scripts/track_r/r0n_1_letter_tables_cpu.py`
runs the complete frozen N1; `scripts/track_r/r0n_1_cpu_block.sh`
is a fail-closed launcher. Source:
`configs/track_r/r0n_letter_analysis_v1.json` (unchanged).

N1 starts with the frozen K3 headline replay:
3,240 total candidates; 2,040 GOOD / 85 ORBIT /
324 consistent / 791 inconsistent; 84/85 prior repairs and
0/2,040 Good25 harm; exactly seven pre-declared P4 Good25 strata.
Any mismatch stops with `STOP_R0N_K3_...`.

Then:
- N1a: image-cohort × family anatomy with image-cluster 10k CIs;
- N1b: exhaustive partition of first-K E0 failures into E1-fixed,
  orbit errors not fixed, selection misses, generation failures,
  plus O_raw, O_orbit, E0/E1/E2 policy statistics;
- N1c: signed and clipped residual distribution quantiles for each
  cohort/family/raw label;
- N1d: exactly 60 saved-loss cost×accuracy configurations
  (10 T sets ×3 J ×2 orbit sets) using original paired draws.
  Choose one configuration on DEV80 only via frozen success, harm,
  compute and tie rules; evaluate it on B22 with 10k image-cluster
  CIs. Missing qualified DEV80 configuration returns an explicit
  unavailability, not a retrospectively chosen B22 winner.
  Computes per-time paired-noise variance ratios for GOOD and
  ORBIT_ERROR with ddof=1;
- N1e: native FAST generation CUDA/wall timing, scorer wall/forwards,
  recorded native solver cost or N/A, projected selected cost;
- N1f: exploratory consistent-failure detection AUROC all families;
- N1g: max-K letter Table 2 by cohort/source/family with
  Good25, mean PSNR and q05 for GT raw/orbit ceilings and
  deployable E0/E1.

Outputs one JSON, one CSV per table, numbers-only MD report,
and three draft figures as PNG/PDF. CPU only, no new model forwards.

**N1 is not yet executed.** The PAC loader verifies the committed N0
table bytes before any experiment; the full tests/track_r suite must
pass on the exact new pinned HEAD. Run N1 only after N0 import,
SHA256SUMS verification, commit and push.

## Next gating

- K1: report exact B22 post-selection fixes for each family from N1b.
- K2: evaluate separately with optional frozen N2 CLIP zero-shot run.
- N3: bounded ImageNet inventory only.
- Planner—not executor—chooses letter go/no-go after N1+N2.
- No R1 is authorized.

Do not reinterpret N1's DEV80 exploratory grid as a primary
confirmatory result: only the preselected DEV80 configuration
evaluated on B22 follows the frozen dev/eval gate.
