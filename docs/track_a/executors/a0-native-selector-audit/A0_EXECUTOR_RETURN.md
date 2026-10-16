# A0_EXECUTOR_RETURN — **source-gated STOP / analysis not yet executed**
Date: 2026-10-15. Track A A0 archival-metadata executor.

**Review status:** `A0.0_RECEIPT_VALIDATED; B24_SITCOM_SOURCE_UNRECONCILED; A0.1_NOT_EXECUTED`. This report is a conservative **interim STOP return**, not a claim that A0's historical candidate comparison is complete. Track A scientific lead should not use missing oracle/native cells as performance evidence.

## 1. Repository and scope
- Only working branch: `codex/prdiff-track-a-a0-native-selector-audit`. Frozen delegation parent: `c61c5029d27220f34ad6b4b93f1b2c7f5a2767ba`; the first A0 pure-selector commit: `5f7ea5eedff19a0b98b4a3fe0135416a9ac02f71`. GitHub issue [#41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41) records delegation and status. Current return lives on this branch, not a historical B21–B26 branch.
- PAC inventory: user-executed, `A0_SCRIPT_RC=0`, no GPU or measurements. ZIP SHA-256 `77342e210f70580f4148860fc2de6bb89441aa92935296ab9cf337d7bccb07bf`.
- PAC output: `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a0_native_selector_audit_20261015/` (existing owned inventory). No other PAC output or new worktree created by A0.

## 2. Frozen source identity and stop
- B22 validated run `B22_2_overnight_20260727_155135`, run `manifest.json` SHA `94184e36...`, validation SHA `8016bb8d...`, source head `191dbd70894b24f0fafa160c08b17d7ab47a17f1`, DiffFPR `a45ffe58f18fed8a63d3446600424e2b08733524`, SITCOM `275ab67efbd8146bffca20155171ba6be1169c09`.
- B24 DEV80 manifest SHA `3945c2ed26e10766a64ab4e7fb64ea673fa2f478fbf080ea8d1d2c85584f6d5f`. External DAPS HEAD `e7a77d094167084faed19b599b96673b7bb11447`; its tracked diff matches frozen `fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69`.
- **Blocking mismatch:** external SITCOM full tracked diff `0a5587945acbbb86da46ca8e49996bb79ff4cf142c821f268515777ad379ab2a` vs B24 frozen `a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e`. Tracked `.pyc` modifications could explain part of it. No source-only diff reconciliation or original native model checkpoint SHA match has been established. Never overwrite/clean historical source to make it match.

## 3. Candidate/seed coverage, field and representation audit
| Exposed cohort | Recorded population | Native selector availability | Offline oracle availability | Identity/metric caveat |
|---|---|---|---|---|
| B21/B22 final100 | DAPS 100 × 2 | Fresh2 K2 exact signed operator loss margin 0.7 | K1/K2 metadata | B21 exact losses and per-arm wall; B21 measurement content hash absent from CSV |
| B22 final100 | SITCOM 100 × 4, sequential RNG master43 | Step160 correction norm K4 | K1–K4 | Native checkpoint hash unverified, though B22 candidate schema confirms step160 field |
| B22 final100 | NP 100 × (4 LF + 4 S2) | LF K1–4; heterogeneous native K8 | LF/S2 K1–4 and mixed K8 | NP8 is not IID eight-restarter policy; signed terminal residual not proven stored |
| B24 DEV80 | DAPS 80 × 4 | Fresh2 exact rep0/1 K2; **no native K4** | K1–K4 | External 8-bit RGB metric |
| B24 DEV80 | SITCOM 80 × 4 | **No documented B22-style step160 native K4** | K1–K4 | Native source unreconciled; external 8-bit RGB metric |
| B24 DEV80 | NP 80 × 4 independent LF | K1–4 postprojection LF score | K1–4 | Native float/tensor PSNR distinct from external PNG quantized PSNR |

B24 DEV80 inventory has 80 complete-image PASS receipts, 320 DAPS terminals, 320 SITCOM terminals, 320 NP roots, 16 Pilot16 reuse and 20 cases per historical screening stratum. The initial B22 inventory counted 100 policy records per family and inspected first/last candidate groups; it **did not** exhaustively certify 1,200 B22 SITCOM/NP candidate files or all finite values. B21 contributes the separate 200 DAPS records.

## 4. Historical headline vs new A0 computation
The following are already published historical records and are **not new A0 replays**:

| Cohort | Native clean-free result | GT candidate ceiling | Availability |
|---|---|---|---|
| B22 DAPS K2 | Good25 **92/100** | K2 Good25 **92/100** | Historical B21 closeout |
| B22 SITCOM K4 | **93/100** | **95/100** | Historical B22 closeout |
| B22 NP mixed K8 | **95/100** | **96/100** | Historical B22 closeout |
| B24 DAPS Fresh2 K2 | **57/80** | DAPS2 oracle **59/80** | Corrected closeout |
| B24 NP4 | **55/80** | **57/80** | B24/B25 reports |
| B24 DAPS native K4 | **UNAVAILABLE** | DAPS4 PSNR oracle logged | No executable native K4 |
| B24 SITCOM native K4 | **UNAVAILABLE** | SITCOM4 PSNR oracle logged | No step160 feature |

No new candidate-level comparison table, independently recomputed mean regret, threshold discordance, bootstrap CI or McNemar result is claimed from the A0.0 inventory. Some historical B22 results include a NP8-vs-Fresh2 McNemar p=0.5488 and historical costs, but these are not freshly replayed A0 evidence.

## 5. Code, tests and future analysis gate
Scoped, branch-only code added: `scripts/track_a/a0_selectors.py`, `scripts/track_a/test_a0_selectors.py`, `scripts/track_a/a0_source_reconcile.py`, `scripts/track_a/a0_historical_compare.py`; frozen `configs/track_a/a0_analysis_spec.json`; reviewed compact source/field manifest in this executor directory.

The pure selector module freezes stable-first GT PSNR oracle, exact DAPS margin, B22 step160 SITCOM, NP LF/S2 lineage, unavailable-policy cells, paired image bootstrap and exact McNemar. Synthetic tests are authored but **not independently executed in this environment**. The PAC extractor is an implementation scaffold **not yet executed against actual candidate files** and may require schema repairs after supervised smoke. It fails closed when the SITCOM tracked source-only SHA or distinct SITCOM checkpoint SHA is missing, all candidate/provenance/metric gates drift, or historical Good25 replay differs. It does not import torch, PIL, models, FFTs, or raw images.

**No A0_POLICY_ROWS.csv / A0_CANDIDATE_ROWS.csv / A0_METRIC_SUMMARY.csv / A0_PAIRED.csv is currently a completed scientific output.** The source-only reconciliation test and model hash precede even a CPU metadata smoke.

## 6. Compute, safety and scientific recommendation
Historic B22 mean charged workload (not remeasured A0): Fresh2 approx 312.6 candidate GPU-s/image, SITCOM4S 196.1, NP8 411.8. B24 NP4 8,800 UNet evaluations/image, DAPS/SITCOM per-candidate GPU-active seconds and group concurrent wall time available. B24 dispatch-supported FLOPs exclude unaccounted FFT/custom work. Never interpolate these numbers into exact cross-family FLOPs or claim compute matching from K alone.

No GPU, dataset/model execution, measurement creation, terminal re-evaluation, operator/FFT or confirmation305 access occurred. No historic source/output modifications; Track B and B26 untouched.

**GO/NO-GO:** **NO-GO for interpreting new historical native/oracle PSNR statistics now; CONDITIONAL GO for one bounded, read-only SITCOM source-only-diff and model-identity check.** If source cannot be reconciled, freeze a missing-data report rather than fabricate a SITCOM native selector. After proven source identity, the Track A lead may authorize CPU metadata-only replay from the new scripts with zero GPU. A0 alone grants no permission to claim prospective NP superiority or to launch P2.

## 7. Receipts and files
- Raw PAC original: `A0_SOURCE_AND_ARTIFACT_MANIFEST.{json,md}`, `A0_FULL_STDOUT_STDERR.log`, `SHA256SUMS.txt`, `A0_INVENTORY_RETURN.zip` in the existing exclusive A0 output tree.
- Reviewed compact JSON/MD: under `docs/track_a/executors/a0-native-selector-audit/` in this branch.
- Code/config/tests: `scripts/track_a/`, `configs/track_a/`.
- Next action: run one bounded **read-only** source-only diff + checkpoint hash audit, then review its actual stdout. Register any final A0 replay return commit in issue #41 and notify the Track A lead; do not open B24 confirmation305.

## Addendum — SITCOM source recheck on 2026-10-15

The user returned `A0_SITCOM_SOURCE_RECHECK.json` (SHA `2c811247ae8cd077aac57ab23ce19f219ef3169a2b69cb04943ffbdccffa16ed`). It **verified** the current SITCOM native checkpoint SHA-256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa` (374417833 bytes), so the initial model-SHA unknown cell is resolved for the currently accessible file. The current SITCOM HEAD is the frozen pinned upstream commit; there is one modified tracked Python file `posterior_sample.py` and ten modified tracked bytecode files.

**Correction to original source STOP reasoning:** Comparing the current source-only diff SHA `d0bb0292...` to the B24 historical **full** tracked diff SHA `a9f0076d...` was invalid because the underlying streams differ. It does **not** prove Python source mismatch. Conversely the bytecode modifications alone do **not** prove unchanged B24 Python source. A bounded direct comparison to the archived B22 `posterior_sample.py` patch is the next step; B24 historical source identity remains *unreconciled* without an independent B24 patch/file hash witness. The A0 source tool and analysis gate have been fixed accordingly. [Assessment](A0_SITCOM_RECHECK_ASSESSMENT.md).

**Updated status:** `MODEL_CURRENT_HASH_VERIFIED / HISTORICAL_B24_SITCOM_SOURCE_UNCERTAIN / A0.1_NOT_RUN`. Neither a source PASS nor new candidate-level scientific result is claimed.

## Addendum — B22 source patch reconciled; B22-only CPU replay path (2026-10-16)

The user returned bounded `A0_SITCOM_PATCH_AUDIT.json`, showing a raw patch digest mismatch versus B22 (`d0bb...` vs `b1d...`). We compared its uploaded `current_patch` with `B22_FULL.identities.sitcom_posterior_sample_diff` in the earlier PAC inventory ZIP. The **actual source diff is identical**: `Path(dir).mkdir()` is replaced with `Path(dir).mkdir(parents=True, exist_ok=True)`; the only textual mismatch is **two trailing unchanged blank-context lines** in the current Git diff. Using `.rstrip("\\n ")` makes the strings equal. See [A0_SITCOM_PATCH_AUDIT_FINDING.md](A0_SITCOM_PATCH_AUDIT_FINDING.md).

This reconciles **B22 historical change-level** code identity (pinned HEAD, exact one-line changed content, verified current model SHA), not an independent proof of B24's historical full tracked diff. **B24 remains SOURCE_UNCERTAIN**. We have now staged `scripts/track_a/run_a0_b22_pac.sh`, which runs synthetic tests and the B22-only frozen candidate-level CPU metadata replay. No PAC tests, historical replay results, confidence intervals or candidate-level tables are claimed yet. The user must execute this explicit script and return its actual logs/ZIP for validation. It cannot read confirmation305 or run image/model code. Its output is exclusively under the existing A0-owned folder.

Updated scientific status: `B22_SOURCE_PATCH_RECONCILED / B22_ANALYSIS_PENDING_PAC / B24_SOURCE_UNCERTAIN`. The earlier generic A0.1 STOP does not bar the now separately justified B22-only retrospective audit.

## Addendum — synthetic B22 selector tests stopped before any candidate replay (2026-10-16)

PAC cloned branch at `cee6664cb9f15100f5ca9bea64cbc0bfa0b180b6` and executed `scripts/track_a/run_a0_b22_pac.sh`. Actual terminal receipt: `A0_SYNTHETIC_TEST_RC=1`; **six of eight synthetic tests passed**; the SITCOM K4 and NP K8 native selector tests failed. No historical candidate analysis started; no A0 new scientific result was generated.

**Root cause is a genuine comparator defect** in A0's `best()`: the minimizer applied `min` to already negated scores, thereby selecting the **maximum** original score. This inverted SITCOM's minimum step160 correction norm and NP's minimum post-projection LF score. This was a correctness fault in this executor's new audit code, not evidence of a defect in the archived SITCOM/NP experiments.

**Committed repair:** use a single `max` over `value` for maximum PSNR and `-value` for minimum native score, with `-index` for earliest-candidate ties. Added tests covering monotonically rising/falling scores, negative scores and stable ties. Strengthened B22 audit to compare every archived B22 winner's native policy identity, oracle policy identity and PSNR, in addition to headline Good25. Changed PAC runner to **V2 distinct output/log/archive paths** so initial failed `A0_B22_SYNTHETIC_TESTS.log` is preserved, and to return nonzero exit status on STOP (the original runner mistakenly returned 0 due to its last `if` shell construct).

**Current status:** `B22_SOURCE_PATCH_RECONCILED / B22_REGRESSION_FIX_COMMITTED / B22_ANALYSIS_PENDING_V2_PAC`; `B24_SOURCE_UNCERTAIN` unchanged. The fixed Python code has **not yet passed user-run PAC tests**; archive verdict remains no new results until actual V2 outcome. All code confined to Track A branch and owned directories.
