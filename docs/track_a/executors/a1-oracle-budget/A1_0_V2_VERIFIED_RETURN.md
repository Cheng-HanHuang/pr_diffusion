# A1.0 V2 — VERIFIED PASS, source and exposed-metadata preflight

**Date:** 2026-10-17. **Executor:** continuing Track A executor. **Disposition:** `PASS_SOURCE_AND_EXPOSED_METADATA_PREFLIGHT` — **only** source identities/checkpoint hashes/exposed metadata; no prospective cohort or GPU execution clearance.

## User PAC receipt (independently verified)

- Executed source commit: `0062498b3393dd7e8946cf254fe1ea541244fbac`.
- Received file: `A1_0_PREFLIGHT_RETURN_V2.zip`, ZIP SHA-256 **`a0d984981838007f67a1f33ae7f19691d80088ce3af0cf2825270f7d356c8f6e`**, exact match to PAC terminal.
- Three exact members: `A1_0_SYNTHETIC_TESTS_V2.log` SHA `551d9da3819e3fb1c147b431b3ba5e7d440c581969152ba21138a6e2889b3e8f`; `A1_0_PREFLIGHT_V2.log` SHA `a2f988df89168ab7915697af804986ea9640982b5e8d9700e95b4f0df17b74e8`; `a1_0_preflight_20261017_v2/A1_0_PREFLIGHT.json` SHA `f28045d6ec0d8ca2a57a0f5d769202295292f81197d5cb22b6ef7c8f7adc856a`. Valid ZIP and JSON.
- **9/9** synthetic tests PASS; `A1_SYNTHETIC_TEST_RC=0`, `A1_PREFLIGHT_RC=0`, `A1_FINAL_STATUS=0`, `A1_V2_RUNNER_RC=0`. JSON status `PASS_SOURCE_AND_EXPOSED_METADATA_PREFLIGHT`; zero issue entries.
- Source snapshots: DAPS HEAD `e7a77d094167084faed19b599b96673b7bb11447` with expected tracked diff SHA `fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69` (source `posterior_sample.py` and `sampler.py`); DiffFPR HEAD `a45ffe58f18fed8a63d3446600424e2b08733524` clean; SITCOM HEAD `275ab67efbd8146bffca20155171ba6be1169c09` with the B22 verified one-line mkdir patch. **Do not claim archived B24 full tracked diff identity is certified**.
- Checkpoint reconciliation: `/egr/research-pac/huang248/external/SITCOM_ODE/checkpoint/ffhq256.pt` is confirmed **symbolic link** whose link text and resolved path both identify `/egr/research-pac/huang248/models/ffhq_10m.pt`. Both named checkpoint routes have `size_bytes=374417833`, SHA-256 **`81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`**, matching frozen. The A1 V1 source/checkpoint STOP was an overly strict checker, not a scientific source drift.
- Metadata SHA checks PASS: exposed B21 Fresh2 CSV, exposed B22 frozen 100 manifest, exposed B24 **DEV80** manifest, previously verified A0 B22 capsule. B21/B22 panel 100, B24 DEV80 panel 80, overlap 0, union **180 already-exposed IDs**. Nothing here certifies a new pristine cohort. **No B24 confirmation305 membership or data accessed.**
- Three absolute conda interpreter paths exist under `/egr/research-pac/huang248/conda-envs`; FFHQ public image directory exists but not enumerated. GPU snapshot: indices 0–3, 4× `NVIDIA RTX PRO 6000 Blackwell Server Edition`, `memory.total=97887` MiB each, `memory.used=0` and `utilization=0` at one instant. GPU UUIDs were recorded in the original preflight JSON; **no GPU allocation/reservation**.
- Machine flags `gpu_work_performed=false`, `inference_performed=false`, `measurements_created=false`, `candidate_images_selected=false`, `locked_confirmation_access=false`; `cohort_status=NOT_RESERVED`, `run_status=NOT_RESERVED`.

## Reconciled status and next stage

**A1.0 source/exposed-metadata preflight is complete.** The first V1 stop and V2 retest artifacts both remain preserved in the dedicated A1 output namespace. A0 B22 forensic work remains frozen. No old source/historical artifacts were changed.

**A1.0→A1.1 transition is NOT an automatic approval to run a 128-trajectory GPU pilot.** Remaining meaningful technical gates:

1. A1-owned and tested fresh independent-restarter wrappers: existing B24 source `scripts/b24/run_b24_1_method_smoke.py` is useful *read-only precedent* (native DAPS and SITCOM each execute separate solver processes with independent seeds), but the B24 study and its outputs remain frozen. Build A1 wrappers rather than invoking/patching B24 production history.
2. New eligible DEV cohort and measurement namespace must be accepted and registered in live issue #41 **before** measurement generation. Avoid exposed B21/B22 100 and B24 DEV80 80. Because the protected confirmation305 registry is excluded, simple exclusion of the 180 exposed IDs **does not certify** absence from the protected pool. Find a provenance-disjoint allowed split/source (or a governance-authorized safe allocation procedure that proves disjointness without disclosing the locked membership); do not hash blindly over all FFHQ IDs and call it fresh.
3. A1 source-preflight of parent native invocation, SEED domain handling (DAPS/SITCOM legacy uint32), metric representation and same signed measurement; a zero-GPU synthetic CLI/run-plan test, ensuring no candidate is silently repeated or extra work hidden in mixing.
4. Only after those gates and Track B resource priority: log explicit `[DATA RESERVATION]` and `[RUN RESERVATION]` physical GPU UUID/time/VRAM envelope; present pinned bounded PAC launch commands to user. No automatic launch from A1.0 PASS.

**Scientific caution:** source-verified SITCOM B22 sequential RNG K4 is not four independently seeded native runs, even though the source/checkpoint are identified. Next study compares true 4-run native baselines and NP LF4/S2-only4 on the *same new noisy observation*; both run-matched and total charged candidate-time comparisons remain separate.

**Decision:** `GO_A1_INDEPENDENT_RESTART_ENGINEERING_CPU_ONLY`; prospective pilot `NOT_READY`.
