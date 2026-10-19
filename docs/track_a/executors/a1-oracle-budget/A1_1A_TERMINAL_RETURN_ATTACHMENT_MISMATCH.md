# A1.1a PAC return — terminal PASS; correct A1.1a archive still requested

**Date:** 2026-10-19. **Owner:** continuing Track A executor. **Disposition:** `PAC_TERMINAL_PASS_ARCHIVE_VERIFICATION_PENDING`, **NOT** `VERIFIED_ARCHIVE_PASS`.

## Source and returned terminal record

User executed A1 dedicated checkout `/egr/research-pac/huang248/pr_diffusion_track_a_a1_oracle_budget` at pinned head `0b94d455f5d82389a400313776bea411c3a488f8`. Checkout was fast-forwarded from `ca17ef8113588ff1b909f0b648d440740c1ab13b` without an error.

Reported machine records:
- `A1_1A_TEST_RC=0`, `A1_1A_MEASUREMENT_RC=0`, `A1_1A_PACK_RC=0`.
- `A1_1A_FINAL_STATUS=0`, `A1_1A_RUNNER_RC=0`.
- Native operator status `A1_1A_SYNTHETIC_NATIVE_MEASUREMENT_PASS`, DAPS frozen HEAD `e7a77d094167084faed19b599b96673b7bb11447`.
- One procedurally created model-range RGB tensor, hash `6eda5926b1cc56e23cdfed5fd70b743bdfe61067c955f778900683a42a6a1123`.
- Native Fourier noisy measurement shape `[1,3,384,384]`, content hash `992a33efa9bfb8e81d97ed61cf28369263ae1f11aeae19273c53973c2470acdb`.
- Report says `gpu_used=false`, `ffhq_accessed=false`. This is a **synthetic engineering probe**, not a new FFHQ study image or diffusion/reconstruction run.
- ZIP returned by PAC at `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/A1_1A_PROBE_METADATA_RETURN_V1.zip`, terminal SHA-256 **`a2f61731638602f38563502c0c86e3d0896fa9a95f8fbf6fbb7937c507bc290f`**.

## Critical attachment discrepancy (independently verified)

The conversation attachment presented for this run is actually `A1_0D_RESTART_SYNTHETIC_RETURN(1).zip`. Directly opened the uploaded file at its provided sandbox backing path, observed ZIP SHA-256 **`88060ce25af74360bb6284d94b472848409d85e70fcbcd77a0d77a458ba170b5`**. Its two exact members are the *old* `A1_0D_RESTART_SYNTHETIC_TESTS.log` (14/14 tests) and `A1_0D_RESTART_SYNTHETIC_SUMMARY.json`. It is a valid duplicate A1.0d receipt, **not** the A1.1a native-measurement return.

Therefore A1.1a is currently **terminal-reported PASS**, but the proper A1.1a ZIP (with its manifest, test log and native measurement log) has **not been received or independently verified**. Request the exact PAC path `A1_1A_PROBE_METADATA_RETURN_V1.zip` above. Do not claim the attached A1.0d ZIP proves A1.1a. Preserve both old archives; no rerun indicated by the terminal log.

## Engineering decision

A1.1a's output is sufficient to justify planning the subsequent pipeline-integration stage (native DAPS/SITCOM/NP with *only* the one procedural probe, no paper-quality metric), but **not** to launch any GPU work. Need the true archived manifest/checksum, A1-owned source-pinned native wrappers/metric conventions, and one explicit approved `[RUN RESERVATION]` including physical GPU UUID, time and VRAM cap with Track B's priority. The full prospective FFHQ experiment still needs a provenance-disjoint approved development cohort; the procedural probe does not establish such a cohort.

Stage: `A1_1A_TERMINAL_PASS_AWAIT_CORRECT_ZIP`. No protected B24 confirmation305 access.
