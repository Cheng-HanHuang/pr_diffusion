# A1.1b integration strategy — one-source, bounded native smoke on procedural observation (NOT LAUNCHED)

**Prepared:** 2026-10-19 by continuing Track A executor. **Prerequisite:** A1.1a proper receipt `A1_1A_PROBE_METADATA_RETURN_V1.zip` independently verified; terminal output alone reported success. **This file is design planning, not a reservation or executable run.**

## Single-source identity

The only admissible first GPU input is the **newly procedural** image and **one** already-frozen CPU-generated native DAPS noisy Fourier magnitude observation from A1.1a under A1-owned `a1_1a_synthetic_probe_v1` directory. It is not an FFHQ sample, not a GT-confirmation pool sample, and **cannot** be used to claim reconstruction-quality generalization. Do not regenerate the measurement for different methods or allow native DAPS to generate its own independent observation. Verify both frozen tensor hashes, raw signed measurement shape/float32 and source HEAD before any candidate.

## Proposed phased execution (each phase needs its own explicit physical-GPU approval)

**Phase A:** one native SITCOM (not four) candidate on the procedural probe; method from frozen `scripts/b22/run_b22_1_sitcom_smoke.py`, one process, fresh per-candidate seed `seed32_numpy("SITCOM",synthetic_probe_token,0)` but use a dedicated synthetic domain (the current `a1_protocol.image_id` only allows five-digit IDs; avoid claiming the dummy token is an actual image ID). Prefer an A1-owned adapter that supplies an `empty` shape-only reference instead of GT to `sampler.get_start`; if historical B22 smoke wrapper reused for initial equivalence, verify its use of GT is **only** shape/reference and offline metrics, never sampler guidance. Measure GPU-active time, wall-clock, peak process VRAM, UNet count, reconstruction finite/shape, measurement raw/used content digest, and frozen source identity.

**Phase B:** one native DAPS candidate on the **same** observation. Native DAPS must be confined to an A1-owned sandbox/writable overlay for its dataset/Hydra config instead of writing temporary records into B19–B26 historical worktrees. Set `B21_START_NOISE_SEED`, `B21_SOURCE_SEED`, and native `args.seed` to the same frozen 32-bit seed, clear continuation env vars, set `B20_LF_ENABLE=0`, `num_runs=1`, native schedule diffusion5 / anneal400. If a symlinked config/dataset in external DAPS repo is unavoidable, log and get separate user approval for that scratch path; do not silently mutate native checked-out sources. Ensure generated observation stays unchanged.

**Phase C:** one NP-LF and one NP-S2 candidate, each with its frozen selector configuration, native DiffFPR checkpoint and **same** saved signed observation. NP input preprocessing `clamp_min(0)` must be recorded as intentional method-specific input modification; compare raw observation hash across all methods but allow differing method-internal representation where historically required. NP configs must be recovered from B22/A1 frozen configuration, with exact scheduler and 1000-step setting, not reconstructed from memory. Record full GPU costs separately per candidate and any shared load cost.

**Phase D:** if all four methods can run one valid candidate on one common observation, repeat exactly one candidate at same seed on same pinned GPU only if needed for deterministic replay; compare tensor-content hashes and full work ledger. Next propose independent K4 restarts only after equivalence/real cost and memory gates pass. No comparison of 4-run success probabilities from one synthetic image.

## Fair metric and GT firewall

One evaluation convention must be frozen and used across all four methods before *any* paper-quality scoring. DAPS native writes 8-bit PNG in frozen code, whereas SITCOM/NP store float tensors. Either capture DAPS prequantization float terminal **noninvasively** or evaluate **all** methods on same 8-bit outputs. Do not compare DAPS PNG PSNR with NP/SITCOM unquantized float PSNR and declare a method winner. Explicitly report unrotated and 180-degree ambiguity-aware metric and threshold, keeping GT oracle for offline comparison only; never use GT during candidate generation, regularization, restart allocation or candidate selection in the proposed deployable policy.

## Governance and STOP conditions

No GPU launch script is committed by this file. A1.1b GPU phases require:
- archived A1.1a true receipt, exact new source HEAD and source/input SHA manifest;
- one A1-owned native execution adapter with audited flags, nonoverwrite scratch paths and static unit tests;
- new recorded `[RUN RESERVATION]` comment in GitHub issue #41 identifying **physical** GPU UUID/index, exact contiguous time window, hard VRAM/process cap, aggregate GPU-seconds cap, stop/fail conditions; Track B has priority;
- explicit bounded PAC command that preserves terminal on failures and a return ZIP containing only requested metadata/logs, not GT/measurement bytes;
- each phase returns a signed machine receipt for independent audit before next phase.

A fully prospective FFHQ quality study separately needs source/provenance-disjoint eligible development images, with no access to the locked B24 confirmation305 set. **No protected metadata may be opened to accomplish the separation.**
