# Track R XP executor return — resource conflict and CPU gate repair

**Branch:** `codex/prdiff-track-r-xp-20261026`
**Planner merged:** `92e4c4d05356fa5bda27273776a2eeaa039c29b4`
**Status:** CPU gates not yet validated on PAC; no new XP GPU jobs launched.

## Operator update, 2026-10-26

The XP-0/R0-N.2 attempt passed the five original exact-estimator unit tests.
The next attempt failed two **new synthetic symlink tests** before reaching B2/A8
or XP-0. The inventory helper still rejected all symlinks unconditionally
(`STOP_SYMLINK`). Fix: permit read-only aliases only when both lexical and
resolved paths avoid prohibited tokens and the real target stays beneath
`/egr/research-pac/huang248`. Re-run all 10 CPU tests before accepting either
CPU receipt. Historical outputs are immutable.

## GPU scheduling clarification from Cheng

On the preceding inventory, GPU memory used (MiB) was 43,370 / 43,312 /
23,708 / 4,156 across the four Blackwell devices. Cheng reports that other
jobs are currently using the GPUs and directs **time-aware XP jobs not to
compete with those jobs**. Reschedule internal XP milestones around available
capacity. Quality-oriented BACKGROUND experiments remain in scope, but must
not preempt existing work or bypass the >2-GiB foreign-process gate without
an explicit operator override. CPU-only A8 and XP-0 are unaffected.
The fixed external CVPR paper/supplement deadlines are not represented as
changed, and all experiment runtime / 400 GPU-hour accounting remains binding.

## Next gate

Commit and push R0-N.2 per-image label counts (2,040 G; 85 orbit;
324 consistent; 791 inconsistent) after CPU replay. Complete the XP-0
measurement-SHA and FFHQ-ID inventory, then implement/validate the full
no-new-code runners and two-lane controller before any XP-T launch.

## XP-0 inventory, first successful PAC pass (2026-10-26)

The PAC executor reports `XP0_RC=0`: `XP_0_NEEDS_IMAGENET`,
180 FFHQ identities and 180 unique measurement SHA files verified, with
oversample 4.0 producing pad 128 and canvas 512. The expected
`test-imagenet` dataset directory is absent (ImageNet manifest count 0);
this blocks ImageNet launches but not the FFHQ input verification.
GPU use remains unchanged (no XP GPU launch).

The initial R0-N.2 B2/A8 full pass reached the aggregation of the
3,240 historical R0.7 records and then raised a Python tuple-unpacking
error. Cause: a previous migration to the 4-key
`(source,cohort,family,category)` stratum retained three-key writes for
other B2 counters. Corrected in the next executor commit; tests for
same-key aggregation and output integrity are added. **No A8 counts
have yet been published**, and the theory executor should await a
successful commit of `R0N2_PER_IMAGE_COUNTS.csv`.

## M1 update — A8 accepted and published

PAC reported 14/14 CPU tests passing, and the R0-N.2 B2/A8 run succeeded:
`R0N2_B2_A8_CPU_PASS`, 760 source/family/image rows and 3,240
historical candidates. Immutable label totals replay as
2,040 GOOD / 85 ORBIT / 324 CONSISTENT / 791 INCONSISTENT.
Committed and pushed on the XP branch at
`83abf2614b30f8e02d63ca15886e6e23c3604187`, with
`R0N2_PER_IMAGE_COUNTS.csv` SHA-256
`197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6`
and B2 membership report SHA-256
`cec48fc01d99eebcb8983f84efd99400fdac8d8cb76aa98a8cbb928086a278da`.

The theory executor can now read the pushed A8 data to fit its
pre-registered FFHQ predictions; this publication is historical
and does not violate the XP run-index embargo.

The CPU-only XP-0 inventory completed:
`XP_0_NEEDS_IMAGENET`, with 180 R0 FFHQ measurements checked for SHA
equality and the exact 512-canvas pad=128 formula verified.
No ImageNet inputs were present at the two inventoried paths.
Next: import only the official DAPS 100-image ImageNet test archive
(File ID `1pqVO-LYrRRL4bVxUidvy-Eb22edpuFCs`), validate every ZIP member
before extraction, then repeat XP-0 against the new asset.
The R0-N.2 A4.3 historical protocol ladder is also queued CPU-only,
without any new inference.

## M1 update: R0-N.2 A4.3 and ImageNet archive correction (2026-10-26)

PAC returned 23/23 CPU unit tests passing and
`R0N2_A43_CPU_PASS`: 14 strata and 224 aggregated policy-ladder rows.
Ladder CSV SHA-256:
`2797220bb489bba620392435480996b663901373081331937907b138e96d0115`.
The four-file return is committed to the XP executor branch at
`b4ca7fe65ca78ca664e92dc11582cb7ee4c0994a`, under
`docs/track_r/executors/r0n/returns/R0N2_A43_20261026T193644Z/`.

Historical cohort-wise anatomy from this frozen analysis, *not* new XP outcomes:
- S1 B22 DAPS (100 images, K2): 95 selected repaired Good25, 5 generation failures;
  SITCOM (100, K4): 94 selected Good25, 4 generation failures, 2 selection misses;
  NP (100, K8): 95 selected Good25, 4 generation failures, 1 selection miss.
- S2 DEV80 DAPS (80, K4): 68 selected Good25, 11 generation failures,
  1 orientation miss; SITCOM: 58 selected Good25, 22 generation failures;
  NP: 56 selected Good25, 23 generation failures, 1 selection miss.
- R0 FAST pooled (180, K4): 155 selected Good25, 24 generation failures,
  1 selection miss. These use the GT-free CLIP orientation selector on
  the E0-residual chosen reconstruction and a GT-only label for analysis.

**ImageNet source mismatch identified.** The official Google Drive archive
for the DAPS `test-imagenet` images
(`1pqVO-LYrRRL4bVxUidvy-Eb22edpuFCs`) was successfully downloaded
once to
`outputs/pr_diffusion/track_r/xp_imagenet_download_20261026T193644Z/test-imagenet.zip`.
Its ZIP member audit found **1,000 ImageNet images**, although DAPS README
describes a supplied subset of 100. The importer correctly refused its
previous exact-100 assumption without extracting any images. The XP
preregistration *already* specifies indices 0 through 99 of the files
sorted in DAPS loader order; the subsequent guarded importer implements
this first 100 of the 1,000, storing original archive SHA, total image
count, canonical relative paths, selected file SHAs, and a manifest.
The importer does **not** assert that this first-100 sample equals the
published DAPS paper's exact evaluation sample; note the distinction when
comparing reported publication PSNRs.

No extra archive download, dataset-bundle FFHQ import, new GPU work, or
time-aware experiment is authorized during this recovery.

## M1 — XP-0 successfully closed; GPU co-tenancy amendment

The PAC CPU return confirms:
- `TEST_RC=0` (26 synthetic tests);
- `XP_IMAGENET100_IMPORTED_PASS`, first 100 of the official
  1,000-image DAPS archive in sorted-loader order, archive SHA-256
  `34af8a05cc5d603465ced5c7b0af97ecd152fc7074dd3519afac4765e0fab5bd`;
- `XP_0_PASS`, 180/180 FFHQ SHA identities plus ImageNet 100/100,
  exact 128-pixel pad and 512 canvas; import manifest crosscheck 100/100;
- commit `6f81d1ed4368d2229e49711d677698fcf5a19e0e` pushed with
  ImageNet file SHA manifest. No GPU inference has run.

Cheng authorizes GPU co-tenancy while other jobs are active. An
otherwise eligible quality run may launch on a device with foreign processes.
Record foreign PIDs, current free/used memory at every launch and repeated
intervals, and tag timing `CONTAMINATED` or `UNVERIFIED` if a foreign
process is seen or process visibility is insufficient. Quality results
remain eligible if all scientific receipts and replay gates pass. A timing
benchmark on a contaminated GPU **must be rerun later**, without presenting
that duration as a valid performance comparison. When no device can hold
the next job while maintaining the 80% device memory ceiling, enter an
automatically polled `WAIT_VRAM` state and resume when feasible. Other
running jobs must never be preempted. Preserve the 400 GPU-hour cap.

The executor implementation of this scheduling amendment is in
`docs/track_r/executors/xp/GPU_COTENANCY_POLICY_20261026.md`.
GPU admission and controller code must pass CPU unit tests and are
**not yet** an accepted XP-T or exchangeability/replay gate; no
scientific execution is triggered by installing these files.

## XP-T scientific entry gate: native FAST historical replay (prepared, not yet run)

The first GPU operation on this branch is a bounded native FAST-DIPS
replay of eight frozen roots from R0.5: the first two sorted R0 image IDs
from B22 and DEV80, with the original R0 `rep = 0,1` seeds and
byte-identical signed measurement inputs. The **generation manifest is
GT-free** and is regenerated from the immutable, SHA-pinned R0.5 720-root
manifest. The single-GPU worker uses the existing pinned FAST-DIPS
source, official environment and R0.5 FFHQ pixel settings.

The gate includes:
1. CPU manifest replay + source/checkpoint/measurement SHA validation
   before model inference;
2. one 8-root native GPU canary on a safe-memory GPU, capturing
   start/during/finish occupancy and process telemetry;
3. isolated GT-accessing **CPU-only offline evaluation** comparing
   each raw PSNR to the frozen R0.7 entry (absolute delta <= 0.1 dB)
   and reproducing its exact failure label;
4. PASS receipt only if all eight replayed roots meet those criteria.
   The new candidate artifacts stay in the Track R XP output root,
   not in existing R0 outputs. A failed gate stops FAST extension
   pending diagnosis; it never auto-continues into K-scaling.

The first canary is not an XP-T full multi-worker timing sweep, and
the corresponding time measurement is only diagnostic. If foreign GPU
processes overlap, mark its timing contaminated without invalidating
an otherwise successful scientific replay. The approved scheduler's
80% VRAM and 400 GPU-hour caps apply. No published XP outcomes are
produced until the replay PASS is verified on PAC.

## Native FAST replay CPU scoring repair (2026-10-26)

The bounded replay launched under commit
`314180b314674a24d5ddc40a0186b24b6082a5e0`
and completed 8/8 native FAST-DIPS roots successfully on GPU 0.
The run is immutable at
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_fast_replay_20261026T203415Z`.
PAC reported `worker_returncode=0`, `R05_CHUNK_PASS`,
8 new roots, zero reused, 29.359 seconds native worker allocated time,
and approximately 0.01991 GPU-hours including model startup.

The subsequent **CPU-only scorer** exited 8 before comparison,
reporting `KeyError: 'measurement_sha256'`. This is a schema-join bug:
the R0.7 evaluated GT JSONL contains labels, PSNR, run index, image,
source and candidate/GT paths but **does not** carry the measurement SHA.
The frozen R0.7 offline catalog carries both measurement path and SHA,
and was itself SHA-verified in XP-0. The original scorer incorrectly
accessed `prev["measurement_sha256"]`; this is not a scientific replay
mismatch. No per-run comparison receipt was written; this replay
remains scientifically unaccepted until CPU scoring succeeds.

The repaired scorer uses explicit `validate_historical_identity` to
cross-check the scored JSONL, offline catalog, frozen source manifest,
run index, candidate path, GT path, measurement path and digest; it
**rehashes** the measurement before comparing the eight reconstructions.
It distinguishes the **generation head** (the already completed
`314180b...` GPU commit) from the newer CPU scoring head.
New regression tests verify that omission of the scored-row
`measurement_sha256` is permitted **only when** the catalog identity
and hash join passes. The repair runs in `CUDA_VISIBLE_DEVICES=""`,
writes a new, auditable recovery receipt instead of modifying the
original `REPLAY_GATE_FAILED` supervisor status, and publishes
a return **only if all eight** historical PSNRs replay to within 0.1 dB
and the R0.7 labels match exactly. Failed science is a STOP; no
GPU rerun or XP K-scaling is automatically authorized.

## M1 PASS — Native FAST replay + bounded XP-T FAST calibration launch design

On 2026-10-26, the CPU recovery of the native FAST-DIPS replay passed:
`XP_FAST_REPLAY_RECOVERY_PASS` with **8/8** historical raw-PSNR
reconstructions matching within `1.9032504212646018e-06` dB, identical
canonical PNG SHA for **8/8**, and matching `GOOD/ORBIT/CONSISTENT/`
R0.7 labels on **8/8**. No GPU reruns were needed. The replay timing was
`CLEAN`, 71.6706793 s allocation elapsed / 0.01990852 GPU-hours.
Return published at GitHub commit
`eb7d5b27c003f6ee9b64db690939414bf03613e8` in
`docs/track_r/executors/xp/returns/XP_FAST_REPLAY_RECOVER_20261026T204735Z/`.
No outcome embargo was breached; the historical runs only were replayed.

**Next authorized step: XP-T FAST FFHQ batch-1 subset.**
The new `xp_t_fast_prepare_cpu.py` freezes an independently seeded
nine-root GT-free timing manifest using the first three sorted B22 image
IDs (3 runs each, original SHA-pinned y). The frozen XP seed rule is
`SHA256('XP|FAST|ffhq|B22_<id>|<rep>')[:8] mod 2^31`.
The official FAST-DIPS source/model/settings and measurement SHA are checked
in a **preflight-only** worker before CUDA is allocated. One-GPU
`xp_t_fast_supervisor.py` records initial/heartbeat/final GPU occupancy,
foreign PIDs, process wall, per-root native and CUDA-event timings,
sampled peak VRAM and 80%-cap compliance. It uses opportunistic
co-tenancy and automatic WAIT_VRAM polling. Nine valid root receipts
and `R0_5_CHUNK_PASS` are required for `XP_T_FAST_FFHQ_BATCH1_PASS`.

**Scientific limitations:** This is **only FAST/FFHQ batch-1 calibration**,
not the full XP-T prereg suite. SITCOM, DAPS-4K, ImageNet, W=1/2/4
throughput, and scoring costs remain pending. No GT is used during
generation; new XP candidate outcome labels are not evaluated, committed,
or reported. Timing from a contaminated GPU must be marked unsuitable
for publication and rerun later. The 400 GPU-hour cap and original
reconstruction run embargo remain unchanged.

## XP-T FAST/FFHQ measured PASS and ImageNet timing pilot (2026-10-26)

PAC completed the bounded FAST FFHQ batch-1 timing pilot under commit
`3e202508dbe2b9dc16ac4ef2f2bd38ee3a1e29a3`:
9/9 successful independent roots, first three sorted B22 image IDs,
`XP_T_FAST_FFHQ_BATCH1_PASS`, median native sample wall
**3.3207844938 s**, median CUDA-event sample **3.3207717285 s**,
whole allocation **61.34695235 s = 0.01704082010 GPU-h**, GPU 0,
**CLEAN** (no foreign PIDs detected at 8 sample points). Receipt SHA-256
`378366c0958238ca832ae29b5f66a8e5d406dcaae8cec401be2dfb9bc17aba4a`
at `outputs/pr_diffusion/track_r/xp_t_fast_20261026T205902Z/XP_T_FAST_COMPLETE.json`.
The sampled GPU-memory peak, 1,603 MiB, is an observation only; it is
not a verified allocation upper bound and **must not** be used by itself
to choose W*. Nine-root calibration is approximately 30 s of sampling
plus model initialization; compare published throughput with individual
CUDA-event sample timings, not end-to-end amortized time.

**Next task:** CPU-only preparation of all 100 common ImageNet signed
measurements, then a FAST-DIPS batch-1 ImageNet pilot (first three
images, 3 independent XP seeds each). This is one of XP-T's required
dataset-family calibrations. It is not historical ImageNet outcome
generation. The preprocessing is the official pinned FAST-DIPS data.py
`ToTensor -> Resize(256) -> CenterCrop(256)`. CPU records new
`y=A(x)+0.05*n`, each with seed
`SHA256('XPMEAS|imagenet|<basename>')[:8]` unsigned mod (2^{63}),
signed-y SHA and physical RMS check. The manifest sent to the GPU has
only y paths/SHAs, seed, uid, run number, and ImageNet model SHA,
**no GT path**. A separate GT-only offline record is never used by
generation. Source files are exactly the 100 selected by XP-0 from
the 1000-image official archive, not a claim to reproduce the paper's
exact original selected 100. FAST sampling parameters are the already
frozen R0.5 native pixel settings and the pinned ImageNet checkpoint.

The XP-T pilot retains per-run CUDA and wall times, 80%-VRAM admission
and foreign-PID telemetry. No new ImageNet success labels or K-curves
are scored. Do not merge with a new planner revision without reviewing
changes; planner head remained
`92e4c4d05356fa5bda27273776a2eeaa039c29b4`
at the time of this preparation.


## ImageNet XP-T FAST completed; post-hoc telemetry accounting audit (2026-10-26)

Under commit \`b3d67d55a569e04f9467dc5dc4c7e796fab1a5f3\`,
the CPU preparation generated the 100 **shared, signed ImageNet phase-retrieval
measurements** using the selected official DAPS ImageNet set, the pinned
FAST-DIPS loader's \`ToTensor -> Resize(256) -> CenterCrop(256)\`,
and independent per-image \`XPMEAS\` seeds. The GT-free shared-y manifest
SHA-256 is
\`26619959e3a17aeb2c0ad413f87b1764ecd8bc7aabfc0415c862cbaf27873b78\`;
the 9-root timing task manifest SHA-256 is
\`2ea2c16a24ecebc3dd62fe852982a807a555bedb8d0afda2c49bb73722b39e61\`.
The official image/checkpoint hashes were verified. Those 100 measurements
are the common input to subsequent FAST, SITCOM and DAPS experiments;
never regenerate y independently for another family.

Native FAST/ImageNet XP-T batch-1 **PASS**: 9/9 distinct new roots;
median native sample wall **7.19089462189 s**, CUDA-event
**7.19086669922 s**, worker elapsed **112.51600675 s =
0.03125444632 GPU-h**, GPU 0. Original completion receipt SHA-256
\`ca66eb9cfcd84d8b567ff407f35e9027520d977d4f1963ea85a9ae94ce7e820d\`.
The original timing monitor classified it \`UNVERIFIED\` because one
HEARTBEAT observed **4,012 MiB GPU memory, utilization 0, and no compute
PID**, after six active heartbeats showed the intended worker PID
\`3833824\`, 4,023 MiB and 81–84% utilization, and before POST_GPU
reported zero memory/processes. No foreign PID was observed anywhere
in 13 snapshots. This sequence is consistent with **post-exit GPU
memory accounting lag**, not proof of another process interfering.

The historical FFHQ XP-T timing remained \`CLEAN\`: 8 snapshots,
four with the owned worker visible under load, and zero foreign PIDs.
The ImageNet and FFHQ GT-free reconstruction receipts both remain
fully valid and immutable regardless of timing classification.

A new CPU-only \`xp_t_timing_reaudit_cpu.py\` audits both exact frozen
runs and **never modifies original status, event, or completion files**.
It only upgrades publication *eligibility* to
\`CLEAN_POSTHOC\` if (a) the trailing orphan-memory sample is unique,
(b) it occurs after all observed owned computation, (c) utilization
is zero, (d) its memory does not exceed the preceding owned peak,
(e) the subsequent POST_GPU sample is idle, (f) no foreign PID or
unknown process visibility is observed, and (g) the original completion
SHA and every saved status identity match. It otherwise stops.
The original label remains \`UNVERIFIED\`. Even an accepted post-hoc
audit cannot exclude short external activity between sampling times;
a prospectively clean repeat remains the best publication timing.

**Next work:** finish the remaining XP-T family/dataset calibrations
(SITCOM and DAPS-4K especially), the W=1,2,4 FAST/SITCOM concurrency
checks, and scoring cost. Then populate and activate the guarded
CRITICAL/BACKGROUND controller. The data cut is November 2 for Tier A.
Only exact GT-free shared measurement manifests may be used by GPU
generators. The original theory prediction embargo remains binding.

## XP-T post-hoc timing audit hotfix (2026-10-26)

The initial CPU-only re-audit at executor commit `68531c40f5c9abeb48419c968e799c940846ada4` STOPPED with `STOP_XPT_AUDIT_TOO_FEW_ACTIVE` before writing any re-audit receipt or publication. Root cause: the audit reused the telemetry classifier's **2,048 MiB threshold for unattributed memory** when identifying *owned active sampling*. The FFHQ FAST native worker's observed working set was only **1,603 MiB**, despite its tracked PID and 65–70% GPU utilization. This created a false negative on FFHQ, not a scientific or timing-source failure. The corrected audit accepts owned active snapshots with tracked worker PID, positive GPU utilization and >512 MiB observed memory; it retains the original >2,048 MiB threshold for unattributed-process memory and otherwise unchanged strict post-exit checks. New synthetic regression tests mirror the observed FFHQ 8-snapshot sequence and reject an idle process with memory but no utilization. No prior timing record, measurement, reconstruction, or original status has been modified. An audit PASS is not claimed until the corrected PAC CPU execution finishes.

## XP-T CPU timing re-audit provenance merge hotfix (2026-10-26)

The second audit attempt passed its 55 CPU tests, and the Python traceback reached the audit entrypoint after the FFHQ helper classified its saved GPU telemetry. It then stopped on an implementation error: `result.update("source_run", name, ...)` passed twelve positional arguments instead of a single dictionary. No audit receipt or publication was created. The revised entrypoint uses `result.update({...})` with explicit named provenance fields and now has a synthetic end-to-end test of both FFHQ and ImageNet saved receipts, their SHA256 identity checks, the qualification output, JSON writing, and source-file immutability. This repair does not touch either original GPU run, their output files or original timing labels. The CPU audit remains unaccepted until PAC reports a new `XP_T_FAST_TIMING_REAUDIT_PASS`.

## XP-SITCOM replay source preflight and manifest preparation (2026-10-26)

PAC read-only provenance preflight showed native SITCOM source `275ab67efbd8146bffca20155171ba6be1169c09` exactly pinned. The frozen R0.7 catalog has 400 B22 SITCOM candidates (100 images) and 320 B24 DEV80 SITCOM candidates (80 images). The first two sorted IDs per source, candidate runs 0 and 1, are 8 exact historical candidates with matching measurement and candidate SHA256. B22 image IDs `60044` and `60140` both use master seed 43, **sequential run-index RNG draws after model loading** and a reset of the same post-model state per image. DEV80 IDs `01113` and `02763` use **independent 32-bit native seeds and separate model loading for each run**; never conflate the two lineages. The pinned SITCOM source's `DAPS.get_start(ref)` is `torch.randn_like(ref) * annealing_sigma_max`, so a shape-only tensor can replace the historical GT reference for a strictly GT-free generation adapter, provided replay verifies equivalence.

Frozen historic labels of selected B22 (60044 run0/1, 60140 run0/1): GOOD, GOOD, ORBIT_ERROR, GOOD. Selected DEV80 (01113 run0/1, 02763 run0/1): all GOOD. No GPU replay has been authorized or executed by this CPU preflight. `xp_sitcom_replay_prepare_cpu.py` creates a separate GT-free eight-row GPU manifest and CPU-only historical reference (candidate file SHA, labels, PSNR) after checking historic y, candidates, B22 native receipts, native solver settings, source and model SHA, and B24 native frozen runtime configurations. This is **only a manifest provenance step**. The next executor step must implement a GT-free SITCOM generator, compare saved historic and new reconstructions offline, and withhold any large-scale SITCOM work until the replay gate passes.

## SITCOM replay integrity gate: tracked Python cache drift (2026-10-26)

PAC diagnostic confirms external SITCOM checkout HEAD
`275ab67efbd8146bffca20155171ba6be1169c09`,
unchanged `sampler.py`, `model.py`, and `forward_operator.py`,
and both copied FFHQ checkpoints SHA-256
`81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`.
Its tracked working tree is modified in **exactly ten** `__pycache__/*.cpython-38.pyc`
cache binaries and **one** `posterior_sample.py` CLI one-line substitution.
These local modifications are not grounds to reset the original external source.

The first XP-SITCOM CPU manifest attempt under
`b2f2fa791b5c6eec4c26af7d2e42612484514620`
correctly STOPPED at the blanket dirty-tree gate
`STOP_XP_SITCOM_NATIVE_SOURCE`, before generating any manifest or
spending GPU time. Follow-up code substitutes a *narrowly allowed
tracked-diff provenance policy* for that blanket check: the native
source must remain on the pinned HEAD, only the eleven exact known paths
may be modified, the unused `posterior_sample.py` edit must have
precisely one insertion and one deletion, and every changed source
under `sampler`, `model`, and `forward_operator` still causes STOP.
The non-imported CLI's current SHA256 and full allowed-dirty list
are recorded in both frozen manifest and CPU receipt.

**Important for subsequent GPU replay:** use pinned
`get_sampler`, `get_model`, and `get_operator` APIs directly;
never execute or import modified `posterior_sample.py`. Set a
fresh `PYTHONPYCACHEPREFIX` under the PAC output/cache tree before
importing any SITCOM Python code, so the tracked Python 3.8 cache
objects are not read; do not delete or reset those files.
`PYTHONDONTWRITEBYTECODE=1` alone does not prevent Python from
*reading* an existing bytecode cache. Replay and timings remain
unverified until separately completed. New synthetic regression tests
reject changed solver source, staged dirty files, changed/unknown
cache paths, and altered CLI diff shape.
