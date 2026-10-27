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

## XP-SITCOM strict checkpoint symlink exemption (2026-10-26)

The retry at executor commit `f0b41af6ed404f5d949f4141522ffe7d556859ed` passed 62 CPU tests and the narrow external checkout integrity gate, then STOPPED before any output was created: `safe(SITCOM/checkpoint/ffhq256.pt)` rejects symlink paths. PAC independently confirmed that both the external checkpoint path and the canonical `models/ffhq_10m.pt` path hash to `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`. The cause is therefore the path-safety policy, not an observed checksum mismatch; the external path may be a symlink. The updated CPU-only preparation exempts **only** the exact external SITCOM checkpoint path, resolves symlinks strictly within the PAC research root, forbids protected target substrings, requires an ordinary regular target file and its exact SHA, and records the resolved target and symlink status in both the GT-free manifest and its receipt. All ordinary candidate and measurement inputs continue to reject symbolic links. Three new synthetic tests cover pinned checkpoint symlink success, direct-file success, outside-root link rejection, and SHA drift rejection. This update does not change any original experiment, external code, or reconstruction, and replay remains pending.

## SITCOM 8-root native GT-free replay implementation (pending PAC execution)

After the successful CPU source/measurement gate under commit
`c055423234441fe907fef8dfda825815632ca15f`, PAC prepared
`xp_sitcom_replay_20261026T223925Z`:
`XP_SITCOM_REPLAY_PREP_PASS`, eight roots, four FFHQ images,
GT-free manifest SHA-256
`82d38a2cadc47c8ad864e97809e9cea9baafd62c261fa26d50fc1d332d48d525`.
The selected B22 images are 60044 and 60140, with runs 0/1 from
a **single per-image reset post-model RNG stream** (seed 43); DEV80
images 01113 and 02763 replay runs 0/1 as **four independently seeded
native model processes**. Historical candidate tensors/PNGs, R0.7
labels, and GT paths remain in a separate CPU-only manifest.

`xp_sitcom_replay_gpu.py` is a GT-free wrapper that calls the pinned
SITCOM `get_operator`, `get_sampler` and `get_model` APIs and
preserves the exact historical construction/seed order. The original
`sampler.get_start(gt)` call is replaced only by a shape/device-matched
empty float32 tensor because the verified native method
`get_start(ref) = torch.randn_like(ref) * sigma_max` has no dependence
on reference pixel values. B22's historical `record=True` and
step-160 lightweight selector are retained; B24's one-run
`record=False` is retained. No GT file or historical candidate path
is read by GPU generation. Each completed root persists float32
reconstruction, RGB8 PNG, SHA-256 hashes, and wall/CUDA timing.

`xp_sitcom_replay_supervisor.py` sequentially dispatches one four-run
B22 process and four independent one-run DEV80 processes; it enforces
clean GPU admission, an 80%-VRAM utilization budget, a 2.0 GPU-hour
execution cap, a stop sentinel, worker-local termination only, and
per-sample GPU occupancy telemetry. Any partial/mismatching original
or new receipt fails closed, with outputs preserved. It invokes
`xp_sitcom_replay_score_cpu.py` in a separate subprocess with
`CUDA_VISIBLE_DEVICES=""` only after all GPU workers succeed.
The scorer verifies SHA-locked candidates and shared y again, computes
raw RGB8 PSNR and frozen R0.7 labels using offline GT, compares each
to the historical score (|delta| <= 0.1 dB plus exact label match),
and separately records exact tensor/PNG equality.

**None of these replay execution or scientific PASS statuses should be
claimed until PAC returns the supervisor and offline receipts.** Never
reset the external SITCOM checkout or touch protected cohort paths.
The worker sets a fresh Python bytecode cache prefix so it cannot
load the ten changed tracked Python 3.8 .pyc files. The edited native
`posterior_sample.py` CLI is never imported or executed.

## XP-SITCOM replay gate: scientific PASS and timing CLEAN (PAC 2026-10-26)

PAC returned the **final**, independently computed GPU and offline receipts for
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_sitcom_replay_20261026T223925Z`.
The supervisor state is `PASS`, `reason=null`, `completed_roots=8`,
`gpu_hours=0.10778908291249537`.
`XP_SITCOM_REPLAY_GPU_SUMMARY.json` is
`XP_SITCOM_REPLAY_GENERATION_PASS`, `n_roots=8`, median native
sampling wall time **31.592317841015756 s**. GPU telemetry was **CLEAN**
(48 samples, foreign PIDs `[]`, reasons `[]`, `timing_valid=true`).
The five native generator groups completed:
- B22: four runs in one seeded, post-model sequential RNG stream,
  `XP_SITCOM_WORKER_PASS`, worker wall 114.73747853701934 s
  (supervised group elapsed 153.11 s including wrapper/admission/observation overhead).
- DEV80: four independent seeded native workers, each one candidate,
  wall 31.76293940236792, 37.070267603732646,
  33.42947893915698, 33.10923477308825 s
  (supervised groups 51.05, 71.50, 61.35, 51.03 s).

The fully separated CPU-only historical equivalence gate
`XP_SITCOM_REPLAY_OFFLINE_SCORE.json` reports **`XP_SITCOM_REPLAY_PASS`**,
`n_roots=8`, `n_label_match=8`, `n_exact_historical=8`,
maximum |raw PSNR difference| **1.971301713865614e-06 dB**.
All four B22 and four DEV80 candidates are exact (native B22
float32 tensor hash equality, native DEV80 PNG SHA256 equality).
B22 image 60140 run0 remains the originally observed
`ORBIT_ERROR`; the other seven remain `GOOD`.
No GT data reached generation; offline GT was used solely for scoring.
Original candidate/measurement/receipt files were never overwritten.

**Evidence SHA256 (verbatim PAC receipt values):**
- frozen GT-free manifest: `82d38a2cadc47c8ad864e97809e9cea9baafd62c261fa26d50fc1d332d48d525`
- GPU summary: `f96f534a820c6c1f59c6e8f63b5eebd91a3dda5b2aec299a6d18410563d816e2`
- offline science: `d59c10a34b1b1986d6e78d86d125306f0b88862122123a20740d0941c6035d4f`
The CPU publisher `xp_sitcom_replay_publish_cpu.py` validates exact hashes
and success fields, including the 48-sample clean timing proof, then stages
only the non-GT final receipts and their SHA-256 sidecar for publication. It
is a separate operation from the already completed native replay.

### Immediate execution priorities under frozen XP prereg v1.1

1. **Close the SITCOM FFHQ exchangeability/replay gate:** PASS, exact 8/8.
   No further SITCOM historical replay GPU work is needed.
2. **XP-T:** native SITCOM on three FFHQ and three ImageNet images,
   three new timing candidates per (family,dataset), using strictly
   source-matched y. ImageNet uses pinned ImageNet checkpoint with the
   FFHQ phase-retrieval settings: prereg A4 established the native
   repository has no separate ImageNet PR hyperparameters.
   **One GPU W=1 baseline is required for per-candidate timing**;
   W={1,2,4} co-tenancy on *one GPU* (FAST and SITCOM, FFHQ) is a
   separate prereg throughput comparison. Four physical GPUs are then
   available for **data parallelism across independent tasks**.
   Do not conflate those distinct experiments or claim that 4 devices
   substitute for the W throughput study.
3. **CRITICAL lane and data cuts:** prioritize Tier A XP-A16, XP-C,
   XP-C2 and XP-D1 before Nov 2; then Tier B XP-B16, XP-D2,
   XP-D3, XP-A64 and XP-F8 before Nov 6. The four-slot scheduler
   admits CRITICAL work first; background work uses genuinely idle
   slots and is never allowed to interrupt active jobs.
4. **Freeze protection:** do not expose FFHQ K>=4 labels or image-level
   outcome data to the prediction planner before the TH1 v1 freeze is
   registered. For SITCOM B22 new runs use the same historical
   post-model RNG stream, continuing sequential stream indices 4–15.
   For DEV80 use independent native seeds, with collision checks
   against the original four. The seven- or nine-root calibration
   cases are not silently merged with historical K populations.
5. **DAPS-4K:** do not substitute DAPS-1K. Published nonlinear PR
   setting is 400 annealing × 10 diffusion, 100 Langevin steps per
   iteration at the preregistered parameters; perform its own native
   timing/replay gates before scaling.

Reconstruction equivalence **does not itself certify timing
comparability across all five historical and future execution modes**.
The clean 8-root result establishes source-faithful replay and provides
one measured SITCOM FFHQ reference. Dedicated XP-T W=1/W=2/W=4
benchmarks are still required.

## XP-T SITCOM batch-1 dual-dataset input freeze (CPU implementation, execution pending)

SITCOM reproduction and timing evidence have been published to the executor
branch at `26204137db23fa80a281e744719450f09ae62ac4`. The PAC commit
contains full `XP_SITCOM_REPLAY_STATUS.json`,
`XP_SITCOM_REPLAY_GPU_SUMMARY.json` and
`XP_SITCOM_REPLAY_OFFLINE_SCORE.json` (plus SHA256SUMS):
**8/8 exactly reproduced native reconstructions, 8/8 historical
labels, max PSNR difference 1.9713e-6 dB; 48 telemetry samples,
CLEAN; 0.1077891 GPU-hours.**

The next ready implementation
`scripts/track_r/xp_t_sitcom_prepare_cpu.py` validates and freezes
**18 new independent SITCOM XP-T batch-1 timing roots**
(three runs per each of the first three sorted FFHQ B22 images and first
three sorted ImageNet benchmark basenames). It reuses only the exact
signed-magnitude measurements from the original frozen FAST XP-T
FFHQ and ImageNet pilot manifests, verifying both source pilot
manifests, each y-file SHA, and the 100-image common ImageNet
measurement manifest SHA
`26619959e3a17aeb2c0ad413f87b1764ecd8bc7aabfc0415c862cbaf27873b78`.
The GPU-safe output contains no GT or historical candidate paths.
These new timing seeds use
`SHA256("XP|SITCOM|dataset|uid|rep")[:8] mod 2^31` and are
**timing-only runs, not silently appended to the B22 or DEV80
historical populations**. Each timing row references exactly the
same y as the corresponding FAST calibration.

Model profiles are pinned separately:
FFHQ DDPM has 128 channels, one residual block and attention
resolution 16 with `ffhq_10m.pt` SHA
`81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`;
ImageNet DDPM has 256 channels, two residual blocks and attention
resolutions `32,16,8`, using the pinned
`models/imagenet256.pt` SHA
`a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1`.
SITCOM's *phase retrieval* sampler configuration remains the frozen
B22 200-step/5-ODE/100-LGVD settings on both datasets, consistent
with XP prereg amendment A4's ImageNet fallback. Native model
loading must be **fail-closed**, since the external loader catches
checkpoint exceptions and otherwise silently uses randomly initialized
weights. The next GPU implementation must validate the native model
preset/checkpoint tensor shapes and require a successful
`load_state_dict` before sampling.

Only the CPU manifest preparation is implemented in this commit.
**Do not claim SITCOM FFHQ or ImageNet XP-T timing calibration PASS
until separate GT-free GPU generation and clean timing receipts exist.**
The eventual multi-GPU experiment may distribute independent roots
over four physical GPUs, but the scientific per-candidate timing
reference uses W=1, and the preregistered one-GPU W={1,2,4}
worker-throughput test remains a distinct experiment.

## XP-T SITCOM FFHQ+ImageNet: 4-GPU W1 calibration implementation (execution pending)

PAC completed CPU stage `XP_T_SITCOM_PREP_PASS`, 77 XP CPU tests,
9 native timing candidates for 3 frozen FFHQ B22 images and 9 for the
first 3 frozen ImageNet-100 benchmark images. Source directory is
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_t_sitcom_20261026T233451Z`.
The immutable GT-free input manifest SHA256 is
`995915c3b7b7a00b1d4d678a4a013c8cc9c2445a513704a92e513d93d6d97131`.
It reuses the very same noisy signed y used by the earlier FFHQ and
ImageNet FAST calibration, and leaves their original source records
untouched.

The new `xp_t_sitcom_worker_gpu.py` and
`xp_t_sitcom_supervisor.py` implement a prospective calibration on
four physical PAC GPUs, **W=1 native process per GPU**, with up to four
concurrent independent SITCOM roots. Each candidate is run in its own
native SITCOM Python 3.8 process with seed set *before* the model
construction, as in validated B24 independent-seed mode. ImageNet uses
the exact pinned SITCOM `imagenet256ddpm.yaml` architecture
(256 channels, 2 residual blocks, attention `32,16,8`) and the
pinned ImageNet checkpoint. FFHQ retains its native
`ffhq256ddpm.yaml` architecture. Both use the frozen B22 200×5
phase-retrieval solver configuration (no ImageNet-specific PR
hyperparameters are published).

**Checkpoint load fail-closed:** the upstream native
`model/ddpm/unet.py` catches `torch.load`/`load_state_dict`
exceptions, prints a warning, and otherwise silently returns a
randomly initialized UNet. This would invalidate ImageNet timing
and quality. The worker first SHA256-verifies the entire checkpoint,
then temporarily instruments `torch.nn.Module.load_state_dict`
only during native `get_model` construction and requires one
successful strict `UNetModel.load_state_dict` call, zero missing
and unexpected parameters, and at least 100 loaded tensors; otherwise
**STOP** before any sampling or receipt. The native code is never
modified. The worker records its successful strict loading proof,
model-config and exact checkpoint hash in every GT-free candidate
receipt.

The supervisor performs clean NVIDIA-GPU admission for each worker,
one worker per GPU, a hard 80% VRAM ceiling, a conservative 2 GPU-hour
aggregate cap, and no killing foreign processes. It preserves any
partial results if a task fails. Per-root timing uses CUDA events
and synchronized process wall from `get_start` through native
`sampler.sample`; model load time is separately reported. Full
GPU-event telemetry is recorded across all used physical GPUs and
co-tenancy/visibility qualify timing independently of generation
quality. The worker records reconstruction PNG/float32 tensor and
SHA256, but no GT evaluation or historical output labels; timing
calibration roots are **never appended** to historical K populations.
This is *not* the separate preregistered W={1,2,4} simultaneous-worker
throughput experiment on one physical GPU.

No new GPU results exist when authoring this implementation. Do not
claim `XP_T_SITCOM_BATCH1_PASS` until PAC reports the exact native
18-root completion receipt. Preserve all existing B22, DEV80,
FAST and SITCOM replay outputs. Do not inspect held-out cohorts,
and do not release embargoed K>=4 outcome data.

## XP-T SITCOM initial four-GPU supervisor pre-GPU failure and exact recovery

The first XP-T SITCOM W1/4-physical-GPU launch under
`709cc92fc757283f0205e585939704afd4197746` passed
82 XP CPU tests and three manifest-only native preflights (indices 0,9,17).
The parent launcher returned `XPT_SITCOM_SUBMITTED` and supervisor
PID `3893497`. The supervisor immediately raised
`TypeError: dump() missing 1 required positional argument: 'fp'`
in `dump_x`, at the first PID JSON serialization, before any worker
start. PAC reported zero GPU memory/utilization on all four GPUs,
and both `XP_T_SITCOM_STATUS.json` and
`XP_T_SITCOM_COMPLETE.json` were absent.

The failure is exclusively implementation, not a native-SITCOM
scientific failure. The failed supervisor did create `gpu/` and
`pycache/` and a zero-byte `XP_T_SITCOM_SUPERVISOR_PID.json`
before crashing; these **must never be deleted, truncated or
rewritten** to permit a normal launch. The new repair fixes the
missing file handle, adds a regression test verifying exclusive
JSON writing and atomic status updates, and supplies a separate,
single-use CPU recovery audit:

- Verify the immutable 18-root manifest SHA256
  `995915c3b7b7a00b1d4d678a4a013c8cc9c2445a513704a92e513d93d6d97131`
  and complete CPU preparation receipt; require exact old error in the
  preserved supervisor log and zero-byte original PID receipt.
- Require original supervisor PID `3893497` no longer running; zero
  candidate content in `gpu/`, empty `pycache/`, and **no** root
  receipts, GPU telemetry, or status receipts.
- If any unexpected artifact exists, STOP with no modification;
  do not reset external Git source or any run.
- On exact clean recovery audit, create a unique
  `XP_T_SITCOM_PRE_GPU_RECOVERY.json` one-time authorization
  recording the original log/PID hashes. The supervisor launched with
  `--recover` rechecks the same evidence and uses a NEW
  `XP_T_SITCOM_RECOVERY_SUPERVISOR.log` and
  `XP_T_SITCOM_RECOVERY_SUPERVISOR_PID.json` rather than overwriting
  the failed launch. The frozen 18 roots and GT-free common-y SHA
  are unchanged.
- Four-GPU W1 native sampling and CPU-only quality embargo remain
  subject to every earlier memory/timing/model-load/provenance gate.

This recovery has NOT been executed yet. A CPU audit failure must be
reported without attempting manual removal of the original artifacts.

## XP-T SITCOM FFHQ+ImageNet native batch-1: completed 4-GPU calibration (2026-10-26 PAC)

The exact fail-closed pre-GPU recovery was executed after the original
supervisor JSON-handle bug. Under committed recovery code
`dacd57d0739ee218490befde50b2bed5991bb80b`,
PAC returned 86 passing CPU tests,
`XP_T_SITCOM_PRE_GPU_RECOVERY_AUTHORIZED`, and supervisor PID 3899197.
Original failed supervisor log SHA
`daf0a649bec6ce971702890663b28c61d7b55003bf735a7a0d8eb1c2a1a56208`
and zero-byte original PID SHA
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
remain preserved.

**Scientific/GPU completion:** 18/18 GT-free native SITCOM sampling roots
completed on four RTX PRO 6000 Blackwell physical GPUs (one native
worker per GPU, W=1). The final PAC supervisor explicitly reports
`XP_T_SITCOM_BATCH1_PASS`, `timing_clean=True`,
aggregate **0.36916 GPU-hours**, and median native W=1 sampling times
**32.104 s (FFHQ)** and **58.343 s (ImageNet)**.
All 18 strict ImageNet/FFHQ checkpoint-load and per-root SHA gates
passed the supervisor. Full final completion receipt SHA-256:
`bb2c3c377f07be2982e412eb96ff1a8b35805146dc021cc8baab3abd6ddc075e`.
Frozen timing manifest SHA-256:
`995915c3b7b7a00b1d4d678a4a013c8cc9c2445a513704a92e513d93d6d97131`.
PAC supervisor log contains all indices 0–17
`XP_T_SITCOM_WORKER_PASS` and the complete W=1 summary.
No GT was loaded for generation or scored here. The exact 8/8
historical SITCOM replay remains a *separate* earlier success.

**Timing interpretation:** the prior XP-T FAST pilots report 3.321 s
FFHQ (original CLEAN) and 7.191 s ImageNet (original UNVERIFIED,
qualified separately CLEAN_POSTHOC). These are small matched-y pilots
and should be reported as runtime observations, not claims of
distribution-level speed dominance. They do not replace the
preregistered W={1,2,4} simultaneous-worker throughput study on one
GPU, which remains pending. Both SITCOM datasets use the same
phase-retrieval hyperparameters; ImageNet loads the distinct pinned
256-channel/two-residual-block network, with strict checkpoint-load
evidence.

**Publication action:** CPU-only `xp_t_sitcom_publish_cpu.py`
independently verifies the frozen manifest and completion exact SHA,
all four GPU timing classifications, all 18 native receipt hashes and
strict model-load proofs, zero GT scoring, and consistent budget and
dataset medians before copying only the final receipts into GitHub.
It must be run on PAC at a clean checkout before publication.
Do not claim publication occurred until PAC prints a pushed commit.

Next highest-priority calibration is the **published DAPS-4K**
configuration (nonlinear PR: 400 annealing, 10 diffusion ODE,
100 Langevin steps per denoising iteration), on exactly the same
frozen signed-y inputs, with no independently regenerated y and
no GT during generation. Follow the frozen XP prereg and critical
Nov 2/Nov 6 data cuts. The four physical GPUs may parallelize
independent sampling tasks; W=1 is the timing baseline.

## Published SITCOM calibration and DAPS-4K follow-on input gate

PAC successfully published the complete SITCOM FFHQ+ImageNet XP-T W=1
timing calibration at executor commit
`142eeac9d5c73bdc13aedbcc9af21756a94dca05`.
The CPU evidence gate passed 90 regression tests and validated
all 18 native roots on the four physical GPUs, with final original PAC
receipt SHA256
`bb2c3c377f07be2982e412eb96ff1a8b35805146dc021cc8baab3abd6ddc075e`.
The precise FFHQ sampling median is
**32.104482267051935 s**, ImageNet **58.342772816773504 s**;
total 0.36916114673742817 GPU-h; timing CLEAN on all four GPUs.
All evidence files are committed to
`docs/track_r/executors/xp/returns/XP_T_SITCOM_BATCH1_20261026T233451Z/`.
Historical SITCOM reproduction (8/8 exact) remains separately published.

**New CPU preparation, not yet executed:** the
`xp_t_daps4k_prepare_cpu.py` stage freezes a separate matched-y
18-root timing pilot for the published DAPS-4K nonlinear PR solver,
three independent new seeds each on the original first three sorted
FFHQ B22 images and the first three ImageNet images. DAPS remains
pinned to the historical B24 checkout
`/egr/research-pac/huang248/pr_diffusion_b19_solver/external/daps`
HEAD `e7a77d094167084faed19b599b96673b7bb11447`.
The CPU preflight verifies exact tracked Git blobs for the sampler,
MCMC, scheduler, Fourier operator, DDPM model, two architecture YAMLs,
and phase-retrieval sampler/task YAMLs; disallows any drift from pinned
HEAD in those files, without resetting modified Python cache binaries.
No bundled FFHQ data or historical candidate file may be read.

**Published nonlinear PR DAPS-4K, not repository 1K commands:**
annealing 400 with sigma 100 to 0.1, poly-7;
diffusion ODE 10 with sigma_min 0.01, poly-7;
pixel Langevin 100 steps, lr 5e-5, tau 0.01,
lr_min_ratio 0.01, gaussian prior;
phase-retrieval oversample 2.0, measurement sigma 0.05;
batch size one, no trajectory. FFHQ uses canonical
`ffhq_10m.pt` SHA256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`;
ImageNet uses the canonical `imagenet256.pt` SHA256
`a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1`.
The CPU manifest references only GT-free, SHA-locked signed y files from
FAST/SITCOM XP-T; it never regenerates y. DAPS runs are new
calibration seeds and never merged with historical K populations.
The next GPU implementation must verify native strict weight loading,
a native W=1 baseline on each physical GPU, clean telemetry, and
per-(dataset,root) file hashes. It must first pass the CPU manifest
check and native-source environment audit.

**Outstanding scientific priorities:** finish XP-T DAPS-4K timing,
one-GPU W={1,2,4} throughput experiments for FAST/SITCOM,
then Tier A critical XP-A16, XP-C, XP-C2, XP-D1 before Nov 2 and
Tier B DAPS/SITCOM/FAST extensions before Nov 6. Do not expose
embargoed K>=4 outcome metrics until frozen theory predictions are
registered; historical holdout remains excluded.

## DAPS-4K integrity gate: B20/B21 modified source; pristine sparse recovery

PAC attempted the native DAPS-4K CPU preparation at
`a83cdf7827c454405ddb9686240de70d4aa546b0`.
All **95** XP CPU tests passed, but the preparation stopped before
freezing a manifest with
`STOP_XPT_DAPS_TRACKED_SOURCE_DRIFT sampler.py`.
The historical DAPS source is at correct pinned Git HEAD
`e7a77d094167084faed19b599b96673b7bb11447`, but the working
`sampler.py` is modified: pinned Git blob
`0e0d1bf9d7893278c8d5589583ba1d3f10bcd972`, local working
blob `ae1fc6908bb1e7f0c0349d4f82cdf20364389db8`, with diff
SHA256 `fdf6d4ae31267cd0833b42c97edfd5a778b6a2b2c0b6a3387d66139283d876a0`
(314 added, 2 deleted lines). The changes contain B20.11
environment-controlled low-frequency Fourier guidance and B21.3
continuation/state-saving machinery. Both advertise default-off
semantics, but leaving them in the 'published DAPS-4K' baseline
would create unnecessary source-provenance ambiguity.

**Resolution: preserve the modified historical DAPS checkout entirely.**
Do not reset, stash, clean or amend the original experimental source.
The new `xp_t_daps4k_native_cpu.py` performs a one-time local
`git clone --local --no-hardlinks --no-checkout`, from the existing
original Git object database into the **new** dedicated path
`/egr/research-pac/huang248/external/DAPS_XP_NATIVE_4K`.
It installs *non-cone sparse-checkout patterns before first checkout*
and materializes only `sampler.py`, solver, Fourier operator, model,
and selected configs. No bundled data or demo directory is checked
out. The cloned checkout has the exact pinned HEAD with a clean
status and all 10 needed native source blobs identical to their Git
HEAD identities; the experimental original sampler's modified blob
is verified unchanged both before and after the operation.

After the pristine source receipt is written to
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_t_daps4k_native_source_20261026/XP_T_DAPS4K_NATIVE_SOURCE.json`,
the updated `xp_t_daps4k_prepare_cpu.py` requires that receipt and
rechecks the pristine sparse tree, all 10 source hashes, canonical
FFHQ/ImageNet checkpoints and frozen common noisy measurements.
Only then can it freeze the 18-root GT-free timing manifest.
A disposable synthetic-Git regression suite tests that a modified
original `sampler.py` is *not* copied, bundled dataset paths are
not materialized, and repeat staging or source drift fail closed.
No GPU work is authorized by either preparation script.

This is a **scientific source-isolation repair**, not a relaxation of
the native-source integrity gate. The next GPU native DAPS-4K
implementation must import from the pristine sparse checkout only,
strictly verify model weight loading (the upstream UNet loader has
a catch-and-continue fallback), and continue to use the same signed
y and per-worker W=1 timing semantics as the prior FAST/SITCOM pilots.

## DAPS-4K pristine-source isolation: PAC PASS; 18-root native W=1 timing runner authored

On PAC the native sparse checkout succeeded after the B20/B21 sampler
drift was isolated. **98 XP regression tests passed**. The receipt
`XP_T_DAPS4K_PRISTINE_NATIVE_PASS` attests that pristine pinned upstream
`e7a77d094167084faed19b599b96673b7bb11447` was materialized
as `/egr/research-pac/huang248/external/DAPS_XP_NATIVE_4K`,
with no bundled data and no modification to the historical experimental
checkout. Source proof SHA256:
`f36a348c26e8dd76a5d72dc84ebe4e100b62d705a821c164e06c28a489433b23`.
The separate CPU frozen shared-y calibration preparation passed at
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_t_daps4k_20261027T011832Z`,
with 18 roots (FFHQ 9, ImageNet 9) and task-manifest SHA256
`1f71f0bff208e1823ceb68fbe58a805c6b28241fe04da577efb35bdf65ed30c0`.
**GPU work has not been done at the time of this update.**

A new `xp_t_daps4k_worker_gpu.py` and
`xp_t_daps4k_supervisor.py` implement the next guarded calibration
against **only the pristine source checkout**. Each independently
seeded root runs its own DAPS Python environment, batch size 1,
a strict checkpoint-load verification, frozen signed Fourier
measurements (no regenerated observations), and the published
DAPS-4K phase-retrieval hyperparameters:

- 400 EDM annealing transitions, sigma_max=100,
  sigma_min=0.1, poly-7; 10 EDM diffusion/Euler ODE steps
  per annealing transition, sigma_min=0.01;
- 100 Langevin MCMC steps per annealing transition,
  learning rate 5e-5, tau 0.01, lr_min_ratio 0.01,
  gaussian prior solver;
- phase-retrieval Fourier operator oversample 2.0,
  measurement sigma 0.05, image size 256,
  model DDPM as in canonical FFHQ/ImageNet architecture YAMLs.
  These are **not** upstream's shorter DAPS-1K commands.

The solver/model/checkpoint itself remains unedited. The model
loader's catch-and-random-initialize behavior is explicitly guarded:
every root must show exactly one successful strict
`UNetModel.load_state_dict` call (>=100 checkpoint tensors,
zero missing/unexpected keys) **before sampling**. The native sampler
uses `get_start(1,model)`, so there is no GT-derived shape placeholder.
Every candidate records synchronized process wall, CUDA-event sampling
time, separate model-loading wall, strict model-load proof, physical
GPU, frozen measurement/model hashes, tensor/PNG reconstruction
hashes, and a GT-free/no-scoring receipt.

The four-GPU supervisor uses **one native worker per physical GPU
(W=1)**, a clean-GPU admission check and hard 80% VRAM requirement,
and a conservative **6 GPU-hour total budget ceiling**.
It initially launches just one FFHQ and one ImageNet candidate,
verifies both exact root receipts/strict model loads, and projects
9 per dataset × 1.75 times observed native sampling duration.
Only if this smoke projection fits the ceiling does it admit the
remaining 16 roots across available devices. On failure it stops new
admissions and preserves all completed outputs; it never kills active
workers or interferes with other experiments. Timing contamination is
classified independently per GPU, so a successful root with uncertain
telemetry is not misrepresented as CLEAN.

The future completion receipt `XP_T_DAPS4K_COMPLETE.json` will contain
separate FFHQ and ImageNet W=1 medians and provenance, plus
`XP_T_DAPS4K_SMOKE_GATE.json` and the exact telemetry classification.
**Do not report XP-T DAPS-4K calibration as complete until PAC returns
these receipts and all 18 source-pinned roots verify successfully.**
No GT evaluation, protected holdout or historical candidate data may
enter these workers. The pre-registered W={1,2,4} same-device
throughput study remains distinct from this four-physical-GPU
independent-task parallelism.

## XP-T DAPS-4K FFHQ + ImageNet calibration: native 18/18 PASS (PAC)

The four-GPU native DAPS-4K run has finished successfully on PAC.
The frozen calibration directory is
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_t_daps4k_20261027T011832Z`;
native source SHA
`f36a348c26e8dd76a5d72dc84ebe4e100b62d705a821c164e06c28a489433b23`;
frozen manifest SHA
`1f71f0bff208e1823ceb68fbe58a805c6b28241fe04da577efb35bdf65ed30c0`;
native runner Git commit
`e3a7f80e713939230fcacf830e0c11a68df0aa28`.
The supervisor reports:

- **`XP_T_DAPS4K_BATCH1_PASS`** with `18/18` roots:
  FFHQ 3 images × 3 independent seeds, ImageNet 3 × 3.
  The first FFHQ and ImageNet roots passed strict native-model-load and
  reconstruction checks in `XP_T_DAPS4K_SMOKE_GATE.json`;
  smoke projected **1.0127296631314675 GPU-hours**.
- The final aggregate was **0.9600079935751596 GPU-hours**,
  substantially below the 6 GPU-hour cap. Every worker completed;
  final `state=PASS`, `phase=COMPLETE`, `reason=null`,
  0 running or active roots; `clean_timing_eligible=true` across all
  four physical GPUs with W=1 worker per GPU.
- **FFHQ median native sampling wall 66.01368511701003 s**;
  CUDA-event median 66.0135234375 s;
  model-loading median 0.4937783391214907 s.
  **ImageNet median native sampling wall 166.23148921877146 s**;
  CUDA-event median 166.2304375 s;
  model-loading median 2.155493638012558 s.
  These *exclude* model loading and are the first 3-image, 3-seed
  source-matched measurement calibration, not an image-population
  quality result or a W={1,2,4} same-device throughput study.
- Native 400-annealing, 10 diffusion and 100 Langevin steps per
  annealing transition were enforced; no DAPS-1K substitution.
  GT and historical candidate images were never loaded or scored.
  The original experimental B20/B21 tree was preserved; the native
  code ran from an isolated pristine pinned sparse checkout.
- **Exact PAC completion receipt SHA256:**
  `6f07bc2672c3d643996c3f5653b1e7eea63a676b3050a10766ae883e06fc2c41`.

**All three generation-family XP-T native timing calibrations are now
complete**, with the following approximate median *sampling wall* times
on the small matched-y pilot:

| Native method (published settings) | FFHQ (s) | ImageNet (s) |
| --- | ---: | ---: |
| FAST-DIPS | 3.321 (original CLEAN) | 7.191 (original UNVERIFIED; separate CLEAN_POSTHOC) |
| SITCOM | 32.104482267051935 (CLEAN) | 58.342772816773504 (CLEAN) |
| DAPS-4K | 66.01368511701003 (CLEAN) | 166.23148921877146 (CLEAN) |

DAPS/SITCOM sampling wall ratio is 2.06× on FFHQ and 2.85× on
ImageNet. These are budget planning/calibration measurements only,
not outcome-aware or population-level speed-quality conclusions.
W=1 independent physical GPU parallelism does not substitute for
the pre-registered FAST/SITCOM simultaneous W={1,2,4} one-GPU
throughput experiment.

**Publication is pending:** `xp_t_daps4k_publish_cpu.py` validates
this exact completion SHA, all 18 checkpoint and reconstructed-file
SHA proofs, 2-root smoke qualification, frozen measurement and source
provenance, GPU safety budget, and all four CLEAN device classifications
before copying only non-GT summary/metadata receipts to the
GitHub XP executor return tree. The publication is not complete
until PAC prints the successful pushed commit.

**Research prioritization:** XP-T's native baseline cost uncertainty is
now substantially reduced. Proceed to Tier A, data cut **Nov 2**:
FAST-FFHQ XP-A16, XP-C and XP-C2, FAST-ImageNet XP-D1
(frozen theory predictions and K>4 embargo remain enforced).
Tier B, Nov 6, includes SITCOM FFHQ XP-B16, SITCOM ImageNet XP-D2,
DAPS-4K ImageNet XP-D3 and FFHQ XP-E′ among other preregistered arms.
The exact planner's locked `XP_PREREG_v1_1.md` remains authoritative
for population IDs, run index semantics, seeds, and scoring rules.
