# Track R · R0 executor handoff — START HERE

**From:** planner (Claude), 2026-10-21
**To:** executor (GPT/Codex session chosen by Cheng). You have no access to the lab machine (PAC); Cheng runs every PAC command you write.
**Binding documents:**
- [R0 spec](../../../planning/track_r/R0_EXPERIMENT_SPEC.md) — FROZEN v1.
- `configs/track_r/r0_frozen_v1.json` — its machine-readable twin.
- `configs/track_r/pac_paths_track_r_v1.json` — paths, unverified.

**Context:** [planner roadmap](../../../planning/track_r/2026-10-21_TRACK_R_PLANNER_ROADMAP.md) and [literature basis](../../../planning/track_r/2026-10-21_LITERATURE_BASIS.md).

---

## 0. Planner's view for Cheng (current snapshot)

- **Where we are.** The NP method line is closed (FAST-DIPS dominates; NP's rule is SCG; its schedule is DiffFPR's). GT-free failure detection already exists (FA-FPR, ISCAS 2025: detect and restart). The open gap we test is **repair, not repeat**: fix exact-symmetry failures in place using the diffusion prior's own likelihood ratio, then use the prior to flag the remaining failures, on a fast engine. This is judged on reliability per compute.
- **Plan.**
  - R0 (this handoff, about 1–2 weeks) measures, mostly on outputs you already have, how many failures are exact-orbit errors and whether the prior repairs and detects them. It also adds cheap FAST runs.
  - R1 builds the FAST-based pipeline, with FA-FPR as the baseline. It runs only if R0's pre-registered gates pass.
  - R2 is a fresh prospective panel. R3 is writing.
- **Publication image.**
  - Working title: "Repair, don't repeat: exact-symmetry repair and prior-consistent failure detection for reliable diffusion phase retrieval."
  - Content: failure anatomy across DAPS/SITCOM/NP/FAST under GT-free selection; a posterior-ratio = prior-ratio proposition with a paired-noise estimator; prior-based detection versus FA-FPR's detector; reliability-per-compute frontier; ablations.
  - Quick venue: IEEE Signal Processing Letters (ICASSP 2027 presentation possible if accepted by 2027-01-06, tight) or ICIP/EUSIPCO 2027. A top venue only with an ImageNet win plus theory.
- **Honest odds:** about 40–50% that R0 justifies a paper-sized R1. The main risk is that exact-orbit errors are a minority of failures. Our older data showed 3 of 42 failed endpoints in A1.3.

---

## 1. Your role and limits

**You may:**
- Create and commit to your executor branch.
- Write Python modules, scripts, configs and tests.
- Run CPU tests in your own sandbox.
- Write PAC command blocks for Cheng.
- Analyze returned logs and summaries.
- Draft the R0 return.

**You may not:**
- Run anything on PAC yourself.
- Ask Cheng to run a GPU job before the gate it depends on has passed.
- Change anything in the frozen spec or config. If you think a change is needed, stop and ask the planner.
- Touch historical outputs except read-only.
- Write outside the Track R namespaces.
- Access B24 confirmation305 in any way.
- Merge into `main`.
- Present GT-based numbers as deployable.

**Escalate to the planner (stop and return a short report) when:**
- any G0 or G1 gate fails;
- a verified path or SHA mismatches the records;
- an environment can't be built without modifying a historical environment;
- the GPU-hour projection exceeds 16;
- the spec is ambiguous for a case you face;
- after R0.7 (final return).

**Planner pre-authorization:** if G0, G1 and the R0.3 smoke all pass, proceed through R0.4–R0.6 without waiting for the planner. Cheng still runs each block.

---

## 2. Branch and checkout

1. Create `codex/prdiff-track-r-r0-<YYYYMMDD>` from the current HEAD of `claude/prdiff-track-r-planner-20261021`. Record that SHA in every return.
2. Put code only under these namespaces:
   - `prdiffusion/track_r/`
   - `scripts/track_r/`
   - `configs/track_r/` — new files only; do not edit the two planner files.
   - `tests/track_r/`
   - `docs/track_r/executors/r0/`
3. PAC checkout. Give Cheng a block that:
   - creates `/egr/research-pac/huang248/pr_diffusion_track_r` with `git clone --branch <your branch> https://github.com/Cheng-HanHuang/pr_diffusion …` **only if the path does not exist**;
   - otherwise fetches and `git checkout --detach <pushed SHA>`, refusing if the tree is dirty.

   Submodules are not needed for R0.
4. Every PAC run must start from a pushed commit with a clean tree. Print `git rev-parse HEAD` and `git status --porcelain` at the top of every log.

---

## 3. Reading order (stop when you have what you need)

1. This handoff, then the R0 spec, then the two config files.
2. Root `AGENTS.md`: Track R override plus the historical safety contract. Shell rules in §7 below.
3. Code to reuse:
   - `scripts/track_a/a13_physics.py` and `scripts/track_a/test_a13_physics.py`. Port the operator, residuals, PNG8 conventions and the centered-FFT witness. The CPU-only guard in that file must become a device-agnostic version.
   - `prdiffusion/guided_backend.py`. Loads `ffhq_10m.pt` via DiffFPR's `guided_diffusion`; ε = first 3 output channels.
   - `scripts/track_a/a15_fastdips_preflight.py`, `scripts/track_a/a15_fastdips_smoke.py`, `configs/track_a/a15_fastdips_exposed_v1.json`. This is the native FAST adapter: checkpoint-load verification, external signed measurement injection, RGB8 PSNR. Generalize it; don't change FAST itself.
4. Where the candidates are (machine-readable first):
   - S1: `configs/b22/b22_2_overnight.json`; `scripts/b22/analyze_b22_3_panel.py` (layout `<family>/row*/candidates/*/result.json`); `docs/b22/b22_3_scientific_analysis_runbook.md`.
   - S2: `configs/b26/b26_dev80_manifest.json` (per-image GT, measurement and task paths); `configs/b24/b24_3_dev80_overnight.json`.
   - S3: `docs/track_a/executors/a1-oracle-budget/A1_PAC_ARTIFACT_INDEX_20261020.json`.
   - S4: §5 of `docs/track_a/executors/a14-a16-planner-handoff/A14_A16_COMPLETE_SCIENTIFIC_PLANNER_RETURN_20261021.md`.
5. Background, only if needed:
   - `docs/b22/b22_3_scientific_closeout.md` (failure taxonomy);
   - `docs/track_a/executors/a13-candidate-audit/A13_EXECUTOR_RETURN.md` (the 68539 orientation case, signed vs clipped, float vs PNG);
   - `docs/b25/B25_FINAL_PLANNER_RETURN.md` (8-mask orbit check on DEV80).

---

## 4. Work packages

### R0.0 — PAC inventory and environment check (CPU; read-only except creating Track R directories)

**Deliverables:** `scripts/track_r/r0_0_inventory.sh` (bash wrapper) and `scripts/track_r/r0_0_inventory.py`.

**Behaviour:**
- Writes the full log and `R0_0_INVENTORY.json` under `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_0_inventory_<UTCSTAMP>/`.
- Prints a summary of at most 40 lines.
- Creates only these directories: Track R output root, logs and cache.

**Checks (each recorded as VERIFIED / MISSING / MISMATCH / WARN):**

1. **Identity:** hostname, UTC time, worktree HEAD and branch, clean status.
2. **Disk:** `df -h` on `/egr/research-pac/huang248`, and free space under the output root.
3. **GPUs:** `nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv`, plus `--query-compute-apps`.
4. **conda:** whether `conda` is available, and its version.
5. **Environments:** every interpreter in `pac_paths_track_r_v1.json:python`.
   - Run with `CUDA_VISIBLE_DEVICES=""` for package checks.
   - Report Python and torch versions, `torch.version.cuda`, and `torch.cuda.get_arch_list()`. Does the arch list contain `sm_120` (Blackwell)?
   - Import checks for: numpy, scipy, PIL, pandas, sklearn, tqdm, ftfy, regex, packaging, `pkg_resources`, json, yaml, and torchvision.
   - Also check that `guided_diffusion` imports from the DiffFPR checkout, and, in the FAST env, that the FAST imports used by A1.5 work.
6. **Models:** `ffhq_10m.pt` SHA-256 must match. Report whether `models/clip/ViT-L-14.pt` exists and its SHA.
7. **External repos:** HEAD and clean status for FAST_DIPS_A15, the A15 adapter checkout, DiffFPR, and UniversalFakeDetect (likely absent).
8. **Sources S1–S4 (bounded checks only):**
   - existence of each root, manifest and CSV;
   - SHA of the B22 full manifest and the A1.2 input-freeze manifest;
   - **counts via bounded globs**, never recursive find on large trees. Examples: `ls -d <B22 stage>/<family>/row* | wc -l`; for DEV80, task directories and one task's `methods/` listing; for A1.2/A1.5, the eight ID folders and their `rep*` folders.
9. **One sample per source:** load one candidate file, one GT and one measurement; print dtype, shape, min and max. Find out whether B22 candidates are saved as `.pt`, `.png` or both, and which field of `result.json` points to them.
10. **FFHQ metadata JSON:** present or absent. Do not parse the image tree.
11. **Forbidden-token self-check:** assert the script itself never touched a path containing `confirmation` or `confirm305`.

**Then:** write `configs/track_r/pac_paths_track_r_verified.json` from the returned JSON and commit it.

### R0.0b — create missing pieces (only if R0.0 reports them missing; Cheng confirms first)

**Deliverable:** `scripts/track_r/r0_0b_create_env_and_assets.sh`, built from functions with explicit return codes. Model it on `scripts/track_a/a15_create_isolated_env_v1.sh`: no `set -e`, no `exit`.

**Steps:**

1. **Environment.** If `/egr/research-pac/huang248/conda-envs/prdiff_track_r` is absent:
   - `conda create --yes --prefix <it> --clone /egr/research-pac/huang248/conda-envs/prdiff_ffhq`;
   - then `pip install` only the missing packages (expected: ftfy, regex, maybe pandas/scikit-learn);
   - if `pkg_resources` is missing, install a setuptools version that still provides it.

   Set `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR`, `TMPDIR`, `XDG_CACHE_HOME`, `TORCH_HOME` and `HF_HOME` under `/egr/research-pac/huang248/outputs/.cache/track_r/`. **Never modify `prdiff_ffhq`, `fast_dips_a15_blackwell`, `daps` or `sitcom_ode_bw`.** Save `pip freeze` and the torch arch list to the log.
2. **UnivFD.** If `/egr/research-pac/huang248/external/UniversalFakeDetect` is absent, clone it and check out `030495aea3300a8b54c0ec37ec7fe1dd7e63c619`. Verify the `fc_weights.pth` SHA-256 is `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847`.
3. **CLIP weights.** If `/egr/research-pac/huang248/models/clip/ViT-L-14.pt` is absent, download it from the URL in the paths config and verify SHA-256 `b8cca3fd…3836`.
   - Use Python `urllib` with an explicit target, the same rule as `clip._download`.
   - **Never** let CLIP write to `~/.cache`.
4. Print a final ENV_READY / ENV_FAILED line with reasons.

### R0.1 — code and CPU tests (in your sandbox; no PAC)

**Modules (suggested interfaces; keep them small and readable):**

`prdiffusion/track_r/physics.py` — device-agnostic port of a13:
- `magnitude(x[B,3,256,256]) -> [B,3,384,384]`
- `rho_signed(x, y)` and `rho_clipped(x, y)`, per-sample, normalized by √m
- `to_rgb8(x) -> uint8`, `from_rgb8(u8) -> model range`
- `load_png_rgb8(path)`, `psnr_rgb8(x8, g8)`
- `ORBIT_PATTERNS` and `apply_orbit(x, pattern)`
- the centered-FFT witness

`prdiffusion/track_r/prior_score.py` — `PriorScorer(eps_model, alphas_cumprod, timesteps, draws, device, batch_size)`:
- `.score_orbit(x_c[1,3,256,256], uid) -> dict` with:
  - S per member; S_mid per member;
  - per-t means per member;
  - s_star, margin, z;
  - `unet_forwards`.
- Implements the seed rule and the paired coupling exactly as in spec §3.7.
- `eps_model` must accept any callable `(x_t, t) -> eps` so tests can use mocks.

`prdiffusion/track_r/univfd.py` — `UnivFD(repo_root, fc_path, clip_path, device)`:
- `.p_fake(x8[B,3,256,256]) -> [B]`.
- Import the repo's `models.clip` (insert the repo root on `sys.path`), call `clip.load(<local ViT-L-14.pt path>, device="cpu")`, build `nn.Linear(768,1)` and load the `fc_weights.pth` state dict (UnivFD does `model.fc.load_state_dict(torch.load(ckpt))`), then move to device.
- Preprocessing as in spec §3.8.

`prdiffusion/track_r/manifest.py`:
- One row schema; per-source loaders S1–S5; the forbidden-token assertion.
- **Row fields:**
  - `uid` (= `{source}|{family}|{image_id}|{run_label}`), `source`, `family`, `image_id`, `run_index`, `run_label`;
  - `candidate_path` and `candidate_format` (`png` | `pt`);
  - `gt_path`, `gt_format`;
  - `measurement_path`, `measurement_sha256` (as recorded, if any);
  - `recorded_psnr_db`, `recorded_psnr_convention`;
  - `native_selector_name`, `native_selector_value`;
  - `recorded_cost_seconds`, `recorded_unet_forwards`;
  - `notes`.

**Scripts:**

| Script | Purpose |
|---|---|
| `scripts/track_r/r0_2_build_manifest.py` | Build the candidate manifest |
| `scripts/track_r/r0_2_validity_gates.py` | Run G0 |
| `scripts/track_r/r0_score_worker.py` | One GPU; shard of the manifest; append-only JSONL with per-candidate rows; resumable by `uid`; refuses a different config hash |
| `scripts/track_r/r0_launch_scoring.sh` | nohup, one worker per GPU, PID file, `status` / `stop` subcommands |
| `scripts/track_r/r0_fast_worker.py` and `scripts/track_r/r0_launch_fast.sh` | FAST generation (see R0.5) |
| `scripts/track_r/r0_7_analyze.py` | CPU; produces every primary and secondary output in spec §4 plus the gate/decision table |

**Tests** (`tests/track_r/`, CPU, no checkpoint, `pytest -q`):

1. Orbit invariance of `magnitude` for random x and for all 8 patterns; exact-pattern bookkeeping.
2. Centered-FFT witness (impulse → constant 1/384; zeros → 0).
3. Paired-noise coupling: with a **rotation-equivariant** mock ε-model (e.g. a pointwise function of x_t), all 8 members score identically to 1e-6.
4. With a deliberately **orientation-biased** mock (e.g. penalize energy in the top rows of channel R), `s_star` recovers the planted pattern.
5. Seed rule determinism and device independence (CPU vs CPU suffices); batch-size invariance.
6. RGB8 round-trip and PSNR against hand-computed values.
7. Manifest schema validation and forbidden-token assertion.
8. Analysis on a synthetic table with known answers: label counts, AUROC, bootstrap reproducibility under a fixed seed, and policy E0–E3 selection logic, including the empty-feasible-set fallback.

If torch is unavailable in your sandbox, still commit the tests and give Cheng a CPU test block (`CUDA_VISIBLE_DEVICES="" <prdiff_track_r python> -m pytest -q tests/track_r`).

### R0.2 — manifest and G0 validity gates (PAC, CPU)

Cheng runs one block that:
1. builds `R0_CANDIDATE_MANIFEST.csv` for S1–S4;
2. runs the G0 checks: ρ_signed(GT) range for every image; orbit invariance on 16 random and 3 real images; recorded-PSNR replay (≥ 99% within 0.01 dB under each source's own convention); forbidden-token assertion.

Outputs: `R0_2_GATES.json` with PASS/WARN/FAIL per gate, plus mismatch lists. Counts per (source, family) must match the expected counts in spec §2 or be explained. **Stop and escalate on any FAIL.**

### R0.3 — GPU smoke, including G1 (PAC, 1 GPU; under 15 min)

1. Check `nvidia-smi` first and pick an idle GPU.
2. **Smoke set (8 candidates):**
   - the G1 canonical case (S3, 68539, DAPS rep2);
   - one GOOD and one NONORBIT failure from S1 DAPS;
   - one SITCOM, one NP, and one S4 FAST candidate;
   - two random others.
3. Run the prior scorer twice: batch sizes 8 and 64, on two different physical GPUs if available. This is the determinism gate.
4. Run UnivFD on all 8 orbit members of each candidate.
5. Run one FAST root through the generalized worker on one S2 image. If A1.5's recorded seed path allows it, also attempt a bit-exact replay of A1.5's `69073/rep0` root.
   - Bit-exact: report PASS.
   - Not bit-exact: report raw-PSNR difference and pixel correlation; proceed only if PSNR is within 0.05 dB, otherwise escalate.
6. **Measure throughput:** UNet forwards/s and CLIP forwards/s at the chosen batch size, plus FAST seconds per root (CUDA-event sampling-only and wall). Project R0.4–R0.6 GPU-hours.
7. Output `R0_3_SMOKE.json`.
8. **If G1 fails, or the projection exceeds 16 GPU-hours → stop and escalate.**

### R0.4 — full scoring of S1–S4 (PAC, 4 GPUs, nohup)

- One worker per idle GPU, sharded by a stable hash of `uid`. Resumable and non-overwriting.
- Per candidate, record:
  - the 8 member S values, S_mid, per-t means;
  - `s_star`, margin, z;
  - p_fake for the 8 members and for the repaired image;
  - ρ_signed and ρ_clipped of the repaired image — identical across members, and a check of invariance;
  - timings and UNet/CLIP forward counts.
- Provide `status` (counts done/total per shard, last error lines) and `stop` (kills only recorded PIDs).

### R0.5 — FAST-DIPS K=4 generation on S1 and S2 measurements (PAC, GPUs, nohup)

- **Interpreter:** `fast_dips_a15_blackwell`. **Source:** `/egr/research-pac/huang248/external/FAST_DIPS_A15` @ `a9758b38…`, unmodified.
- **Long-lived worker:** one process per GPU loads the model once (reuse A1.5's `load_checkpoint_verified`), then loops over its assigned (source, image_id, rep).
- **Per root:**
  1. Load the locked signed y (verify the recorded SHA when available).
  2. Seed by the spec rule.
  3. Run the native sampler with the frozen config.
  4. Save `reconstruction.pt` (float32, model range), `reconstruction.png` (canonical RGB8), and `R0_FAST_ROOT_COMPLETE.json`, which records:
     - seed, source SHAs, config hash;
     - CUDA-event sampling-only seconds and wall seconds;
     - denoiser forwards (expected 149 for T=150; count, don't assume) and operator calls if accessible.
- **The worker never reads GT.**
- Output root: `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_5_fast_k4_<UTCSTAMP>/<source>/<image_id>/rep<r>/`.

### R0.6 — score the FAST outputs (S5)

Append S5 rows to the manifest (as a new manifest version, not an in-place edit) and run the same scoring worker.

### R0.7 — analysis and return (CPU)

- `r0_7_analyze.py` produces everything in spec §4:
  - `R0_SUMMARY.json` (every primary number with CI and n; gate and decision table);
  - CSV tables;
  - figures: anatomy stacked bars per source × family; repair confusion; ROC curves; policy table;
  - contact sheets: all ORBIT_ERROR candidates with raw / repaired / GT-best orbit member, and a random 24 NONORBIT failures;
  - a numbers-only report draft.
- Package metadata only (no tensors) as `R0_RETURN_<UTCSTAMP>.zip`, under about 50 MB, and give its SHA-256.
- Commit compact JSON/CSV/MD outputs under `docs/track_r/executors/r0/returns/R0_<UTCSTAMP>/`.
- Write `R0_EXECUTOR_RETURN.md` per the [return template](R0_RETURN_TEMPLATE.md).

---

## 5. Data rules (repeat of spec §2, operational)

- Read historical outputs only through manifests and recorded paths. Never move, rename, chmod or delete them.
- **B24 confirmation305 is off limits:** no open, list, glob, hash, or membership check. `assert_no_forbidden(path)` runs on every path the manifest emits and every path a worker opens.
- GT is used only in labeling and analysis code paths (R0.2, R0.7). The prior scorer, UnivFD scorer and FAST worker never load GT.

---

## 6. Cost accounting (always on)

Record per candidate:
- UNet forwards for prior scoring (expected 256);
- CLIP forwards (expected 9);
- FAST denoiser forwards and seconds per root.

Report aggregate GPU-hours per stage from worker wall-clock times. The ceiling is 16 aggregate GPU-hours for R0.

---

## 7. PAC command conventions (write every block this way)

- One consolidated block per step. Set variables at the top, save the full output to a log under `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/logs/`, print a short summary and the log path.
- **Never** use `set -e`, `set -u`, `set -o pipefail`, `exit`, `logout`, `exec`, or `pkill`/`killall` patterns. Stop only the recorded PIDs.
- Long jobs use `nohup … > "$LOG" 2>&1 &`, write `$RUN/PIDS.txt`, and come with `status` and `stop` commands.
- Before any GPU block: print `nvidia-smi`; set `CUDA_VISIBLE_DEVICES` explicitly per worker; refuse to start on a GPU with foreign processes using more than 2 GiB unless Cheng passes an explicit override.
- Verify HEAD equals the pushed SHA and the tree is clean before running.
- **Never write under `/home`.** Export the cache variables to `/egr/research-pac/huang248/outputs/.cache/track_r/`.
- Template:

```bash
# === Track R R0.x <name> (<CPU|GPU>; purpose: ...; expected time: ...) ===
ROOT=/egr/research-pac/huang248
WT=$ROOT/pr_diffusion_track_r
PY=$ROOT/conda-envs/prdiff_track_r/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$ROOT/outputs/pr_diffusion/track_r/r0_x_<name>_$STAMP
LOG=$ROOT/outputs/pr_diffusion/track_r/logs/r0_x_<name>_$STAMP.log
mkdir -p "$OUT" "$(dirname "$LOG")"
export XDG_CACHE_HOME=$ROOT/outputs/.cache/track_r TORCH_HOME=$ROOT/outputs/.cache/track_r/torch PIP_CACHE_DIR=$ROOT/outputs/.cache/track_r/pip
cd "$WT" && git rev-parse HEAD && git status --porcelain
CUDA_VISIBLE_DEVICES="" "$PY" scripts/track_r/<script>.py --out "$OUT" > "$LOG" 2>&1
echo "rc=$? out=$OUT log=$LOG"
tail -n 30 "$LOG"
```

---

## 8. Known pitfalls (read before coding)

1. **Measurement conventions.** B22 measurements came from the B21.11 run (`meas5401`), DEV80 from B24 inputs, A1.x from the A1.2 freeze. The G0 ρ(GT) check is how you prove each GT/measurement pairing and the operator convention are right. NP internally clipped y (`clamp_min(0)`); R0 metrics never clip except for ρ_clipped.
2. **PSNR conventions differ by source.** B22 records float PSNR after clamp; A1.x and B24 record RGB8. Replay each with its own convention for G0. Use RGB8 canonical for everything new.
3. **B22 SITCOM's 4 candidates are sequential-RNG runs,** not independent. Keep `run_index` order. B22 NP's 8 candidates are 2 configs × 4 seeds; keep the config label.
4. **UnivFD's `CLIPModel` calls `clip.load("ViT-L/14")`, which downloads into `~/.cache/clip`.** Do not use that class; load CLIP from the local file path.
5. **UnivFD's bundled `clip.py` imports `pkg_resources`,** which recent setuptools versions no longer ship.
6. **A1.5 FAST ran under `fast_dips_a15_blackwell`,** not the cu121 `fast_dips_a15` env. Confirm `sm_120` in the arch list.
7. **The guided-diffusion UNet has learned sigma:** take channels `[:3]` as ε. Check against `prdiffusion/guided_backend.py`.
8. **Don't recompute anything with GT inside workers;** keep GT in analysis only.

---

## 9. What to return, and when

| When | What |
|---|---|
| After R0.0 / R0.0b | Short note to Cheng (summary block plus `pac_paths_track_r_verified.json` commit). Escalate to the planner only on MISSING/MISMATCH that blocks R0. |
| After R0.3 | `R0_3_SMOKE.json` summary with gate results and the GPU-hour projection. |
| After R0.7 | `R0_EXECUTOR_RETURN.md` per the template. The planner then writes the R0 decision with a "Planner's view for Cheng". |

Every return starts with a plain-language "Status for Cheng" paragraph of at most 8 lines: what ran, what passed or failed, what is next. Leave scientific interpretation and decisions to the planner. You may add a clearly labeled "Executor opinion" paragraph.
