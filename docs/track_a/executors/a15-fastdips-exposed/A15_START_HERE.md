# Track A A1.5 — source-frozen FAST-DIPS native comparison on eight A1.2 exposed B22 measurements
**Protocol prepared 2026-10-20. STATUS: IMPLEMENTATION COMMITTED; PAC NOT INSPECTED; NO SOURCE/NUMERIC PREFLIGHT PASS; NO GPU EXECUTED.**

**Branch:** `codex/prdiff-track-a-a15-fastdips-exposed-20261020` (derived ONLY from accepted A1.4 desk branch; no modifications to A1.2, A1.3, B21–B26 or Track B). Work happens in this chat with the user as the sole PAC operator.

## 1. Scientific purpose and experimental contract

Question: With the exact eight **previously exposed** FFHQ B22 images and saved **signed** A1.2 measurements, does original FAST-DIPS have better **raw RGB8 PSNR versus full K-candidate compute** than our prior DAPS/SITCOM/NP LF/S2 K1–4 independent-root portfolios, under the same nominal FFHQ pretrained checkpoint file?

This is an **engineering retrospective paired benchmark**, **not** fresh validation or evidence that NP wins on new images. Existing B24 confirmation305 remains entirely excluded, **including membership**, and there is no other dataset discovery or new noise. GT is available for offline raw PNG8 PSNR only after the FAST sampler has produced its candidate and is never handed to the sampler. `GT best-of-K` is an **offline non-deployable oracle**. 180-degree rotated PSNR is auxiliary.

Main figures: per-image FAST roots rep0..3, K1/K2/K4 prefix best raw RGB8 PSNR, Good25/Good30/Bad20, native and rounded PNG8 export comparison, all per-root wall sampling and CUDA-event timing; A1.2 existing offline oracle baselines read only. FAST native output includes float and PNG8; original A1.2 native baselines are already frozen. Report input/measurement/model/source SHA, exact seed, model load proof and all candidate costs. Distinguish FAST paper published **time per run** from K-run actual budget. **Do not** rerun DAPS, SITCOM or NP.

## 2. Upstream FAST source and compatibility

- Original official repository: https://github.com/ququlza/FAST-DIPS (ICLR 2026, Table1 and Appendix A.6).
- Exact required upstream commit: `a9758b38906b7c7950fa04491f37f6634948db26` (2026-09-13).
- FAST original pixel PR released example: `edm_FAST_DIPS`, `T=150`, `K=2` ADMM iterations **per step**, `S=5` x-gradient steps, `rho=200`, `epsilon=.05`, batch=1, `num_runs=4` for paper best4. ***K_ADMM=2 is not independent restart count***.
- Native `PhaseRetrieval`: `pad=int(oversample/8*256)=64`, `x_model*.5+.5`, centered `fft2c_new(...,norm='ortho')` magnitude, 384²×3; `Operator.measure` adds signed `.05 * randn`. **We do not invoke `measure`** because frozen A1 observation is supplied directly; this is a valid input substitution, not an algorithm change.
- FAST official `model.py` catches a checkpoint-loading exception and may **silently continue with random weights**. New wrapper pins A1 FFHQ source checkpoint SHA and post-load checks 7 param/buffer tensors against its original state dict, with exact key shapes/count. Model source and algorithm remain native FAST; no attempt to transplant NP state or alter sampling.
- FAST official `misc.tensor_to_pils` truncates uint8; our wrapper records both an explicit *uniform rounded RGB8 candidate* PNG (primary A1 metric convention) and an **original FAST native truncating PNG** with separate PSNR. No choosing which one based on favorable GT outcomes.

## 3. Exact PAC roles/paths, no overwriting

- New A15 source checkout: `/egr/research-pac/huang248/pr_diffusion_track_a_a15_fastdips`.
- Separate pinned upstream source checkout: `/egr/research-pac/huang248/external/FAST_DIPS_A15`.
- Inputs READ ONLY: `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_2_exposed_b22_inputs_v1/A1_2_INPUT_FREEZE_COMPLETE.json`, exactly IDs `69073,68539,67948,60140,67293,67174,61101,66631`; source/contents SHA pinned in `configs/track_a/a15_fastdips_exposed_v1.json`. A1.2 oracle summary read-only. A1.3 source-input PASS receipt read-only.
- Model READ ONLY `/egr/research-pac/huang248/models/ffhq_10m.pt`, expected SHA256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`.
- **Dedicated required interpreter:** `/egr/research-pac/huang248/conda-envs/fast_dips_a15/bin/python`, pinned Python 3.10 and torch2.3.0/cu121, torchvision0.18.0; never install into `prdiff_ffhq`. After actual A15.0 PAC check reported `STOP_MISSING_PYTHON_MODULES:omegaconf,imageio,wandb,setproctitle,einops,pytorch_lightning,transformers` in the **old** `prdiff_ffhq` Python, A15 made an isolated pixel-only curated environment installer `scripts/track_a/a15_create_isolated_env_v1.sh` and pinned `configs/track_a/a15_fastdips_pixel_requirements.txt`. Optional LDM/StableDiffusion packages, including `pytorch_lightning`, `diffusers`, `transformers` and CLIP, are **not required** by the pixel native sampling path and omitted to avoid introducing conflicts. New `a15_fastdips_preflight.py` verifies only immediate native pixel imports, exact new interpreter and torch version. GitHub source changed after initial failed PAC preflight; user must fast-forward A15 source HEAD before rerunning. An existing or partial conda environment is preserved and blocks the installer rather than being overwritten.
- New **exclusive** output tree: `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a15_fastdips_exposed_20261020/`. Stages `a15_0_cpu_preflight_v1`, `a15_1_single_native_fast_smoke_v1`, `a15_2_native_fast_k4_exposed_v1/<ID>/rep<R>`, `a15_3_report_k1_v1`, `a15_5_report_k2_v1` and `a15_4_report_k4_v1`. Every path is mkdir exclusive; failures preserve partial artifacts. Never `rm -rf`, reset native PAC repos, or overwrite original A1/A1.3 outputs.

## 4. Stage transitions — fail closed

**Stage A15.env (NOW, after observed missing modules):** User fast-forwards the clean isolated A15 checkout to the latest committed HEAD. With `conda` available, run `scripts/track_a/a15_create_isolated_env_v1.sh <PINNED_A15_HEAD>` to create **only** `conda-envs/fast_dips_a15` with Python 3.10, official PyTorch2.3 CUDA121 wheel pair and curated pixel native imports. All pip/conda caches and tmp go under `/egr/research-pac/huang248/outputs/.cache/fast_dips_a15`, no home env. No `set -e`; STOP if any install/import check fails. Do not delete/overwrite partial env; return log.

**Stage A15.0 (after env import PASS, safe CPU command):** Inspect non-overlap and reuse the separately cloned Git checkouts. Pin both Git SHAs. Run `scripts/track_a/a15_fastdips_preflight.py` under CPU-only `CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`. It first verifies source HEAD/clean tracked worktree, native Python modules and CPU Fourier equation against independent witness, all frozen eight input byte and tensor SHA, accepted A1.3 receipt and FFHQ checkpoint exact SHA. Only after PASS does it write new stage0 receipt. If missing packages, STOP and ask for environment-specific fix, **do not launch GPU**.

**Stage A15.1 (NOT YET AUTHORIZED UNTIL A15.0 PASS + issue41 GPU lease):** One FAST native pixel root, `image=69073`, `rep=0`, exactly one physically allocated RTX6000 GPU in `CUDA_VISIBLE_DEVICES`; separate program `a15_fastdips_smoke.py --mode smoke`. Full 150-step native FAST with official ADMM params, no new data, no GT during inference. Records actual run/step/event timing and peak VRAM. Need one `[RUN RESERVATION]` issue41 comment with actual GPU physical ID, source HEAD, worktree/output and max wall budget after checking live free device and Track B reservations; then user executes command. **No auto-advance**.

**Stage A15.2 (NOT YET AUTHORIZED):** After a scientifically valid smoke/measurement/weight/quality and timing receipt, run eight independent rep0 K1 grid roots; pause for analysis. Then rep1–3 for exact K4 only with separate GPU budget/ledger. The grid runner is present but requires `--mode grid`, an exact disjoint output and explicit `A15_GRID_LEASE_RECORDED_ISSUE41`; its presence does not authorize execution. No hidden restart, mutation, control tuning or other method inference.

**Stage A15.3/4/5 (CPU reporting after complete prefixes):** `scripts/track_a/a15_fastdips_report.py --k 1|2|4` consumes exactly 8 or 32 completed grid receipts, recomputes paired historical A1.2 offline-oracle statistics, writes only fresh metadata/CSV. It does **not** generate any candidate itself.

## 5. Stop and evaluation rules

STOP on missing FFHQ checksum, incompatible native FAST DDPM state-dict shape/load, wrong source HEAD, any unexpected changed input byte, FFT geometry mismatch, CPU environment mismatch, no recorded GPU lease or occupied output, failed/no receipt, NaN, GPU OOM, or 150-step runtime substantially beyond planned lease. No silent weight reinitialization, no new measurement generation or changing FAST's official epsilon/schedule to win.

Publication: even if FAST loses on eight exposed ID set, do **not** claim NP population superiority. This panel is development and was already used in NP design. A truly independent paper benchmark, if needed, remains a separate planner decision with independent eligible data, updated strong baseline and Track B GPU ownership considerations.

**Current state:** Implementation committed; no PAC clone/installation/check/FFT run has been observed in A15, and no GPU reservation has been posted. Next user response should return Stage0 PASS/STOP stdout only.
