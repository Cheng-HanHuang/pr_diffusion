# B22.0 fixed-baseline inventory sign-off

## Status

**Signed off.**

B22.0 established the exact PAC-side source, environment, checkpoint,
measurement, policy, and cost-accounting interfaces required before any fixed
baseline GPU run.

```text
B22.0 inventory: SIGNED OFF
B22.1 one-image smoke: AUTHORIZED
Full 100-image baseline panel: BLOCKED pending B22.1 review
```

This sign-off does not authorize tuning, detector fitting, baseline
configuration search, or a full-panel launch.

## Audit inputs

The sign-off is grounded in:

- the B21-to-B22 GitHub checkpoint;
- the scientific PAC checkpoint
  `/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints/B21_to_B22_20260727_033521`;
- the source snapshot
  `/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints/B21_source_snapshot_20260727_040208`;
- the safe closeout archive
  `B22_0_closeout_safe_20260727_122753.tar.gz`.

All captured closeout steps passed:

1. preflight;
2. SITCOM effective-source identity;
3. dependency identity;
4. historical policy extraction;
5. locked-measurement clipping audit;
6. historical runtime inventory;
7. operator source audit.

## Frozen source identities

| Component | Frozen identity |
|---|---|
| B22 base branch | `codex/project-checkpoint-b21-to-b22` |
| B22 base commit | `0c3c2ec972a50d462b37af7742011ed2a2c5a20a` |
| official SITCOM | `sjames40/SITCOM_ODE@275ab67efbd8146bffca20155171ba6be1169c09` |
| NP/SITCOM handoff fork | `Cheng-HanHuang/SITCOM_ODE_npsitcom@52f2c37e587576d02e2b27ac971e247f2899fc5e` |
| DiffFPR | `Chilie/DiffFPR@a45ffe58f18fed8a63d3446600424e2b08733524` |
| DAPS base | `zhangbingliang2019/DAPS@e7a77d094167084faed19b599b96673b7bb11447` plus preserved B21 local modifications |
| FFHQ checkpoint SHA-256 | `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa` |

The tracked PAC-side SITCOM source difference is only:

```diff
- Path(dir).mkdir()
+ Path(dir).mkdir(parents=True, exist_ok=True)
```

This is an output-directory robustness patch, not a reconstruction-policy
change. The untracked `forward_operator/motionblur/` tree was hashed and
preserved as part of the effective checkout identity.

## Verified PAC environments

### NP

```text
environment: prdiff_ffhq
Python: 3.11.15
PyTorch: 2.11.0+cu128
torchvision: 0.26.0+cu128
NumPy: 2.4.6
SciPy: 1.17.1
Pillow: 12.1.1
diffusers: 0.37.1
```

### SITCOM

```text
environment: sitcom_ode_bw
Python: 3.11.15
PyTorch: 2.10.0+cu128
torchvision: 0.25.0+cu128
NumPy: 1.26.4
SciPy: 1.10.1
Pillow: 10.3.0
Hydra: 1.3.2
OmegaConf: 2.3.0
PIQ: 0.8.0
```

### Existing DAPS/Fresh2

```text
environment: daps
Python: 3.11.15
PyTorch: 2.10.0+cu128
torchvision: 0.25.0+cu128
NumPy: 1.26.4
SciPy: 1.13.1
Pillow: 10.3.0
Hydra: 1.3.2
OmegaConf: 2.3.0
PIQ: 0.8.0
pandas: 2.2.3
diffusers: 0.38.0
```

B22 reports these observed environments rather than upstream README
recommendations.

## Locked measurement interface

B21.11 contains exactly 100 locked tensors:

```text
shape: (1, 3, 384, 384)
dtype: torch.float32
finite: true
noise setting: sigma_y = 0.05
oversampling: 2.0
```

All 100 tensors contain negative entries. Across the panel:

| Frequency region | Negative-entry fraction | Squared energy removed by clipping |
|---|---:|---:|
| all coefficients | `0.3699419714` | `0.0070969769` |
| radius `0.2` | `0.1301508850` | `0.0002352924` |
| radius `0.6` | `0.3644202097` | `0.0066024322` |

The historical NP policy explicitly used `clip_noisy_magnitude=True`.
Therefore NP must:

1. load the exact raw B21.11 tensor;
2. verify and record its identity;
3. apply `clamp_min(0)` in memory;
4. never regenerate measurement noise;
5. never overwrite the locked tensor.

SITCOM consumes the exact raw tensor without clipping.

## Frozen SITCOM policies

The official historical configuration is:

```text
model: FFHQ DDPM
operator: phase retrieval
oversample: 2.0
sigma_y: 0.05
annealing steps: 200
diffusion steps: 5
LGVD steps: 100
LGVD learning rate: 5e-5
LGVD tau: 0.01
seed: 43
historical num_runs: 4
```

B22 separates:

- `SITCOM-1`: one fixed trajectory initialized from seed 43;
- `SITCOM-4S`: four trajectories with the previously frozen executable
  correction-norm selector;
- `SITCOM-oracle4`: best ground-truth PSNR among four, diagnostic only.

B22.1 exercises `SITCOM-1`.

## Frozen NP policies

Each NP trajectory uses:

```text
backend: guided DiffFPR
variant: np_canonical_soft5_hard1
diffusion steps: 1000
projection start: 300
soft candidates: 5
hard candidates: 1
score radius: 0.6
projection radius: 0.2
oversample: 2.0
seeds for portfolio: 100,101,102,103
measurement preprocessing: clamp_min(0) in memory
```

The two portfolio arms are:

- `lf`;
- `s2_preproj_lam001`, with lambda `0.01` before projection only.

Historical selectors are classified as follows:

- `global_run_by_selector`: executable global residual selector over all eight
  LF/S2 trajectories;
- `selected_config_seed_by_selector`: executable secondary diagnostic;
- `selected_config_bestofk`: ground-truth PSNR oracle, diagnostic only.

B22.1 exercises `NP-1`: LF with seed 100. The eventual executable portfolio
row is `NP-8-RS`, using `global_run_by_selector`.

## Frozen comparison matrix

### Executable rows

| Row | Policy | Native cost |
|---|---|---:|
| Fresh1 | existing first DAPS trajectory | 1 DAPS trajectory |
| Fresh2 | frozen two-trajectory loss-margin selector | 2 DAPS trajectories |
| NP-1 | LF, seed 100 | 1 NP trajectory |
| NP-8-RS | LF/S2 × seeds 100–103, residual-selected | 8 NP trajectories |
| SITCOM-1 | official fixed configuration, seed 43 | 1 SITCOM trajectory |
| SITCOM-4S | four trajectories, correction-norm selected | 4 SITCOM trajectories |

### Diagnostic rows

- NP selected-configuration best-of-four PSNR oracle;
- SITCOM best-of-four PSNR oracle;
- optional NP unclipped-measurement ablation, explicitly noncanonical.

## Evaluation rules

Every output is reevaluated through one common offline metric path:

1. clamp reconstruction and ground truth to `[-1,1]`;
2. map both to `[0,1]`;
3. compute RGB MSE;
4. report `-10 log10(MSE)`.

Raw PSNR is primary. `max(raw, rot180)` is auxiliary and must be labeled
offline and ground-truth-assisted.

Report separately:

- native trajectory count;
- summed GPU-seconds;
- observed wall time;
- model-load time;
- peak allocated and reserved GPU memory;
- Fresh1-normalized cost.

A run of NP, SITCOM, and DAPS must not be treated as equal merely because each
is called one trajectory.

## B22.1 authorization conditions

B22.1 may run exactly one nondiscretionarily selected locked image through
`SITCOM-1` and `NP-1`.

The implementation must:

- select the image without inspecting any method outcome;
- consume the exact locked tensor;
- verify source and model identities;
- record seeds and complete frozen configurations;
- save finite reconstruction tensors and PNGs;
- recompute raw and rot180 PSNR offline;
- record runtime and GPU memory;
- package compact logs and results;
- leave full-panel authorization false.

Any mismatch stops the checkpoint. It must not be repaired by regenerating
measurements or changing a method.
