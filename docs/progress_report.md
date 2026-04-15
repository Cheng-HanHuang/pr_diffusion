# prdiffusion progress report

This note records the current experimental status of the **prdiffusion** project, including the early institution-cluster runs, the migration to PAC, the setup/debugging steps that were required, the main experimental findings so far, and the current recommended next steps.

---

## 1. Current objective

The current goal is to build a reliable and scalable experimental pipeline for the phase-retrieval diffusion project using:

- the face-prior setup,
- `google/ddpm-celebahq-256`,
- magnitude-only Fourier measurements, and
- two main reconstruction methods:
  - **SITCOM**
  - **Noise Picking**

The current overall strategy is:

- use **PAC** as the active experiment machine,
- use the institution cluster as **background confirmation** when available,
- avoid unnecessary delays from long queue times, and
- keep PAC naming neutral and non-venue-specific.

---

## 2. Early institution-cluster experiment status

### Initial submission order

The first wave of institution-cluster experiments was submitted in the following order:

1. **Phase 0** — sanity check
2. **Phase 2** — SITCOM tuning
3. **Phase 1** — radius validation

### What happened

- **Phase 0** failed early.
- **Phase 2** completed successfully.
- **Phase 1** failed once due to a bug, then later timed out, and later remained in a long queue.

### Interpretation of these outcomes

#### Phase 0

This was only a sanity check and is not scientifically important. It can be safely ignored.

#### Phase 1 initial failure

This was traced to a bug in `neurips_canonical_compare.py`, not to data paths, memory, or environment problems.

The cause was:

- config rows from both SITCOM and Noise Picking were being written into one CSV,
- only the first row's fieldnames were used,
- Noise Picking rows had additional fields,
- which caused Python to raise:

```text
ValueError: dict contains fields not in fieldnames
```

This was fixed by changing the CSV-writing logic to use the **union of all row keys**.

#### Phase 1 timeout

A later rerun with a 16-hour walltime timed out. This suggested that the job itself was valid but 16 hours was too short.

#### Phase 1 queueing delay

A later rerun with a 24-hour walltime remained in queue for a long time with:

```text
ReqNodeNotAvail, May be reserved for other job
```

This indicated that the bottleneck was **H200 long-GPU scheduling**, not correctness of the experiment code.

---

## 3. Phase 2 SITCOM tuning results

Phase 2 completed successfully on the institution cluster.

### Main findings

#### Learning-rate sweep

- `lr_inner = 0.02` was the best overall choice.
- `lr_inner = 0.1` was clearly too large.

#### `eta_scale × init_scale` sweep

- `eta = 0.5` improved PSNR,
- but worsened measurement error substantially,
- producing a clear quality / consistency tradeoff.

### Practical takeaway

For SITCOM:

- move away from the earlier `lr_inner = 0.05`,
- use **`lr_inner = 0.02`** as the current best choice,
- keep both a quality-oriented and a more balanced `(eta, init)` combination in mind.

This tuning was useful, but SITCOM tuning is no longer the main bottleneck in the project.

---

## 4. Migration to PAC

Because the H200 queue became too slow, PAC was adopted as the active development machine.

### PAC path choices

**Repo**

- `/egr/research-pac/huang248/pr_diffusion_repo`

**Small probe data**

- `/egr/research-pac/huang248/data/celeba_hq_256_probe`

**Main PAC output root**

- `/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411`

### PAC setup issues that were solved

#### 1. Repo access

The private GitHub repo could not be cloned directly on PAC via HTTPS because password authentication is not supported for private Git operations.

**Fix:**
- copy the repo from the institution machine to PAC.

#### 2. Dataset availability

PAC initially had no phase-retrieval dataset prepared.

**Fix:**
- copy only the required image subset first,
- rather than trying to move or download the full dataset immediately.

#### 3. Hugging Face model loading

Model loading on PAC eventually worked directly.

The following messages were observed but were only warnings, not failures:

- missing `safetensors`,
- unauthenticated HF access warning,
- deprecated symlink argument warning.

#### 4. Python module import

Direct script execution initially failed with:

```text
ModuleNotFoundError: No module named 'prdiffusion'
```

**Fix:**
- set:

```bash
export PYTHONPATH=$REPO_ROOT:$PYTHONPATH
```

#### 5. Naming concerns

Because words like `neurips` were visible in process names and output paths, PAC-local neutral copies were made, such as:

- `pr_canonical_compare.py`

and neutral output roots were used, such as:

- `phase_retrieval_20260411`

### tmux usage

PAC experiments were moved into `tmux`, which is the correct practice and should remain standard for all longer PAC runs.

---

## 5. PAC 10-image Phase-1 probe

A smaller PAC-based probe replaced waiting for the full cluster validation.

### Probe configuration

- split: `validation_10`
- seeds: `100,101,102,103,104`
- radii: `0.1, 0.2, 0.5`
- comparison:
  - Noise Picking masked
  - SITCOM unmasked

### Main outcome

The PAC probe completed successfully and clearly validated the PAC setup.

### Main conclusion from the probe

The 10-image probe showed:

- `r = 0.5` is the best **working default**,
- `r = 0.2` is the best **secondary / conservative check**,
- `r = 0.1` is no longer needed as a main candidate.

This was a stronger signal than the earlier 5-image study, where `0.2` looked best by mean and `0.5` mainly by best-of-R.

### Working interpretation

The current rule became:

- **primary radius = 0.5**
- **secondary check radius = 0.2**

This was enough to move beyond tiny-data radius studies.

---

## 6. PAC schedule tuning at radius 0.5

After the working radius was frozen at `0.5`, schedule tuning was run on PAC.

### Sweep dimensions

At `r = 0.5`, the following were swept:

- `num_candidates_soft ∈ {3, 5, 7}`
- `num_candidates_hard ∈ {1, 2, 3}`
- `proj_start ∈ {200, 400, 600}`

The sweep was postprocessed into:

- `run_level.csv`
- `image_level.csv`
- `split_summary.csv`

### Main findings

The strongest signals were:

- `num_candidates_hard = 1` is best,
- `proj_start = 400` is best,
- `num_candidates_soft` is a tradeoff:
  - `7` slightly improves some summary values but costs substantially more runtime,
  - `5` looks like the best practical compromise,
  - `3` is clearly cheaper but worse in quality.

### Provisional schedule after this sweep

- `soft = 5`
- `hard = 1`
- `proj_start = 400`

A smaller direct confirmation run was then used to validate this choice.

---

## 7. Combined schedule confirmation run

A direct three-setting confirmation run was performed on PAC at `r = 0.5`.

### Compared settings

1. **balanced**  
   `soft = 5, hard = 1, proj_start = 400`

2. **quality**  
   `soft = 7, hard = 1, proj_start = 400`

3. **fast**  
   `soft = 3, hard = 1, proj_start = 400`

### Results

The result was clear:

#### Balanced

Best overall:

- best mean PSNR,
- best median PSNR,
- best max PSNR,
- moderate runtime.

#### Fast

Clearly cheaper, but significantly worse in quality.

#### Quality

More expensive, but did **not** outperform balanced.
It appeared less stable overall.

### Final schedule conclusion

The main Noise Picking schedule is now frozen as:

- `radius = 0.5`
- `num_candidates_soft = 5`
- `num_candidates_hard = 1`
- `proj_start = 400`

This is now the current **main PAC configuration**.

---

## 8. Why “quality” lost to “balanced”

This became an important conceptual point.

### Core explanation

Using more soft candidates does **not** guarantee a better final reconstruction.

Why:

- the same seed values do **not** imply the same reconstruction trajectory once candidate count changes,
- the method optimizes a **local low-frequency proxy score**, not final PSNR directly,
- increasing the candidate count can make the policy too greedy toward the proxy,
- which can hurt later denoising or global image recovery.

So “quality” losing to “balanced” is scientifically sensible.

### Working interpretation

The method story is not:

- “the more soft guidance the better”

but rather:

- “soft guidance helps, but must be balanced.”

This is actually a stronger and more believable result.

---

## 9. Current main PAC configuration

At this checkpoint, the main PAC Noise Picking setting is:

- **radius = 0.5**
- **num_candidates_soft = 5**
- **num_candidates_hard = 1**
- **proj_start = 400**

Secondary / backup settings:

- `radius = 0.2` as a secondary check,
- `soft = 3, hard = 1, proj_start = 400` as a fast / budget point.

---

## 10. Current PAC data status

### Already present on PAC

- repo copy,
- working environment,
- Hugging Face model availability,
- `validation_10.txt`,
- the 10 corresponding probe images,
- working canonical comparison pipeline,
- working schedule-tuning pipeline.

### Recently copied or being prepared

On the institution machine, split files and larger staged subsets were being prepared for PAC, including:

- `dev_10.txt`
- `validation_10.txt`
- `validation_20.txt`
- `validation_25.txt`
- `test_20.txt`
- `test_50.txt`
- `seed_list_10.txt`

The next image subsets planned / copied for PAC were:

- `validation_25`
- `test_20`

The PAC plan remains a **staged migration**, not a full 5400-image migration all at once.

---

## 11. Current experiment state

### Active on PAC

- **mechanism ablation** has been launched.

This is the correct next experiment because it tests whether the method really needs:

- masked score,
- masked projection,
- or both.

### Institution machine

- split files and the next image subsets are being copied to PAC.

### Institution cluster

- the full larger Phase-1 validation remains queued,
- and now functions only as a **confirmation run**, not a blocker.

---

## 12. What has already been established scientifically

At this checkpoint, the following statements are already supported:

1. **Noise Picking with masking clearly beats unmasked SITCOM** on the current PAC validation slices.
2. **Masking matters**.
3. A larger soft candidate count does **not** automatically improve final reconstruction.
4. The best current PAC working setting is:
   - `r = 0.5`
   - `soft = 5`
   - `hard = 1`
   - `proj_start = 400`
5. PAC is now a viable active experiment machine for this project.

---

## 13. Recommended next steps

### Immediate

1. let **mechanism ablation** finish on PAC,
2. verify PAC has the larger split files and the next staged image subsets.

### Next

3. run a **PAC main-comparison pilot on `test_20`** using the frozen balanced setting.

### Later

4. use the queued full cluster Phase-1 run as larger-split confirmation,
5. decide whether PAC should absorb more later runs,
6. only then consider broader subset migration or larger-scale PAC expansion.

---

## 14. What should not be repeated now

At this point, the following are no longer the priority:

- more 5-image or 10-image radius studies,
- more schedule sweeps,
- waiting for the cluster before progressing,
- copying the full 5400-image dataset to PAC immediately.

That tuning branch is sufficiently settled.

---

## 15. One-paragraph summary

The project has successfully moved from fragile setup/debugging into a real PAC-based experiment workflow. Early institution-cluster runs revealed a canonical-comparison CSV-writing bug and then severe queueing delays, which made PAC the active machine. On PAC, the repo, environment, model loading, subset-data transfer, neutral naming, and tmux-based workflow were all made to work. A 10-image validation probe established `r = 0.5` as the main working radius, and schedule tuning plus a direct three-setting confirmation established the balanced Noise Picking schedule `(soft=5, hard=1, proj_start=400)` as the best overall PAC setting. Mechanism ablation is now running on PAC, larger split files and staged subsets are being transferred, and the project is ready to move into the next PAC main-comparison phase while the queued full cluster validation remains only as a confirmation experiment.
