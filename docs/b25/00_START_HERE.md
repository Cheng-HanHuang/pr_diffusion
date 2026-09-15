# B25 — noise-selection bias and phase-retrieval failure mechanisms

Status: **B25 COMPLETE. CPU evidence accepted with reporting qualifications; control returned to the scientific planner. No new execution is authorized.**

B25 started from immutable B24 commit `ed162c2f97430804fddb5d9a0bfec7abde201ca0`. The signed B23.1 ancestor is `27505e6328157ac9296c95dc5e611cbeef80de98`. B24 remains closed under `STOP_B24_METHOD_REFINEMENT`; confirmation remains locked.

B25 branch: `codex/b25-noise-selection-mechanisms`. Draft PR: `#39`, based on `codex/b24-bestof4-failure-sweep`.

## Completed scientific identity

- pushed pre-run scientific commit: `32453db6445acec4fc19a4a928142a412d67f1ae`
- PAC run: `/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z`
- sealed capsule: `/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z.tar.gz`
- capsule SHA-256: `5b1e785ba46e6be952ffdbc768c5d9bab365e495cf9efd5b154b075ce0081354`
- capsule verification: 26 safe members; all 25 internal checksums verified
- scientific execution: CPU-only with `CUDA_VISIBLE_DEVICES=""`
- GPU work: none
- pretrained-model inference: none
- new FFHQ measurement/reconstruction generation: none
- confirmation305 payload access: none

The existing capsule is final and must remain unchanged. The planner explicitly required **no rerun and no repackaging** for the reporting corrections.

## Accepted interpretation

B25 provides a defensible next conditional-inference hypothesis. It does **not** establish a better FFHQ reconstruction method or a PSNR improvement.

Accepted findings:

1. In the controlled iid proposal model, hard best-of-K selection changes the small-step behavior: the tested hard-selection displacements scale approximately as `h^0.5`, while the tested likelihood-weighted displacements scale approximately as `h`. Matching the order alone does not prove that weighted selection has correct posterior dynamics.
2. In the near-ambiguity synthetic priors, exact-intermediate likelihood weighting is much closer to the exact posterior probabilities than hard selection or denoised-point weighting at `K=8`. This is posterior-fidelity evidence, not a reconstruction-accuracy result: hard selection has lower MSE to the single frozen truth in both near-ambiguity families. The distinguishable family prevents any universal claim that weighting wins.
3. The exact-ambiguity hard-selection row is tie-sensitive at floating-point precision and is excluded from evidence for an intrinsic hard-selection defect until explicit tie handling is examined.
4. NP's `clamp_min(0)` materially changes all 80 locked DEV observations (median relative L2 change approximately `0.08223`). This establishes an observation/preprocessing difference, not its effect on reconstruction quality.
5. The tested eight channelwise reversal transformations produce no new Good25 recoveries for DAPS or SITCOM on the frozen 10-image shared-failure subset. This conclusion is limited to the tested transformations and does not rule out every phase-retrieval ambiguity.
6. NP's `57/80` figure in the symmetry analysis is the **four-candidate oracle**. Historical clean-free NP selection is `55/80` Good25.

Correct full process accounting is approximately `166.263 s` total stage-process wall time with maximum recorded RSS approximately `0.649 GiB`. The smaller `159.669 s` / `0.550 GiB` values are inner measurements retained only for provenance.

## Final returned direction

The single B25 direction remains:

**a precisely defined NP conditional-selection correction, investigated prospectively at fixed compute**

The unresolved question is:

> **Can we estimate the intermediate conditional likelihood well enough to improve reconstruction at a fixed computational budget?**

Before any new FFHQ experiment, the scientific planner must specify the implementable intermediate-likelihood estimator, proposal distribution, finite-proposal approximation/resampling rule, lineage handling, and complete compute cost.

Any eventual FFHQ comparison must isolate:

- frozen historical NP;
- raw-measurement correction only;
- selection-rule/intermediate-likelihood correction only;
- combined raw-measurement + selection correction.

Fresh2 remains an important efficiency comparator. No new GPU stage is authorized by B25.

## Read this completed stage in this order

1. root `AGENTS.md`
2. this file
3. `docs/b25/B25_FINAL_PLANNER_RETURN.md`
4. `docs/b25/B25_FINAL_CHECKPOINT.md`
5. `docs/b25/B25_POSTRUN_NATIVE_NP_AUDIT.md`
6. `docs/b25/evidence/B25_FINAL_EVIDENCE.json`
7. `docs/b25/B25_ARCHIVE_SHA256.txt`
8. `docs/b25/B25_EXECUTOR_CONTRACT.md`
9. `docs/b25/B25_MATHEMATICAL_MODEL.md`
10. `docs/b25/B25_NATIVE_NP_AUDIT.md`
11. `docs/b25/B25_LITERATURE_AND_NOVELTY.md`
12. `configs/b25/b25_cpu_spec.json`
13. `configs/b25/b25_synthetic_templates.json`
14. historical B24 final-return records as needed for provenance

## Historical execution/provenance entrypoints

These files remain for reproduction/audit of the completed stage; they are not a launch authorization:

- synthetic Experiments 1–2: `scripts/b25/run_b25_synthetic.py`
- DEV80 Experiments 3–4: `scripts/b25/run_b25_dev_diagnostics.py`
- deterministic analysis: `scripts/b25/analyze_b25_results.py`
- integrity tests: `scripts/b25/test_b25.py`
- resource wrapper: `scripts/b25/run_cpu_stage.py`
- launcher/worker/status/resume/stop: `scripts/b25/launch_b25_cpu.sh`, `run_b25_cpu_worker.sh`, `status_b25_cpu.sh`, `resume_b25_cpu.sh`, `stop_b25_cpu.sh`
- capsule packager: `scripts/b25/package_b25_run.py`

Historical native NP semantics remain important to interpretation: candidate zero may reuse the previously selected noise when `eps_prev` exists and `K>1`, while the remaining candidates are newly sampled. B25's iid proposal study is therefore a controlled mechanism diagnostic, not a direct model of native NP.

## Hard restrictions remain in force

- Do not reopen B24 refinement.
- Do not expose confirmation305 payloads.
- Do not treat the GT-assisted symmetry oracle as deployable selection.
- Do not claim B25 established a reconstruction-quality improvement.
- Do not launch a new FFHQ/GPU stage without a new planner authorization and a prospectively frozen design.
- Preserve historical worktrees, outputs, the sealed B25 capsule, and PR history; do not force-push or rewrite them.
- Use `/egr/research-pac/huang248`, never `/home`, for any future PAC stage.

B25 is closed on the executor side pending a separately authorized follow-up design from the scientific planner.
