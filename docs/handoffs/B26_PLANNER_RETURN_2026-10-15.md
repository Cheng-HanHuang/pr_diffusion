# B26 PLANNER_RETURN — 2026-10-15

**Disposition:** `MACHINE_COMPLETE_PASS` · **Executor recommendation:** `stop`.

## 1. Frozen execution identity and scope

- Repository: `Cheng-HanHuang/pr_diffusion`.
- Frozen B26 execution branch: `codex/b26-np-conditional-correction`.
- Local and remote B26 HEAD: `5beb7f6485c5a7e44be19bcf4dc1cc5cd47603d2` (clean working tree at launch).
- Frozen B25 base: `codex/b25-noise-selection-mechanisms` at `c906d36e0e396a6abbf761e3d65c91433428170f`.
- PAC run: `/egr/research-pac/huang248/outputs/pr_diffusion/b26/B26_20260915T153700Z`.
- Run was created on 2026-09-15. The CPU arm was already complete when recovered on 2026-10-15; GPU smoke, DEV16, and DEV80 were executed in that original run with cumulative job reuse. No new run was created.
- `PRE_RUN_IDENTITY.json` PASS. B26 spec SHA-256: `c5355afddca9028dc006958cf12eee7f2918bb59f6ef9d5aad3f654fa5261ecd`. Frozen DEV80 manifest SHA-256: `c31e1ab9c8b21ff3ded632baea02234b1cae518022dac4eab3c2aff978d0c445`. Both hashes independently checked against the copies in the compact evidence archive.
- B25 capsule SHA-256 recorded and checked before GPU smoke: `5b1e785ba46e6be952ffdbc768c5d9bab365e495cf9efd5b154b075ce0081354`.
- Frozen panel: 80 DEV images (20 each from screening strata A, B, C, D); four original NP roots per image per arm; no confirmation images or new FFHQ measurements.

## 2. Execution results and resource gates

| Stage | Cumulative required PASS jobs | Newly attempted jobs | Reused PASS jobs | Gate |
|---|---:|---:|---:|---|
| Smoke | 8 | 8 | 0 | PASS |
| DEV16 | 128 | 120 | 8 | PASS |
| DEV80 | 640 | 512 | 128 | PASS |

- No recorded GPU job failures; all four workers finished PASS at each stage.
- Every GPU trajectory had exactly 2,200 UNet evaluations: 640 total trajectories, **1,408,000 total UNet evaluations**. H and R each contributed 320 trajectories and 704,000 UNet evaluations. Each arm used 703,680 scoring FFT evaluations and 703,680 diagnostic FFT evaluations.
- All 320 H historical-replay records have matched terminal SHA, RNG SHA, selected-noise SHA and permitted selector-score tolerance. The official smoke, DEV16 and DEV80 gates all attest historical H replay PASS.
- Every one of 640 completion receipts has a result JSON with a matching recorded SHA-256 digest; independently checked in the compact evidence archive.
- Aggregate worker reservation: **15,951.757513 seconds = 4.431043754 GPU-hours** (24 GPU-hour ceiling). Model loading across workers: 7.904699 seconds, for **4.433239503 GPU-hours** of reservation plus separately accounted model loading. Reservation is worker wall accounting, not independently measured kernel-active GPU time. Sum of synchronized per-root job wall times: 15,858.789085 seconds.
- Maximum monitored process GPU memory: **2,976 MiB** vs 52,452 MiB hard limit; four specified GPU IDs (0–3) used; no conflicting compute jobs reported at launch.
- CPU-only stage: 128 synthetic observations; 768 method–observation combinations; 512 reverse trajectories each; **393,216 completed CPU reverse trajectories**. CPU resource PASS, return code 0, 12.739551-second wrapper wall time, peak reported RSS 0.042194 GiB vs 16 GiB limit, CUDA disabled.
- Official `B26_COMPLETE.json`: **PASS**. No confirmation payload access, no new FFHQ measurements, no weighted-FFHQ reconstruction, no PSNR-directed stage advancement. There are no preserved GPU failure records.
- GPU autopilot was successful, ending `B26_AUTOPILOT_END_RC=0` on 2026-10-15 at 22:45:56 UTC. Stage transition and machine completion were performed with the frozen scripts. No code or frozen-config changes were made.

## 3. B26.1 scientific findings: historical H vs signed-score R

**Experimental definition.** H scores and selects against `y_plus = max(y_raw,0)` and projects against `y_plus`. R scores and selects against signed `y_raw`, but still projects against `y_plus`. Both use the same frozen native NP settings, root seeds and workload. The primary executable terminal selector is H+/Rraw, respectively; the alternatives Hraw and R+ and four-root GT oracles are analysis only.

| DEV80 endpoint | H | R |
|---|---:|---:|
| Mean selected raw-tensor PSNR | 25.7424 dB | 25.5209 dB |
| Good25 | 55/80 | 53/80 |
| Bad20 | 16/80 | 15/80 |
| Mean four-root GT-oracle PSNR | 26.0779 dB | 25.9763 dB |
| Four-root GT-oracle Good25 | 57/80 | 56/80 |

- Paired mean R−H selected PSNR: **−0.2214856 dB**; stratified image bootstrap 95% percentile CI **[−1.7165184, +1.2363824] dB** (10,000 replicates, frozen A/B/C/D strata); development-only inference.
- Good25 transitions: 47 succeed under both, 19 fail under both, **6 R rescues**, **8 R harms**. No defensible overall improvement from signed-score correction.
- Cross-target terminal selection: H+ mean 25.7424, Good25 55; Hraw mean 25.8242, Good25 55; Rraw mean 25.5209, Good25 53; R+ mean 25.5196, Good25 53. Selector index differences H+/Hraw on 3 images and Rraw/R+ on 1 image. These off-diagonal selectors are not a new prospective policy.
- Pre-projection winner-target disagreement counts: **6,825** over H trajectories and **6,410** over R trajectories. This is a diagnostic count of score-target candidate disagreement, not an estimate of scientific efficacy or independent proof of improvement.
- Stratified mean R−H PSNR: A −0.0570 dB, B −2.1679 dB, C +0.8653 dB, D +0.4737 dB. Good25 H→R: A 20→20, B 17→15, C 9→10, D 9→8.
- Frozen 10-image joint-baseline-failure subset: H mean 17.9900, R mean 20.8461 (R−H +2.8561 dB), but Good25 **2/10 under both**, with 2 rescues and 2 harms; this is exploratory and heterogeneous, not a stable Good25 rescue result.
- Illustrative large paired switches: image `08547` R−H −24.083 dB; image `38516` R−H +16.343 dB. The signed-score target can change trajectory basin assignment and cause extreme individual outcomes in both directions, even where the selected root index agrees.

**Conclusion:** The hypothesized signed-measurement-score correction does not yield an aggregate DEV80 benefit under the frozen four-root native NP policy. The result is useful as a controlled negative finding and as evidence of substantial per-image trajectory sensitivity; the wide bootstrap interval should not be presented as proof of exact equivalence. No population/held-out inference is authorized.

## 4. B26.2 scientific findings: finite inner-likelihood estimator

Finite-support CPU experiment: four frozen 6×6 template families, 32 independent truth/noise observations per family, six selection methods, K=5 outer proposals and 512 full reverse trajectories per method–observation. Exact reference posterior is available for this controlled synthetic problem. Reported mean TV distances of terminal empirical law to the exact posterior:

| Family | Categorical exact intermediate | Categorical Lhat M=1 | M=4 | M=16 |
|---|---:|---:|---:|---:|
| Ambiguity unequal | 0.06311 | 0.08197 | 0.06470 | 0.06531 |
| Distinguishable equal | 0.11676 | 0.15387 | 0.11670 | 0.11633 |
| Near-ambiguity equal | 0.03245 | 0.03219 | 0.02925 | 0.03098 |
| Near-ambiguity unequal | 0.02504 | 0.02467 | 0.02348 | 0.02366 |

- Four inner likelihood draws already approach the exact-intermediate categorical comparator in all four families; M=16 offers no consistent additional terminal-TV improvement. Each M=1,4,16 family uses 327,680; 1,310,720; and 5,242,880 inner template evaluations respectively, aggregated over 32 observations.
- Hard exact-intermediate selection omitted a reference-posterior mode in **32/32** near-ambiguity-equal observations and **25/32** near-ambiguity-unequal observations; categorical exact-intermediate incurred **0** such omissions in these families.
- Exact intermediate scores do not eliminate the finite K=5 outer proposal mismatch. Distinguishable/equal categorical exact-intermediate TV was 0.11676 despite an exact reference posterior; hard exact-intermediate had much lower TV (0.00171) in that particular family but failed severely in near-ambiguous families.
- Posterior fidelity and sampled-truth MSE are different objectives. For ambiguity/unequal, denoised-point categorical selection had worse TV (0.18018 vs 0.06311) but better sampled-truth MSE (0.022585 vs 0.032032) than exact-intermediate categorical sampling.

**Conclusion:** The finite Monte Carlo intermediate-likelihood estimator is computationally tractable on this finite-support synthetic model, and small M can approximate the exact-intermediate categorical comparator. This does **not** validate the true intermediate likelihood or a corrected posterior sampler for pretrained FFHQ diffusion, and does not authorize a weighted-NP production pilot.

## 5. Evidence artifacts and checksums

**Full official PAC archive**
- PAC: `/egr/research-pac/huang248/outputs/pr_diffusion/b26/B26_20260915T153700Z.tar.gz`.
- SHA-256 (PAC packaging/sidecar verified): `22acb5896072e80c4c86b5cb8c85f4abbcdc402defb72b0c01415ade373b6760`.
- 2,616 archive members; 2,615 internal file checksums verified by the official packaging script; ~456 MiB on PAC.
- Google Drive file: https://drive.google.com/file/d/1mQj3pQbhd8yollrmgVc10_jkiazdiEsY/view?usp=drivesdk ; Drive-reported size 477,402,020 bytes. Its bytes were **not** independently downloaded and rehashed in this executor session; checksum is from the successful PAC verification.

**Compact planner-evidence archive**
- PAC: `/egr/research-pac/huang248/outputs/pr_diffusion/B26_PLANNER_EVIDENCE_20261015.tar.gz`.
- SHA-256: `c803b284fd8bafea3be5cb754a511939895ce140f1b076e4d8d34b25d7752503` (PAC sidecar, attached file locally, and independently checked).
- 597,362 bytes, 1,325 archive members (1,324 with verified SHA-256 checksums). The 640 complete-job records and 640 result JSONs are present and cross-hash-validated; no tensor or trace payloads (full archive retains those).
- Google Drive file: https://drive.google.com/file/d/14ogPXk4JK-1esNxLiPnprBXMvdeGATZV/view?usp=drivesdk ; Drive-reported size matches 597,362 bytes.
- Parent Google Drive folder: https://drive.google.com/drive/folders/1j2cNKrzTluLhimqM40YNoMLBfGpaRVsd (`prdiffusion`). Both archive names present in the final folder listing.

**SHA-256 of key machine files (PAC postrun review)**

| File relative to run | SHA-256 |
|---|---|
| `PRE_RUN_IDENTITY.json` | `acad73e51e30960e8f4121d7c9873d354c109e8da26a4e431f3d8a5c9be61229` |
| `gpu/GATE_smoke.json` | `c0f65a039f39a10dbe99836676aea3fc778cb360b24fd29c55baed4c7ca6c25b` |
| `gpu/GATE_dev16.json` | `3571866f96188e641a5668c00b4307d196cd071916de07a9dcb97cabfb4b5bd3` |
| `gpu/GATE_dev80.json` | `7e2f075597cee77ceda1ea08e613cb087d242743e5fe4d353d1a6ed243caff6a` |
| `gpu/ANALYSIS_dev80.json` | `06cf8d0cb6155343ea7b25054cd9e5a03dc26d9d32348da53b6d99546cfa2dea` |
| `cpu/experiment/CPU_SUMMARY.json` | `89f921500be1af3a7b763d8eb89dbebf744d8fdcbb261f2d2a4afce18d419f76` |
| `cpu/CPU_RESOURCE.json` | `c4e497f6c2341d79ce3c8522d35c26756966d8ae5494a6149e65b763023fcf5c` |
| `B26_COMPLETE.json` | `99614372b914df2952f88b11e3664a2ceb3a9bb0895994838742b9c14ec5eac5` |

## 6. Recommendation and limits

**Select exactly: `stop`.** B26 has achieved the frozen development experiment and technical evidence goals. The corrected score target fails to produce an aggregate DEV80 gain and causes both major rescues and major harms. The synthetic likelihood work is informative but is not sufficient validation to jump to a much larger weighted-FFHQ campaign. No further B26 technical defect is identified in the delivered PASS evidence that requires a scoped repair. Preserve the negative result, hand it back to the scientific planner, and keep confirmation305 closed.

Not authorized without a new prospective planner specification: confirmation runs or payload access, any weighted-NP FFHQ campaign, new FFHQ measurements, additional DAPS/SITCOM/Fresh baseline execution, changes to frozen scientific code, stage or selector settings, or repository history rewriting.

**Planner action:** accept the complete B26 capsule and evidence, close this frozen executor task, and treat any new algorithmic hypothesis as a separate study with its own authorization and budget.
