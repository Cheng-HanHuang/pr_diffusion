# A1.0c — DAPS native start-noise RNG audit: passed

Date: 2026-10-18. Track A continuing executor. **Status:** `DAPS_SOURCE_SEED_GATE_PASS` for Gaussian initial state, not a complete prospective diffusion-run reproducibility test. The user returned bounded PAC native source excerpts and Git HEAD; no new GPU, measurements or image/GT data used.

## Observations (literal PAC source excerpts)

- `sampler.py` defines `_b21_seed_all(seed)` using `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`.
- DAPS `get_start(self,batch_size,model)` resolves device and input shape from `model`, reads `B21_START_NOISE_SEED`, and **when present** invokes `_b21_seed_all(int(start_seed))` *before* drawing `torch.randn(batch_size,*in_shape,device=device)*self.annealing_scheduler.get_prior_sigma()`.
- Consequently, changing a 32-bit-compatible per-restart `B21_START_NOISE_SEED` in **separate fresh processes** changes the initial Gaussian RNG stream (subject to the usual pseudorandom seed caveat), not just metadata. It does not draw the initial start from ground-truth pixels.
- The parent `posterior_sample.py:main(args)` separately seeds NumPy, torch CPU and CUDA with `args.seed` and enables deterministic cuDNN. The frozen source HEAD is `e7a77d094167084faed19b599b96673b7bb11447` as in A1.0 V2.
- A1.0b already established that native SITCOM `get_start(ref)` only computes `torch.randn_like(ref)*sigma_max`, using reference shape/device/dtype, not GT pixel values; its entrypoint seeds CPU/CUDA/NumPy.

**Engineering inference, not a GPU result:** per-candidate process isolation, unique 32-bit-compatible seeds, fixed measurement and no continuation environment variables are a credible basis for prospective independent DAPS4 and SITCOM4 native-start sampling. However independence of generated *full trajectories*, deterministic replay, source signature at execution time, and end-to-end source/evaluation parity must still be checked on a bounded authorized run.

## Required A1 execution semantics

- The existing A1 `seed63` metadata identifier is *not automatically a valid native `np.random.seed` input*. A1 `seed32_numpy(tag,image_id,k)` produces disjoint SHA-256-derived uint32 seeds. Use **the same derived uint32 for DAPS/SITCOM native `seed`, `B21_START_NOISE_SEED`, `B21_SOURCE_SEED`**, with separate `DAPS` and `SITCOM` namespace tags. Record both canonical long ID and actual native uint32. Explicitly check per-image uniqueness rather than assuming no 32-bit collision.
- DAPS process must be fresh per restart, `num_runs=1`; clear B21 continuation envs and disable B20 LF modifications as in frozen B21/B24 source, with native schedule diffusion 5 and annealing 400.
- SITCOM process must be fresh per restart, native seed uint32, native schedule anneal 200, diff 5 and LGVD steps 100. The future A1-owned worker should pass a shape-only dummy reference instead of a GT tensor to `get_start` to make the non-GT-generation firewall explicit; offline GT PSNR only after reconstruction.
- NP LF/S2 seed normalization and complete generator call must be audited separately. NP mixture is reuse of 4 LF and 4 S2 full candidates, **never extra inference**.
- DAPS outputs as PNG vs NP/SITCOM float tensor is a material representation risk; implement one scored float metric convention and prohibit silent 8-bit vs float comparisons.

## Remaining scientific/resource block

No prospective image cohort has been allocated; 180 B21/B22/DEV80 images are exposed, while the 305-protected confirmation roster is excluded (never read or used to filter). Pure FFHQ IDs excluding only exposed panels cannot certify separation. Use an independently provenance-disjoint allowed cohort, or an approved new source for engineering-only development, before the first real-image measurement. Full A1.1 GPU campaign remains STOP until the new cohort, native runners, source/measurement/scope audit, and issue #41 DATA and RUN reservations.

**Next:** CPU-only A1-native K4 independent-restarter **dry-run manifest and source gate** on synthetic five-digit ID, with no external-image read or actual candidate execution. Preserve read-only historical source trees and all previous A1 STOP/PASS archives.
