# Planner notes to the XP executor, round 17 (2026-11-03)

**Responds to:** the A20 budget stop and compute request (XP `aa87f74`, Issue #46), the Nov 4 publication (`953e2f0`) and XP-I2 production (100/100 bags).

**Governing amendment:** prereg **A21**.

## 1. XP-C3: approved as A21

Your stop was correct. A20's order made G5 unreachable once XP-N's projection exceeded 8 GPU-h, and continuing silently would have broken the rule.

**Cheng's instruction (Nov 3):** several workers per GPU, longer limits, all four GPUs. A21 implements it. Your questions (i)–(v):

| Your question | A21 answer |
|---|---|
| (i) Budget and schedule | **Ceiling 56 GPU-h in total** (8 + your 48); 400 cap unchanged. Main text: panels returned by **Nov 6, 12:00 ET**. Hard stop: ceiling or **Nov 15, 12:00 ET** |
| (ii) GPU admission, GPU 0 | **GPUs 0–3**, clean or co-tenanted (no XP-C3 timing is reported). Free memory ≥ measured peak × 1.25 + **8 GiB** to start a worker; below **4 GiB** free, start nothing there and stop one worker after its current chunk. Replaces the 24,576 MiB request and the 80% limit |
| (iii) Workers and batching | **Up to 8 workers per GPU** (32 in all). Calls of B ∈ {8, 32} examples of a **single t** (any j, both orientations, several candidates). No fp16/bf16, no TF32/cuDNN/cuBLAS flag changes, no compilation |
| (iv) Pilot and accounting | **Equivalence:** rescore the 64 R2 smoke candidates. Pass if max \|ΔL_000\|, max \|ΔL_111\| ≤ 1e-5 and `prior_choice` agrees wherever R2's \|margin\| > 1e-5. Fail → production runs R2's code with the concurrency. **Accounting:** GPU-occupancy hours, the per-GPU union of XP-C3 process lifetimes summed over GPUs; worker-hours reported for information. Pilot ≤ 30 min and 2 GPU-h |
| (v) Order, skip, deadline | XP-N → G5 → XP-D4 as a priority queue. A panel that cannot finish within the remaining ceiling and before the hard stop is skipped, so G5 may run when XP-N cannot. Complete panels only |

**Not adopted:**
- **Filling memory to the limit.** Your smoke shows GPU 2 had about 75 GB free with seven co-tenants, so memory is not the constraint. Speed comes from the share of compute time, which more busy workers and larger calls increase. Memory held beyond need adds nothing and invites out-of-memory exits, ours and theirs.
- **Scoring G and O_g only.** Exact, but unnecessary at this ceiling.

## 2. Implementation suggestions (yours to decide; none changes the estimator)

1. **Batch at fixed t.**
   - One call per t covers 4 j × 2 orientations for each candidate: B = 8 is one candidate, B = 32 is four candidates.
   - R2's `exact_index` gate then holds unchanged.
2. **Keep R2's numerics path.**
   - Per-(uid, t, j) CPU generators, as now.
   - Per-example means on the GPU in fp32, as now.
   - Then accumulate in Python floats in R2's order (t outer, j inner). The only remaining difference is the convolution algorithm cuDNN picks for the batch size, which the equivalence test measures.
3. **Fewer synchronizations.**
   - R2 calls `float()` twice per call: 16 synchronizations per candidate, each one an idle gap that co-tenants fill.
   - Move each call's per-example means to the CPU once.
   - Verify and load the next chunk's `.pt` files on CPU threads while the GPU works.
4. **Concurrency.**
   - Claim chunks with exclusive-create claim files.
   - Write results as now (exclusive create), so two workers can never write the same image.
   - Reclaim a chunk only when its claimant's PID is dead.
5. **Ramp.** For example: one worker per GPU first. Then add workers while the measured candidates per GPU-occupancy hour on that GPU keeps rising and admission holds.

**Rough expectation (not a measurement).** Each extra busy process gets roughly its own share of the GPU's time slices. On GPU 2, with seven co-tenants:
- one worker gets about 1/8;
- four get about 4/11;
- eight get about 8/15.

Larger calls also use each slice better. The 13,600 candidates should then take hours on four GPUs, far below the ceiling. Your pilot decides.

## 3. One PAC block, from a pushed clean commit

1. The controller checks the freeze: code, A21 constants and equivalence rule.
2. It runs the equivalence pilot (a) on one GPU.
3. It picks the configuration (b).
4. It runs production over the priority queue.
5. It ends with a receipt (A21.10) and, per complete panel, the six-column CSV in `returns/` and `theory_inputs/`.

**Please return:**
- XP-N as soon as it completes, without waiting for the other panels;
- G5 and XP-D4 when they complete.

I compute E1-prior against E1-CLIP on each panel.

## 4. Nov 4 publication (`953e2f0`): received

- I am reproducing A9.2(d) (exact E1 face-CLIP curves; 120 comparisons with v1) from your CSV and building Fig. 3 from the 15 PNGs.
- Results come in round 18. Your headline, S2 7/7 and race 0–3/7 per cohort on test 1's E1 points, goes into Table 2 as registered, whatever it does to the tally.

## 5. XP-I2: please score it now (CPU, in parallel with XP-C3)

Production is complete (100/100 bags, 3.05 GPU-h). **Scoring is due Nov 5, 12:00 ET for the main text (A15.5),** otherwise the supplement. Same form as XP-I (A15.3; your per-particle table at `82d1e4a`), on the 100 ImageNet images, GOOD = PSNR ≥ 20 dB.

**(a) Same particles.**
- Per particle: R0 label, PC-DAPS `pr_outcomes` label (A4.1 rule), ρ, G1 CLIP choice and margin, and PSNR, SSIM and VGG-LPIPS at the chosen orientation.
- Per policy: first particle, mean over the bag (PC-DAPS's reporting), E0, E1 (G1), O_raw and O_orbit.

**(b) Compute-labeled comparison against DAPS-4K D2-8.**
- Comparators:
  - run 0;
  - fixed E1 at K = 1 and 2;
  - the frozen protocol's pooled point on G5.
- Compute: NFE from the run receipts. Clean wall-clock for one PC-DAPS ImageNet bag on 3 ids only if a clean GPU exists; otherwise NFE only, stated as such.
- No matched-compute claim.

**Files:**
- the per-particle CSV and the policy table in `returns/`;
- strings and G1 decisions in `theory_inputs/`.

I reproduce the intervals (image bootstrap) myself.

## 6. Unchanged

- **Queue after XP-C3 and XP-I2:**
  - Tab. 3 E1 columns (A7 ZIP, Nov 6);
  - XP-K;
  - the rest of G7;
  - CP4 (Nov 15).
- No protocol v2, no new selector, no matched-compute claim.
- Stop and ask only under A13.4.
