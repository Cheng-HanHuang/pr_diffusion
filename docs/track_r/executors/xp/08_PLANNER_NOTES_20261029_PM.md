# Planner notes to the XP executor, round 8 (2026-10-29, 12:35 ET)

**Responds to:** issue #42 comments `6501694685` (F8 complete) and `6501968376` (GPU scheduling request); B16 admission fix `27359a6`.

**Governing amendment:** XP prereg **A12**, which **authorizes GPU work**. It replaces the "no new GPU stage" line of round 7 §3.

## 1. Accepted

- **F8:** 920/920 with zero failures; A10 total 55.55 GPU-h.
- **B16 admission fix.** Preserving the original receipt and verifying its SHA is the right way to do it.
- **Protocol v1 is certified independently.** The theory executor derived the exact random-order value of the stop-at-M rule on a finished panel: E T = M(n+1)/(L+1), and the first M passing runs form a uniform M-subset. Your frozen evaluator agrees with it within 3e-4 on all five dev panels. The exact form is a cross-check only; the frozen script stays the reference.

## 2. GPU: one campaign, authorized now (A12)

- **The 400 GPU-h project cap is unchanged** (about 123 used).
- **A10's 70 GPU-h sub-cap is replaced** by a 150 GPU-h A12 campaign cap.
- **Accounting stays wall-hours per occupied GPU.** Ceilings assume co-tenancy at 2× the clean per-run time, plus 25%. Every stage stops at its ceiling and reports a partial result.
- **In the first A12 receipt,** publish the cumulative project ledger: every stage since XP-0, in GPU-h.

**Launch order** (each stage starts when a GPU is free; FAST stages first, because they are cheap):

| # | Stage | What | Runs | Ceiling (GPU-h) | Gate | Done by (ET) |
|---|---|---|---|---|---|---|
| G1a | XP-D4a | FAST ImageNet, runs 16–31 | 1,600 | 25 | 2-root replay of D1 roots | Nov 3, 12:00 |
| G2 | XP-G | FAST FFHQ, σ ∈ {0.02, 0.10}, n = 8, 180 ids, paired-noise y (§1) | 2,880 | 10 | smoke per σ | Nov 3, 12:00 |
| G3 | XP-I | PC-DAPS @ `59b2f046`, own FFHQ PR command, 1 run × 4 particles, 180 ids; save all particles and its own estimate | 180 | 45 | **Only if the code is present.** If absent, post the exact clone command for Cheng. 2-id smoke; if the projection > 45, cohort A only | Nov 3, 12:00 |
| G4 | XP-E′8 | DAPS-4K FFHQ, runs 4–7 | 720 | 30 | 2-root replay of E′ | Nov 4, 12:00 |
| G5 | XP-D2-8 | DAPS-4K ImageNet, runs 4–7 | 400 | 40 | 2-root replay of D2 | Nov 4, 12:00 |
| G1b | XP-D4b | FAST ImageNet, runs 32–63 | 3,200 | 40 | only after G1a–G5 have launched and the campaign projection fits | Nov 4, 12:00 |
| G6 | XP-T clean | DAPS 400 × 5 FFHQ and FAST ImageNet batch-1 timing, if missing | 18 | 1 | clean GPUs only | opportunistic |
| G7 | XP-C GPU, XP-C2 | orientation tests and the equivariance defect on GT (supplement) | — | 4 | — | Nov 15 |

**Not authorized:** B32, XP-K, XP-H, XP-G2, XP-J, XP-I2.

**Gate tolerances.** Replays pass bitwise, or at 0.1 dB raw PSNR with an identical label (A10.1). Seeds follow §1; for XP-G the rule is `XP|FAST|ffhq_s<σ>|<uid>|<r>`.

**Seals and scoring:**
- **XP-D4 outcomes (runs ≥ 16) stay sealed** until a theory `v4` is registered or declined, or Nov 2, 12:00 ET.
- **Generation may finish earlier;** do not score it.
- **The other stages are scored on completion,** under the protocol hypotheses frozen in A12.4:
  - H1/H3/H4 on XP-D4;
  - H1 on E′8 and D2-8;
  - H-G1/H-G2 on XP-G;
  - XP-I is descriptive at matched NFE.

## 3. CPU order (unchanged in substance; one swap)

1. **D1-P2 strings and per-candidate CLIP decisions, exported first** (round 7 §2.2). This is a quick export, and it unblocks two things: the first held-out protocol result, and the theory `v4` fit, which needs the 16-run strings by about Oct 31. Copy them to `theory_inputs/` once committed (A11.4). Then run the six protocol commands.
2. **B16 against v1/v3 plus A9.2,** then B16 under the protocol.
3. **F8 against v1/v3 plus A9.2,** then F8 under the protocol.
4. **The H1–H4 table, the unsolved-images PNG,** E′/A8.6, A5.5.

## 4. Return when

- the D1-P2 protocol outputs are committed;
- the B16 comparison is committed;
- any A12 stage passes its gate, or hits its ceiling;
- the cumulative GPU ledger is published (first A12 receipt);
- any stop condition fires.

Every return: a digest of at most 20 lines.
