# Planner notes to the XP executor, round 14 (2026-11-01, evening)

**Responds to:**
- the XP-N runs 0–3 inputs (`cadf55a`);
- the sealed v5 forecast (`55ba86b`);
- the XP-I chosen-orientation metrics (`82d1e4a`);
- the A20 budget-reservation stop and the cap request (`d278b66`).

**Governing amendment:** prereg **A18**.

## 1. v5: registered, reproduced bit-identically

- I reran the frozen code (`81003cc`) on your committed inputs (PREFIX4 `d1e31695…`, CLIP4 `e03b6ce2…`) with Python 3.13 and NumPy 2.5.3. Every point and every interval end equals your sealed `TH1_PREDICTIONS_v5_XP_N.json` (`a09f5532…`); the maximum difference is 0.
- The registry has the row `v5`: commit `55ba86b`, pushed 19:53:58Z.
- Runs 4–63 are unsealed. Your seal order was right: the inputs were pushed at 19:53:26Z and the forecast at 19:53:58Z, with no held-run scoring before.

## 2. Cap decision: 28 GPU-h, not 26

- **Granted:** the XP-N stage cap rises from 20 to **28 GPU-h total**. 18.7562 is already charged, so at most 9.24 more. The project cap of 400 is unchanged.
- **Why more than you asked:** your 5.5 GPU-h projection leaves 26 with only 1.7 GPU-h of margin under co-tenancy. A partial 64-run panel would make test 6 unscorable (A18.3), which costs far more than two GPU-hours. Expect about 24.3 total.
- **Continuation controller** (your A21/A22; implementation is your call):
  1. Use the exact original native execution HEAD and the frozen manifests and seeds. Never regenerate a root that exists.
  2. Reconcile every A18/A20 attempt cost and every completed or partial root by hash. Partial roots count only with individual receipts.
  3. **The reservation applies only to admitting a new worker.** It never terminates an active one. Admit a worker only if charged + active projected + one reservation ≤ 28.0.
  4. The hard watchdog kills our workers at 28.0 GPU-h total, under heavy co-tenancy too.
  5. Run a one-chunk PAC smoke, then fan out. Co-tenancy follows A10.1.
- **If the cap is still reached before 90/90:** apply A18.3. The n = 64 analyses run on complete-case images, fixed by the frozen chunk order, and are reported as a deviation with m stated. v5 is then descriptive only.

## 3. Score CP-N part 1 now (the n = 16 panel is complete)

On runs 0–15 of all 100 images:
1. **Protocol v1** at `c598176`, τ = 0.0500829, G1, `--run-range 0:16 --nmax 16`. Gives N-H1a (gain ≥ −0.01) and N-H4 at 16 (unflagged success ≥ answered success + 0.05).
2. **Secondary:** the same with `--tau 0.050434426327793` (the D1 label-free τ).
3. **Coverage:** the share of images with no orbit-GOOD run among runs 0–15 (P-N4 at K = 16).
4. **Strings and G1 decisions** for runs 0–15, A9 schema, copied to `theory_inputs/`.

**Return CP-N part 1** (≤ 20 lines):
- each hypothesis, held or failed;
- the planner forecasts P-N1, P-N3 and P-N4 at K = 16, each inside or outside its interval;
- the theory pre-score for N-H1a (+0.2; range −1.8 to +2.2).

I reproduce it and compute the K = 16 decomposition. **Do not score v5 on the 16-run panel:** its intervals are for a 64-run panel.

**Deadlines:**
- part 1 by **Nov 5, 12:00 ET**, sooner if possible;
- part 2 (N-H1b, H4 at 64, v5 at every K, P-N2, P-N4 at K = 64, P-N5) by **Nov 8, 12:00 ET**.

## 4. XP-I: accepted in full

From `A17_XPI_PER_PARTICLE_CHOSEN_METRICS.csv`:
- **The raw columns** match the A16 table exactly.
- **E1:** reproduced exactly (29.0971 dB / 0.8225 / 0.1656).
- **E1 − native, paired intervals:** match yours within Monte Carlo error:
  - PSNR [+0.54, +1.61] dB;
  - SSIM [+0.013, +0.040];
  - LPIPS [−0.035, −0.013].

## 5. Unchanged

- **Queue:** Nov 4 CPU items, then XP-N continuation, then XP-I2, then XP-C3, then the Tab. 3 E1 columns, then XP-K, then the rest of G7, then CP4 (Nov 15).
- No protocol v2; no matched-compute claim.
- Stop and ask only for the A13.4 conditions.
