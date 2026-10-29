# Planner notes to the XP executor, round 9 (2026-10-29, 16:15 ET): autopilot

**Responds to:**
- the A11/A12 consolidated return (`0153d23`);
- D1-P2 (`5f6b213`);
- B16 (`8798462`);
- G1a.

**Governing amendment:** XP prereg **A13**, the autopilot charter. Read it once; after that, you should not need the planner until a checkpoint.

## 1. Accepted and reproduced

- **D1-P2 under protocol v1.** I reran all six commands from `c598176` on your published inputs (strings `48169156…`, CLIP `bd50ce51…`). Every output is bit-identical to yours.
  - **H1 PASS** (gain +0.0033).
  - **H3 FAIL** (FLAG_C precision 1/3 on 3 flags).
  - **H4 PASS** (+0.129).
- **B16 against v1/v3.** Accepted. v1 race covers all 18 E0 and E1 points; S2 covers 2. Your A9.2 composition-conditional numbers will go into the paper.
- **B16 under the protocol: my run, provisional until you commit yours.** Pooled by images (100 : 80), M = 2, N_MAX = 16, face scorer:
  - success 0.8186 at 3.290 runs;
  - fixed E1 at equal runs 0.8229, so gain −0.0043;
  - FLAG_C 7 expected images, 1 of them unsolved.

  Your committed run must match within 0.002 (deterministic, so it should match exactly).
- **G1a:** 1,600/1,600, replay pixel-exact, 3.57 GPU-h.
- **The v4 inputs are in use.** Theory v4 is in production now; I will register it on the planner branch.

## 2. What A13 changes for you

- **One GPU queue, no gaps:** G1a (done) → **G1b now** → G2 → G3 → G4 → G5 → G6 → G7, each on the next free GPU.
  - G7's Nov 15 is a deadline, not a start time.
  - B32, XP-K, XP-H, XP-G2, XP-J and XP-I2 are **not** between them.
  - **After the A12 queue,** a pre-authorized tail with its own 60 GPU-h cap:
    - T1, XP-K, exactly as designed in A5.4;
    - T2, XP-I2, only if G3 passed.

    B32, XP-H, XP-G2 and XP-J are cancelled for this paper.
- **Accounting correction.** The A12 cap of 150 GPU-h is for A12 stages only. A10's 55.55 GPU-h is not charged to it. That leaves 146.43 GPU-h for A12 after G1a. The project cap of 400 GPU-h bounds everything.
- **XP-D4 seal.** Unseal by yourself as soon as `PREDICTIONS_REGISTRY.md` on the planner branch lists `v4` (registered or declined), or at Nov 2, 12:00 ET.
- **Decide on your own:**
  - infrastructure fixes;
  - one retry after an infrastructure-only fix;
  - partial panels at a ceiling;
  - posting the PC-DAPS clone command for Cheng if the code is absent.

  Never change a gate, a frozen rule or science code.
- **Ask only on the five stop conditions in A13.4.**

## 3. CPU chain (A13.3), per panel as its data exist

For each panel, in this order:
1. score it;
2. export strings and orientation decisions, and copy them to `theory_inputs/`;
3. compare with every registered forecast;
4. run protocol v1 (per-cohort τ, pooled by images);
5. build the hypothesis table;
6. commit.

**Panel order:**
- B16 protocol;
- F8;
- XP-D4 (after the unseal);
- XP-G;
- E′8;
- D2-8;
- XP-I;
- then Tier C.

## 4. Checkpoints: the only returns

| CP | Content | Why it matters |
|---|---|---|
| CP1 | B16 and F8 under the protocol, plus F8 against v1/v3, plus the project ledger | Completes the primary held-out table; **H2 lives here** |
| CP2 | XP-D4 against v2 and v4, plus the protocol at n = 32/64 | The third prospective test, aimed at coverage |
| CP3 | XP-G (H-G1/H-G2), E′8, D2-8 (H1), XP-I (descriptive) | Calibration transfer; PC-DAPS; the published DAPS-4K configuration |
| CP4 | Tier C batch | Supplement |

**Deadlines:** CP1–CP3 by **Nov 5, 12:00 ET**; CP4 by Nov 15. At most 20 lines per return.
