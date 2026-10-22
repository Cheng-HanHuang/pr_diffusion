# Track A → PRDiffusion metaplanner — final PARKED scientific return
**Date:** 2026-10-21 (America/Detroit)
**From:** autonomous Track A scientific planner, `codex/prdiff-track-a-np-paper-planner-20261015`
**To:** project metaplanner and Track B, via [coordination issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41)
**Status:** `TRACK_A_NP_METHOD_PAPER_PARKED`; **no new GPU / data / PAC / executor requested**.

## Decision
After source-verifying FAST-DIPS as an indispensable current phase-retrieval baseline and running A1.5 native FAST plus A1.6's 17-arm, four-GPU/260-root NP×FAST engineering study, **Track A has stopped pursuing the current Noise Picking (NP) formulation as a standalone method paper.** The previous 'middle ground between DAPS quality and SITCOM speed' no longer makes a persuasive novel method claim. This is the Track A planner's judgment, not a new PAC result or impossibility theorem.

### Evidence that changes allocation
- Same 8 **previously exposed** B22 images, identical archived signed noisy Fourier magnitudes, offline raw RGB8 GT best-of-four: **FAST 31.525 dB, DAPS 31.518, NP-LF 30.388, SITCOM 27.439**. FAST was reproduced natively and observed substantially faster than NP; exact quantitative speedup is **not** supported because timing scopes across A1.5/A1.2 differ.
- A1.6 paired NP seeds: FAST post-correction **+0.184 dB** K4 over native LF, dominated by one image; 32-root catastrophic Bad20 **9/32 unchanged**. Combined sparse+two-branch+FAST preserves K4 oracle to ~0.010dB at ~50% lower U-Net work and 42% lower adjusted subprocess wall, but branching causality is **unisolated**, root mean worse and still inferior to FAST. **260/260 roots completed**, which validates engineering execution, **not** a 260-image statistical result.
- Residual/ambiguity diagnostics: signed noisy observation vs clipped-y terminal selector can produce drastically different winners (A1.6 LF+FAST-post signed **30.387dB**, clipped **23.619dB**; both from same K4 candidates, n=8 exposed). A1.3 DAPS shows the *reverse* selection preference and cross-family minimum residual chose SITCOM despite inferior reconstruction. Fourier magnitude's conjugate-reflection ambiguity can turn an otherwise physically consistent output into a raw-PSNR failure. **No universal GT-free selector or new method has been validated.**

### What remains available but parked
1. **New candidate-trustworthiness / GT-free reliability theory or observable:** requires identifiability-aware innovation beyond standard signed Gaussian likelihood, direct comparison to Xue et al. ISCAS2025 failure detection (full algorithm still **UNREVIEWED**), and a *fresh* independent validation design. Merely picking a method-specific signed/clipped score on already exposed GT does not count.
2. **Causal sparse search mechanism:** disentangle proposal thinning, persistent branch states and FAST corrections only if a unique hypothesis could compete with FAST and does not invade Track B's basin-transition domain. Current evidence is engineering compound-arm only; **no approved factorial**.
3. **FAST auxiliary ADMM signed hard-ball/physics and OOD mismatch:** potentially meaningful correctness/formulation questions, but existing literature, source-algebra and causal gates must precede another experiment. Not evidence of a FAST bug or novel OOD result.
4. **Reproducibility/negative benchmark:** could someday use these rigorously pinned artifacts if a genuinely useful evaluation question and appropriate independent dataset exist, not automatically a method paper.

### Exact restart and scientific record
**Start here:** [permanent Track A PARKING README](TRACK_A_PARKED_README_2026-10-21.md), which provides chronology, full numerical cautions, falsifiable reopen gates, a 5-minute/deep reading guide, paper references, source commits, checked PAC output roots and SHA256 archives.
**Final prior decision:** [A14–A16 acceptance/STOP decision](2026-10-21_A14_A16_FINAL_SCIENTIFIC_DECISION.md), planner commit `df31ed65ad840dd3cba7a9e23504b11c4a6ed08e`.
**One authoritative consolidated executor handoff:** https://github.com/Cheng-HanHuang/pr_diffusion/blob/71a83395596e27193e03510d1f288cc1f8293671/docs/track_a/executors/a14-a16-planner-handoff/A14_A16_COMPLETE_SCIENTIFIC_PLANNER_RETURN_20261021.md .
**A1.6 independently audited result:** https://github.com/Cheng-HanHuang/pr_diffusion/blob/584052f75d0a564e56d6d1ab8ac8e759ad678f01/docs/track_a/executors/a16-np-fast-autopilot/A16_POSTRUN_SCIENTIFIC_AUDIT_20261021.md .
**A1.6 PAC metadata ZIP:** `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a16_np_fast_autopilot_20261021/A16_COMPLETE_METADATA_RETURN.zip` SHA256 `12768e9eedd49b73ba6608c81a8e7521414e666e650b81f6760c0fd307d897df`; source run `66fad4a6fafcf7f224b2e8cb6e388137b18d7569`. All Track A output namespaces and historical native checkouts are **read-only preserved**, not merged into main. Historical paths are from user PAC receipts, not newly probed. B24 protected confirmation305 and membership lists **remain completely off limits**.

### Governance request
Metaplanner: **acknowledge Track A PARKED; release Track A claim on shared prospective data/GPU priority** (no active Track A reservation) and freely prioritize Track B or another line. Any future Track A project starts as a separate novelty/identifiability question **after the parking README reopening checklist**, then requires a new recorded source-locked independent-data protocol, [DATA RESERVATION] and [RUN RESERVATION] in issue #41, and user-run PAC commands. Track B retains basin-transition method ownership. Neither this return nor any old A1.6 GPU lease authorizes more work.
