# NeurIPS experiment log

This file is a lightweight running log for the current NeurIPS-oriented `pr_diffusion` experiment campaign.

## 2026-04-11

### HPC submission record
Submitted the following phases to the HPC cluster in this order:

1. **Phase 0** — sanity run
2. **Phase 2** — SITCOM tuning
3. **Phase 1** — radius validation

### Notes
- The submission order followed the current experiment plan.
- Phase 1 requests the most walltime among this first wave, so it was submitted after the smaller initial jobs were prepared.
- Current status: **waiting for results** before choosing the working radius and proceeding to the next radius-dependent phases.

### Next intended decision point
After these jobs finish, inspect:
- Phase 0 outputs for sanity / pipeline confirmation
- Phase 2 outputs for SITCOM tuning summaries
- Phase 1 outputs for radius selection

Then decide the radius for:
- Phase 3 — NP schedule tuning
- Phase 4 — budget study
- Phase 5 — mechanism ablation
- Phase 6 — main held-out benchmark
- Phase 7 — masked-SITCOM secondary ablation
