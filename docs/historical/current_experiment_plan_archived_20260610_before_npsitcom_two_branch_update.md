# Archived experiment plan: pre NP-SITCOM two-branch update

Archived: 2026-06-10

This file preserves the active experiment-roadmap state before the June 2026 NP-SITCOM two-branch update.  The active `docs/current_experiment_plan.md` has been rewritten around the two current branches:

- Branch A: NP-SITCOM candidate selection and eventual per-step defect/OOD controller.
- Branch B: NP-to-SITCOM sigma handoff and continuation diagnostics.

The previous active plan focused on moving from multi-lambda NP selection to adaptive reliability.  Its main ideas were:

1. Estimate per-image candidate-generation success probability from full FFHQ-25 multi-lambda traces.
2. Simulate adaptive compute policies that add seeds/configs only when confidence is low.
3. Target recurring hard images such as `00005`, `00013`, `00027`, `00028`, `00032`, and `00034`.
4. Treat fixed two-seed validation as insufficient because no good candidate may exist for some images/seeds.
5. Keep `score_radius=0.6`, `proj_radius=0.2`, `proj_start=300`, and multi-lambda LF/S2 configs as the NP baseline.

The exact old file content is preserved in Git history before this archive/update commit.  This archive is kept as a navigation snapshot so the active plan can focus on the current NP-SITCOM hybrid work.
