# B24.2 cumulative 7424 Class-B top-up authorization

Status: **AUTHORIZED BY USER/PLANNER**.

The completed cumulative-6144 baseline screen is the immutable parent checkpoint. The current cumulative census is A=5733, B=89, C=251, D=71. Class B remains below the desired 100-case balanced-cohort quota.

Authorize one fixed deterministic cumulative extension to **7424 total rows**:

- preserve rows 0--6143 exactly;
- execute only rows 6144--7423;
- 1280 new rows total;
- exactly 320 new rows per fixed physical GPU;
- keep Good25 and A/B/C/D definitions unchanged;
- keep the 10,240-MiB calibrated baseline free-memory admission gate;
- do not early-stop individual shards when the 100th B case appears;
- do not execute any in-project/NP-native method during this top-up;
- after the fixed tranche completes, recount all classes and, if B>=100, freeze exactly 100 A + 100 B + 100 C by the pre-existing deterministic B24 class-rank rule.

The 7424 renderer/launcher/worker/status/resume entrypoints are:

- `scripts/b24/render_b24_baseline_manifest.py --count 7424`
- `scripts/b24/launch_b24_2_7424.sh`
- `scripts/b24/run_b24_2_7424_extension_shard.py`
- `scripts/b24/status_b24_2_7424.sh`
- `scripts/b24/resume_b24_2_7424.sh`

This authorization does not modify external DAPS or SITCOM implementations, does not authorize method-development execution, and does not authorize merge/rebase/squash/retarget/force-push/history rewrite.
