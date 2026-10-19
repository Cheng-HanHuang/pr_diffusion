# A1.2 CPU-only exposed B22 DEV8 inventory handoff

**Status:** Prepared, not yet run. **Owner:** continuing Track A executor. **Frozen intent:** no GPU, no model loading, no GT or measurement payload reads, no protected B24 confirmation305 registration, IDs, list, files or metadata access.

A1.1b V2 complete return is verified in `A1_1B_V2_VERIFIED_GPU_RETURN.md` (SHA256 `875fae9cc3a546c82d72c15cf31964dc8c5728c52528b4e2f5005b5f7b63df3a`; 4/4 pipeline methods complete). The corrected next scope is separate from original A1 prospective pilot:

**A1.2 EXPOSED DEVELOPMENT ENGINEERING ONLY.** Select 8 of the already exposed B22 FFHQ validation 100 via fixed `SHA256("A1_2_EXPOSED_B22_DEV8_FIXED_RANK_20261019|"+image_id)` ordering. No GT, original result, algorithm performance, or failure taxonomy is used for this choice. The candidates are **already exposed**, hence this protocol is *not new held-out data*. Reuse exact B22 archived noisy observation later; avoid regenerating measurement. This is only a proposed execution/independent-K4 engineering extension (128 candidate trajectories), not the primary prospective publication comparison.

Input fixed, allowed archive ONLY:
`/egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines/B22_2_overnight_20260727_155135/full/manifest.json`
with SHA256 `94184e36b27aac0defb93b31ae7b47ea332909bcb0c0a80445be2949f422a219`. Source preflight previously validated this exact metadata signature and 100 known exposed rows. The inventory code reads only this manifest and checks known file *existence and path location*, not contents; GT source must be under `.../data/ffhq/ffhq-dataset/images1024x1024/` and measurement under `.../outputs/pr_diffusion/b21_solver/`. Does not ever open forbidden B24 confirmation records. Returns selected known-exposed IDs and source hashes for later development-only reservation.

A1-owned code:
- `scripts/track_a/a1_2_exposed_b22_inventory.py` (stdlib-only + git);
- `scripts/track_a/test_a1_2_exposed_b22_inventory.py`;
- `scripts/track_a/a1_2_run_exposed_b22_inventory.sh` (CPU no-CUDA tests and source inventory, STOP/ZIP).

New exclusive output:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_2_exposed_b22_dev8_inventory_v1/A1_2_EXPOSED_DEV8_INVENTORY.json`
plus `A1_2_EXPOSED_B22_INVENTORY_RETURN_V1.zip` (logs and metadata only, no image/measurement content). Refuse overwrite. Tests reject invalid count/duplicates/out-of-range IDs; source inventory fails closed if archived SHA, allowed path, required metadata keys or file existence differ. No GPU allocation/reservation needed for inventory.

On a clean machine PASS receipt, use those eight **already-exposed** IDs for an explicitly labeled A1.2 engineering pilot to validate independent K4 seeds, full GPU cost, and prefix-oracle accounting, after developing tested A1-owned per-image K4 launch/controller. Projected 8×16 candidates ≈1.53 GPUh at one-probe timing, with ~3 GPUh proposed conservative ceiling, but real-image timing unknown; no launch code nor new reservation yet.

**Important scientific interpretation:** The original A1 preregistration targets common unquantized float raw PSNR, while our native DAPS output is saved 8-bit PNG. The separate A1.2 engineering extension will use **all methods' canonical saved RGB8 PNG PSNR**, raw orientation primary and rot180-aware auxiliary; this cannot be mixed numerically with archived B22 float PSNR or substituted silently into confirmatory held-out results. A genuine pristine A1 pilot needs disjoint provenance; B24 screen selection spans `range(70_000)`, so FFHQ numeric train/test ranges alone cannot certify non-overlap with locked confirmation305. An independent source/custodian must resolve that without disclosing forbidden membership.

**Gate:** `GO_A1_2_METADATA_INVENTORY_CPU_ONLY`; `NO_LONG_GPU_RUN_YET`. Track B GPU priority unchanged; only user invokes PAC, with no shell `set -e` etc.
