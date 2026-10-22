# Track R R0.0 inventory review — 2026-10-22

## Status for Cheng

R0.0 inventory completed on PAC at 2026-10-22T03:04:01Z, checkout
`700a347a7e8c46cfb3291206f03064e22b805e4b`, clean detached tree,
return code 0. This is *inventory collection*, not G0 validity or a GPU
authorization. The original JSON reports 45 VERIFIED, 4 MISSING, 4 WARN,
1 MISMATCH. Do not infer that all source schemas are reconciled.

## Verified

- `ffhq_10m.pt` SHA256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`.
- B22 full manifest SHA256 `94184e36b27aac0defb93b31ae7b47ea332909bcb0c0a80445be2949f422a219`.
- A1.2 input-freeze receipt SHA256 `fda3b5c3a26d3ed761b81cee026eea8dc2eb134b490d6369c2feda29965c2490`.
- FAST-DIPS checkout `a9758b38...`, DiffFPR `a45ffe58...`, and A1.5
  adapter `d51170f7...` all at pinned Git commits with clean worktrees.
- S1 B22 NP and SITCOM: 100 bounded row folders each; Fresh CSV exists.
- S2 DEV80: 80 source-index entries, saved GT and signed y sample present.
- S3: 8 image IDs, each DAPS/SITCOM/NP_LF/NP_S2 with 4 rep dirs.
- S4: same 8 image IDs, each with 4 FAST rep dirs.
- FFHQ metadata JSON exists. No FFHQ tree scan performed.
- Disk ~8.4 TiB available at inventory time.

## Missing, as anticipated for opt-in R0.0b

The isolated `prdiff_track_r` Python environment, pinned CLIP weights,
UnivFD checkout, and its fc head are absent. All legacy environments
remain untouched. The source FFHQ env has missing optional `ftfy`
and `sklearn` imports, to be installed only into the clone.

## Do not mistake an incomplete probe for an architecture mismatch

The all-import FAST Blackwell interpreter probe timed out (65 s).
The status reporter then assigned MISMATCH because no `sm_120` data
was returned. Separately, `torch.cuda.get_arch_list()` returns `[]`
when CUDA is disabled, regardless of compiled wheel capability.

Consequently, the specific Blackwell claim is **UNVERIFIED**, *not*
a demonstrated wrong CUDA wheel. R0.0a rechecks the pinned FAST
interpreter's compiled CUDA flags via `torch._C._cuda_getArchFlags()`
with GPU visibility disabled, in an isolated targeted subprocess.
An actual non-`sm_120` compiled wheel is a real mismatch requiring
escalation; a repeat timeout remains WARN.

## Unresolved candidate layouts

- **S1:** the inventory's `sample_tensors/S1.candidate` was accidentally
  selected from the `measurement_path` field. The B22 SITCOM receipt
  separately records `reconstruction_png_path` and
  `reconstruction_tensor_path`. The inventory code is corrected,
  but this particular receipt was not an eligible candidate image.
  The S1 Fresh1/Fresh2 per-run path and saved-GT construction remain open.
- **S2:** the inspected `NP4_INDEPENDENT` DEV80 method directory
  contains `terminal_0.pt`. The sibling methods listing does not include
  DAPS4 or SITCOM4; the B24 source config says those are **fresh external
  baselines**, so locate their individual candidate roots independently.
  Do not fabricate their K4 rows or silently drop those families.
- **S3/S4:** explicit A1.2/A1.5 candidate PNG and GT paths were
  observed at the expected model/RGB8 shapes. Per-source full replay
  remains a future G0 check, not an inventory result.

The frozen R0 spec and two planner configs were not altered.
`configs/track_r/pac_paths_track_r_verified.json` records the observed
statuses and unresolved probes without declaring overall PASS.

## Resource snapshot (not a lease)

At 03:04Z physical GPU0 was idle; GPUs1–3 had foreign
~75.5-GiB processes. The use of GPU0 (or any GPU) still requires
G0, G1, smoke-stage approval and a fresh nvidia-smi check.

## Next CPU operations

1. Run `scripts/track_r/r0_0a_reconcile.py` on the new clean pushed
   executor commit to settle FAST compiled arch flags and source directories.
2. R0.0b only after Cheng explicitly opts in; the script clones the
   environment and creates only missing pinned assets.
3. Then run `pytest -q tests/track_r` using the newly cloned interpreter.
4. Resolve S1 Fresh1/Fresh2 and S2 fresh DAPS4/SITCOM4 record
   formats before materializing a complete S1–S4 manifest.
5. Perform G0 only after those mappings are audited; no GPU scoring or
   FAST generation yet. Protected cohort stays locked through R1.
