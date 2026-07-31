# PAC inventory and freeze

## Returned zero-GPU inventory

The attached inventory capsule `B23_0_inventory_20260731T140541Z.tar.gz` was safety-checked before
extraction. Its SHA-256 is
`babcbee34247ec9c0d52c5eaed3fa4e82e502332626d70a3fd4e837771614aa9`; all internal checksums pass.

Repository identity gate:

- remote planning head: `d1119e37fa688ac07f48ffc87ce19b13dbfb1c27` — match;
- accepted-plan merge base: `ed4f46e8f116648eda76d387388d762d7cb8f3d7` — match;
- planning head ahead/behind accepted plan: `4/0`;
- remote `codex/b23-execution`: absent at inventory time;
- protected remote `b19_solver_integration`: `819c400846aa4f812c832117afab4c600f2c3b80`;
- proposed PAC worktree/output layout: absent/available at inventory time.

The dirty historical checkout did not locally contain the two newer planning objects. This is not an
ancestry contradiction: GitHub verified ancestry and the checkout had intentionally not been fetched.

## Source identities

| Source | Head | Tracked worktree diff SHA-256 | State |
|---|---|---|---|
| historical project | `0c3c2ec972a50d462b37af7742011ed2a2c5a20a` | `b57536b4d8c7b89b6ed7fcc5deaba55087b09d6494f5c93f390d6f218e16ca9c` | 75 status rows; preserve |
| DAPS submodule | `e7a77d094167084faed19b599b96673b7bb11447` | `fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69` | 18,095-byte B20/B21 patch plus untracked artifacts |
| official SITCOM | `275ab67efbd8146bffca20155171ba6be1169c09` | `a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e` | source change is output-dir robustness; cache binaries also dirty |
| NP/SITCOM fork | `52f2c37e587576d02e2b27ac971e247f2899fc5e` | empty diff hash | seven untracked rows only |
| DiffFPR | `a45ffe58f18fed8a63d3446600424e2b08733524` | empty diff hash | clean |

The DAPS diff matches `B21_source_snapshot_20260727_040208/daps/local.patch` in size and expected
content. The PAC B23.0 collector re-hashes both sides; it never edits these sources.

## Model, environments, and hardware

- model: `/egr/research-pac/huang248/models/ffhq_10m.pt`, 374,417,833 bytes,
  SHA-256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`;
- `daps`: Python 3.11.15, PyTorch 2.10.0+cu128, NumPy 1.26.4, SciPy 1.13.1;
- `prdiff_ffhq`: Python 3.11.15, PyTorch 2.11.0+cu128, NumPy 2.4.6, SciPy 1.17.1;
- `sitcom_ode_bw`: Python 3.11.15, PyTorch 2.10.0+cu128, NumPy 1.26.4, SciPy 1.10.1;
- four NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, 97,887 MiB each;
- driver 580.126.20, CUDA compiler 13.0.88, Ubuntu 24.04.4.

Environment paths are under `/egr/research-pac/huang248/conda-envs`; default Python is prohibited on
PAC. B23.0 probes clear `CUDA_VISIBLE_DEVICES` and require `torch.cuda.is_initialized()==False`.

## Dataset and historical evidence

The FFHQ root exists. Its bounded top-level count was one because the dataset is sharded; this is not
used as an image count. B21 manifests and known shard paths resolve referenced images without a broad
recursive scan. The B21.11 benchmark and source snapshot roots are present.

## Pre-run items that the PAC collector must close

1. create/verify the clean execution worktree on the pushed execution head;
2. hash the official SITCOM `checkpoint/ffhq256.pt` and match the frozen model identity;
3. re-hash current source diffs and the source-snapshot DAPS patch;
4. probe all three environments with CUDA hidden;
5. merge exact B21 locked-measurement and historical exposure evidence;
6. publish the compact extracted evidence, checksums, and bounded archive.

Until the post-run evidence commit passes those checks, B23.1 remains blocked. No discrepancy may be
resolved by modifying the historical checkout, external repos, model, or measurements.
