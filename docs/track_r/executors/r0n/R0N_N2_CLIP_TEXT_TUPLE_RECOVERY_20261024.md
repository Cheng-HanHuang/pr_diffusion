# R0-N N2: exact UnivFD CLIP text-tower fix and N2-only recovery

**Date:** 2026-10-24 EDT
**Previous completed science:** N0/N1/N3 and planner return are committed at
`5af47dbb4f6dba5ba5aa140b9b30677039b31e39`.
Do not rerun or modify N0/N1/N3.

## Evidence

PAC attempt: `r0n_all_20261024T061111Z`, stage `N2_CLIP_GPU`
returned RC=8 in 6.83 s. N2 CPU preparation succeeded
with 3,240 losslessly round-tripped canonical RGB8 PNGs and frozen GT-free
task-manifest SHA
`7be734de4cd2e6f395add521717506be9260c37708f8902e248c1848b68b705b`;
preparer Git HEAD was
`b1637f7c27a3caaa03a8b204171fca0d3fccfcd5`.

GPU device 0 admitted with **97,250 MiB free**, **0 MiB used**.
The failure arose during the *first six text prompts*, before
any candidate scoring:

```
AttributeError: 'tuple' object has no attribute 'permute'
  ... UniversalFakeDetect/models/clip/model.py, encode_text:
      x = x.permute(1, 0, 2)
```

Scientific root cause, verified in the external **pinned** source
`WisconsinAIVision/UniversalFakeDetect` commit
`030495aea3300a8b54c0ec37ec7fe1dd7e63c619`:
`Transformer.forward` was changed to return
`(per_layer_features_dict, final_LND_tensor)` to support intermediate
*image* representations. `VisionTransformer.forward` unpacks this
pair correctly. `CLIP.encode_text` still assumes the older
tensor-only transformer API and calls `.permute` on the pair.

This is not caused by occupied GPUs, an invalid CLIP checkpoint,
a changing text prompt, a faulty N1 analysis or numerical divergence.

## Narrow repair

`prdiffusion/track_r/r0n_clip_compat.py` reproduces the
canonical `CLIP.encode_text` arithmetic with one explicit difference:
unpack and select the *second tensor* from the pinned
`Transformer.forward` two-item return. It asserts the first item
is a dict and the second is a tensor of unchanged `[L,N,D]` shape.
It does NOT modify or monkeypatch external UnivFD/CLIP source code,
model weights, selected zero-shot prompts, preprocessing, normalization,
or decision rule.

`r0n_2_clip_orientation.py` now uses the wrapper and supports **only**
a narrowly pinned reuse of the original CPU-prepared task file:
exact original path, SHA256, preparer HEAD and successful preparation
receipt must agree. The scoring payload still contains **no GT** or
measurement paths; the worker loads only verified canonical PNGs.

## Time cap and automatic return

The planner permits `≤0.25 GPU-h = 900 s` for N2.
The previous failed GPU stage consumed at most 6.83 seconds of
wall/allocation. The N2-only retry permits **at most 870 further
worker seconds**, including load and preflight, giving aggregate
≤876.83 s, strictly below the frozen budget. There is no GPU budget
extension and no new experiments outside N2.

`scripts/track_r/r0n_2_recovery_launch.sh`:
- fetches and checks out the exact new implementation HEAD;
- runs all Track R CPU tests, including a synthetic reproduction of
  the tuple transformer interface;
- checks the unchanged 3,240-task SHA;
- selects a GPU with measured ≥40 GiB free;
- launches one `nohup` N2-only controller.

`scripts/track_r/r0n_2_recovery_autopilot.py`:
- verifies N0/N1/N3 committed ancestry and original N2 preparation;
- starts N2 GT-free CLIP scoring, with an **870-second** external
  wall timeout and frozen original task SHA;
- if scoring passes, evaluates all 3,240 CLIP choices on CPU
  against *already computed* R0.7 per-orbit PSNR;
- records GLOBAL-orbit successes, GOOD harm and image-cluster CI for
  CLIP, prior-global, prior-mid-global and UnivFD-global;
- writes a compact **supplemental** R0-N letter handoff to a
  **new** `docs/track_r/executors/r0n/returns/R0N_N2_RECOVERY_<UTC>`
  directory and one PAC ZIP, leaving original N1/N3 files intact;
- commits/pushes the supplemental return, recording
  `_LOCAL_PUSH_PENDING` if PAC Git credentials fail.

**No R1, no new candidate generation, no new measurements, and no
B24 confirmation305 file enumeration or access.**

The planner retains the decision on K2 after inspecting the frozen
CLIP comparison; successful N2 does not itself revive the method paper.
