# R0-N N2 recovery controller: stage invocation correction (2026-10-24)

## Operator-returned evidence

The first N2-only retry from `df031132e678c1305dce1184b3b2d2d4b91d84ec`
passed the frozen 3,240-task SHA and **105 PAC CPU tests**, selected physical GPU 0 with
97,250 MiB free, then failed immediately in the **controller itself**:

```text
STOP_R0N_N2_RECOVERY TypeError("stage() got multiple values for argument 'gpu'")
  ... r0n_2_recovery_autopilot.py, line 276
  rc,wall=stage(out,statefile,state,"N2_CLIP_GPU_RETRY",cmd,
                gpu=args.gpu,timeout=args.budget_seconds)
```

`stage` is defined as
`stage(out, status, name, args, gpu=None, timeout=1200)`.
The extra `statefile` positional parameter shifted the remaining arguments,
binding `gpu` positionally and again by keyword. The second offline-CPU
`stage()` call had the same extra argument. Both have been corrected to
`stage(out,state,name,cmd,...)`.

**No GPU child process was spawned** in this attempt. The original
frozen N2 CPU preparation and 3,240 canonical PNGs are unchanged;
the 870-second recovery allowance has not been consumed.
The original failed N2 attempt from the all-stage run was 6.83 seconds
of GPU-stage wall time, and the previous 900-second aggregate ceiling
still applies without extension.

## Regression testing and recovery

Two new tests:
1. Parse the controller AST and bind **each** `stage()` call against
   `inspect.signature(controller.stage)`; require exactly four
   positional arguments, catching a future duplicate `gpu` binding.
2. Call the **actual** `stage` function on temporary files with a fake
   subprocess, separately testing GPU and CPU visibility,
   receipt/status bookkeeping, and no real GPU or child job.

These tests run inside the PAC launcher's
`pytest -q -p no:cacheprovider tests/track_r` gate.

To recover:
- Fetch exact corrected GitHub HEAD onto the clean PAC worktree.
- Launch the unchanged `r0n_2_recovery_launch.sh` script.
- It reuses the same SHA-pinned 3,240 PNG manifest, generates a **new**
  recovery run root, and starts only N2, not N0/N1/N3.
- Inspect stage stdout; the next expected milestone is
  `R0N_N2_SHARED_GPU_ADMISSION`, followed by a
  `N2_MODEL_LOAD_SECONDS` 64-candidate budget canary.
- An actual successful CLIP scorer must write
  `R0N_N2_CLIP_GPU_PASS 3240` before offline evaluation.
  Planner K2 stays undecided until the offline scientific receipt is
  present and verified.

The accepted R0 and published R0-N N1/N3 results remain intact,
and no R1 or protected B24 confirmation305 access is authorized.
