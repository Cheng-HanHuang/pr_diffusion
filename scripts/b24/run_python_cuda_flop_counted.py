#!/usr/bin/env python3
"""Run a Python entrypoint in-process with development-only FLOP instrumentation.

Primary count is PyTorch FlopCounterMode's dispatch-supported dynamic FLOPs.
Unsupported operations are not silently treated as free: FFT call shapes and
optimizer/backward diagnostics are recorded separately in the output.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from collections import Counter
from pathlib import Path


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--script",type=Path,required=True)
    ap.add_argument("--cwd",type=Path,required=True)
    ap.add_argument("--flop-json",type=Path,required=True)
    ap.add_argument("script_args",nargs=argparse.REMAINDER)
    args=ap.parse_args()
    if args.script_args[:1]==["--"]: args.script_args=args.script_args[1:]
    if args.flop_json.exists(): raise FileExistsError(args.flop_json)

    import torch
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as exc:
        raise RuntimeError("torch.utils.flop_counter.FlopCounterMode unavailable") from exc
    if not torch.cuda.is_available(): raise RuntimeError("FLOP audit requires CUDA")

    counters={"torch_randn_calls":0,"torch_randn_like_calls":0,"torch_fft2_calls":0,"torch_ifft2_calls":0,"sgd_step_calls":0,"autograd_backward_calls":0}
    fft_shapes=Counter(); ifft_shapes=Counter()
    orig_randn,orig_randn_like=torch.randn,torch.randn_like
    orig_fft2,orig_ifft2=torch.fft.fft2,torch.fft.ifft2
    orig_sgd=torch.optim.SGD.step
    orig_backward=torch.autograd.backward
    def c_randn(*a,**k): counters["torch_randn_calls"]+=1; return orig_randn(*a,**k)
    def c_randn_like(*a,**k): counters["torch_randn_like_calls"]+=1; return orig_randn_like(*a,**k)
    def _shape_key(a,k):
        x=a[0] if a else k.get("input")
        shape=tuple(int(v) for v in getattr(x,"shape",()))
        dim=k.get("dim",a[1] if len(a)>1 else (-2,-1))
        if isinstance(dim,int): dim=(dim,)
        return f"shape={shape};dim={tuple(dim)}"
    def c_fft2(*a,**k): counters["torch_fft2_calls"]+=1; fft_shapes[_shape_key(a,k)]+=1; return orig_fft2(*a,**k)
    def c_ifft2(*a,**k): counters["torch_ifft2_calls"]+=1; ifft_shapes[_shape_key(a,k)]+=1; return orig_ifft2(*a,**k)
    def c_sgd(*a,**k): counters["sgd_step_calls"]+=1; return orig_sgd(*a,**k)
    def c_backward(*a,**k): counters["autograd_backward_calls"]+=1; return orig_backward(*a,**k)
    torch.randn=c_randn; torch.randn_like=c_randn_like; torch.fft.fft2=c_fft2; torch.fft.ifft2=c_ifft2; torch.optim.SGD.step=c_sgd; torch.autograd.backward=c_backward

    device=torch.device("cuda:0")
    start_event=torch.cuda.Event(enable_timing=True); end_event=torch.cuda.Event(enable_timing=True)
    old_cwd=Path.cwd(); old_argv=sys.argv; old_path=list(sys.path)
    status=0; error=None; total_flops=None
    wall_start=time.perf_counter(); start_event.record()
    flop_mode=FlopCounterMode(display=False)
    try:
        os.chdir(args.cwd); sys.path.insert(0,str(args.script.resolve().parent)); sys.argv=[str(args.script),*args.script_args]
        with flop_mode:
            try:
                runpy.run_path(str(args.script),run_name="__main__")
            except SystemExit as exc:
                status=int(exc.code or 0)
                if status: raise
        total_flops=int(flop_mode.get_total_flops())
    except BaseException as exc:
        if status==0: status=1
        error=f"{type(exc).__name__}: {exc}"
        try: total_flops=int(flop_mode.get_total_flops())
        except Exception: total_flops=None
        raise
    finally:
        end_event.record(); torch.cuda.synchronize(device)
        gpu_seconds=float(start_event.elapsed_time(end_event))/1000.0; wall_seconds=time.perf_counter()-wall_start
        os.chdir(old_cwd); sys.argv=old_argv; sys.path[:]=old_path
        torch.randn=orig_randn; torch.randn_like=orig_randn_like; torch.fft.fft2=orig_fft2; torch.fft.ifft2=orig_ifft2; torch.optim.SGD.step=orig_sgd; torch.autograd.backward=orig_backward
        args.flop_json.parent.mkdir(parents=True,exist_ok=True)
        args.flop_json.write_text(json.dumps({
          "schema_version":"b24.dynamic-flop-audit.v1","status":status,
          "dispatch_supported_dynamic_flops":total_flops,
          "flop_counter":"torch.utils.flop_counter.FlopCounterMode(display=False)",
          "unsupported_operation_policy":"FFT and other unsupported kernels are not assumed zero-cost; explicit runtime counters/shapes are reported separately.",
          "runtime_counters":counters,"fft2_shape_counts":dict(sorted(fft_shapes.items())),"ifft2_shape_counts":dict(sorted(ifft_shapes.items())),
          "gpu_active_seconds":gpu_seconds,"wall_seconds":wall_seconds,"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),
          "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES",""),"device_name":torch.cuda.get_device_name(device),"error":error,
        },indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return status

if __name__=="__main__": raise SystemExit(main())
