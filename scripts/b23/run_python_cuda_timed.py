#!/usr/bin/env python3
"""Run a Python entrypoint in-process with paired CUDA-event and wall timing."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--timing-json", type=Path, required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.script_args[:1] == ["--"]:
        args.script_args = args.script_args[1:]
    if args.timing_json.exists():
        raise FileExistsError(f"refusing to overwrite timing record: {args.timing_json}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("authorized B23.1 timed entrypoint requires CUDA")
    runtime_counters = {
        "torch_randn_calls": 0,
        "torch_randn_like_calls": 0,
        "torch_fft2_calls": 0,
        "torch_ifft2_calls": 0,
        "sgd_step_calls": 0,
    }
    original_randn = torch.randn
    original_randn_like = torch.randn_like
    original_fft2 = torch.fft.fft2
    original_ifft2 = torch.fft.ifft2
    original_sgd_step = torch.optim.SGD.step

    def counted_randn(*values, **kwargs):
        runtime_counters["torch_randn_calls"] += 1
        return original_randn(*values, **kwargs)

    def counted_randn_like(*values, **kwargs):
        runtime_counters["torch_randn_like_calls"] += 1
        return original_randn_like(*values, **kwargs)

    def counted_fft2(*values, **kwargs):
        runtime_counters["torch_fft2_calls"] += 1
        return original_fft2(*values, **kwargs)

    def counted_ifft2(*values, **kwargs):
        runtime_counters["torch_ifft2_calls"] += 1
        return original_ifft2(*values, **kwargs)

    def counted_sgd_step(*values, **kwargs):
        runtime_counters["sgd_step_calls"] += 1
        return original_sgd_step(*values, **kwargs)

    torch.randn = counted_randn
    torch.randn_like = counted_randn_like
    torch.fft.fft2 = counted_fft2
    torch.fft.ifft2 = counted_ifft2
    torch.optim.SGD.step = counted_sgd_step
    device = torch.device("cuda:0")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    old_cwd = Path.cwd()
    old_argv = sys.argv
    old_path = list(sys.path)
    status = 0
    wall_start = time.perf_counter()
    start_event.record()
    try:
        os.chdir(args.cwd)
        sys.path.insert(0, str(args.script.resolve().parent))
        sys.argv = [str(args.script), *args.script_args]
        try:
            runpy.run_path(str(args.script), run_name="__main__")
        except SystemExit as exc:
            status = int(exc.code or 0)
            if status:
                raise
        except BaseException:
            status = 1
            raise
    finally:
        end_event.record()
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - wall_start
        gpu_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path[:] = old_path
        torch.randn = original_randn
        torch.randn_like = original_randn_like
        torch.fft.fft2 = original_fft2
        torch.fft.ifft2 = original_ifft2
        torch.optim.SGD.step = original_sgd_step
        args.timing_json.parent.mkdir(parents=True, exist_ok=True)
        args.timing_json.write_text(
            json.dumps(
                {
                    "schema_version": "b23.cuda-timing.v1",
                    "status": status,
                    "gpu_active_seconds": gpu_seconds,
                    "wall_seconds": wall_seconds,
                    "timer_method": "CUDA_EVENTS_SYNCHRONIZED_PLUS_PERF_COUNTER",
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "device_name": torch.cuda.get_device_name(device),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "runtime_counters": runtime_counters,
                    "determinism_audit": {
                        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "UNSET"),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
