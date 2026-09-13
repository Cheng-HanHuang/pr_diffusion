#!/usr/bin/env python3
"""Run frozen DEV80 NP arms on one locked development measurement.

This wrapper reuses the already-tested B24.3 parent runner and EPP321
implementation. It changes no algorithmic semantics. Its only execution-level
extension is to allow a frozen DEVELOPMENT row outside Pilot16.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
REF_PATH = REPO / "scripts" / "b24" / "run_b24_3_epp321_refinement.py"


def load_refinement():
    spec = importlib.util.spec_from_file_location("b24_3_epp321_refinement_dev80", REF_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import refinement runner from {REF_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ref = load_refinement()
ref.install_refinement()
base = ref.base

ARM_ORDER = (
    "NP4_INDEPENDENT",
    "NP_EPP_321",
    "NP_EPP_321_RANDOM_PRUNE",
    "NP_EPP_321_NO_REALLOCATION",
)
base.ARM_ORDER = ARM_ORDER


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--role-row", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--physical-gpu", type=int, required=True)
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    args = ap.parse_args()

    base.validate_spec()
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    if not arms or any(a not in ARM_ORDER for a in arms):
        raise RuntimeError(f"invalid DEV80 arms {arms}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            f"physical binding mismatch CUDA_VISIBLE_DEVICES={visible!r} expected {args.physical_gpu}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("B24.3 DEV80 NP execution requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    manifest = base.read_json(args.input_manifest.resolve())
    role = base.parse_role_row(args.role_row.resolve())
    image_id = str(role["image_id"]).zfill(5)
    pilot16 = str(role.get("pilot16", "")).upper() == "TRUE"
    if str(manifest["image_id"]).zfill(5) != image_id:
        raise RuntimeError("role/input image identity mismatch")
    if str(role.get("method_role", "")).upper() != "DEVELOPMENT":
        raise RuntimeError("DEV80 NP runner accepts only frozen DEVELOPMENT rows")
    if int(manifest["measurement_seed"]) != int(role["dev_measurement_seed"]):
        raise RuntimeError("development measurement seed mismatch")
    if int(role["dev_measurement_seed"]) == int(role["source_measurement_seed"]):
        raise RuntimeError("development measurement reused screening seed")
    if base.sha256_file(base.MODEL_PATH) != base.MODEL_SHA:
        raise RuntimeError("model SHA mismatch")

    meas_path = Path(manifest["measurement_path"]).resolve()
    gt_path = Path(manifest["ground_truth_tensor_path"]).resolve()
    if base.sha256_file(meas_path) != manifest["measurement_file_sha256"]:
        raise RuntimeError("measurement file SHA mismatch")
    measurement_raw = base.unwrap_tensor(
        meas_path, ("measurement", "y", "observation"), device
    ).to(dtype=torch.float32)
    if (
        base.tensor_sha256(measurement_raw) != manifest["measurement_tensor_sha256"]
        or tuple(measurement_raw.shape) != (1, 3, 384, 384)
        or not bool(torch.isfinite(measurement_raw).all())
    ):
        raise RuntimeError("measurement tensor identity/schema mismatch")
    measurement_np = measurement_raw.clamp_min(0.0)

    gt = base.unwrap_tensor(gt_path, ("ground_truth", "image", "x"), device).to(
        dtype=torch.float32
    )
    base.require_model_range(gt, "ground_truth")
    if base.tensor_sha256(gt) != manifest["ground_truth_tensor_sha256"]:
        raise RuntimeError("ground-truth SHA mismatch")

    selector = base.load_module("b24_3_selector_parent_dev80", base.SELECTOR_PATH)
    load_start = time.perf_counter()
    bundle = selector.load_guided_diffusion_model(
        model_path=str(base.MODEL_PATH),
        device=device,
        preset="difffpr_ffhq_10m",
        guided_diffusion_dir=str(base.DIFFFPR_ROOT),
        strict=True,
    )
    torch.cuda.synchronize(device)
    model_load_s = time.perf_counter() - load_start
    bundle.scheduler.set_timesteps(base.NP_STEPS, device=device)
    timesteps = bundle.scheduler.timesteps
    if len(timesteps) != base.NP_STEPS:
        raise RuntimeError(f"scheduler timestep drift: {len(timesteps)}")

    ctx = base.RunContext(
        selector,
        bundle.unet,
        bundle.scheduler,
        device,
        measurement_np,
        selector.base.oversample_pad(256, 2.0),
        timesteps,
        selector.base.parse_radius_schedule(base.PROJ_SCHEDULE, base.PROJ_RADIUS),
        image_id,
        gt,
    )
    roots = [int(role[f"np_root_seed_{i}"]) for i in range(4)]
    if len(set(roots)) != 4:
        raise RuntimeError("NP root seed collision")

    out = args.output_root.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    base.write_json_atomic(
        out / "RUN_IDENTITY.json",
        {
            "schema_version": "b24.dev80-np-run-identity.v1",
            "image_id": image_id,
            "class_label": role["class_label"],
            "method_role": role["method_role"],
            "pilot16": pilot16,
            "development_measurement_seed": int(role["dev_measurement_seed"]),
            "screening_measurement_seed": int(role["source_measurement_seed"]),
            "measurement_file_sha256": manifest["measurement_file_sha256"],
            "measurement_tensor_sha256": manifest["measurement_tensor_sha256"],
            "np_root_seeds": roots,
            "screen_manifest_sha256": base.SCREEN_MANIFEST_SHA,
            "method_spec_sha256": base.sha256_file(base.SPEC_PATH),
            "model_sha256": base.MODEL_SHA,
            "physical_gpu": args.physical_gpu,
            "cuda_visible_devices": visible,
            "model_load_seconds": model_load_s,
            "arms": arms,
        },
    )

    all_results = {}
    max_process_peak = 0
    max_torch_reserved = 0.0
    total_wall_start = time.perf_counter()
    for arm in arms:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        bundle.scheduler.set_timesteps(base.NP_STEPS, device=device)
        ctx.timesteps = bundle.scheduler.timesteps
        monitor = base.GPUMonitor(args.physical_gpu, 1.0)
        monitor.start()
        start = time.perf_counter()
        try:
            with torch.no_grad():
                terminals, events, proposal_evals, total_evals = base.RUNNERS[arm](ctx, roots)
            torch.cuda.synchronize(device)
            wall_s = time.perf_counter() - start
            torch_mem = base.torch_memory_snapshot(device)
        finally:
            gpu_mem = monitor.stop()
        result = base.finalize_arm(
            arm,
            ctx,
            terminals,
            events,
            proposal_evals,
            total_evals,
            out / arm,
            wall_s,
            torch_mem,
            gpu_mem,
        )
        all_results[arm] = result
        if gpu_mem.get("max_b24_process_gpu_mib") is not None:
            max_process_peak = max(
                max_process_peak, int(gpu_mem["max_b24_process_gpu_mib"])
            )
        max_torch_reserved = max(
            max_torch_reserved, float(result["torch_memory"]["peak_reserved_mib"])
        )
        del terminals
        gc.collect()
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "arm": arm,
                    "proposal_unet_evals": proposal_evals,
                    "total_unet_evals": total_evals,
                    "selected_psnr_raw_db": result["clean_free_selected_psnr_raw_db"],
                    "oracle_psnr_raw_db": result["oracle_best_psnr_raw_db"],
                    "wall_seconds": wall_s,
                    "max_process_gpu_mib": gpu_mem.get("max_b24_process_gpu_mib"),
                    "peak_torch_reserved_mib": result["torch_memory"]["peak_reserved_mib"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    total_wall = time.perf_counter() - total_wall_start
    if max_process_peak > base.HARD_CEILING_MIB:
        raise RuntimeError(
            f"process peak {max_process_peak} > {base.HARD_CEILING_MIB}"
        )
    recommended_gate = max(
        10240,
        int(math.ceil((max_process_peak + 4096) / 1024.0) * 1024)
        if max_process_peak
        else 16384,
    )
    if recommended_gate > base.HARD_CEILING_MIB:
        raise RuntimeError(f"derived gate {recommended_gate} > hard ceiling")

    complete = {
        "schema_version": "b24.dev80-np-image.v1",
        "status": "PASS",
        "image_id": image_id,
        "class_label": role["class_label"],
        "method_role": role["method_role"],
        "pilot16": pilot16,
        "gpu_work_performed": True,
        "measurement_generation_performed_by_runner": False,
        "runtime_ground_truth_decisions": False,
        "all_requested_arms_passed": list(all_results) == arms,
        "arms": arms,
        "arm_results": {
            arm: {
                "proposal_unet_evals": all_results[arm]["proposal_unet_evals"],
                "total_unet_evals": all_results[arm]["total_unet_evals"],
                "terminal_count": all_results[arm]["terminal_count"],
                "clean_free_selected_psnr_raw_db": all_results[arm][
                    "clean_free_selected_psnr_raw_db"
                ],
                "oracle_best_psnr_raw_db": all_results[arm]["oracle_best_psnr_raw_db"],
                "selector_gap_psnr_db": all_results[arm]["selector_gap_psnr_db"],
                "wall_seconds": all_results[arm]["wall_seconds"],
                "peak_torch_reserved_mib": all_results[arm]["torch_memory"][
                    "peak_reserved_mib"
                ],
                "max_b24_process_gpu_mib": all_results[arm]["gpu_monitor"].get(
                    "max_b24_process_gpu_mib"
                ),
            }
            for arm in arms
        },
        "max_b24_process_gpu_mib": max_process_peak,
        "max_torch_reserved_mib": max_torch_reserved,
        "hard_ceiling_mib": base.HARD_CEILING_MIB,
        "recommended_min_free_mib": recommended_gate,
        "total_wall_seconds_excluding_model_load": total_wall,
        "model_load_seconds": model_load_s,
        "next": "DEV80_AGGREGATION_ONLY_AFTER_ALL_80_BASELINES_AND_NP_RESULTS_PASS",
    }
    base.write_json_atomic(out / "SMOKE_COMPLETE.json", complete)
    print(json.dumps(complete, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
