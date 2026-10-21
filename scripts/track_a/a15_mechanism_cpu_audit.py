#!/usr/bin/env python3
"""A1.5 retrospective native FAST-DIPS mechanics audit (CPU ONLY; no new inference).

Read all 32 original K4 candidate floats and frozen signed measurements.  Quantify
the exact nonnegativity lower bound on observation feasibility and the actual
terminal measurement residual.  Does not access GT or modify frozen artifacts.
Runs only from the disjoint audit checkout, leaving the pinned A15 execution HEAD
and its original 2026-10-20 Stage0 receipt untouched.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys

from scripts.track_a.a15_fastdips_preflight import CFG, file_ok, git, load, load_tensor, sha, tensor_hash

ORIGINAL_A15_HEAD = "d51170f77ee672990c22f1703c5997e949dda9d6"
ORIGINAL_STAGE0_SHA = "963a9e4b2c3ed7b2c8557ef5ff759d86df22425a00bf4ce2cf3752984ccbb8aa"
AUDIT_BRANCH = "codex/prdiff-track-a-a15-mechanism-audit-20261021"
AUDIT_CHECKOUT = Path("/egr/research-pac/huang248/pr_diffusion_track_a_a15_mechanism_audit")
OUTPUT_STAGE = "a15_6_mechanism_cpu_audit_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-audit-head", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("STOP: CPU audit requires CUDA_VISIBLE_DEVICES=''")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("STOP: no bytecode writes allowed")
    if not AUDIT_CHECKOUT.is_dir() or AUDIT_CHECKOUT.resolve() != Path.cwd().resolve():
        raise RuntimeError("STOP: must run in isolated audit checkout")
    if git(AUDIT_CHECKOUT, "branch", "--show-current") != AUDIT_BRANCH:
        raise RuntimeError("STOP: wrong audit branch")
    if git(AUDIT_CHECKOUT, "rev-parse", "HEAD") != args.expected_audit_head:
        raise RuntimeError("STOP: audit checkout source commit mismatch")
    if git(AUDIT_CHECKOUT, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("STOP: dirty tracked audit checkout")

    cfg = load(CFG)
    if git(Path(cfg["a15_checkout"]), "rev-parse", "HEAD") != ORIGINAL_A15_HEAD:
        raise RuntimeError("STOP: original validated A15 execution source drifted")
    if git(Path(cfg["a15_checkout"]), "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("STOP: original A15 execution checkout dirty")
    if git(Path(cfg["fast_repo"]), "rev-parse", "HEAD") != cfg["fast_head"]:
        raise RuntimeError("STOP: official FAST source HEAD drifted")
    if git(Path(cfg["fast_repo"]), "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("STOP: official FAST source modified")

    root = Path(cfg["output_root"])
    target = root / OUTPUT_STAGE
    if target.exists() or target.is_symlink():
        raise RuntimeError("STOP: output occupied; audit is nonoverwriting")

    preflight_path = root / cfg["preflight_stage"] / "A15_PREFLIGHT_PASS.json"
    file_ok(preflight_path, ORIGINAL_STAGE0_SHA, "original CPU source/data preflight")
    preflight = load(preflight_path)
    if preflight.get("status") != "A15_CPU_PREFLIGHT_PASS":
        raise RuntimeError("STOP: original preflight status invalid")
    file_ok(Path(cfg["frozen_input_manifest"]), cfg["frozen_input_manifest_sha256"], "frozen input manifest")

    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("STOP: unexpectedly initialized CUDA")
    if torch.__version__.split("+")[0] != "2.3.0":
        raise RuntimeError("STOP: this read-only audit targets validated CPU torch 2.3")
    sys.path.insert(0, cfg["fast_repo"])
    from forward_operators import get_operator
    op = get_operator(name="phase_retrieval", oversample=2.0, resolution=256, sigma=0.05)

    # Independently confirm upstream implementation features that motivate this audit.
    main_text = (Path(cfg["fast_repo"]) / "main.py").read_text(encoding="utf-8")
    required = (
        "x = x0.clone()",
        "u = torch.zeros_like(measurement_dc)",
        "v = operator(x0)",
        "b = v - u",
        "gamma_t = (sigma ** 2)",
        "scale = (self.epsilon / nrm).clamp_max(1.0)",
        "if is_last_step:\n                xt = x0hat",
    )
    if any(fragment not in main_text for fragment in required):
        raise RuntimeError("STOP: native sampling control flow changed")
    if "use_exact_jvp=False" not in main_text or "A_step = A(x + eta * g)" not in main_text:
        raise RuntimeError("STOP: native finite-difference code drifted")

    frozen = load(Path(cfg["frozen_input_manifest"]))
    if [str(q["image_id"]) for q in frozen["rows"]] != cfg["ids"]:
        raise RuntimeError("STOP: original eight-image order changed")

    eps = float(cfg["fast_native_pixel"]["epsilon"])
    image_rows, root_rows = [], []
    for info in frozen["rows"]:
        ident = str(info["image_id"])
        y_path = Path(info["measurement_path"])
        file_ok(y_path, info["measurement_file_sha256"], "signed frozen observation")
        y = load_tensor(y_path, ("measurement", "y", "observation"), (1, 3, 384, 384))
        if tensor_hash(y) != info["measurement_tensor_sha256"]:
            raise RuntimeError(f"STOP: signed tensor-content SHA mismatch {ident}")

        # Because Fourier magnitude A(x) >= 0 componentwise,
        # ||A(x)-y||_2 >= ||min(y, 0)||_2 for every candidate x.
        n = y.numel()
        negative_floor = float(torch.linalg.vector_norm(y.clamp(max=0).double()))
        negative_fraction = float((y < 0).double().mean())
        nobs_sqrt = math.sqrt(n)
        noise_scale_euclid = float(cfg["operator"]["noise_sigma"]) * nobs_sqrt
        row_results = []
        for rep in range(4):
            loc = root / cfg["grid_stage"] / ident / f"rep{rep}"
            receipt_path = loc / "A15_SMOKE_COMPLETE.json"
            if not receipt_path.is_file():
                raise RuntimeError(f"STOP: missing saved FAST terminal {ident}/{rep}")
            r = load(receipt_path)
            if r.get("status") != "A15_FAST_CANDIDATE_PASS" or r.get("image_id") != ident or r.get("rep") != rep:
                raise RuntimeError(f"STOP: incorrect candidate receipt {ident}/{rep}")
            if r.get("source_head") != ORIGINAL_A15_HEAD or r.get("fast_head") != cfg["fast_head"]:
                raise RuntimeError(f"STOP: mismatched native candidate source {ident}/{rep}")
            if r.get("measurement_tensor_sha256") != info["measurement_tensor_sha256"]:
                raise RuntimeError(f"STOP: mismatched observation {ident}/{rep}")
            if r.get("native_checkpoint_load_validation", {}).get("checkpoint_sha256") != cfg["checkpoint_sha256"]:
                raise RuntimeError(f"STOP: mismatched FFHQ checkpoint {ident}/{rep}")
            float_path = loc / "reconstruction.pt"
            png_path = loc / "reconstruction.png"
            file_ok(float_path, r["reconstruction_float_sha256"], "native float terminal")
            file_ok(png_path, r["reconstruction_png_sha256"], "native rounded PNG")
            x = load_tensor(float_path, ("reconstruction",), (1, 3, 256, 256))
            with torch.no_grad():
                Ax = op(x)
                residual = Ax - y
                residual_norm = float(torch.linalg.vector_norm(residual.double()))
                residual_rms = residual_norm / nobs_sqrt
            if not math.isfinite(residual_norm):
                raise RuntimeError("STOP: nonfinite measurement residual")
            if residual_norm + 1e-5 < negative_floor:
                raise RuntimeError("STOP: nonnegative Fourier magnitude feasibility bound violated")
            record = {
                "image_id": ident,
                "rep": rep,
                "raw_rgb8_psnr_db_from_pinned_receipt": float(r["metric"]["raw_rgb8_psnr_db"]),
                "obs_negative_fraction": negative_fraction,
                "obs_negative_norm_lower_bound": negative_floor,
                "epsilon": eps,
                "hard_feasibility_impossible": negative_floor > eps,
                "actual_output_measurement_l2": residual_norm,
                "actual_output_measurement_rms": residual_rms,
                "terminal_l2_over_epsilon": residual_norm / eps,
                "candidate_sample_only_s": float(r["sample_only_wall_seconds"]),
                "candidate_program_wall_after_imports_s": float(r["program_wall_after_imports_seconds"]),
                "native_torch_peak_reserved_mib": float(r["peak_torch_reserved_mib"]),
            }
            row_results.append(record)
            root_rows.append(record)
        image_rows.append({
            "image_id": ident, "measurement_shape": list(y.shape),
            "epsilon": eps, "negative_coordinate_fraction": negative_fraction,
            "strict_nonnegative_forward_residual_floor_l2": negative_floor,
            "floor_to_epsilon_ratio": negative_floor / eps,
            "nominal_noise_l2_scale_sigma_sqrt_m": noise_scale_euclid,
            "hard_constraint_infeasible_for_any_image": negative_floor > eps,
            "best_of_4_raw_psnr_db": max(q["raw_rgb8_psnr_db_from_pinned_receipt"] for q in row_results),
            "min_output_residual_l2": min(q["actual_output_measurement_l2"] for q in row_results),
            "mean_output_residual_l2": statistics.mean(q["actual_output_measurement_l2"] for q in row_results),
            "total_4_root_sampling_seconds": sum(q["candidate_sample_only_s"] for q in row_results),
            "total_4_root_program_wall_after_imports_seconds": sum(q["candidate_program_wall_after_imports_s"] for q in row_results)
        })
    summary = {
        "status": "A15_CPU_MECHANISM_AUDIT_PASS",
        "audit_head": args.expected_audit_head,
        "a15_execution_head": ORIGINAL_A15_HEAD,
        "native_fast_head": cfg["fast_head"],
        "original_cpu_gate_sha256": ORIGINAL_STAGE0_SHA,
        "cohort": cfg["cohort"],
        "image_count": len(image_rows),
        "candidate_count": len(root_rows),
        "gpu_inference_count": 0,
        "ground_truth_tensors_read": 0,
        "nominal_observation_noise_sigma": cfg["operator"]["noise_sigma"],
        "epsilon": eps,
        "measurement_shape": [1, 3, 384, 384],
        "observation_components_per_image": 3 * 384 * 384,
        "images_where_hard_constraint_infeasible_by_y_negative_bound":
            sum(v["hard_constraint_infeasible_for_any_image"] for v in image_rows),
        "negative_bound_mean_l2": statistics.mean(v["strict_nonnegative_forward_residual_floor_l2"] for v in image_rows),
        "all_root_mean_sample_seconds": statistics.mean(v["candidate_sample_only_s"] for v in root_rows),
        "all_root_mean_program_wall_after_imports_seconds": statistics.mean(v["candidate_program_wall_after_imports_s"] for v in root_rows),
        "per_image": image_rows,
        "mechanism_source_findings": {
            "unet_calls_per_root": 150,
            "requested_admm_iterations": 2,
            "first_admm_x_gradient_zero_under_exact_arithmetic": True,
            "pixel_x_update_uses_finite_difference_jvp_fallback": True,
            "final_diffusion_step_returns_x0hat_not_x0y": True,
            "projected_auxiliary_variable_is_not_identical_to_terminal_image": True
        },
        "interpretation": "Infeasibility certifies exact image-space hard residual constraint is unachievable for signed observation with epsilon=0.05; it does not imply the finite-K approximate ADMM sample is invalid. Source findings are implementation semantics, not measured causal ablations."
    }
    # Save only after all 32 independent source/input/receipt/float and FFT checks pass.
    target.mkdir(parents=True, exist_ok=False)
    with (target / "A15_PER_CANDIDATE_MECHANISM.csv").open("x", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(root_rows[0]))
        writer.writeheader()
        writer.writerows(root_rows)
    with (target / "A15_MECHANISM_AUDIT_COMPLETE.json").open("x", encoding="utf8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "status": summary["status"],
        "images": summary["image_count"],
        "candidates": summary["candidate_count"],
        "infeasible_images": summary["images_where_hard_constraint_infeasible_by_y_negative_bound"],
        "mean_lower_bound_l2": summary["negative_bound_mean_l2"],
        "mean_K4_full_program_wall_after_imports_seconds":
            statistics.mean(v["total_4_root_program_wall_after_imports_seconds"] for v in image_rows),
        "receipt_sha256": sha(target / "A15_MECHANISM_AUDIT_COMPLETE.json")
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"A15_CPU_MECHANISM_AUDIT_STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(11)
