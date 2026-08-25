#!/usr/bin/env python3
"""Execute one bounded B23.1 native or typed-wrapper parent trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PARENTS = ("Fresh1", "LF-v1", "NP-1", "SITCOM-1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(header + b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"parent command failed rc={result.returncode}; inspect {log}")


def compatible_input_manifest(item: dict[str, Any], repo: Path, model_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_rule": "B23.1 signed outcome-free registry",
        "selection_uses_method_outcome": False,
        "image_id": item["image_id"],
        "measurement_path": item["measurement_path"],
        "measurement_filename": Path(item["measurement_path"]).name,
        "measurement_file_sha256": item["measurement_file_sha256"],
        "measurement_content_sha256": item["measurement_tensor_sha256"],
        "measurement_shape": item["measurement_shape"],
        "measurement_dtype": item["measurement_dtype"],
        "ground_truth_tensor_path": item["ground_truth_tensor_path"],
        "ground_truth_tensor_content_sha256": item["ground_truth_tensor_sha256"],
        "ground_truth_source_path": item["ground_truth_source_path"],
        "ground_truth_source_sha256": item["ground_truth_source_sha256"],
        "repo_root": str(repo),
        "repo_branch": "codex/b23-execution",
        "repo_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "model_path": str(model_path),
        "model_sha256": "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa",
    }


def b22_compatible_config(parent: str, seed: int, paths: dict[str, Any]) -> dict[str, Any]:
    config = {
        "schema_version": 1,
        "problem": {"resolution": 256, "oversample": 2.0, "sigma_y": 0.05},
        "pac_paths": {
            "difffpr_root": paths["difffpr"],
            "sitcom_root": paths["official_sitcom"],
            "model_path": paths["ffhq_model"],
        },
    }
    if parent == "NP-1":
        config["np1"] = {
            "config_tag": "lf",
            "guided_preset": "difffpr_ffhq_10m",
            "guided_strict": True,
            "hard_candidates": 1,
            "log_every": 100,
            "measurement_preprocessing": "clamp_min_zero_in_memory",
            "num_steps": 1000,
            "proj_radius": 0.2,
            "proj_radius_schedule": "300:0.2",
            "proj_start": 300,
            "score_huber_delta": 0.05,
            "score_mode": "lf",
            "score_radius": 0.6,
            "score_reg_lambda": 0.0,
            "score_reg_lambda_schedule": "constant",
            "seed": seed,
            "soft_candidates": 5,
            "variant": "np_canonical_soft5_hard1",
        }
    elif parent == "SITCOM-1":
        config["sitcom1"] = {
            "anneal_schedule": "linear",
            "anneal_sigma_max": 100.0,
            "anneal_sigma_min": 0.1,
            "anneal_steps": 200,
            "diff_schedule": "linear",
            "diff_sigma_min": 0.01,
            "diff_steps": 5,
            "lgvd_lr": 5.0e-5,
            "lgvd_lr_min_ratio": 0.01,
            "lgvd_steps": 100,
            "lgvd_tau": 0.01,
            "measurement_preprocessing": "none",
            "model_path_relative_to_sitcom_root": "checkpoint/ffhq256.pt",
            "num_runs": 1,
            "seed": seed,
            "timestep": "poly-7",
        }
    else:
        raise ValueError(parent)
    return config


def run_np_or_sitcom(
    *, parent: str, seed: int, repo: Path, paths: dict[str, Any], item: dict[str, Any],
    run_dir: Path, env: dict[str, str]
) -> dict[str, Any]:
    input_dir = run_dir / "input"
    input_dir.mkdir()
    model_path = Path(paths["ffhq_model"])
    write_json(input_dir / "input_manifest.json", compatible_input_manifest(item, repo, model_path))
    config_path = run_dir / "frozen_runtime_config.json"
    write_json(config_path, b22_compatible_config(parent, seed, paths))
    trace_path = run_dir / "native_trace.json"
    timing_path = run_dir / "cuda_timing.json"
    env = dict(env)
    env["B23_TRACE_PATH"] = str(trace_path)
    if parent == "NP-1":
        python_bin = paths["python"]["prdiff_ffhq"]
        script = repo / "scripts/b22/run_b22_1_np_smoke.py"
        method_dir = run_dir / "np1"
    else:
        python_bin = paths["python"]["sitcom_ode_bw"]
        script = repo / "scripts/b22/run_b22_1_sitcom_smoke.py"
        method_dir = run_dir / "sitcom1"
    command = [
        python_bin,
        str(repo / "scripts/b23/run_python_cuda_timed.py"),
        "--script", str(script),
        "--cwd", str(repo / "scripts/b22"),
        "--timing-json", str(timing_path),
        "--",
        "--config", str(config_path),
        "--repo_root", str(repo),
        "--run_root", str(run_dir),
    ]
    run_checked(command, cwd=repo, env=env, log=run_dir / "parent.log")
    result = read_json(method_dir / "result.json")
    return {
        "result_path": str(method_dir / "result.json"),
        "reconstruction_path": result["reconstruction_tensor_path"],
        "reconstruction_sha256": result["reconstruction_content_sha256"],
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "timing_path": str(timing_path),
        "timing": read_json(timing_path),
    }


def run_daps(
    *, parent: str, seed: int, repo: Path, paths: dict[str, Any], item: dict[str, Any],
    run_dir: Path, env: dict[str, str]
) -> dict[str, Any]:
    daps = Path(paths["historical_checkout"]) / "external/daps"
    input_dir = run_dir / "input"
    input_dir.mkdir()
    write_json(
        input_dir / "input_manifest.json",
        compatible_input_manifest(item, repo, Path(paths["ffhq_model"])),
    )
    data_name = f"b23-1-{item['image_id']}"
    data_dir = daps / "dataset" / data_name
    config_path = daps / "configs/data" / f"{data_name}.yaml"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"{item['image_id']}.png"
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(item["ground_truth_source_path"])
    config_path.write_text(
        "\n".join([
            "name: image",
            f"root: 'dataset/{data_name}'",
            "resolution: 256",
            "start_id: 0",
            "end_id: 1",
            "",
        ]),
        encoding="utf-8",
    )
    save_dir = run_dir / "daps_results"
    timing_path = run_dir / "cuda_timing.json"
    name = "parent"
    env = dict(env)
    env.update({
        "B21_START_NOISE_SEED": str(seed),
        "B21_SOURCE_SEED": str(seed),
        "B21_MEASUREMENT_PATH": item["measurement_path"],
        "B20_LF_ENABLE": "1" if parent == "LF-v1" else "0",
        "B20_LF_ALPHA": "0.50" if parent == "LF-v1" else "0.0",
        "B20_LF_FRAC": "0.35",
        "B20_LF_RADIUS_FRAC": "0.12",
        "B20_LF_VERBOSE": "1",
    })
    for key in ("B21_CONT_ENABLE", "B21_CONT_STATE_PATH", "B21_CONT_NOISE_SEED", "B21_SAVE_STATE_STEPS"):
        env.pop(key, None)
    posterior = daps / "posterior_sample.py"
    command = [
        paths["python"]["daps"], str(repo / "scripts/b23/run_python_cuda_timed.py"),
        "--script", str(posterior), "--cwd", str(daps), "--timing-json", str(timing_path), "--",
        f"+data={data_name}", f"+measurement_path={item['measurement_path']}",
        "+model=ffhq256ddpm", "+sampler=edm_daps", "+task=phase_retrieval",
        "task_group=pixel", "batch_size=1", "data.start_id=0", "data.end_id=1", "gpu=0",
        f"seed={seed}", "num_runs=1", f"name={name}", f"save_dir={save_dir}",
        "sampler.diffusion_scheduler_config.num_steps=5",
        "sampler.annealing_scheduler_config.num_steps=400",
        "save_samples=true", "save_traj=true", "save_traj_raw_data=true", "save_traj_video=false",
    ]
    run_checked(command, cwd=repo, env=env, log=run_dir / "parent.log")
    samples = sorted((save_dir / name / "samples").glob("*.png"))
    if len(samples) != 1:
        raise RuntimeError(f"expected one DAPS terminal sample, found {samples}")
    raw_paths = sorted((save_dir / name / "trajectory/raw").glob("trajectory_run*.pth"))
    if len(raw_paths) != 1:
        raise RuntimeError(f"expected one DAPS raw trajectory, found {raw_paths}")
    sys.path.insert(0, str(daps))
    import torch
    try:
        trajectory = torch.load(raw_paths[0], map_location="cpu", weights_only=False)
    except TypeError:
        trajectory = torch.load(raw_paths[0], map_location="cpu")
    tensor_names = ("x0hat", "x0y", "xt")
    if any(name not in trajectory.tensor_data for name in tensor_names):
        raise RuntimeError("DAPS trajectory omitted a required native tensor")
    step_count = len(trajectory.tensor_data["x0hat"])
    if step_count != 400 or any(len(trajectory.tensor_data[name]) != step_count for name in tensor_names):
        raise RuntimeError(f"DAPS trajectory step count mismatch: {step_count}")
    trace_steps = []
    for index in range(step_count):
        trace_row = {"step_index": index}
        for tensor_name in tensor_names:
            tensor = trajectory.tensor_data[tensor_name][index]
            trace_row[f"{tensor_name}_sha256"] = tensor_sha256(tensor)
            trace_row[f"{tensor_name}_mean"] = float(tensor.float().mean().item())
            trace_row[f"{tensor_name}_l2"] = float(torch.linalg.norm(tensor.float()).item())
        if "sigma" in trajectory.value_data:
            sigma = trajectory.value_data["sigma"][index]
            trace_row["sigma"] = float(sigma.item()) if torch.is_tensor(sigma) else float(sigma)
        trace_steps.append(trace_row)
    trace_manifest = {
        "schema_version": "b23.daps-native-trace.v1",
        "parent_id": parent,
        "image_id": item["image_id"],
        "seed": seed,
        "steps": trace_steps,
        "step_count": step_count,
        "annealing_transitions_expected": 400,
        "diffusion_substeps_per_transition_expected": 5,
        "mcmc_iterations_per_transition_expected": 100,
        "terminal_candidates": 1,
    }
    trace_path = run_dir / "native_trace.json"
    write_json(trace_path, trace_manifest)
    return {
        "result_path": None,
        "reconstruction_path": str(samples[0]),
        "reconstruction_sha256": sha256_file(samples[0]),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "raw_trace_path": str(raw_paths[0]),
        "raw_trace_sha256": sha256_file(raw_paths[0]),
        "timing_path": str(timing_path),
        "timing": read_json(timing_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", choices=PARENTS, required=True)
    parser.add_argument("--mode", choices=("native", "wrapper", "smoke"), required=True)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--split", choices=("B23.1-SMOKE-1", "B23.1-SMOKE-4"), required=True)
    parser.add_argument("--row-id", type=int, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    run_dir = args.output.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite parent run: {run_dir}")
    run_dir.mkdir(parents=True)
    sys.path.insert(0, str(repo))
    from prdiffusion.b23_protocol import derive_seed

    config = read_json(repo / "configs/b23/b23_1a_b_execution.yaml")
    paths = read_json(repo / "configs/b23/pac_paths.yaml")
    inputs = read_json(args.inputs.resolve() / "INPUTS.json")
    matches = [
        row for row in inputs["rows"]
        if row["split"] == args.split and int(row["row_id"]) == args.row_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"signed input row not found: split={args.split} row={args.row_id}")
    item = matches[0]
    canonical_seed = derive_seed(
        int(item["solver_base_seed"]),
        stream_name="native_start_noise",
        image_id=item["image_id"],
        measurement_id=item["measurement_id"],
        parent_id=args.parent,
        branch_id="root",
        draw_index=0,
    )
    seed = canonical_seed % (2 ** 32)
    if not 0 <= seed <= (2 ** 32 - 1):
        raise RuntimeError("native seed adapter did not produce a uint32-compatible value")
    env = os.environ.copy()
    if env.get("CUDA_VISIBLE_DEVICES", "") == "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must name exactly one authorized physical GPU")
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if args.parent in {"Fresh1", "LF-v1"}:
        details = run_daps(
            parent=args.parent, seed=seed, repo=repo, paths=paths, item=item, run_dir=run_dir, env=env
        )
    else:
        details = run_np_or_sitcom(
            parent=args.parent, seed=seed, repo=repo, paths=paths, item=item, run_dir=run_dir, env=env
        )
    native_trace = read_json(Path(details["trace_path"]))
    if args.parent in {"Fresh1", "LF-v1"}:
        if native_trace.get("step_count") != 400:
            raise RuntimeError(f"{args.parent} trace did not reconcile to 400 annealing transitions")
    elif args.parent == "NP-1":
        steps = native_trace.get("steps", [])
        candidate_counts = [int(step["candidate_count"]) for step in steps]
        if (
            native_trace.get("step_count") != 999
            or candidate_counts[:300] != [5] * 300
            or candidate_counts[300:] != [1] * 699
            or sum(candidate_counts) + 1 != 2200
        ):
            raise RuntimeError("NP-1 trace did not reconcile scheduler, candidates, and denoiser calls")
    else:
        if (
            native_trace.get("step_count") != 200
            or native_trace.get("denoiser_forwards_expected") != 1000
            or native_trace.get("lgvd_optimizer_iterations_expected") != 20000
        ):
            raise RuntimeError("SITCOM-1 trace did not reconcile annealing, denoiser, and LGVD counts")
    runtime_counters = details["timing"]["runtime_counters"]
    observed_random_calls = runtime_counters["torch_randn_calls"] + runtime_counters["torch_randn_like_calls"]
    expected_random_calls = {
        "Fresh1": 40401,
        "LF-v1": 40401,
        "NP-1": 1901,
        "SITCOM-1": 20201,
    }[args.parent]
    if observed_random_calls != expected_random_calls:
        raise RuntimeError(
            f"{args.parent} runtime RNG-call mismatch: observed={observed_random_calls} expected={expected_random_calls}"
        )
    if args.parent == "SITCOM-1" and runtime_counters["sgd_step_calls"] != 20000:
        raise RuntimeError("SITCOM-1 runtime SGD-step count did not reconcile to 20000")
    if args.parent in {"Fresh1", "LF-v1"}:
        operation_counts = {
            "denoiser_forward": 2000,
            "mcmc_iterations": 40000,
            "measurement_gradient_calls": 40000,
            "annealing_transitions": 400,
            "diffusion_substeps": 2000,
            "lf_guidance_active_steps": "SOURCE_FORMULA_RUNTIME_RECONCILE" if args.parent == "LF-v1" else 0,
            "terminal_candidates": 1,
            "runtime_counters": runtime_counters,
        }
        rng_streams = [
            {
                "name": "legacy_native_global_torch_rng",
                "draw_calls": 40401,
                "branch_id": "root",
                "actual_seed": seed,
                "canonical_parent_seed": canonical_seed,
            },
        ]
    elif args.parent == "NP-1":
        operation_counts = {
            "denoiser_forward": 2200,
            "diffusion_transitions": 999,
            "soft_candidate_steps": 300,
            "soft_candidates_evaluated": 1500,
            "hard_candidate_steps": 699,
            "hard_candidates_evaluated": 699,
            "fresh_random_proposals": 1900,
            "terminal_candidates": 1,
            "runtime_counters": runtime_counters,
        }
        rng_streams = [
            {
                "name": "legacy_native_global_torch_rng",
                "draw_calls": 1901,
                "branch_id": "root",
                "actual_seed": seed,
                "canonical_parent_seed": canonical_seed,
            },
        ]
    else:
        operation_counts = {
            "denoiser_forward": 1000,
            "diffusion_substeps": 1000,
            "lgvd_optimizer_iterations": 20000,
            "measurement_error_calls": 21000,
            "forward_renoising": 200,
            "annealing_transitions": 200,
            "terminal_candidates": 1,
            "runtime_counters": runtime_counters,
        }
        rng_streams = [
            {
                "name": "legacy_native_global_torch_rng",
                "draw_calls": 20201,
                "branch_id": "root",
                "actual_seed": seed,
                "canonical_parent_seed": canonical_seed,
            },
        ]
    operation_path = run_dir / "operation_counts.json"
    rng_path = run_dir / "rng_ledger.json"
    write_json(operation_path, {
        "schema_version": "b23.operation-count-audit.v1",
        "parent_id": args.parent,
        "source_derived_before_run": True,
        "counts": operation_counts,
    })
    write_json(rng_path, {
        "schema_version": "b23.rng-ledger.v1",
        "parent_id": args.parent,
        "streams": rng_streams,
        "stream_isolation_status": "NATIVE_GLOBAL_STREAM_PRESERVED_NOT_ISOLATED",
        "native_parent_changed": False,
        "hidden_retry_or_candidate": False,
    })
    record = {
        "schema_version": "b23.parent-run.v1",
        "status": "PASS",
        "parent_id": args.parent,
        "mode": args.mode,
        "repeat_index": args.repeat_index,
        "split": args.split,
        "row_id": args.row_id,
        "image_id": item["image_id"],
        "measurement_id": item["measurement_id"],
        "derived_parent_seed": canonical_seed,
        "native_entrypoint_seed": seed,
        "native_seed_adapter": "canonical_parent_seed modulo 2**32",
        "gpu_work_performed": True,
        "terminal_candidates": 1,
        "b23_2_authorized": False,
        "adaptive_schedule_used": False,
        "operation_counts_path": str(operation_path),
        "operation_counts_sha256": sha256_file(operation_path),
        "rng_ledger_path": str(rng_path),
        "rng_ledger_sha256": sha256_file(rng_path),
        **details,
    }
    write_json(run_dir / "RUN.json", record)
    print(json.dumps({"status": "PASS", "parent": args.parent, "mode": args.mode, "output": str(run_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
