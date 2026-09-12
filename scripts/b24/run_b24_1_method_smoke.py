#!/usr/bin/env python3
"""B24.1 exposed-image serial-vs-concurrent independent baseline smoke.

Each candidate is a separate native solver process with its own preregistered
B24 seed. Concurrent mode is throughput scheduling only: it changes neither
DAPS nor SITCOM solver semantics and is not a B24 method-novelty claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path("/egr/research-pac/huang248/pr_diffusion_b24")
INPUTS_DEFAULT = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b23/B23_1_run_20260825T051021Z/inputs/INPUTS.json")
DAPS = Path("/egr/research-pac/huang248/pr_diffusion_b19_solver/external/daps")
SITCOM = Path("/egr/research-pac/huang248/external/SITCOM_ODE")
DAPS_PY = Path("/egr/research-pac/huang248/conda-envs/daps/bin/python")
SITCOM_PY = Path("/egr/research-pac/huang248/conda-envs/sitcom_ode_bw/bin/python")
MODEL = Path("/egr/research-pac/huang248/models/ffhq_10m.pt")
MODEL_SHA = "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
SIGNED_B24_0 = "0ed429cf579ec201c1f9b3dbd6c531f46a4e3ea3"
DAPS_HEAD = "e7a77d094167084faed19b599b96673b7bb11447"
SITCOM_HEAD = "275ab67efbd8146bffca20155171ba6be1169c09"
GPU_UUIDS = {
    0: "GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3",
    1: "GPU-883c037a-34d2-48c4-467f-9a352fd8fdff",
    2: "GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe",
    3: "GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a",
}
HARD_MIB = 52_452
TARGET_MIB = 48_000
MIN_FREE_MIB = 52_096
PLAN_MIB = 44_000
MIB = 1024 * 1024


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def gpu_state(gpu: int) -> tuple[str, int, int]:
    raw = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    uuid, free, used = [part.strip() for part in raw.split(",")]
    return uuid, int(free), int(used)


def process_memory() -> dict[int, int]:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"
    ], text=True).strip()
    result: dict[int, int] = {}
    if not raw:
        return result
    for line in raw.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            result[int(fields[0])] = int(fields[1])
        except ValueError:
            continue
    return result


def preflight(repo: Path, gpu: int, method: str, inputs: Path) -> dict[str, Any]:
    if git(repo, "branch", "--show-current") != "codex/b24-bestof4-failure-sweep":
        raise RuntimeError("B24 worktree is on the wrong branch")
    head = git(repo, "rev-parse", "HEAD")
    if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", SIGNED_B24_0, head], check=False).returncode:
        raise RuntimeError("B24.1 head does not descend from signed-off B24.0")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError("B24 worktree must be clean before B24.1")
    if not inputs.is_file():
        raise FileNotFoundError(inputs)
    if not MODEL.is_file() or sha256_file(MODEL) != MODEL_SHA:
        raise RuntimeError("FFHQ model identity mismatch")
    source = DAPS if method == "DAPS" else SITCOM
    expected = DAPS_HEAD if method == "DAPS" else SITCOM_HEAD
    if git(source, "rev-parse", "HEAD") != expected:
        raise RuntimeError(f"{method} source HEAD mismatch")
    uuid, free, used = gpu_state(gpu)
    if uuid != GPU_UUIDS[gpu]:
        raise RuntimeError(f"GPU {gpu} UUID mismatch: {uuid}")
    if free < MIN_FREE_MIB:
        raise RuntimeError(f"GPU {gpu} free={free} MiB < required {MIN_FREE_MIB} MiB")
    return {
        "b24_head": head, "gpu_id": gpu, "gpu_uuid": uuid,
        "preflight_free_mib": free, "preflight_used_mib": used,
        "source_head": expected,
    }


def choose_input(inputs_path: Path, image_id: str) -> dict[str, Any]:
    rows = read_json(inputs_path).get("rows", [])
    matches = [row for row in rows if str(row.get("image_id", "")).zfill(5) == image_id]
    if not matches:
        raise RuntimeError(f"accepted B23 inputs contain no exposed image {image_id}")
    matches.sort(key=lambda row: (
        0 if row.get("split") == "B23.1-SMOKE-1" else 1,
        str(row.get("split", "")), int(row.get("row_id", 10**9)),
    ))
    return matches[0]


def compatible_input_manifest(item: dict[str, Any], repo: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_rule": "B24.1 exposed-image equivalence smoke; accepted B23.1 locked input reused",
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
        "repo_branch": "codex/b24-bestof4-failure-sweep",
        "repo_head": git(repo, "rev-parse", "HEAD"),
        "model_path": str(MODEL),
        "model_sha256": MODEL_SHA,
    }


def sitcom_config(seed: int, repo: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "problem": {"resolution": 256, "oversample": 2.0, "sigma_y": 0.05},
        "pac_paths": {"difffpr_root": str(repo), "sitcom_root": str(SITCOM), "model_path": str(MODEL)},
        "sitcom1": {
            "anneal_schedule": "linear", "anneal_sigma_max": 100.0, "anneal_sigma_min": 0.1,
            "anneal_steps": 200, "diff_schedule": "linear", "diff_sigma_min": 0.01, "diff_steps": 5,
            "lgvd_lr": 5.0e-5, "lgvd_lr_min_ratio": 0.01, "lgvd_steps": 100, "lgvd_tau": 0.01,
            "measurement_preprocessing": "none", "model_path_relative_to_sitcom_root": "checkpoint/ffhq256.pt",
            "num_runs": 1, "seed": seed, "timestep": "poly-7",
        },
    }


def prepare_daps_dataset(item: dict[str, Any], data_name: str) -> tuple[Path, Path]:
    data_dir = DAPS / "dataset" / data_name
    config_path = DAPS / "configs/data" / f"{data_name}.yaml"
    if data_dir.exists() or config_path.exists():
        raise RuntimeError(f"refusing pre-existing B24 DAPS data config: {data_name}")
    data_dir.mkdir(parents=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{item['image_id']}.png").symlink_to(item["ground_truth_source_path"])
    config_path.write_text("\n".join([
        "name: image", f"root: 'dataset/{data_name}'", "resolution: 256", "start_id: 0", "end_id: 1", ""
    ]), encoding="utf-8")
    return data_dir, config_path


def child_spec(method: str, repo: Path, item: dict[str, Any], data_name: str | None,
               parent: Path, rep: int, canonical_seed: int, gpu: int) -> dict[str, Any]:
    native_seed = canonical_seed % (2**32)
    run_dir = parent / f"rep{rep}"
    run_dir.mkdir(parents=True, exist_ok=False)
    timing = run_dir / "cuda_timing.json"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
    })
    if method == "DAPS":
        if data_name is None:
            raise RuntimeError("missing DAPS data name")
        save_dir = run_dir / "daps_results"
        env.update({
            "B21_START_NOISE_SEED": str(native_seed), "B21_SOURCE_SEED": str(native_seed),
            "B21_MEASUREMENT_PATH": item["measurement_path"], "B20_LF_ENABLE": "0", "B20_LF_ALPHA": "0.0",
        })
        for key in ("B21_CONT_ENABLE", "B21_CONT_STATE_PATH", "B21_CONT_NOISE_SEED", "B21_SAVE_STATE_STEPS"):
            env.pop(key, None)
        command = [
            str(DAPS_PY), str(repo / "scripts/b23/run_python_cuda_timed.py"),
            "--script", str(DAPS / "posterior_sample.py"), "--cwd", str(DAPS), "--timing-json", str(timing), "--",
            f"+data={data_name}", f"+measurement_path={item['measurement_path']}",
            "+model=ffhq256ddpm", "+sampler=edm_daps", "+task=phase_retrieval", "task_group=pixel",
            "batch_size=1", "data.start_id=0", "data.end_id=1", "gpu=0", f"seed={native_seed}",
            "num_runs=1", f"name=rep{rep}", f"save_dir={save_dir}",
            "sampler.diffusion_scheduler_config.num_steps=5", "sampler.annealing_scheduler_config.num_steps=400",
            "save_samples=true", "save_traj=false", "save_traj_raw_data=false", "save_traj_video=false",
        ]
        terminal = save_dir / f"rep{rep}" / "samples" / "00000_run0000.png"
        result_path = None
    else:
        input_dir = run_dir / "input"
        input_dir.mkdir()
        write_json(input_dir / "input_manifest.json", compatible_input_manifest(item, repo))
        config_path = run_dir / "frozen_runtime_config.json"
        write_json(config_path, sitcom_config(native_seed, repo))
        command = [
            str(SITCOM_PY), str(repo / "scripts/b23/run_python_cuda_timed.py"),
            "--script", str(repo / "scripts/b22/run_b22_1_sitcom_smoke.py"),
            "--cwd", str(repo / "scripts/b22"), "--timing-json", str(timing), "--",
            "--config", str(config_path), "--repo_root", str(repo), "--run_root", str(run_dir),
        ]
        terminal = run_dir / "sitcom1" / "reconstruction.pt"
        result_path = run_dir / "sitcom1" / "result.json"
    return {
        "method": method, "rep": rep, "canonical_seed": canonical_seed, "native_seed": native_seed,
        "run_dir": run_dir, "timing": timing, "terminal": terminal, "result": result_path,
        "command": command, "env": env, "log": run_dir / "run.log",
    }


def start_task(spec: dict[str, Any]) -> tuple[subprocess.Popen, Any]:
    handle = spec["log"].open("w", encoding="utf-8")
    proc = subprocess.Popen(spec["command"], cwd=spec["run_dir"], env=spec["env"],
                            stdout=handle, stderr=subprocess.STDOUT, text=True)
    return proc, handle


def collect_task(spec: dict[str, Any]) -> dict[str, Any]:
    if not spec["terminal"].is_file() or not spec["timing"].is_file():
        raise RuntimeError(f"missing terminal/timing for {spec['method']} rep {spec['rep']}")
    timing = read_json(spec["timing"])
    if int(timing.get("status", 1)) != 0:
        raise RuntimeError(f"nonzero timing status for {spec['method']} rep {spec['rep']}")
    if spec["method"] == "DAPS":
        terminal_hash = sha256_file(spec["terminal"])
        metrics = {}
    else:
        result = read_json(spec["result"])
        terminal_hash = result["reconstruction_content_sha256"]
        metrics = result.get("metrics", {})
    return {
        "rep": spec["rep"], "canonical_seed": spec["canonical_seed"], "native_seed": spec["native_seed"],
        "terminal_content_sha256": terminal_hash,
        "gpu_active_seconds": float(timing["gpu_active_seconds"]), "wall_seconds": float(timing["wall_seconds"]),
        "peak_allocated_mib": float(timing["peak_allocated_bytes"]) / MIB,
        "peak_reserved_mib": float(timing["peak_reserved_bytes"]) / MIB,
        "metrics": metrics, "terminal_path": str(spec["terminal"]),
    }


def terminate_own(active: dict[int, tuple[dict[str, Any], subprocess.Popen, Any]]) -> None:
    for _, proc, _ in active.values():
        if proc.poll() is None:
            proc.terminate()


def run_group(specs: list[dict[str, Any]], concurrency: int, gpu: int, samples_path: Path) -> dict[str, Any]:
    uuid, free, used = gpu_state(gpu)
    if uuid != GPU_UUIDS[gpu] or free < MIN_FREE_MIB:
        raise RuntimeError(f"pre-group GPU fit gate failed: uuid={uuid} free={free}")
    pending = list(specs)
    active: dict[int, tuple[dict[str, Any], subprocess.Popen, Any]] = {}
    max_b24_mib = 0
    max_device_used_mib = used
    started = time.perf_counter()
    try:
        with samples_path.open("w", encoding="utf-8") as sample_file:
            sample_file.write("elapsed_s\tdevice_used_mib\tdevice_free_mib\tb24_process_mib\tpids\n")
            while pending or active:
                while pending and len(active) < concurrency:
                    spec = pending.pop(0)
                    proc, handle = start_task(spec)
                    active[proc.pid] = (spec, proc, handle)
                mem = process_memory()
                _, free_now, used_now = gpu_state(gpu)
                b24_mib = sum(mem.get(pid, 0) for pid in active)
                max_b24_mib = max(max_b24_mib, b24_mib)
                max_device_used_mib = max(max_device_used_mib, used_now)
                sample_file.write(
                    f"{time.perf_counter()-started:.3f}\t{used_now}\t{free_now}\t{b24_mib}\t"
                    + ",".join(map(str, sorted(active))) + "\n"
                )
                sample_file.flush()
                if b24_mib > HARD_MIB:
                    terminate_own(active)
                    raise RuntimeError(f"B24 hard-cap violation: {b24_mib} MiB > {HARD_MIB}")
                finished: list[int] = []
                for pid, (spec, proc, handle) in active.items():
                    rc = proc.poll()
                    if rc is not None:
                        handle.close()
                        if rc != 0:
                            terminate_own(active)
                            raise RuntimeError(f"{spec['method']} rep {spec['rep']} failed rc={rc}; see {spec['log']}")
                        finished.append(pid)
                for pid in finished:
                    del active[pid]
                if pending or active:
                    time.sleep(1.0)
    finally:
        for _, proc, handle in active.values():
            if proc.poll() is None:
                proc.terminate()
            try:
                handle.close()
            except Exception:
                pass
    return {
        "concurrency": concurrency,
        "group_wall_seconds": time.perf_counter() - started,
        "candidate_rows": [collect_task(spec) for spec in specs],
        "max_observed_b24_process_mib": max_b24_mib,
        "device_used_mib_at_group_start": used,
        "max_observed_device_used_mib": max_device_used_mib,
        "memory_samples_tsv": str(samples_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("DAPS", "SITCOM"), required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--image-id", default="65082")
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--inputs", type=Path, default=INPUTS_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    pre = preflight(repo, args.gpu, args.method, args.inputs)
    sys.path.insert(0, str(repo))
    from prdiffusion.b24_protocol import seed_row
    seeds = seed_row(args.image_id)["solver_seeds"][args.method]
    item = choose_input(args.inputs, args.image_id)
    data_dir = config_path = None
    data_name = None
    try:
        if args.method == "DAPS":
            data_name = f"b24-1-{output.parent.name}-{args.image_id}".replace("_", "-").lower()
            data_dir, config_path = prepare_daps_dataset(item, data_name)
        serial_specs = [child_spec(args.method, repo, item, data_name, output / "serial", rep, seed, args.gpu)
                        for rep, seed in enumerate(seeds)]
        serial = run_group(serial_specs, 1, args.gpu, output / "serial_memory.tsv")
        peak_reserved = max(row["peak_reserved_mib"] for row in serial["candidate_rows"])
        planned_concurrency = max(1, min(4, int(PLAN_MIB // max(1.0, peak_reserved))))
        concurrent_specs = [child_spec(args.method, repo, item, data_name, output / "concurrent", rep, seed, args.gpu)
                            for rep, seed in enumerate(seeds)]
        concurrent = run_group(concurrent_specs, planned_concurrency, args.gpu, output / "concurrent_memory.tsv")
        serial_hashes = {row["rep"]: row["terminal_content_sha256"] for row in serial["candidate_rows"]}
        concurrent_hashes = {row["rep"]: row["terminal_content_sha256"] for row in concurrent["candidate_rows"]}
        exact = serial_hashes == concurrent_hashes
        conservative_reserved = planned_concurrency * peak_reserved
        memory_pass = (
            concurrent["max_observed_b24_process_mib"] <= TARGET_MIB
            and conservative_reserved <= TARGET_MIB
        )
        summary = {
            "schema_version": 1, "stage": "B24.1", "method": args.method, "image_id": args.image_id,
            "accepted_b23_input": {
                "split": item.get("split"), "row_id": item.get("row_id"),
                "measurement_path": item["measurement_path"],
                "measurement_file_sha256": item["measurement_file_sha256"],
                "measurement_tensor_sha256": item["measurement_tensor_sha256"],
            },
            "preflight": pre,
            "resource_contract": {
                "hard_ceiling_mib": HARD_MIB, "normal_target_mib": TARGET_MIB,
                "minimum_free_before_launch_mib": MIN_FREE_MIB,
                "concurrency_planning_budget_mib": PLAN_MIB,
            },
            "serial": serial, "concurrent": concurrent,
            "planned_concurrency": planned_concurrency,
            "conservative_concurrent_peak_reserved_mib": conservative_reserved,
            "exact_terminal_hash_equivalence": exact, "equivalence_pass": exact,
            "memory_pass": memory_pass,
            "speedup_serial_wall_over_concurrent_wall": serial["group_wall_seconds"] / concurrent["group_wall_seconds"],
            "overall_pass": bool(exact and memory_pass),
            "full_daps_trajectory_generated": False,
            "scientific_note": "Concurrent mode schedules independent native single-trajectory processes; it is protocol engineering, not solver or method novelty."
        }
        write_json(output / "METHOD_SUMMARY.json", summary)
        print(json.dumps({
            "status": "PASS" if summary["overall_pass"] else "FAIL", "method": args.method,
            "gpu": args.gpu, "concurrency": planned_concurrency, "exact": exact,
            "memory_pass": memory_pass, "serial_wall_s": serial["group_wall_seconds"],
            "concurrent_wall_s": concurrent["group_wall_seconds"],
            "speedup": summary["speedup_serial_wall_over_concurrent_wall"],
            "summary": str(output / "METHOD_SUMMARY.json"),
        }, sort_keys=True))
        return 0 if summary["overall_pass"] else 2
    finally:
        if config_path is not None and config_path.exists():
            config_path.unlink()
        if data_dir is not None and data_dir.exists():
            shutil.rmtree(data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
