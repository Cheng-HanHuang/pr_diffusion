#!/usr/bin/env python3
"""Execute one authorized B24.3 DEV80 image.

For every development image this runs fresh DAPS-4 and pinned SITCOM-4 on the
locked development measurement. For non-Pilot16 rows it additionally runs the
frozen NP4/EPP321 portfolio. Pilot16 NP outputs are reused and only validated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SMOKE_PATH = REPO / "scripts" / "b24" / "run_b24_1_method_smoke.py"
CTRL_PY = Path("/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python")
DEV80_NP = REPO / "scripts" / "b24" / "run_b24_3_dev80_np.py"
EVALUATE = REPO / "scripts" / "b24" / "evaluate_b24_baseline_image.py"
HARD_CEILING_MIB = 52452
NP_ARMS = (
    "NP4_INDEPENDENT",
    "NP_EPP_321",
    "NP_EPP_321_RANDOM_PRUNE",
    "NP_EPP_321_NO_REALLOCATION",
)
EXPECTED_NP_TOTALS = {
    "NP4_INDEPENDENT": 8800,
    "NP_EPP_321": 8800,
    "NP_EPP_321_RANDOM_PRUNE": 8800,
    "NP_EPP_321_NO_REALLOCATION": 6900,
}


def load_smoke():
    spec = importlib.util.spec_from_file_location("b24_smoke_dev80", SMOKE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SMOKE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_role(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if len(rows) != 1:
        raise RuntimeError(f"expected one role row: {path}")
    return rows[0]


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
        raise RuntimeError(f"command failed rc={result.returncode}; see {log}")


def validate_np_result(path: Path, arm: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = read_json(path)
    if value.get("status") != "PASS" or value.get("arm") != arm:
        raise RuntimeError(f"bad NP result identity: {path}")
    if int(value.get("total_unet_evals", -1)) != EXPECTED_NP_TOTALS[arm]:
        raise RuntimeError(f"NP work-count drift for {arm}: {path}")
    if bool(value.get("runtime_decisions_use_ground_truth", True)):
        raise RuntimeError(f"runtime GT decision flag drift for {arm}: {path}")
    if bool(value.get("terminal_selection_uses_ground_truth", True)):
        raise RuntimeError(f"terminal selection GT flag drift for {arm}: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", type=Path, required=True)
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--role-row", type=Path, required=True)
    ap.add_argument("--task-root", type=Path, required=True)
    ap.add_argument("--physical-gpu", type=int, choices=range(4), required=True)
    ap.add_argument("--min-free-mib", type=int, default=10240)
    args = ap.parse_args()

    task = read_json(args.task_json.resolve())
    role = read_role(args.role_row.resolve())
    item = read_json(args.input_manifest.resolve())
    task_root = args.task_root.resolve()
    image_id = str(task["image_id"]).zfill(5)
    if str(role["image_id"]).zfill(5) != image_id or str(item["image_id"]).zfill(5) != image_id:
        raise RuntimeError("task/role/input image mismatch")
    if str(role["method_role"]).upper() != "DEVELOPMENT":
        raise RuntimeError("DEV80 accepts DEVELOPMENT rows only")
    if int(role["dev_measurement_seed"]) != int(item["measurement_seed"]):
        raise RuntimeError("development measurement seed mismatch")
    if int(role["source_measurement_seed"]) == int(item["measurement_seed"]):
        raise RuntimeError("development measurement reused screening seed")
    if str(task["class_label"]) != str(role["class_label"]):
        raise RuntimeError("screening-stratum mismatch")
    if bool(task["pilot16"]) != (str(role["pilot16"]).upper() == "TRUE"):
        raise RuntimeError("Pilot16 identity mismatch")

    daps_seeds = [int(x) for x in task["daps_solver_seeds"]]
    sitcom_seeds = [int(x) for x in task["sitcom_solver_seeds"]]
    if len(daps_seeds) != 4 or len(sitcom_seeds) != 4:
        raise RuntimeError("expected four DAPS and four SITCOM seeds")
    if len({x % (2**32) for x in daps_seeds}) != 4 or len({x % (2**32) for x in sitcom_seeds}) != 4:
        raise RuntimeError("native baseline seed collision")
    if args.min_free_mib > HARD_CEILING_MIB:
        raise RuntimeError("admission gate exceeds hard ceiling")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r} != physical GPU {args.physical_gpu}"
        )

    smoke = load_smoke()
    uuid, free, _ = smoke.gpu_state(args.physical_gpu)
    if uuid != smoke.GPU_UUIDS[args.physical_gpu]:
        raise RuntimeError("GPU UUID mismatch")
    if free < args.min_free_mib:
        raise RuntimeError(f"pre-image free={free} MiB < {args.min_free_mib} MiB")
    smoke.MIN_FREE_MIB = args.min_free_mib

    baseline = task_root / "baseline"
    methods = task_root / "methods"
    if baseline.exists():
        raise FileExistsError(baseline)
    baseline.mkdir(parents=True)
    logs = task_root / "logs"
    logs.mkdir(exist_ok=True)

    token = hashlib.sha256(str(task_root).encode("utf-8")).hexdigest()[:8]
    data_name = f"b24-dev80-g{args.physical_gpu}-t{int(task['task_index']):03d}-{image_id}-{token}"
    data_dir = config_path = None
    started = time.perf_counter()

    try:
        data_dir, config_path = smoke.prepare_daps_dataset(item, data_name)
        daps_specs = [
            smoke.child_spec(
                "DAPS",
                REPO,
                item,
                data_name,
                baseline / "daps",
                rep,
                seed,
                args.physical_gpu,
            )
            for rep, seed in enumerate(daps_seeds)
        ]
        daps_group = smoke.run_group(
            daps_specs,
            4,
            args.physical_gpu,
            baseline / "daps_memory.tsv",
        )
        daps_payload = {
            "schema_version": "b24.dev80-baseline-group.v1",
            "status": "PASS",
            "method": "DAPS",
            "image_id": image_id,
            "solver_seeds": daps_seeds,
            **daps_group,
        }
        daps_group_path = baseline / "DAPS_GROUP.json"
        write_json(daps_group_path, daps_payload)
    finally:
        if config_path is not None and config_path.exists():
            config_path.unlink()
        if data_dir is not None and data_dir.exists():
            shutil.rmtree(data_dir)

    sitcom_specs = [
        smoke.child_spec(
            "SITCOM",
            REPO,
            item,
            None,
            baseline / "sitcom",
            rep,
            seed,
            args.physical_gpu,
        )
        for rep, seed in enumerate(sitcom_seeds)
    ]
    sitcom_group = smoke.run_group(
        sitcom_specs,
        4,
        args.physical_gpu,
        baseline / "sitcom_memory.tsv",
    )
    sitcom_payload = {
        "schema_version": "b24.dev80-baseline-group.v1",
        "status": "PASS",
        "method": "SITCOM",
        "image_id": image_id,
        "solver_seeds": sitcom_seeds,
        **sitcom_group,
    }
    sitcom_group_path = baseline / "SITCOM_GROUP.json"
    write_json(sitcom_group_path, sitcom_payload)

    env = os.environ.copy()
    eval_out = baseline / "metrics"
    run_checked(
        [
            str(CTRL_PY),
            str(EVALUATE),
            "--image-id",
            image_id,
            "--ground-truth",
            str(Path(item["ground_truth_tensor_path"]).resolve()),
            "--daps-group",
            str(daps_group_path.resolve()),
            "--sitcom-group",
            str(sitcom_group_path.resolve()),
            "--output",
            str(eval_out),
        ],
        cwd=REPO,
        env=env,
        log=logs / "baseline_evaluation.log",
    )
    metrics_path = eval_out / "METRICS.json"
    metrics = read_json(metrics_path)
    if metrics.get("image_id") != image_id:
        raise RuntimeError("baseline metrics image mismatch")

    np_paths: dict[str, str] = {}
    if bool(task["run_np"]):
        run_checked(
            [
                sys.executable,
                str(DEV80_NP),
                "--input-manifest",
                str(args.input_manifest.resolve()),
                "--role-row",
                str(args.role_row.resolve()),
                "--output-root",
                str(methods),
                "--physical-gpu",
                str(args.physical_gpu),
                "--arms",
                ",".join(NP_ARMS),
            ],
            cwd=REPO,
            env=env,
            log=logs / "np_methods.log",
        )
        smoke_complete = methods / "SMOKE_COMPLETE.json"
        if read_json(smoke_complete).get("status") != "PASS":
            raise RuntimeError("new NP portfolio summary not PASS")
        for arm in NP_ARMS:
            path = methods / arm / "result.json"
            validate_np_result(path, arm)
            np_paths[arm] = str(path.resolve())
    else:
        source = task.get("source_np_results", {})
        if set(source) != set(NP_ARMS):
            raise RuntimeError("Pilot16 NP source map incomplete")
        for arm in NP_ARMS:
            path = Path(source[arm]).resolve()
            validate_np_result(path, arm)
            np_paths[arm] = str(path)

    daps_rows = daps_payload["candidate_rows"]
    sitcom_rows = sitcom_payload["candidate_rows"]
    completion = {
        "schema_version": "b24.dev80-image-complete.v1",
        "status": "PASS",
        "task_index": int(task["task_index"]),
        "image_id": image_id,
        "screening_stratum": role["class_label"],
        "fresh_baseline_class": metrics["class_label"],
        "pilot16": bool(task["pilot16"]),
        "input_mode": task["input_mode"],
        "measurement_seed": int(item["measurement_seed"]),
        "measurement_file_sha256": item["measurement_file_sha256"],
        "measurement_tensor_sha256": item["measurement_tensor_sha256"],
        "daps_solver_seeds": daps_seeds,
        "sitcom_solver_seeds": sitcom_seeds,
        "baseline_metrics": str(metrics_path.resolve()),
        "daps_group": str(daps_group_path.resolve()),
        "sitcom_group": str(sitcom_group_path.resolve()),
        "np_results": np_paths,
        "np_execution_performed": bool(task["run_np"]),
        "np_pilot_result_reused": not bool(task["run_np"]),
        "fresh_daps4_best_psnr_raw_rgb_db": float(metrics["methods"]["DAPS"]["best_psnr_raw_rgb_db"]),
        "fresh_sitcom4_best_psnr_raw_rgb_db": float(metrics["methods"]["SITCOM"]["best_psnr_raw_rgb_db"]),
        "compute_ledger": {
            "DAPS": {
                "native_trajectory_count": 4,
                "sum_gpu_active_seconds": sum(float(r["gpu_active_seconds"]) for r in daps_rows),
                "group_wall_seconds": float(daps_payload["group_wall_seconds"]),
                "max_observed_b24_process_mib": int(daps_payload["max_observed_b24_process_mib"]),
            },
            "SITCOM": {
                "native_trajectory_count": 4,
                "sum_gpu_active_seconds": sum(float(r["gpu_active_seconds"]) for r in sitcom_rows),
                "group_wall_seconds": float(sitcom_payload["group_wall_seconds"]),
                "max_observed_b24_process_mib": int(sitcom_payload["max_observed_b24_process_mib"]),
            },
            "NP4_INDEPENDENT_total_unet_evals": 8800,
            "NP_EPP_321_total_unet_evals": 8800,
            "NP_EPP_321_RANDOM_PRUNE_total_unet_evals": 8800,
            "NP_EPP_321_NO_REALLOCATION_total_unet_evals": 6900,
            "cross_family_equivalence_claimed": False,
        },
        "wall_seconds_image_stage": time.perf_counter() - started,
        "confirmation_exposed": False,
    }
    write_json(task_root / "IMAGE_COMPLETE.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
