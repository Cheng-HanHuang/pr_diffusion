#!/usr/bin/env python3
"""Run one B24.2 cumulative-2048 extension shard (rows 256--2047 only).

The current cumulative-256 campaign is the immutable parent checkpoint. This
worker executes only the deterministic new rows assigned to one fixed physical
GPU. It preserves the B24.1-validated four-independent-process scheduling form
for DAPS-4 and SITCOM-4.

A worker may be launched while its parent-256 shard is still running. It waits
for that exact parent shard to finish and validates its PASS summary before
starting any row >=256. This is used for the GPU-2 handoff so no completed
parent work is discarded or recomputed.

The baseline-screen admission gate is calibrated separately from the global B24
hard ceiling. B24_BASELINE_MIN_FREE_MIB defaults to 10240 MiB for this stage.
Transient lack of free GPU memory causes a recorded wait, not shard failure.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from prdiffusion.b24_protocol import GPU_UUIDS, shard_rows  # noqa: E402

DAPS_PY = Path("/egr/research-pac/huang248/conda-envs/daps/bin/python")
CTRL_PY = Path("/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python")
PARENT_COUNT = 256
TARGET_COUNT = 2048
EXPECTED_ROWS_PER_SHARD = (TARGET_COUNT - PARENT_COUNT) // 4
POLL_SECONDS = 60
CALIBRATED_MIN_FREE_MIB = int(os.environ.get("B24_BASELINE_MIN_FREE_MIB", "10240"))
if CALIBRATED_MIN_FREE_MIB < 8192:
    raise RuntimeError(
        f"refusing baseline admission gate below 8192 MiB: {CALIBRATED_MIN_FREE_MIB}"
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gpu_state(gpu: int):
    raw = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}",
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits",
    ], text=True).strip()
    uuid, free = [x.strip() for x in raw.split(",")]
    return uuid, int(free)


def pid_alive(pid: int) -> bool:
    return subprocess.run(["kill", "-0", str(pid)], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False).returncode == 0


def wait_for_parent_shard(parent_runroot: Path, shard: int) -> dict:
    summary_path = parent_runroot / f"shard{shard}/SHARD_COMPLETE.json"
    pid_path = parent_runroot / f"pids/gpu{shard}.pid"
    announced = False
    while not summary_path.is_file():
        if pid_path.is_file():
            try:
                pid = int(pid_path.read_text().strip())
            except ValueError as exc:
                raise RuntimeError(f"invalid parent PID file: {pid_path}") from exc
            if pid_alive(pid):
                if not announced:
                    print(
                        f"PARENT_WAIT|shard={shard}|pid={pid}|summary={summary_path}",
                        flush=True,
                    )
                    announced = True
                time.sleep(POLL_SECONDS)
                continue
        raise RuntimeError(
            f"parent shard {shard} is not complete and no live parent worker remains: {summary_path}"
        )
    value = read_json(summary_path)
    if value.get("status") != "PASS" or int(value.get("completed", -1)) != 48:
        raise RuntimeError(f"bad parent-256 shard summary: {summary_path}: {value}")
    print(
        f"PARENT_READY|shard={shard}|completed={value['completed']}|summary={summary_path}",
        flush=True,
    )
    return value


def wait_for_fit(gpu: int, label: str) -> float:
    start = time.perf_counter()
    last_print = -1
    while True:
        uuid, free = gpu_state(gpu)
        if uuid != GPU_UUIDS[gpu]:
            raise RuntimeError(f"GPU UUID mismatch on {gpu}: {uuid} != {GPU_UUIDS[gpu]}")
        if free >= CALIBRATED_MIN_FREE_MIB:
            waited = time.perf_counter() - start
            if waited > 0.5:
                print(
                    f"GPU_FIT_READY|gpu={gpu}|label={label}|free_mib={free}|"
                    f"required_mib={CALIBRATED_MIN_FREE_MIB}|wait_s={waited:.1f}",
                    flush=True,
                )
            return waited
        elapsed = int(time.perf_counter() - start)
        if elapsed // 300 != last_print:
            last_print = elapsed // 300
            print(
                f"GPU_WAIT|gpu={gpu}|label={label}|free_mib={free}|"
                f"required_mib={CALIBRATED_MIN_FREE_MIB}|elapsed_s={elapsed}",
                flush=True,
            )
        time.sleep(POLL_SECONDS)


def run_logged(command, *, cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}; see {log}")


def run_group_after_fit(smoke, specs, gpu: int, memory_path: Path, label: str):
    # run_b24_1_method_smoke.py retains the original generic 52-GiB default.
    # Override only the imported helper instance used by this calibrated stage.
    smoke.MIN_FREE_MIB = CALIBRATED_MIN_FREE_MIB
    while True:
        wait_for_fit(gpu, label)
        try:
            return smoke.run_group(specs, 4, gpu, memory_path)
        except RuntimeError as exc:
            if "pre-group GPU fit gate failed" not in str(exc):
                raise
            print(f"GPU_FIT_RACE|gpu={gpu}|label={label}|error={exc}", flush=True)
            time.sleep(POLL_SECONDS)


def completion_matches(value: dict, row: dict, manifest_sha: str, shard: int) -> bool:
    expected = {
        "status": "PASS",
        "stage": "B24.2_2048_EXTENSION",
        "manifest_file_sha256": manifest_sha,
        "shard_id": shard,
        "gpu_id": shard,
        "row_index": int(row["row_index"]),
        "image_id": str(row["image_id"]).zfill(5),
        "measurement_seed": int(row["measurement_seed"]),
        "daps_solver_seeds": [int(x) for x in row["daps_solver_seeds"]],
        "sitcom_solver_seeds": [int(x) for x in row["sitcom_solver_seeds"]],
    }
    return all(value.get(k) == v for k, v in expected.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--parent-runroot", type=Path, required=True)
    ap.add_argument("--shard", type=int, choices=range(4), required=True)
    ap.add_argument("--gpu", type=int, choices=range(4), required=True)
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    parent_runroot = args.parent_runroot.resolve()
    parent_manifest_path = parent_runroot / "B24_2_baseline_256.json"
    if args.gpu != args.shard:
        raise RuntimeError("B24.2 requires shard_id == physical gpu_id")
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(parent_manifest_path)

    uuid, _ = gpu_state(args.gpu)
    if uuid != GPU_UUIDS[args.gpu]:
        raise RuntimeError(f"GPU UUID mismatch: {uuid} != {GPU_UUIDS[args.gpu]}")

    manifest = read_json(args.manifest)
    parent = read_json(parent_manifest_path)
    rows = manifest.get("rows")
    parent_rows = parent.get("rows")
    if not isinstance(rows, list) or len(rows) != TARGET_COUNT:
        raise RuntimeError(f"expected cumulative {TARGET_COUNT}-row manifest")
    if not isinstance(parent_rows, list) or len(parent_rows) != PARENT_COUNT:
        raise RuntimeError(f"expected cumulative {PARENT_COUNT}-row parent manifest")
    if rows[:PARENT_COUNT] != parent_rows:
        raise RuntimeError("cumulative-2048 manifest does not preserve exact 256-row prefix")

    # Each shard waits only for its own parent-256 shard. Thus GPUs 0/1/3 can
    # begin the 2048 extension while GPU 2 finishes its existing parent shard.
    wait_for_parent_shard(parent_runroot, args.shard)

    selected = [
        r for r in shard_rows(rows)[args.shard]
        if int(r["row_index"]) >= PARENT_COUNT
    ]
    if len(selected) != EXPECTED_ROWS_PER_SHARD:
        raise RuntimeError(
            f"expected {EXPECTED_ROWS_PER_SHARD} extension rows in shard {args.shard}, got {len(selected)}"
        )

    out = args.output_root.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    if args.resume and not out.exists():
        raise FileNotFoundError(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    manifest_sha = sha256_file(args.manifest)
    parent_sha = sha256_file(parent_manifest_path)
    b24_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    smoke = load_module(
        repo / "scripts/b24/run_b24_1_method_smoke.py", "b24_smoke_reuse_2048"
    )
    smoke.MIN_FREE_MIB = CALIBRATED_MIN_FREE_MIB

    shard_start = time.perf_counter()
    completed = reused = new = 0
    total_wait = 0.0
    preserved = []

    for row in selected:
        row_index = int(row["row_index"])
        image_id = str(row["image_id"]).zfill(5)
        image_dir = out / f"row{row_index:04d}_{image_id}"
        completion_path = image_dir / "IMAGE_COMPLETE.json"

        if args.resume and completion_path.is_file():
            value = read_json(completion_path)
            if not completion_matches(value, row, manifest_sha, args.shard):
                raise RuntimeError(f"resume identity mismatch: {completion_path}")
            completed += 1
            reused += 1
            print(
                f"IMAGE_REUSE|shard={args.shard}|row={row_index}|image={image_id}|"
                f"class={value['class_label']}|daps={float(value['daps_best_psnr_raw_rgb_db']):.4f}|"
                f"sitcom={float(value['sitcom_best_psnr_raw_rgb_db']):.4f}",
                flush=True,
            )
            continue

        if image_dir.exists():
            if not args.resume:
                raise FileExistsError(image_dir)
            partial = out / "partial_attempts"
            partial.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            dest = partial / f"{image_dir.name}.pre_resume_{stamp}"
            image_dir.rename(dest)
            preserved.append(str(dest))
            print(
                f"PARTIAL_PRESERVED|shard={args.shard}|row={row_index}|image={image_id}|path={dest}",
                flush=True,
            )

        image_dir.mkdir()
        image_start = time.perf_counter()
        print(f"IMAGE_START|shard={args.shard}|row={row_index}|image={image_id}", flush=True)

        total_wait += wait_for_fit(args.gpu, f"row{row_index}:input")
        input_dir = image_dir / "input"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        run_logged([
            str(DAPS_PY), str(repo / "scripts/b24/generate_b24_locked_input.py"),
            "--image-id", image_id,
            "--measurement-seed", str(int(row["measurement_seed"])),
            "--output", str(input_dir),
        ], cwd=repo, env=env, log=image_dir / "logs/input_generation.log")
        item = read_json(input_dir / "input_manifest.json")
        if int(item["measurement_seed"]) != int(row["measurement_seed"]):
            raise RuntimeError("measurement seed drift")

        data_name = (
            f"b24-2048-{out.parent.name}-s{args.shard}-r{row_index:04d}-{image_id}"
        ).lower()
        data_dir = config_path = None
        try:
            data_dir, config_path = smoke.prepare_daps_dataset(item, data_name)
            daps_specs = [
                smoke.child_spec(
                    "DAPS", repo, item, data_name, image_dir / "daps",
                    rep, int(seed), args.gpu,
                )
                for rep, seed in enumerate(row["daps_solver_seeds"])
            ]
            before = time.perf_counter()
            daps = run_group_after_fit(
                smoke, daps_specs, args.gpu, image_dir / "daps_memory.tsv",
                f"row{row_index}:DAPS",
            )
            total_wait += max(
                0.0, time.perf_counter() - before - float(daps["group_wall_seconds"])
            )
        finally:
            if config_path is not None and Path(config_path).exists():
                Path(config_path).unlink()
            if data_dir is not None and Path(data_dir).exists():
                shutil.rmtree(data_dir)
        write_atomic(image_dir / "DAPS_GROUP.json", daps)

        sitcom_specs = [
            smoke.child_spec(
                "SITCOM", repo, item, None, image_dir / "sitcom",
                rep, int(seed), args.gpu,
            )
            for rep, seed in enumerate(row["sitcom_solver_seeds"])
        ]
        for spec in sitcom_specs:
            mpath = Path(spec["run_dir"]) / "input/input_manifest.json"
            mval = read_json(mpath)
            mval["selection_rule"] = (
                "B24.2 deterministic cumulative-2048 screen row; "
                "frozen image/measurement/solver seeds"
            )
            mval["selection_uses_method_outcome"] = False
            mval["b24_stage"] = "B24.2_2048_EXTENSION"
            write_atomic(mpath, mval)
        before = time.perf_counter()
        sitcom = run_group_after_fit(
            smoke, sitcom_specs, args.gpu, image_dir / "sitcom_memory.tsv",
            f"row{row_index}:SITCOM",
        )
        total_wait += max(
            0.0, time.perf_counter() - before - float(sitcom["group_wall_seconds"])
        )
        write_atomic(image_dir / "SITCOM_GROUP.json", sitcom)

        total_wait += wait_for_fit(args.gpu, f"row{row_index}:metrics")
        metric_env = os.environ.copy()
        metric_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        run_logged([
            str(CTRL_PY), str(repo / "scripts/b24/evaluate_b24_baseline_image.py"),
            "--image-id", image_id,
            "--ground-truth", item["ground_truth_tensor_path"],
            "--daps-group", str(image_dir / "DAPS_GROUP.json"),
            "--sitcom-group", str(image_dir / "SITCOM_GROUP.json"),
            "--output", str(image_dir / "metrics"),
        ], cwd=repo, env=metric_env, log=image_dir / "logs/metrics.log")
        metrics = read_json(image_dir / "metrics/METRICS.json")

        completion = {
            "schema_version": "b24.image-completion.v1",
            "stage": "B24.2_2048_EXTENSION",
            "manifest_file_sha256": manifest_sha,
            "parent_256_manifest_file_sha256": parent_sha,
            "manifest_payload_sha256": manifest.get("manifest_sha256"),
            "b24_head": b24_head,
            "shard_id": args.shard,
            "gpu_id": args.gpu,
            "gpu_uuid": uuid,
            "row_index": row_index,
            "image_id": image_id,
            "measurement_seed": int(row["measurement_seed"]),
            "measurement_file_sha256": item["measurement_file_sha256"],
            "measurement_tensor_sha256": item["measurement_tensor_sha256"],
            "daps_solver_seeds": [int(x) for x in row["daps_solver_seeds"]],
            "sitcom_solver_seeds": [int(x) for x in row["sitcom_solver_seeds"]],
            "daps_group_wall_seconds": float(daps["group_wall_seconds"]),
            "sitcom_group_wall_seconds": float(sitcom["group_wall_seconds"]),
            "class_label": metrics["class_label"],
            "daps_best_psnr_raw_rgb_db": float(
                metrics["methods"]["DAPS"]["best_psnr_raw_rgb_db"]
            ),
            "sitcom_best_psnr_raw_rgb_db": float(
                metrics["methods"]["SITCOM"]["best_psnr_raw_rgb_db"]
            ),
            "image_wall_seconds": time.perf_counter() - image_start,
            "baseline_min_free_mib": CALIBRATED_MIN_FREE_MIB,
            "status": "PASS",
        }
        write_atomic(completion_path, completion)
        completed += 1
        new += 1
        print(
            f"IMAGE_COMPLETE|shard={args.shard}|row={row_index}|image={image_id}|"
            f"class={completion['class_label']}|daps={completion['daps_best_psnr_raw_rgb_db']:.4f}|"
            f"sitcom={completion['sitcom_best_psnr_raw_rgb_db']:.4f}|"
            f"wall_s={completion['image_wall_seconds']:.1f}",
            flush=True,
        )

    summary = {
        "schema_version": "b24.shard-summary.v1",
        "stage": "B24.2_2048_EXTENSION",
        "status": "PASS",
        "shard_id": args.shard,
        "gpu_id": args.gpu,
        "gpu_uuid": uuid,
        "manifest_file_sha256": manifest_sha,
        "parent_256_manifest_file_sha256": parent_sha,
        "row_count": len(selected),
        "completed": completed,
        "resume": bool(args.resume),
        "reused_completed": reused,
        "newly_completed": new,
        "baseline_min_free_mib": CALIBRATED_MIN_FREE_MIB,
        "gpu_fit_wait_seconds_approx": total_wait,
        "preserved_partial_attempts": preserved,
        "shard_wall_seconds": time.perf_counter() - shard_start,
    }
    write_atomic(out / "SHARD_COMPLETE.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
