#!/usr/bin/env python3
"""Run one deterministic B24.2 baseline shard.

One shard is fixed to one physical GPU. Images are sequential within the shard;
for each image the four DAPS candidates run concurrently, followed by the four
SITCOM candidates concurrently. This is exactly the scheduling form validated
by B24.1, not solver-internal batching.

With --resume, only atomically completed rows whose manifest/row/seed identity
matches the frozen tranche are reused. Any incomplete row directory is preserved
under partial_attempts before that row is retried from its frozen input seed.
"""
from __future__ import annotations

import argparse
import hashlib
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
from prdiffusion.b24_protocol import GPU_UUIDS, MIN_FREE_BEFORE_LAUNCH_MIB, shard_rows  # noqa: E402

DAPS_PY = Path("/egr/research-pac/huang248/conda-envs/daps/bin/python")
CTRL_PY = Path("/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python")


def load_smoke_module(repo: Path):
    path = repo / "scripts/b24/run_b24_1_method_smoke.py"
    spec = importlib.util.spec_from_file_location("b24_1_method_smoke_reuse", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def gpu_state(gpu: int):
    raw = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"
    ], text=True).strip()
    uuid, free = [x.strip() for x in raw.split(",")]
    return uuid, int(free)


def run_logged(command, *, cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}; see {log}")


def validate_completed_row(completion: dict, row: dict, *, manifest_file_sha: str, shard: int, gpu: int) -> None:
    expected = {
        "status": "PASS",
        "stage": "B24.2_64",
        "manifest_file_sha256": manifest_file_sha,
        "shard_id": shard,
        "gpu_id": gpu,
        "row_index": int(row["row_index"]),
        "image_id": str(row["image_id"]).zfill(5),
        "measurement_seed": int(row["measurement_seed"]),
        "daps_solver_seeds": [int(x) for x in row["daps_solver_seeds"]],
        "sitcom_solver_seeds": [int(x) for x in row["sitcom_solver_seeds"]],
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise RuntimeError(
                f"resume completion identity mismatch for row {row['row_index']} field {key}: "
                f"observed={completion.get(key)!r} expected={value!r}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--shard", type=int, choices=range(4), required=True)
    ap.add_argument("--gpu", type=int, choices=range(4), required=True)
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    if args.gpu != args.shard:
        raise RuntimeError("B24.2 requires shard_id == physical gpu_id")
    uuid, free = gpu_state(args.gpu)
    if uuid != GPU_UUIDS[args.gpu]:
        raise RuntimeError(f"GPU UUID mismatch: {uuid} != {GPU_UUIDS[args.gpu]}")
    if free < MIN_FREE_BEFORE_LAUNCH_MIB:
        raise RuntimeError(f"GPU {args.gpu} free={free} MiB < {MIN_FREE_BEFORE_LAUNCH_MIB} MiB")

    manifest = read_json(args.manifest)
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 64:
        raise RuntimeError("B24.2 first tranche requires exactly 64 manifest rows")
    selected = list(shard_rows(rows)[args.shard])
    if len(selected) != 16:
        raise RuntimeError(f"expected 16 rows in shard {args.shard}, got {len(selected)}")

    out = args.output_root.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    if not out.exists() and args.resume:
        raise FileNotFoundError(f"resume output root does not exist: {out}")
    if not out.exists():
        out.mkdir(parents=True)
    (out / "logs").mkdir(exist_ok=True)
    manifest_file_sha = sha256_file(args.manifest)
    b24_head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    smoke = load_smoke_module(repo)

    shard_start = time.perf_counter()
    completed = 0
    reused_completed = 0
    newly_completed = 0
    preserved_partial = []
    for row in selected:
        image_id = str(row["image_id"]).zfill(5)
        image_dir = out / f"row{int(row['row_index']):03d}_{image_id}"
        completion_path = image_dir / "IMAGE_COMPLETE.json"

        if args.resume and completion_path.is_file():
            completion = read_json(completion_path)
            validate_completed_row(
                completion, row, manifest_file_sha=manifest_file_sha,
                shard=args.shard, gpu=args.gpu,
            )
            completed += 1
            reused_completed += 1
            print(
                f"IMAGE_REUSE|shard={args.shard}|row={row['row_index']}|image={image_id}|"
                f"class={completion['class_label']}|daps={float(completion['daps_best_psnr_raw_rgb_db']):.4f}|"
                f"sitcom={float(completion['sitcom_best_psnr_raw_rgb_db']):.4f}",
                flush=True,
            )
            continue

        if image_dir.exists():
            if not args.resume:
                raise FileExistsError(image_dir)
            partial_root = out / "partial_attempts"
            partial_root.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            destination = partial_root / f"{image_dir.name}.pre_resume_{stamp}"
            if destination.exists():
                raise FileExistsError(destination)
            image_dir.rename(destination)
            preserved_partial.append(str(destination))
            print(
                f"PARTIAL_PRESERVED|shard={args.shard}|row={row['row_index']}|image={image_id}|path={destination}",
                flush=True,
            )

        image_dir.mkdir()
        image_start = time.perf_counter()
        print(f"IMAGE_START|shard={args.shard}|row={row['row_index']}|image={image_id}", flush=True)

        # Generate the manifest-frozen locked input with the accepted DAPS operator.
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

        # DAPS-4 using the exact B24.1-validated terminal-only scheduling primitive.
        data_name = f"b24-2-{out.parent.name}-s{args.shard}-r{int(row['row_index']):03d}-{image_id}".lower()
        data_dir = config_path = None
        try:
            data_dir, config_path = smoke.prepare_daps_dataset(item, data_name)
            daps_specs = [
                smoke.child_spec("DAPS", repo, item, data_name, image_dir / "daps", rep, int(seed), args.gpu)
                for rep, seed in enumerate(row["daps_solver_seeds"])
            ]
            daps = smoke.run_group(daps_specs, 4, args.gpu, image_dir / "daps_memory.tsv")
        finally:
            if config_path is not None and Path(config_path).exists():
                Path(config_path).unlink()
            if data_dir is not None and Path(data_dir).exists():
                shutil.rmtree(data_dir)
        write_atomic(image_dir / "DAPS_GROUP.json", daps)

        # SITCOM-4, again using the B24.1-validated four-independent-process primitive.
        sitcom_specs = [
            smoke.child_spec("SITCOM", repo, item, None, image_dir / "sitcom", rep, int(seed), args.gpu)
            for rep, seed in enumerate(row["sitcom_solver_seeds"])
        ]
        # child_spec is intentionally reused from B24.1 for numerical identity;
        # correct its B24.1-only provenance label before any SITCOM process starts.
        for spec in sitcom_specs:
            manifest_path = Path(spec["run_dir"]) / "input/input_manifest.json"
            manifest_value = read_json(manifest_path)
            manifest_value["selection_rule"] = (
                "B24.2 deterministic unexposed FFHQ screen row; manifest-frozen image/measurement/solver seeds"
            )
            manifest_value["selection_uses_method_outcome"] = False
            manifest_value["b24_stage"] = "B24.2_64"
            write_atomic(manifest_path, manifest_value)
        sitcom = smoke.run_group(sitcom_specs, 4, args.gpu, image_dir / "sitcom_memory.tsv")
        write_atomic(image_dir / "SITCOM_GROUP.json", sitcom)

        # Common raw-RGB metrics. Use the assigned GPU only after solver children finish.
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
            "stage": "B24.2_64",
            "manifest_file_sha256": manifest_file_sha,
            "manifest_payload_sha256": manifest.get("manifest_sha256"),
            "b24_head": b24_head,
            "shard_id": args.shard,
            "gpu_id": args.gpu,
            "gpu_uuid": uuid,
            "row_index": int(row["row_index"]),
            "image_id": image_id,
            "measurement_seed": int(row["measurement_seed"]),
            "measurement_file_sha256": item["measurement_file_sha256"],
            "measurement_tensor_sha256": item["measurement_tensor_sha256"],
            "daps_solver_seeds": [int(x) for x in row["daps_solver_seeds"]],
            "sitcom_solver_seeds": [int(x) for x in row["sitcom_solver_seeds"]],
            "daps_group_wall_seconds": float(daps["group_wall_seconds"]),
            "sitcom_group_wall_seconds": float(sitcom["group_wall_seconds"]),
            "class_label": metrics["class_label"],
            "daps_best_psnr_raw_rgb_db": float(metrics["methods"]["DAPS"]["best_psnr_raw_rgb_db"]),
            "sitcom_best_psnr_raw_rgb_db": float(metrics["methods"]["SITCOM"]["best_psnr_raw_rgb_db"]),
            "image_wall_seconds": time.perf_counter() - image_start,
            "status": "PASS",
        }
        write_atomic(image_dir / "IMAGE_COMPLETE.json", completion)
        completed += 1
        newly_completed += 1
        print(
            f"IMAGE_COMPLETE|shard={args.shard}|row={row['row_index']}|image={image_id}|class={completion['class_label']}|"
            f"daps={completion['daps_best_psnr_raw_rgb_db']:.4f}|sitcom={completion['sitcom_best_psnr_raw_rgb_db']:.4f}|"
            f"wall_s={completion['image_wall_seconds']:.1f}",
            flush=True,
        )

    summary = {
        "schema_version": "b24.shard-summary.v1",
        "stage": "B24.2_64",
        "status": "PASS",
        "shard_id": args.shard,
        "gpu_id": args.gpu,
        "gpu_uuid": uuid,
        "manifest_file_sha256": manifest_file_sha,
        "row_count": len(selected),
        "completed": completed,
        "resume": bool(args.resume),
        "reused_completed": reused_completed,
        "newly_completed": newly_completed,
        "preserved_partial_attempts": preserved_partial,
        "shard_wall_seconds": time.perf_counter() - shard_start,
    }
    write_atomic(out / "SHARD_COMPLETE.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
