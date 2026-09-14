#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


DAPS_EXPECTED = {
    "head": "e7a77d094167084faed19b599b96673b7bb11447",
    "tree": "e63f9715e4704d9cd7a43a166559496d9d94e781",
    "index_sha256": "d5487cdba570dbaac0c1909e549da361a0a0fc3fed81e5c13f59fa12925876b6",
    "diff_sha256": "fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69",
}
SITCOM_EXPECTED = {
    "head": "275ab67efbd8146bffca20155171ba6be1169c09",
    "tree": "80263442e3606824a06dc003504c28da5c59c2c5",
    "index_sha256": "3ef63a8a29d0ba65cc642027a57ec102257fd9b387b0e9a5b4aae7f46d6a949f",
    "diff_sha256": "a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e",
}
DIFFFPR_EXPECTED_HEAD = "a45ffe58f18fed8a63d3446600424e2b08733524"
SOURCE_PATTERN = re.compile(
    r"(measurement|phase.?retrieval|forward|operator|clamp|clip|abs\(|absolute|fft|norm|sigma|likelihood|loss|project)",
    re.IGNORECASE,
)


def readj(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def writej(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    x = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(x.dtype), "shape": list(x.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    h = hashlib.sha256(header + b"\0")
    h.update(x.numpy().tobytes(order="C"))
    return h.hexdigest()


def unwrap_tensor(path: Path, keys: tuple[str, ...]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            if torch.is_tensor(payload.get(key)):
                return payload[key]
    raise RuntimeError(f"no expected tensor in {path}")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def source_identity(repo: Path, expected: dict[str, str]) -> dict[str, Any]:
    observed = {
        "head": git(repo, "rev-parse", "HEAD"),
        "tree": git(repo, "rev-parse", "HEAD^{tree}"),
        "index_sha256": sha256_bytes(git_bytes(repo, "ls-files", "-s")),
        "diff_sha256": sha256_bytes(git_bytes(repo, "diff", "--binary", "HEAD", "--", ".")),
    }
    observed["expected"] = expected
    observed["pass"] = all(observed[k] == expected[k] for k in expected)
    return observed


def bounded_source_snippets(root: Path, relpaths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in relpaths:
        path = root / rel
        if not path.is_file():
            out.append({"path": str(path), "status": "MISSING"})
            continue
        if path.stat().st_size > 2_000_000:
            raise RuntimeError(f"source file unexpectedly large: {path}")
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        hits = [
            {"line": i, "text": line[:500]}
            for i, line in enumerate(text, start=1)
            if SOURCE_PATTERN.search(line)
        ]
        out.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "line_count": len(text),
            "matching_lines": hits[:160],
            "matching_line_count": len(hits),
            "status": "PASS",
        })
    return out


def canonical_quantized01_model_range(x: torch.Tensor) -> np.ndarray:
    q = torch.round(((x.float().clamp(-1, 1) + 1.0) * 0.5) * 255.0).clamp(0, 255) / 255.0
    arr = q.detach().cpu().numpy()
    if arr.shape == (1, 3, 256, 256):
        arr = arr[0]
    if arr.shape != (3, 256, 256):
        raise RuntimeError(f"unexpected canonical tensor shape: {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def png01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    if arr.shape != (256, 256, 3):
        raise RuntimeError(f"unexpected PNG shape {arr.shape}: {path}")
    return np.transpose(arr, (2, 0, 1)).copy()


def psnr01(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x - y) ** 2))
    mse = max(mse, 1.0e-12)
    return 10.0 * math.log10(1.0 / mse)


def centered_magnitude_rgb(x01: np.ndarray, pad: int = 64) -> np.ndarray:
    if x01.shape != (3, 256, 256):
        raise RuntimeError(f"operator expects 3x256x256, got {x01.shape}")
    xp = np.pad(x01, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    shifted = np.fft.ifftshift(xp, axes=(-2, -1))
    freq = np.fft.fftn(shifted, axes=(-2, -1), norm="ortho")
    freq = np.fft.fftshift(freq, axes=(-2, -1))
    return np.abs(freq)


def transform_mask(x: np.ndarray, mask: int) -> np.ndarray:
    parts = []
    for c in range(3):
        xc = x[c]
        parts.append(np.flip(xc, axis=(-2, -1)).copy() if ((mask >> c) & 1) else xc.copy())
    return np.stack(parts, axis=0)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1.0e-300))


def synthetic_invariance(tol: float) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(2509140103))
    x = rng.random((3, 256, 256), dtype=np.float64)
    ref = centered_magnitude_rgb(x)
    values = []
    for mask in range(8):
        value = rel_l2(centered_magnitude_rgb(transform_mask(x, mask)), ref)
        values.append(value)
    return {"relative_l2_by_mask": values, "max_relative_l2": max(values), "tolerance": tol, "pass": max(values) <= tol}


def find_input_manifest(task: dict[str, Any], taskdir: Path) -> Path:
    if bool(task["pilot16"]):
        path = Path(task["source_input_manifest"]).resolve()
    else:
        path = (taskdir / "input" / "input_manifest.json").resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def baseline_metric_lookup(metrics: dict[str, Any]) -> dict[tuple[str, int], float]:
    return {
        (str(r["method"]), int(r["rep"])): float(r["psnr_raw_rgb_db"])
        for r in metrics["terminal_rows"]
    }


def load_np_terminal(term: dict[str, Any]) -> tuple[np.ndarray, Path]:
    path = Path(term["reconstruction_path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = unwrap_tensor(path, ("reconstruction", "image", "x"))
    if tensor_sha256(raw) != str(term["reconstruction_sha256"]):
        raise RuntimeError(f"NP terminal tensor hash mismatch: {path}")
    return canonical_quantized01_model_range(raw), path


def load_sitcom_terminal(cand: dict[str, Any]) -> tuple[np.ndarray, Path]:
    tensor_path = Path(cand["terminal_path"]).resolve()
    if not tensor_path.is_file():
        raise FileNotFoundError(tensor_path)
    raw = unwrap_tensor(tensor_path, ("reconstruction", "image", "x"))
    if tensor_sha256(raw) != str(cand["terminal_content_sha256"]):
        raise RuntimeError(f"SITCOM tensor hash mismatch: {tensor_path}")
    png = tensor_path.with_name("reconstruction.png")
    if not png.is_file():
        raise FileNotFoundError(png)
    return png01(png), png


def load_daps_terminal(cand: dict[str, Any]) -> tuple[np.ndarray, Path]:
    path = Path(cand["terminal_path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(cand["terminal_content_sha256"]):
        raise RuntimeError(f"DAPS PNG hash mismatch: {path}")
    return png01(path), path


def terminal_diagnostics(
    *,
    image_id: str,
    screening_stratum: str,
    method: str,
    candidate_index: int,
    rec01: np.ndarray,
    source_path: Path,
    source_hash: str,
    gt01: np.ndarray,
    measurement_raw: np.ndarray,
    measurement_clamped: np.ndarray,
    expected_psnr: float | None,
    psnr_tol: float,
    invariance_tol: float,
) -> dict[str, Any]:
    original_mag = centered_magnitude_rgb(rec01)
    raw_res = float(np.linalg.norm(original_mag - measurement_raw))
    clamp_res = float(np.linalg.norm(original_mag - measurement_clamped))
    raw_norm = max(float(np.linalg.norm(measurement_raw)), 1e-300)
    clamp_norm = max(float(np.linalg.norm(measurement_clamped)), 1e-300)
    original_psnr = psnr01(rec01, gt01)
    if expected_psnr is not None and abs(original_psnr - expected_psnr) > psnr_tol:
        raise RuntimeError(
            f"canonical PSNR drift {method}/{image_id}/candidate{candidate_index}: "
            f"{original_psnr} vs {expected_psnr}"
        )
    pvals, discrepancies = [], []
    for mask in range(8):
        transformed = transform_mask(rec01, mask)
        pvals.append(psnr01(transformed, gt01))
        discrepancies.append(rel_l2(centered_magnitude_rgb(transformed), original_mag))
    max_disc = max(discrepancies)
    if max_disc > invariance_tol:
        raise RuntimeError(
            f"real terminal symmetry invariance failed {method}/{image_id}/candidate{candidate_index}: "
            f"{max_disc} > {invariance_tol}"
        )
    best_mask = max(range(8), key=lambda m: (pvals[m], -m))
    return {
        "image_id": image_id,
        "screening_stratum": screening_stratum,
        "method": method,
        "candidate_index": candidate_index,
        "source_path": str(source_path),
        "source_content_sha256": source_hash,
        "original_psnr_raw_rgb_db": original_psnr,
        "original_good25": original_psnr >= 25.0,
        "measurement_residual_raw_l2": raw_res,
        "measurement_residual_raw_normalized_l2": raw_res / raw_norm,
        "measurement_residual_clamped_l2": clamp_res,
        "measurement_residual_clamped_normalized_l2": clamp_res / clamp_norm,
        "max_verified_transform_measurement_relative_l2": max_disc,
        "transform_measurement_relative_l2_by_mask": json.dumps(discrepancies, separators=(",", ":")),
        "psnr_by_mask_db": json.dumps(pvals, separators=(",", ":")),
        "gt_assisted_symmetry_oracle_psnr_db": pvals[best_mask],
        "gt_assisted_symmetry_gain_db": pvals[best_mask] - original_psnr,
        "gt_assisted_symmetry_best_mask": best_mask,
        "gt_assisted_symmetry_good25": pvals[best_mask] >= 25.0,
    }


def dist(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()) if len(x) else None,
        "median": float(np.median(x)) if len(x) else None,
        "min": float(x.min()) if len(x) else None,
        "max": float(x.max()) if len(x) else None,
        "q10": float(np.quantile(x, 0.10)) if len(x) else None,
        "q90": float(np.quantile(x, 0.90)) if len(x) else None,
    }


def summarize_by_image(rows: list[dict[str, Any]], shared_ids: set[str]) -> dict[str, Any]:
    per: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per[(row["image_id"], row["method"])].append(row)
    per_image: list[dict[str, Any]] = []
    for (image, method), vals in sorted(per.items()):
        original = max(float(v["original_psnr_raw_rgb_db"]) for v in vals)
        sym = max(float(v["gt_assisted_symmetry_oracle_psnr_db"]) for v in vals)
        per_image.append({
            "image_id": image,
            "screening_stratum": vals[0]["screening_stratum"],
            "method": method,
            "candidate_count": len(vals),
            "original_candidate_oracle_psnr_db": original,
            "symmetry_candidate_oracle_psnr_db": sym,
            "symmetry_oracle_gain_db": sym - original,
            "original_good25": original >= 25.0,
            "symmetry_good25": sym >= 25.0,
            "good25_recovery": original < 25.0 <= sym,
            "shared_failure_subset": image in shared_ids,
        })
    return {"rows": per_image}


def aggregate(per_image_rows: list[dict[str, Any]], subset_ids: set[str] | None = None) -> dict[str, Any]:
    rows = per_image_rows if subset_ids is None else [r for r in per_image_rows if r["image_id"] in subset_ids]
    out: dict[str, Any] = {}
    for method in ["DAPS", "SITCOM", "NP4_INDEPENDENT"]:
        vals = [r for r in rows if r["method"] == method]
        gains = [float(r["symmetry_oracle_gain_db"]) for r in vals]
        out[method] = {
            "images": len(vals),
            "original_good25": sum(bool(r["original_good25"]) for r in vals),
            "symmetry_good25": sum(bool(r["symmetry_good25"]) for r in vals),
            "good25_recoveries": sum(bool(r["good25_recovery"]) for r in vals),
            "gain_db": dist(gains),
            "gain_ge_1db": sum(g >= 1.0 for g in gains),
            "gain_ge_3db": sum(g >= 3.0 for g in gains),
            "gain_ge_5db": sum(g >= 5.0 for g in gains),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--role-dir", type=Path, required=True)
    ap.add_argument("--dev80-run", type=Path, required=True)
    ap.add_argument("--closeout", type=Path, required=True)
    ap.add_argument("--daps-root", type=Path, required=True)
    ap.add_argument("--sitcom-root", type=Path, required=True)
    ap.add_argument("--difffpr-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("B25 requires CUDA_VISIBLE_DEVICES=''")
    started = time.perf_counter()
    spec = readj(args.spec.resolve())
    cfg3 = spec["experiment3"]

    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    # ID-only allowlist gate. No DEV tensor/image/measurement payload is opened before this passes.
    role_csv = (args.role_dir / "B24_METHOD_IMAGE_ROLES.csv").resolve()
    role_rows = read_csv(role_csv)
    dev_rows = [r for r in role_rows if r["method_role"] == "DEVELOPMENT"]
    confirm_rows = [r for r in role_rows if r["method_role"] == "CONFIRMATION"]
    dev_ids = {str(r["image_id"]).zfill(5) for r in dev_rows}
    confirm_ids = {str(r["image_id"]).zfill(5) for r in confirm_rows}
    if len(dev_ids) != 80 or len(confirm_ids) != 305 or dev_ids & confirm_ids:
        raise RuntimeError("DEV80/confirmation305 registry gate failed")

    manifest = readj((args.dev80_run / "DEV80_MANIFEST.json").resolve())
    tasks = manifest["tasks"]
    task_ids = {str(t["image_id"]).zfill(5) for t in tasks}
    if len(tasks) != 80 or task_ids != dev_ids or manifest.get("confirmation_exposed") is not False:
        raise RuntimeError("DEV80 manifest allowlist mismatch")
    allowlist = {
        "schema_version": "b25.dev-allowlist.v1",
        "status": "PASS",
        "role_csv": str(role_csv),
        "role_csv_sha256": sha256_file(role_csv),
        "dev_count": len(dev_ids),
        "confirmation_count": len(confirm_ids),
        "intersection_count": len(dev_ids & confirm_ids),
        "dev_ids": sorted(dev_ids),
        "confirmation_ids_sha256": sha256_bytes("\n".join(sorted(confirm_ids)).encode()),
        "confirmation_payloads_accessed": False,
        "dev80_manifest": str((args.dev80_run / "DEV80_MANIFEST.json").resolve()),
        "dev80_manifest_sha256": sha256_file(args.dev80_run / "DEV80_MANIFEST.json"),
    }
    writej(out / "DEV_ALLOWLIST.json", allowlist)

    # Accepted dirty source trees are verified by exact HEAD/tree/index/diff identities.
    source = {
        "DAPS": source_identity(args.daps_root.resolve(), DAPS_EXPECTED),
        "SITCOM": source_identity(args.sitcom_root.resolve(), SITCOM_EXPECTED),
        "DIFFFPR": {
            "head": git(args.difffpr_root.resolve(), "rev-parse", "HEAD"),
            "expected_head": DIFFFPR_EXPECTED_HEAD,
        },
    }
    source["DIFFFPR"]["pass"] = source["DIFFFPR"]["head"] == DIFFFPR_EXPECTED_HEAD
    if not source["DAPS"]["pass"] or not source["SITCOM"]["pass"] or not source["DIFFFPR"]["pass"]:
        raise RuntimeError("pinned external source identity drift")
    source["snippets"] = {
        "DAPS": bounded_source_snippets(args.daps_root.resolve(), ["forward_operator.py", "posterior_sample.py"]),
        "SITCOM": bounded_source_snippets(args.sitcom_root.resolve(), ["forward_operator.py", "sampler.py"]),
        "NP_REPO": bounded_source_snippets(args.repo.resolve(), [
            "scripts/b24/run_b24_3_dev80_np.py",
            "scripts/b24/run_b24_3_np_branching.py",
            "scripts/pr_external_difffpr_np_benchmark.py",
            "scripts/b22/run_b22_1_sitcom_smoke.py",
        ]),
    }
    writej(out / "SOURCE_AUDIT.json", source)

    syn = synthetic_invariance(float(cfg3["invariance_tolerances"]["synthetic_relative_l2"]))
    if not syn["pass"]:
        raise RuntimeError(f"synthetic symmetry invariance failed: {syn}")
    writej(out / "SYMMETRY_SYNTHETIC_VALIDATION.json", syn)

    hard_rows = read_csv((args.closeout / "HARD_SUBSET.csv").resolve())
    shared_ids = {str(r["image_id"]).zfill(5) for r in hard_rows}
    if len(shared_ids) != 10:
        raise RuntimeError(f"corrected shared-failure subset drift: {len(shared_ids)}")
    closeout_rows = {
        str(r["image_id"]).zfill(5): r
        for r in read_csv((args.closeout / "DEV80_CLOSEOUT_PER_IMAGE.csv").resolve())
    }
    if set(closeout_rows) != dev_ids:
        raise RuntimeError("closeout DEV80 identity drift")
    fresh2_rows = {
        str(r["image_id"]).zfill(5): r
        for r in read_csv((args.closeout / "FRESH2_PER_IMAGE.csv").resolve())
    }

    terminal_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    psnr_tol = float(cfg3["invariance_tolerances"]["psnr_recompute_abs_db"])
    real_inv_tol = float(cfg3["invariance_tolerances"]["real_relative_l2"])

    for task in sorted(tasks, key=lambda t: (str(t["class_label"]), str(t["image_id"]))):
        image_id = str(task["image_id"]).zfill(5)
        taskdir = (args.dev80_run / "workers" / f"gpu{int(task['assigned_gpu'])}" / task["output_subdir"]).resolve()
        complete_path = taskdir / "IMAGE_COMPLETE.json"
        if not complete_path.is_file():
            raise FileNotFoundError(complete_path)
        comp = readj(complete_path)
        if comp.get("status") != "PASS" or str(comp.get("image_id")).zfill(5) != image_id:
            raise RuntimeError(f"bad DEV completion: {complete_path}")

        input_manifest_path = find_input_manifest(task, taskdir)
        inp = readj(input_manifest_path)
        if str(inp["image_id"]).zfill(5) != image_id:
            raise RuntimeError("input manifest image identity drift")
        measurement_path = Path(inp["measurement_path"]).resolve()
        gt_path = Path(inp["ground_truth_tensor_path"]).resolve()
        if not measurement_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(f"missing allowlisted DEV payload for {image_id}")
        if sha256_file(measurement_path) != str(inp["measurement_file_sha256"]):
            raise RuntimeError(f"measurement file hash mismatch {image_id}")
        measurement_tensor = unwrap_tensor(measurement_path, ("measurement", "y", "observation"))
        if tensor_sha256(measurement_tensor) != str(inp["measurement_tensor_sha256"]):
            raise RuntimeError(f"measurement tensor hash mismatch {image_id}")
        if tuple(measurement_tensor.shape) != (1, 3, 384, 384):
            raise RuntimeError(f"measurement shape drift {image_id}: {tuple(measurement_tensor.shape)}")
        measurement_raw = measurement_tensor.float().numpy()[0].astype(np.float64)
        measurement_clamped = np.maximum(measurement_raw, 0.0)

        gt_tensor = unwrap_tensor(gt_path, ("ground_truth", "image", "x"))
        if tensor_sha256(gt_tensor) != str(inp["ground_truth_tensor_sha256"]):
            raise RuntimeError(f"GT tensor hash mismatch {image_id}")
        gt01 = canonical_quantized01_model_range(gt_tensor)

        negative = measurement_raw < 0
        preprocessing_rows.append({
            "image_id": image_id,
            "screening_stratum": str(task["class_label"]),
            "measurement_path": str(measurement_path),
            "measurement_tensor_sha256": str(inp["measurement_tensor_sha256"]),
            "element_count": int(measurement_raw.size),
            "negative_count": int(negative.sum()),
            "negative_fraction": float(negative.mean()),
            "minimum_raw_amplitude": float(measurement_raw.min()),
            "raw_to_clamped_l2": float(np.linalg.norm(measurement_raw - measurement_clamped)),
            "raw_to_clamped_relative_l2": rel_l2(measurement_clamped, measurement_raw),
        })

        metrics = readj(Path(comp["baseline_metrics"]).resolve())
        expected = baseline_metric_lookup(metrics)

        daps_group = readj(Path(comp["daps_group"]).resolve())
        sitcom_group = readj(Path(comp["sitcom_group"]).resolve())
        daps_candidates = sorted(daps_group["candidate_rows"], key=lambda r: int(r["rep"]))
        sitcom_candidates = sorted(sitcom_group["candidate_rows"], key=lambda r: int(r["rep"]))
        if len(daps_candidates) > 4 or len(sitcom_candidates) > 4:
            raise RuntimeError("baseline candidate coverage exceeds B25 cap")

        for cand in daps_candidates:
            rep = int(cand["rep"])
            rec, source_path = load_daps_terminal(cand)
            terminal_rows.append(terminal_diagnostics(
                image_id=image_id,
                screening_stratum=str(task["class_label"]),
                method="DAPS",
                candidate_index=rep,
                rec01=rec,
                source_path=source_path,
                source_hash=str(cand["terminal_content_sha256"]),
                gt01=gt01,
                measurement_raw=measurement_raw,
                measurement_clamped=measurement_clamped,
                expected_psnr=expected[("DAPS", rep)],
                psnr_tol=psnr_tol,
                invariance_tol=real_inv_tol,
            ))

        for cand in sitcom_candidates:
            rep = int(cand["rep"])
            rec, source_path = load_sitcom_terminal(cand)
            terminal_rows.append(terminal_diagnostics(
                image_id=image_id,
                screening_stratum=str(task["class_label"]),
                method="SITCOM",
                candidate_index=rep,
                rec01=rec,
                source_path=source_path,
                source_hash=str(cand["terminal_content_sha256"]),
                gt01=gt01,
                measurement_raw=measurement_raw,
                measurement_clamped=measurement_clamped,
                expected_psnr=expected[("SITCOM", rep)],
                psnr_tol=psnr_tol,
                invariance_tol=real_inv_tol,
            ))

        np_result_path = Path(comp["np_results"]["NP4_INDEPENDENT"]).resolve()
        np_result = readj(np_result_path)
        np_terms = np_result.get("terminals", [])
        if len(np_terms) > 4 or len(np_terms) == 0:
            raise RuntimeError(f"NP4 candidate coverage drift {image_id}: {len(np_terms)}")
        np_image_rows = []
        for term in sorted(np_terms, key=lambda r: int(r["terminal_index"])):
            idx = int(term["terminal_index"])
            rec, source_path = load_np_terminal(term)
            row = terminal_diagnostics(
                image_id=image_id,
                screening_stratum=str(task["class_label"]),
                method="NP4_INDEPENDENT",
                candidate_index=idx,
                rec01=rec,
                source_path=source_path,
                source_hash=str(term["reconstruction_sha256"]),
                gt01=gt01,
                measurement_raw=measurement_raw,
                measurement_clamped=measurement_clamped,
                expected_psnr=None,
                psnr_tol=psnr_tol,
                invariance_tol=real_inv_tol,
            )
            terminal_rows.append(row)
            np_image_rows.append(row)
        selected_index = int(np_result["clean_free_selected_terminal_index"])
        selected_row = next(r for r in np_image_rows if int(r["candidate_index"]) == selected_index)
        historical_np4 = float(closeout_rows[image_id]["NP4_SELECTED"])
        if abs(float(selected_row["original_psnr_raw_rgb_db"]) - historical_np4) > psnr_tol:
            raise RuntimeError(
                f"NP4 canonical selected PSNR drift {image_id}: "
                f"{selected_row['original_psnr_raw_rgb_db']} vs {historical_np4}"
            )

        if len(daps_candidates) + len(sitcom_candidates) + len(np_terms) > int(cfg3["max_unique_terminals_per_image"]):
            raise RuntimeError(f"terminal cap exceeded on {image_id}")

    if len(terminal_rows) > 80 * int(cfg3["max_unique_terminals_per_image"]):
        raise RuntimeError("global terminal cap exceeded")
    write_csv(out / "EXP3_TERMINALS.csv", terminal_rows)

    per_image = summarize_by_image(terminal_rows, shared_ids)
    writej(out / "EXP3_PER_IMAGE.json", per_image)
    pirows = per_image["rows"]
    strata = {}
    for label in "ABCD":
        ids = {image for image in dev_ids if closeout_rows[image].get("screening_stratum", closeout_rows[image].get("class_label", "")) == label}
        if not ids:
            ids = {r["image_id"] for r in pirows if r["screening_stratum"] == label}
        strata[label] = aggregate(pirows, ids)
    symmetry_summary = {
        "schema_version": "b25.exp3-summary.v1",
        "status": "PASS",
        "dev_images": len(dev_ids),
        "terminal_rows": len(terminal_rows),
        "max_unique_terminals_per_image": int(cfg3["max_unique_terminals_per_image"]),
        "fresh2_counted_as_new_candidate": False,
        "fresh2_pointer_rows_present": len(fresh2_rows) == 80,
        "shared_failure_definition": cfg3["shared_failure_definition"],
        "shared_failure_count": len(shared_ids),
        "shared_failure_ids": sorted(shared_ids),
        "all_dev80": aggregate(pirows),
        "by_screening_stratum": strata,
        "shared_failure_subset": aggregate(pirows, shared_ids),
        "synthetic_invariance": syn,
        "gt_assisted_oracle_only": True,
    }
    writej(out / "EXP3_SYMMETRY_SUMMARY.json", symmetry_summary)

    write_csv(out / "EXP4_MEASUREMENT_NEGATIVES.csv", preprocessing_rows)
    neg_counts = [int(r["negative_count"]) for r in preprocessing_rows]
    rel_changes = [float(r["raw_to_clamped_relative_l2"]) for r in preprocessing_rows]
    values = np.array(spec["experiment4"]["synthetic_values"], dtype=float)
    clipped = np.maximum(values, 0.0)
    exp4 = {
        "schema_version": "b25.exp4-audit.v1",
        "status": "SOURCE_AUDIT_COMPLETE_NUMERICAL_CONSEQUENCE_MEASURED",
        "known_repo_facts_before_run": spec["experiment4"]["known_before_run"],
        "source_identity": {k: v for k, v in source.items() if k != "snippets"},
        "dev80_measurement_preprocessing": {
            "images": len(preprocessing_rows),
            "images_with_negative_stored_amplitudes": sum(n > 0 for n in neg_counts),
            "total_negative_elements": int(sum(neg_counts)),
            "negative_count_distribution": dist([float(n) for n in neg_counts]),
            "raw_to_clamped_relative_l2_distribution": dist(rel_changes),
        },
        "synthetic_clipping_demo": {
            "raw": values.tolist(),
            "clamp_min_zero": clipped.tolist(),
            "changed_indices": np.flatnonzero(values != clipped).tolist(),
            "l2_change": float(np.linalg.norm(values - clipped)),
        },
        "interpretation_gate": "Use SOURCE_AUDIT.json line excerpts to classify DAPS preprocessing; do not infer it from the numerical demo alone.",
        "confirmation_payloads_accessed": False,
        "new_ffhq_measurements_generated": False,
        "new_ffhq_reconstructions_generated": False,
    }
    writej(out / "EXP4_PREPROCESSING_AUDIT.json", exp4)

    ru = resource.getrusage(resource.RUSAGE_SELF)
    resource_payload = {
        "schema_version": "b25.dev-resource.v1",
        "wall_seconds": time.perf_counter() - started,
        "max_rss_raw": ru.ru_maxrss,
        "max_rss_note": "Linux ru_maxrss is KiB",
        "max_rss_gib": float(ru.ru_maxrss) / (1024.0 * 1024.0),
        "gpu_work_performed": False,
        "pretrained_model_inference_performed": False,
        "confirmation_payloads_accessed": False,
        "dev_payload_images_opened": 80,
        "terminal_rows_evaluated": len(terminal_rows),
    }
    writej(out / "DEV_RESOURCE.json", resource_payload)
    print(json.dumps({
        "status": "PASS",
        "dev_images": 80,
        "terminal_rows": len(terminal_rows),
        "shared_failure_count": len(shared_ids),
        "images_with_negative_measurements": exp4["dev80_measurement_preprocessing"]["images_with_negative_stored_amplitudes"],
        "wall_seconds": resource_payload["wall_seconds"],
        "max_rss_gib": resource_payload["max_rss_gib"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
