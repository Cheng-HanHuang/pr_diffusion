#!/usr/bin/env python3
"""A14 prospective validation for dual predeclared frozen Branch A policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


LOWFREQ_FRAC = 0.125
RESIDUAL_WINDOW_FRAC = 0.80
SAMPLE_RE = re.compile(r"_(\d{5})_run(\d{4})\.png$")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def mean_or_nan(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else math.nan


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


def merge_chunk_csvs(outdir: Path, chunk_dirs: Sequence[Path]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    step_rows: List[Dict[str, str]] = []
    run_rows: List[Dict[str, str]] = []
    for chunk in chunk_dirs:
        step_rows.extend(read_csv(chunk / "trajectory_step_metrics.csv"))
        run_rows.extend(read_csv(chunk / "run_level_summary.csv"))
    step_rows.sort(key=lambda r: (str(r["image_id"]), int(r["run_index"]), int(r["step"])))
    run_rows.sort(key=lambda r: (str(r["image_id"]), int(r["run_index"])))
    write_csv(outdir / "trajectory_step_metrics.csv", step_rows)
    write_csv(outdir / "run_level_summary.csv", run_rows)
    return step_rows, run_rows


def copy_samples(outdir: Path, chunk_dirs: Sequence[Path]) -> None:
    sample_dir = outdir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunk_dirs:
        src_dir = chunk / "samples"
        if not src_dir.exists():
            continue
        for png in sorted(src_dir.glob("*.png")):
            dst = sample_dir / f"{chunk.name}_{png.name}"
            if not dst.exists():
                shutil.copy2(png, dst)


def rank_within_group(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    ranks[order] = np.arange(1, arr.size + 1, dtype=float)
    return [float(x) for x in ranks]


def add_interrun_rank_features(step_rows: List[Dict[str, str]]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["step"]))].append(row)
    base = "x0y_full_residual_normed"
    for rows in grouped.values():
        vals = np.asarray([to_float(r.get(base)) for r in rows], dtype=float)
        if vals.size == 0 or np.any(~np.isfinite(vals)):
            continue
        ranks = rank_within_group(vals)
        for row, rank in zip(rows, ranks):
            row[f"{base}__interrun_rank"] = rank


def slope_or_nan(vals: Sequence[float]) -> float:
    xs = np.asarray(vals, dtype=float)
    if xs.size < 2 or np.any(~np.isfinite(xs)):
        return math.nan
    t = np.linspace(0.0, 1.0, xs.size)
    t = t - t.mean()
    y = xs - xs.mean()
    denom = float(np.dot(t, t))
    if denom <= 0.0:
        return math.nan
    return float(np.dot(t, y) / denom)


def parse_sample_paths(sample_dir: Path) -> Dict[Tuple[str, int], Path]:
    out: Dict[Tuple[str, int], Path] = {}
    for path in sorted(sample_dir.glob("*.png")):
        m = SAMPLE_RE.search(path.name)
        if not m:
            continue
        out[(m.group(1), int(m.group(2)))] = path
    return out


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def lowfreq_repr(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(fft).astype(np.float32)
    h, w = mag.shape
    hh = max(1, int(round(h * LOWFREQ_FRAC / 2.0)))
    ww = max(1, int(round(w * LOWFREQ_FRAC / 2.0)))
    cy, cx = h // 2, w // 2
    return mag[cy - hh : cy + hh, cx - ww : cx + ww]


def l2_normed(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(diff.ravel()) / math.sqrt(diff.size))


def build_run_feature_rows(step_rows: List[Dict[str, str]], run_rows: List[Dict[str, str]], sample_dir: Path) -> List[Dict[str, object]]:
    add_interrun_rank_features(step_rows)
    grouped_steps: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped_steps[(str(row["image_id"]), int(row["run_index"]))].append(row)
    for rows in grouped_steps.values():
        rows.sort(key=lambda r: int(r["step"]))

    run_meta = {(str(r["image_id"]), int(r["run_index"])): r for r in run_rows}
    sample_paths = parse_sample_paths(sample_dir)
    by_image: Dict[str, List[Tuple[int, Dict[str, str], Path]]] = defaultdict(list)
    for key, meta in run_meta.items():
        if key not in sample_paths:
            continue
        by_image[key[0]].append((key[1], meta, sample_paths[key]))

    out_rows: List[Dict[str, object]] = []
    for image_id, items in sorted(by_image.items()):
        items.sort(key=lambda t: t[0])
        if len(items) != 4:
            raise ValueError(f"Expected 4 runs for image {image_id}, found {len(items)}")
        rgbs = [load_rgb(p) for _, _, p in items]
        lowfs = [lowfreq_repr(rgb) for rgb in rgbs]
        lowf_pair = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(i + 1, 4):
                d = l2_normed(lowfs[i], lowfs[j])
                lowf_pair[i, j] = lowf_pair[j, i] = d

        for i, (run_index, meta, sample_path) in enumerate(items):
            step_key = (image_id, run_index)
            rows = grouped_steps[step_key]
            k = max(1, int(math.ceil(len(rows) * RESIDUAL_WINDOW_FRAC)))
            wr = rows[:k]
            vals = [to_float(r.get("x0y_full_residual_normed__interrun_rank")) for r in wr]
            final_psnr = to_float(meta["final_psnr"])
            out_rows.append(
                {
                    "image_id": image_id,
                    "run_index": run_index,
                    "final_psnr": final_psnr,
                    "bad25": int(final_psnr < 25.0),
                    "bad20": int(final_psnr < 20.0),
                    "x0y_full_residual_normed__interrun_rank__first80pct__slope": slope_or_nan(vals),
                    "x0y_full_residual_normed__interrun_rank__first80pct__last_in_window": vals[-1] if vals else math.nan,
                    "lowfreq_dist_to_nearest_neighbor": float(np.min(np.delete(lowf_pair[i], i))),
                    "sample_path": str(sample_path),
                }
            )
    out_rows.sort(key=lambda r: (str(r["image_id"]), int(r["run_index"])))
    return out_rows


def load_np_fallbacks(np_csv: Path, noise: float, image_ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
    rows = read_csv(np_csv)
    candidates = []
    targets = set(image_ids)
    for row in rows:
        if row.get("alignment_mode") != "resolve":
            continue
        if abs(to_float(row.get("measurement_noise_std")) - noise) > 1e-12:
            continue
        image_id = image_id_from_np_basename(str(row["image_basename"]))
        if image_id not in targets:
            continue
        candidates.append(
            {
                "image_id": image_id,
                "config_tag": row["config_tag"],
                "seed": int(row["seed"]),
                "psnr": to_float(row["psnr"]),
                "selector_post_winner_lf_mse_mean": to_float(row["selector_post_winner_lf_mse_mean"]),
            }
        )
    out: Dict[str, Dict[str, object]] = {}
    for image_id in image_ids:
        cur = [r for r in candidates if r["image_id"] == image_id]
        if not cur:
            raise ValueError(f"No NP fallback candidates for {image_id} at noise {noise}")
        best = min(
            cur,
            key=lambda r: (
                r["selector_post_winner_lf_mse_mean"],
                -r["psnr"],
                str(r["config_tag"]),
                int(r["seed"]),
            ),
        )
        out[image_id] = {
            "np_selected_psnr": best["psnr"],
            "np_selected_config_tag": best["config_tag"],
            "np_selected_seed": best["seed"],
            "np_selected_selector_post_lf_mse": best["selector_post_winner_lf_mse_mean"],
        }
    return out


def apply_threshold(value: float, direction: str, threshold: float) -> bool:
    if direction == "high_is_risky":
        return value >= threshold
    if direction == "low_is_risky":
        return value <= threshold
    raise ValueError(direction)


def summarize_policy(policy_rows: List[Dict[str, object]], image_rows: List[Dict[str, object]]) -> Dict[str, object]:
    psnrs = [to_float(r["policy_final_psnr"]) for r in policy_rows]
    fp_losses = [to_float(r["delta_vs_sitcom"]) for r in policy_rows if bool(r["false_positive_replacement"])]
    remaining_bad = [to_float(r["sitcom_final_psnr"]) for r in policy_rows if bool(r["false_negative_remaining_bad25"])]
    remaining_cat = [to_float(r["sitcom_final_psnr"]) for r in policy_rows if bool(r["false_negative_remaining_bad20"])]
    return {
        "policy_name": policy_rows[0]["policy_name"],
        "policy_kind": policy_rows[0]["policy_kind"],
        "run_level_mean_psnr": mean_or_nan(psnrs),
        "run_level_min_psnr": min(psnrs),
        "run_level_num_below25": sum(1 for v in psnrs if v < 25.0),
        "run_level_num_below20": sum(1 for v in psnrs if v < 20.0),
        "num_replaced": sum(1 for r in policy_rows if bool(r["was_replaced"])),
        "num_false_positive_replacements": sum(1 for r in policy_rows if bool(r["false_positive_replacement"])),
        "num_true_positive_replacements": sum(1 for r in policy_rows if bool(r["true_positive_replacement"])),
        "num_false_negative_remaining_bad25": sum(1 for r in policy_rows if bool(r["false_negative_remaining_bad25"])),
        "num_false_negative_remaining_bad20": sum(1 for r in policy_rows if bool(r["false_negative_remaining_bad20"])),
        "image_level_best_of_4_mean_psnr": mean_or_nan(to_float(r["best_of_4_psnr"]) for r in image_rows),
        "image_level_best_of_4_min_psnr": min(to_float(r["best_of_4_psnr"]) for r in image_rows),
        "worst_false_positive_psnr_loss": min(fp_losses) if fp_losses else math.nan,
        "mean_false_positive_psnr_loss": mean_or_nan(fp_losses),
        "worst_remaining_bad25_miss": min(remaining_bad) if remaining_bad else math.nan,
        "worst_remaining_bad20_miss": min(remaining_cat) if remaining_cat else math.nan,
    }


def build_image_level_rows(policy_rows: List[Dict[str, object]], sitcom_image_ref: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in policy_rows:
        grouped[str(row["image_id"])].append(row)
    out = []
    for image_id, rows in sorted(grouped.items()):
        vals = [to_float(r["policy_final_psnr"]) for r in rows]
        base = sitcom_image_ref[image_id]
        out.append(
            {
                "policy_name": rows[0]["policy_name"],
                "policy_kind": rows[0]["policy_kind"],
                "image_id": image_id,
                "best_of_4_psnr": max(vals),
                "mean_of_4_psnr": mean_or_nan(vals),
                "min_of_4_psnr": min(vals),
                "num_runs_below25": sum(1 for v in vals if v < 25.0),
                "num_runs_below20": sum(1 for v in vals if v < 20.0),
                "replaced_run_indices": ",".join(str(int(r["run_index"])) for r in rows if bool(r["was_replaced"])),
                "sitcom_only_best_of_4_psnr": base["best_of_4_psnr"],
                "sitcom_only_min_of_4_psnr": base["min_of_4_psnr"],
                "delta_best_of_4_vs_sitcom": max(vals) - base["best_of_4_psnr"],
                "delta_min_of_4_vs_sitcom": min(vals) - base["min_of_4_psnr"],
            }
        )
    return out


def make_sitcom_image_ref(run_rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["image_id"])].append(to_float(row["policy_final_psnr"]))
    out = {}
    for image_id, vals in grouped.items():
        out[image_id] = {
            "best_of_4_psnr": max(vals),
            "min_of_4_psnr": min(vals),
        }
    return out


def baseline_policy(
    name: str,
    kind: str,
    feature_rows: List[Dict[str, object]],
    np_fallbacks: Dict[str, Dict[str, object]],
    mode: str,
) -> List[Dict[str, object]]:
    out_rows: List[Dict[str, object]] = []
    for row in feature_rows:
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        sitcom_psnr = to_float(row["final_psnr"])
        if mode == "sitcom_only":
            replace = False
        elif mode == "replace_all":
            replace = True
        elif mode == "oracle_bad25":
            replace = sitcom_psnr < 25.0
        else:
            raise ValueError(mode)
        fb = np_fallbacks[image_id]
        policy_psnr = float(fb["np_selected_psnr"]) if replace else sitcom_psnr
        out_rows.append(
            {
                "policy_name": name,
                "policy_kind": kind,
                "image_id": image_id,
                "run_index": run_index,
                "sitcom_final_psnr": sitcom_psnr,
                "policy_final_psnr": policy_psnr,
                "delta_vs_sitcom": policy_psnr - sitcom_psnr,
                "sitcom_bad_below25": sitcom_psnr < 25.0,
                "sitcom_bad_below20": sitcom_psnr < 20.0,
                "final_bad_below25": policy_psnr < 25.0,
                "final_bad_below20": policy_psnr < 20.0,
                "was_flagged": replace if mode != "sitcom_only" else False,
                "was_replaced": replace,
                "replacement_source": "np_selected" if replace else "sitcom",
                "replacement_detail": f"{fb['np_selected_config_tag']} seed={fb['np_selected_seed']}" if replace else "",
                "true_positive_replacement": replace and sitcom_psnr < 25.0,
                "false_positive_replacement": replace and sitcom_psnr >= 25.0,
                "false_negative_remaining_bad25": (not replace) and sitcom_psnr < 25.0,
                "false_negative_remaining_bad20": (not replace) and sitcom_psnr < 20.0,
                "missed_catastrophic_run": (not replace) and sitcom_psnr < 20.0,
            }
        )
    return out_rows


def apply_conservative_policy(
    feature_rows: List[Dict[str, object]],
    cfg: Dict[str, object],
    np_fallbacks: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    spec = cfg["policy_spec"]
    feature_name = str(spec["feature_name"])
    direction = str(spec["direction"])
    threshold = float(spec["threshold"])
    out_rows: List[Dict[str, object]] = []
    for row in feature_rows:
        image_id = str(row["image_id"])
        sitcom_psnr = to_float(row["final_psnr"])
        value = to_float(row[feature_name])
        flagged = apply_threshold(value, direction, threshold)
        fb = np_fallbacks[image_id]
        policy_psnr = float(fb["np_selected_psnr"]) if flagged else sitcom_psnr
        out_rows.append(
            {
                "policy_name": str(cfg["policy_name"]),
                "policy_kind": "primary_prospective_frozen_policy",
                "policy_role": "primary_conservative",
                "image_id": image_id,
                "run_index": int(row["run_index"]),
                "sitcom_final_psnr": sitcom_psnr,
                "policy_final_psnr": policy_psnr,
                "delta_vs_sitcom": policy_psnr - sitcom_psnr,
                "sitcom_bad_below25": sitcom_psnr < 25.0,
                "sitcom_bad_below20": sitcom_psnr < 20.0,
                "final_bad_below25": policy_psnr < 25.0,
                "final_bad_below20": policy_psnr < 20.0,
                "was_flagged": flagged,
                "was_replaced": flagged,
                "replacement_source": "np_selected" if flagged else "sitcom",
                "replacement_detail": f"{fb['np_selected_config_tag']} seed={fb['np_selected_seed']}" if flagged else "",
                "true_positive_replacement": flagged and sitcom_psnr < 25.0,
                "false_positive_replacement": flagged and sitcom_psnr >= 25.0,
                "false_negative_remaining_bad25": (not flagged) and sitcom_psnr < 25.0,
                "false_negative_remaining_bad20": (not flagged) and sitcom_psnr < 20.0,
                "missed_catastrophic_run": (not flagged) and sitcom_psnr < 20.0,
                "combine_mode": str(cfg.get("combine_mode", "single")),
                "consensus_feature_name": feature_name,
                "consensus_feature_value": value,
                "consensus_direction": direction,
                "consensus_threshold": threshold,
                "consensus_flag": flagged,
            }
        )
    return out_rows


def apply_aggressive_policy(
    feature_rows: List[Dict[str, object]],
    cfg: Dict[str, object],
    np_fallbacks: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    spec = cfg["policy_spec"]
    residual_spec = spec["residual_spec"]
    consensus_spec = spec["consensus_spec"]
    residual_features = list(residual_spec["feature_names"])
    residual_dirs = {str(k): str(v) for k, v in residual_spec["feature_directions"].items()}
    residual_thr = {str(k): float(v) for k, v in residual_spec["thresholds"].items()}
    consensus_feature = str(consensus_spec["feature_name"])
    consensus_direction = str(consensus_spec["direction"])
    consensus_threshold = float(consensus_spec["threshold"])
    top_combine = str(cfg.get("combine_mode", spec.get("combine_mode", "or"))).lower()
    residual_combine = str(residual_spec.get("combine_mode", "and")).lower()
    out_rows: List[Dict[str, object]] = []
    for row in feature_rows:
        image_id = str(row["image_id"])
        sitcom_psnr = to_float(row["final_psnr"])
        residual_values = {name: to_float(row[name]) for name in residual_features}
        residual_flags = {
            name: apply_threshold(residual_values[name], residual_dirs[name], residual_thr[name])
            for name in residual_features
        }
        if residual_combine != "and":
            raise ValueError(f"Unsupported residual combine mode {residual_combine}")
        residual_arm_flag = all(residual_flags.values())
        consensus_value = to_float(row[consensus_feature])
        consensus_flag = apply_threshold(consensus_value, consensus_direction, consensus_threshold)
        if top_combine != "or":
            raise ValueError(f"Unsupported top-level combine mode {top_combine}")
        flagged = residual_arm_flag or consensus_flag
        fb = np_fallbacks[image_id]
        policy_psnr = float(fb["np_selected_psnr"]) if flagged else sitcom_psnr
        record: Dict[str, object] = {
            "policy_name": str(cfg["policy_name"]),
            "policy_kind": "secondary_prospective_frozen_policy",
            "policy_role": "secondary_aggressive_higher_replacement_budget",
            "image_id": image_id,
            "run_index": int(row["run_index"]),
            "sitcom_final_psnr": sitcom_psnr,
            "policy_final_psnr": policy_psnr,
            "delta_vs_sitcom": policy_psnr - sitcom_psnr,
            "sitcom_bad_below25": sitcom_psnr < 25.0,
            "sitcom_bad_below20": sitcom_psnr < 20.0,
            "final_bad_below25": policy_psnr < 25.0,
            "final_bad_below20": policy_psnr < 20.0,
            "was_flagged": flagged,
            "was_replaced": flagged,
            "replacement_source": "np_selected" if flagged else "sitcom",
            "replacement_detail": f"{fb['np_selected_config_tag']} seed={fb['np_selected_seed']}" if flagged else "",
            "true_positive_replacement": flagged and sitcom_psnr < 25.0,
            "false_positive_replacement": flagged and sitcom_psnr >= 25.0,
            "false_negative_remaining_bad25": (not flagged) and sitcom_psnr < 25.0,
            "false_negative_remaining_bad20": (not flagged) and sitcom_psnr < 20.0,
            "missed_catastrophic_run": (not flagged) and sitcom_psnr < 20.0,
            "combine_mode": top_combine,
            "residual_arm_combine_mode": residual_combine,
            "residual_arm_flag": residual_arm_flag,
            "consensus_arm_flag": consensus_flag,
            "consensus_feature_name": consensus_feature,
            "consensus_feature_value": consensus_value,
            "consensus_direction": consensus_direction,
            "consensus_threshold": consensus_threshold,
        }
        for name in residual_features:
            record[f"{name}__value"] = residual_values[name]
            record[f"{name}__direction"] = residual_dirs[name]
            record[f"{name}__threshold"] = residual_thr[name]
            record[f"{name}__flagged"] = residual_flags[name]
        out_rows.append(record)
    return out_rows


def make_policy_image_and_summary(policy_rows: List[Dict[str, object]], sitcom_image_ref: Dict[str, Dict[str, float]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    image_rows = build_image_level_rows(policy_rows, sitcom_image_ref)
    return image_rows, summarize_policy(policy_rows, image_rows)


def filter_failure_rows(policy_rows: List[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    return [row.copy() for row in policy_rows if bool(row[key])]


def build_replacement_loss_summary(policy_rows_by_name: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for policy_name, rows in policy_rows_by_name.items():
        fp_rows = [r for r in rows if bool(r["false_positive_replacement"])]
        losses = [to_float(r["delta_vs_sitcom"]) for r in fp_rows]
        worst_row = min(fp_rows, key=lambda r: to_float(r["delta_vs_sitcom"])) if fp_rows else None
        out.append(
            {
                "policy_name": policy_name,
                "num_false_positive_replacements": len(fp_rows),
                "mean_false_positive_psnr_loss": mean_or_nan(losses),
                "worst_false_positive_psnr_loss": min(losses) if losses else math.nan,
                "worst_loss_image_id": worst_row["image_id"] if worst_row else "",
                "worst_loss_run_index": worst_row["run_index"] if worst_row else "",
                "worst_loss_sitcom_final_psnr": worst_row["sitcom_final_psnr"] if worst_row else "",
                "worst_loss_policy_final_psnr": worst_row["policy_final_psnr"] if worst_row else "",
            }
        )
    return out


def render_summary(
    outdir: Path,
    policy_summaries: List[Dict[str, object]],
    chunk_dirs: Sequence[Path],
    conservative_src: Path,
    aggressive_src: Path,
    git_commit: str,
    step_rows: List[Dict[str, str]],
    run_rows: List[Dict[str, str]],
    exact_commands: Sequence[str],
) -> None:
    by_name = {str(r["policy_name"]): r for r in policy_summaries}
    lines = [
        "# A14 Prospective Dual-Policy Validation",
        "",
        "This analysis uses a fresh SITCOM trajectory run and evaluates the two predeclared frozen A14 Branch A policies without any retuning.",
        "",
        "## Policy Summary",
        "",
        "| policy | kind | run mean | run min | bad25 | bad20 | image best-of-4 mean | image best-of-4 min | replaced | TP repl | FP repl | FN bad25 | FN bad20 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "sitcom_only",
        "consensus_lowfreq_nn",
        "residual_or_lowfreq_nn",
        "replace_all_np_selected",
        "oracle_risk_np_selected",
    ]
    for name in order:
        row = by_name[name]
        lines.append(
            "| {policy} | {kind} | {run_mean:.3f} | {run_min:.3f} | {b25} | {b20} | {img_mean:.3f} | {img_min:.3f} | {repl} | {tp} | {fp} | {fn25} | {fn20} |".format(
                policy=row["policy_name"],
                kind=row["policy_kind"],
                run_mean=to_float(row["run_level_mean_psnr"]),
                run_min=to_float(row["run_level_min_psnr"]),
                b25=int(to_float(row["run_level_num_below25"])),
                b20=int(to_float(row["run_level_num_below20"])),
                img_mean=to_float(row["image_level_best_of_4_mean_psnr"]),
                img_min=to_float(row["image_level_best_of_4_min_psnr"]),
                repl=int(to_float(row["num_replaced"])),
                tp=int(to_float(row["num_true_positive_replacements"])),
                fp=int(to_float(row["num_false_positive_replacements"])),
                fn25=int(to_float(row["num_false_negative_remaining_bad25"])),
                fn20=int(to_float(row["num_false_negative_remaining_bad20"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `consensus_lowfreq_nn` is the primary conservative prospective result.",
            "- `residual_or_lowfreq_nn` is the secondary aggressive higher-replacement-budget prospective result.",
            "- `replace_all_np_selected` is a degenerate baseline and not an acceptable solver policy.",
            "- `oracle_risk_np_selected` is diagnostic only and not executable.",
            "- The copied frozen configs were used unchanged; no A14 result was used to alter thresholds, features, directions, or fallback source.",
            "",
            "## Provenance",
            "",
            f"- Git commit: `{git_commit}`",
            f"- GPU IDs: `3`",
            f"- Conservative config source: `{conservative_src}`",
            f"- Aggressive config source: `{aggressive_src}`",
            f"- Conservative config SHA256: `{sha256_file(outdir / 'frozen_policy_conservative.json')}`",
            f"- Aggressive config SHA256: `{sha256_file(outdir / 'frozen_policy_aggressive.json')}`",
            f"- Trajectory step row count: `{len(step_rows)}`",
            f"- Run summary row count: `{len(run_rows)}`",
            f"- Row-count sanity: `trajectory_step_metrics.csv == 20000 -> {len(step_rows) == 20000}`",
            f"- Row-count sanity: `run_level_summary.csv == 100 -> {len(run_rows) == 100}`",
            "",
            "Exact commands:",
            "",
        ]
    )
    for cmd in exact_commands:
        lines.append(f"- `{cmd}`")
    lines.extend(["", "Chunk configs:"])
    for chunk in chunk_dirs:
        cfg = json.loads((chunk / "config.json").read_text(encoding="utf-8"))
        lines.append(
            f"- `{chunk.name}`: seed `{cfg['seed']}`, gpu `{cfg['gpu']}`, image_ids `{','.join(cfg['image_ids'])}`"
        )
    write_text(outdir / "SUMMARY.md", "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chunk_dirs", nargs="+", required=True)
    ap.add_argument("--conservative_json", required=True)
    ap.add_argument("--aggressive_json", required=True)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--git_commit", required=True)
    ap.add_argument("--exact_command", action="append", default=[])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_dirs = [Path(p) for p in args.chunk_dirs]
    conservative_src = Path(args.conservative_json)
    aggressive_src = Path(args.aggressive_json)

    conservative_cfg = json.loads(conservative_src.read_text(encoding="utf-8"))
    aggressive_cfg = json.loads(aggressive_src.read_text(encoding="utf-8"))
    shutil.copy2(conservative_src, outdir / "frozen_policy_conservative.json")
    shutil.copy2(aggressive_src, outdir / "frozen_policy_aggressive.json")

    step_rows, run_rows = merge_chunk_csvs(outdir, chunk_dirs)
    if len(step_rows) != 20000:
        raise ValueError(f"Expected 20000 step rows, found {len(step_rows)}")
    if len(run_rows) != 100:
        raise ValueError(f"Expected 100 run rows, found {len(run_rows)}")
    copy_samples(outdir, chunk_dirs)

    feature_rows = build_run_feature_rows(step_rows, run_rows, outdir / "samples")
    image_ids = sorted({str(r["image_id"]) for r in feature_rows})
    np_fallbacks = load_np_fallbacks(Path(conservative_cfg["fallback_source_csv"]), args.noise, image_ids)

    sitcom_rows = baseline_policy(
        "sitcom_only",
        "executable_baseline",
        feature_rows,
        np_fallbacks,
        mode="sitcom_only",
    )
    sitcom_image_ref = make_sitcom_image_ref(sitcom_rows)
    sitcom_image_rows, sitcom_summary = make_policy_image_and_summary(sitcom_rows, sitcom_image_ref)

    conservative_rows = apply_conservative_policy(feature_rows, conservative_cfg, np_fallbacks)
    conservative_image_rows, conservative_summary = make_policy_image_and_summary(conservative_rows, sitcom_image_ref)

    aggressive_rows = apply_aggressive_policy(feature_rows, aggressive_cfg, np_fallbacks)
    aggressive_image_rows, aggressive_summary = make_policy_image_and_summary(aggressive_rows, sitcom_image_ref)

    replace_all_rows = baseline_policy(
        "replace_all_np_selected",
        "degenerate_replace_all_baseline",
        feature_rows,
        np_fallbacks,
        mode="replace_all",
    )
    replace_all_image_rows, replace_all_summary = make_policy_image_and_summary(replace_all_rows, sitcom_image_ref)

    oracle_rows = baseline_policy(
        "oracle_risk_np_selected",
        "diagnostic_oracle_risk",
        feature_rows,
        np_fallbacks,
        mode="oracle_bad25",
    )
    oracle_image_rows, oracle_summary = make_policy_image_and_summary(oracle_rows, sitcom_image_ref)

    all_image_rows = (
        sitcom_image_rows
        + conservative_image_rows
        + aggressive_image_rows
        + replace_all_image_rows
        + oracle_image_rows
    )
    policy_summaries = [
        sitcom_summary,
        conservative_summary,
        aggressive_summary,
        replace_all_summary,
        oracle_summary,
    ]

    write_csv(outdir / "conservative_policy_applied_runs.csv", conservative_rows)
    write_csv(outdir / "aggressive_policy_applied_runs.csv", aggressive_rows)
    write_csv(outdir / "controller_policy_summary.csv", policy_summaries)
    write_csv(outdir / "controller_policy_image_level.csv", all_image_rows)
    write_csv(
        outdir / "detector_missed_bad_runs.csv",
        filter_failure_rows(conservative_rows, "false_negative_remaining_bad25")
        + filter_failure_rows(aggressive_rows, "false_negative_remaining_bad25"),
    )
    write_csv(
        outdir / "detector_false_positive_runs.csv",
        filter_failure_rows(conservative_rows, "false_positive_replacement")
        + filter_failure_rows(aggressive_rows, "false_positive_replacement"),
    )
    write_csv(
        outdir / "replacement_loss_summary.csv",
        build_replacement_loss_summary(
            {
                "consensus_lowfreq_nn": conservative_rows,
                "residual_or_lowfreq_nn": aggressive_rows,
                "replace_all_np_selected": replace_all_rows,
                "oracle_risk_np_selected": oracle_rows,
            }
        ),
    )

    render_summary(
        outdir,
        policy_summaries,
        chunk_dirs,
        conservative_src,
        aggressive_src,
        args.git_commit,
        step_rows,
        run_rows,
        args.exact_command,
    )


if __name__ == "__main__":
    main()
