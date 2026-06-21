#!/usr/bin/env python3
"""Offline A6 risk-screen analysis from A5 trajectory CSVs plus A5.5 sanity.

This script reads the existing A5 SITCOM hard-image trajectory outputs and:

1. Recomputes early-window detector aggregates from `trajectory_step_metrics.csv`.
2. Adds simple inter-run features by comparing the four SITCOM runs at each
   image/step.
3. Scores threshold-based offline detectors for bad-final-run labels.
4. Compares the instrumented A5 run against the official SITCOM noise=0.05 run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


WINDOWS: List[Tuple[str, float]] = [
    ("first10pct", 0.10),
    ("first20pct", 0.20),
    ("first30pct", 0.30),
    ("first50pct", 0.50),
]

BASE_FEATURES: List[str] = [
    "x0y_full_residual_normed",
    "x0y_lowfreq_residual_normed",
    "correction_norm",
    "x0hat_x0y_disagreement",
    "xt_step_jump",
    "x0y_step_jump",
]

INTERRUN_BASES: List[str] = [
    "x0y_full_residual_normed",
    "x0y_lowfreq_residual_normed",
    "correction_norm",
    "x0hat_x0y_disagreement",
]

AGGREGATES: List[str] = ["max", "mean", "final_in_window", "slope"]


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


def finite(vals: Iterable[float]) -> List[float]:
    out = []
    for v in vals:
        if math.isfinite(v):
            out.append(v)
    return out


def mean_or_nan(vals: Iterable[float]) -> float:
    xs = finite(vals)
    return float(np.mean(xs)) if xs else math.nan


def make_run_key(row: Dict[str, str]) -> Tuple[str, int]:
    return str(row["image_id"]), int(row["run_index"])


def load_run_level_labels(run_level_rows: List[Dict[str, str]]) -> Dict[Tuple[str, int], Dict[str, object]]:
    labels: Dict[Tuple[str, int], Dict[str, object]] = {}
    for row in run_level_rows:
        key = make_run_key(row)
        final_psnr = to_float(row["final_psnr"])
        labels[key] = {
            "image_id": key[0],
            "run_index": key[1],
            "final_psnr": final_psnr,
            "bad25": int(final_psnr < 25.0),
            "bad20": int(final_psnr < 20.0),
        }
    return labels


def add_interrun_features(step_rows: List[Dict[str, str]]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["step"]))].append(row)

    for _, rows in grouped.items():
        for base in INTERRUN_BASES:
            vals = np.array([to_float(r[base]) for r in rows], dtype=float)
            if np.any(~np.isfinite(vals)):
                continue
            median = float(np.median(vals))
            order = np.argsort(np.argsort(vals))
            ranks_high = order + 1
            if len(vals) > 1:
                ranks_high = len(vals) - order
            for i, row in enumerate(rows):
                v = float(vals[i])
                row[f"{base}__interrun_rank_high"] = float(ranks_high[i])
                row[f"{base}__interrun_minus_median"] = v - median
                denom = median if abs(median) > 1e-12 else math.nan
                row[f"{base}__interrun_div_median"] = v / denom if math.isfinite(denom) else math.nan


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


def aggregate_window(vals: Sequence[float]) -> Dict[str, float]:
    xs = np.asarray(vals, dtype=float)
    if xs.size == 0 or np.any(~np.isfinite(xs)):
        return {agg: math.nan for agg in AGGREGATES}
    return {
        "max": float(np.max(xs)),
        "mean": float(np.mean(xs)),
        "final_in_window": float(xs[-1]),
        "slope": slope_or_nan(xs),
    }


def build_detector_rows(
    step_rows: List[Dict[str, str]],
    run_labels: Dict[Tuple[str, int], Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["run_index"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["step"]))

    sample_row = step_rows[0]
    features = [f for f in BASE_FEATURES if f in sample_row]
    for key in sample_row:
        if "__interrun_" in key:
            features.append(key)

    detector_rows: List[Dict[str, object]] = []
    for run_key, rows in sorted(grouped.items()):
        meta = run_labels[run_key]
        n_steps = len(rows)
        if n_steps == 0:
            continue
        row_out: Dict[str, object] = dict(meta)
        row_out["num_steps"] = n_steps
        for window_name, frac in WINDOWS:
            k = max(1, int(math.ceil(n_steps * frac)))
            window_rows = rows[:k]
            for feature in features:
                vals = [to_float(r.get(feature, math.nan)) for r in window_rows]
                for agg_name, agg_val in aggregate_window(vals).items():
                    detector_name = f"{feature}__{window_name}__{agg_name}"
                    row_out[detector_name] = agg_val
        detector_rows.append(row_out)
    return detector_rows


def average_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def auroc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = average_ranks(scores.astype(float))
    sum_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / max(tp + fp, 1)
        if label == 1:
            ap += precision * (recall - prev_recall)
            prev_recall = recall
    return ap


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def balanced_accuracy(counts: Dict[str, int]) -> float:
    pos = counts["tp"] + counts["fn"]
    neg = counts["tn"] + counts["fp"]
    if pos == 0 or neg == 0:
        return math.nan
    tpr = counts["tp"] / pos
    tnr = counts["tn"] / neg
    return 0.5 * (tpr + tnr)


def precision(counts: Dict[str, int]) -> float:
    denom = counts["tp"] + counts["fp"]
    return counts["tp"] / denom if denom else math.nan


def recall(counts: Dict[str, int]) -> float:
    denom = counts["tp"] + counts["fn"]
    return counts["tp"] / denom if denom else math.nan


def evaluate_thresholds(
    scores: np.ndarray,
    y_true: np.ndarray,
    detector_id: str,
    label_name: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    unique = sorted({float(x) for x in scores if math.isfinite(float(x))})
    if not unique:
        raise ValueError(f"no finite scores for {detector_id}")

    directions = {
        "high_is_risky": lambda s, thr: s >= thr,
        "low_is_risky": lambda s, thr: s <= thr,
    }
    curve_rows: List[Dict[str, object]] = []
    best_row: Dict[str, object] | None = None
    best_key: Tuple[float, int, int, float] | None = None

    auroc_high = auroc_score(y_true, scores)
    auprc_high = average_precision(y_true, scores)
    auroc_low = auroc_score(y_true, -scores)
    auprc_low = average_precision(y_true, -scores)
    auc_direction = "high_is_risky"
    auc_auroc = auroc_high
    auc_auprc = auprc_high
    if (math.isfinite(auroc_low) and not math.isfinite(auroc_high)) or (
        math.isfinite(auroc_low) and math.isfinite(auroc_high) and auroc_low > auroc_high
    ):
        auc_direction = "low_is_risky"
        auc_auroc = auroc_low
        auc_auprc = auprc_low

    for direction_name, pred_fn in directions.items():
        for threshold in unique:
            pred = pred_fn(scores, threshold).astype(int)
            counts = confusion_counts(y_true, pred)
            bal_acc = balanced_accuracy(counts)
            row = {
                "detector_id": detector_id,
                "label": label_name,
                "direction": direction_name,
                "threshold": threshold,
                **counts,
                "balanced_accuracy": bal_acc,
                "precision": precision(counts),
                "recall": recall(counts),
                "tpr": recall(counts),
                "fpr": counts["fp"] / (counts["fp"] + counts["tn"]) if (counts["fp"] + counts["tn"]) else math.nan,
                "is_best_balanced_accuracy": False,
            }
            curve_rows.append(row)
            key = (
                bal_acc if math.isfinite(bal_acc) else -math.inf,
                counts["tp"],
                -counts["fp"],
                -abs(threshold),
            )
            if best_row is None or key > best_key:
                best_row = row
                best_key = key

    assert best_row is not None
    for row in curve_rows:
        if row["direction"] == best_row["direction"] and row["threshold"] == best_row["threshold"]:
            row["is_best_balanced_accuracy"] = True

    summary = {
        "detector_id": detector_id,
        "label": label_name,
        "auc_direction": auc_direction,
        "auroc": auc_auroc,
        "auprc": auc_auprc,
        "best_threshold_direction": best_row["direction"],
        "best_threshold": best_row["threshold"],
        "best_balanced_accuracy": best_row["balanced_accuracy"],
        "best_tp": best_row["tp"],
        "best_fp": best_row["fp"],
        "best_tn": best_row["tn"],
        "best_fn": best_row["fn"],
    }
    return summary, curve_rows


def choose_best_threshold(scores: np.ndarray, y_true: np.ndarray) -> Dict[str, object]:
    summary, _ = evaluate_thresholds(scores, y_true, "tmp", "tmp")
    return summary


def apply_threshold(scores: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    if direction == "high_is_risky":
        return (scores >= threshold).astype(int)
    if direction == "low_is_risky":
        return (scores <= threshold).astype(int)
    raise ValueError(direction)


def leave_one_image_out(
    detector_rows: List[Dict[str, object]],
    detector_id: str,
    label_name: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    images = sorted({str(r["image_id"]) for r in detector_rows})
    eval_rows: List[Dict[str, object]] = []
    bal_accs: List[float] = []
    total = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for image_id in images:
        train = [r for r in detector_rows if str(r["image_id"]) != image_id]
        test = [r for r in detector_rows if str(r["image_id"]) == image_id]
        train_scores = np.asarray([to_float(r[detector_id]) for r in train], dtype=float)
        train_labels = np.asarray([int(r[label_name]) for r in train], dtype=int)
        test_scores = np.asarray([to_float(r[detector_id]) for r in test], dtype=float)
        test_labels = np.asarray([int(r[label_name]) for r in test], dtype=int)
        choice = choose_best_threshold(train_scores, train_labels)
        pred = apply_threshold(test_scores, str(choice["best_threshold_direction"]), float(choice["best_threshold"]))
        counts = confusion_counts(test_labels, pred)
        bal_acc = balanced_accuracy(counts)
        if math.isfinite(bal_acc):
            bal_accs.append(bal_acc)
        for key in total:
            total[key] += counts[key]
        eval_rows.append(
            {
                "image_id": image_id,
                "label": label_name,
                "detector_id": detector_id,
                "train_best_threshold_direction": choice["best_threshold_direction"],
                "train_best_threshold": choice["best_threshold"],
                **counts,
                "balanced_accuracy": bal_acc,
            }
        )
    summary = {
        "loo_mean_balanced_accuracy": mean_or_nan(bal_accs),
        "loo_tp": total["tp"],
        "loo_fp": total["fp"],
        "loo_tn": total["tn"],
        "loo_fn": total["fn"],
    }
    return summary, eval_rows


def detector_catalog(detector_rows: List[Dict[str, object]]) -> List[str]:
    reserved = {"image_id", "run_index", "final_psnr", "bad25", "bad20", "num_steps"}
    return [k for k in detector_rows[0] if k not in reserved]


def parse_detector_id(detector_id: str) -> Dict[str, object]:
    parts = detector_id.split("__")
    feature = "__".join(parts[:-2]) if len(parts) > 3 else parts[0]
    window = parts[-2]
    aggregate = parts[-1]
    return {
        "feature": feature,
        "window": window,
        "aggregate": aggregate,
        "is_interrun_feature": "__interrun_" in feature,
    }


def score_detectors(detector_rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    auc_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []
    loo_store: Dict[str, List[Dict[str, object]]] = {}

    for detector_id in detector_catalog(detector_rows):
        scores = np.asarray([to_float(r[detector_id]) for r in detector_rows], dtype=float)
        if np.any(~np.isfinite(scores)):
            continue
        feature_meta = parse_detector_id(detector_id)
        for label_name in ("bad25", "bad20"):
            labels = np.asarray([int(r[label_name]) for r in detector_rows], dtype=int)
            summary, curve = evaluate_thresholds(scores, labels, detector_id, label_name)
            loo_summary, loo_rows = leave_one_image_out(detector_rows, detector_id, label_name)
            auc_rows.append(
                {
                    **feature_meta,
                    **summary,
                    **loo_summary,
                    "n_runs": len(detector_rows),
                    "n_positive": int(np.sum(labels == 1)),
                    "n_negative": int(np.sum(labels == 0)),
                }
            )
            threshold_rows.extend([{**feature_meta, **row} for row in curve])
            loo_store[f"{label_name}::{detector_id}"] = loo_rows
    return auc_rows, threshold_rows, loo_store


def choose_best_detectors(auc_rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for label_name in ("bad25", "bad20"):
        candidates = [r for r in auc_rows if r["label"] == label_name]
        if not candidates:
            continue
        candidates.sort(
            key=lambda r: (
                r["loo_mean_balanced_accuracy"] if math.isfinite(to_float(r["loo_mean_balanced_accuracy"])) else -math.inf,
                r["auroc"] if math.isfinite(to_float(r["auroc"])) else -math.inf,
                r["auprc"] if math.isfinite(to_float(r["auprc"])) else -math.inf,
                r["best_balanced_accuracy"] if math.isfinite(to_float(r["best_balanced_accuracy"])) else -math.inf,
            ),
            reverse=True,
        )
        out[label_name] = candidates[0]
    return out


def collect_failure_cases(
    detector_rows: List[Dict[str, object]],
    best_detectors: Dict[str, Dict[str, object]],
    loo_store: Dict[str, List[Dict[str, object]]],
) -> List[Dict[str, object]]:
    by_key = {(str(r["image_id"]), int(r["run_index"])): r for r in detector_rows}
    rows: List[Dict[str, object]] = []
    images = sorted({str(r["image_id"]) for r in detector_rows})

    for label_name, det in best_detectors.items():
        detector_id = str(det["detector_id"])
        threshold = float(det["best_threshold"])
        direction = str(det["best_threshold_direction"])
        for run_row in detector_rows:
            score = to_float(run_row[detector_id])
            pred = int(apply_threshold(np.asarray([score]), direction, threshold)[0])
            label = int(run_row[label_name])
            if pred != label:
                rows.append(
                    {
                        "policy_type": "global_best_threshold",
                        "label": label_name,
                        "detector_id": detector_id,
                        "image_id": run_row["image_id"],
                        "run_index": run_row["run_index"],
                        "final_psnr": run_row["final_psnr"],
                        "feature_value": score,
                        "threshold_direction": direction,
                        "threshold": threshold,
                        "label_value": label,
                        "predicted_positive": pred,
                        "error_type": "false_positive" if pred == 1 else "false_negative",
                    }
                )

        loo_rows = loo_store.get(f"{label_name}::{detector_id}", [])
        for image_eval in loo_rows:
            holdout = str(image_eval["image_id"])
            holdout_runs = [r for r in detector_rows if str(r["image_id"]) == holdout]
            thr = float(image_eval["train_best_threshold"])
            dirn = str(image_eval["train_best_threshold_direction"])
            for run_row in holdout_runs:
                score = to_float(run_row[detector_id])
                pred = int(apply_threshold(np.asarray([score]), dirn, thr)[0])
                label = int(run_row[label_name])
                if pred != label:
                    rows.append(
                        {
                            "policy_type": "leave_one_image_out_threshold",
                            "label": label_name,
                            "detector_id": detector_id,
                            "image_id": run_row["image_id"],
                            "run_index": run_row["run_index"],
                            "final_psnr": run_row["final_psnr"],
                            "feature_value": score,
                            "threshold_direction": dirn,
                            "threshold": thr,
                            "label_value": label,
                            "predicted_positive": pred,
                            "error_type": "false_positive" if pred == 1 else "false_negative",
                        }
                    )
    return rows


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def official_psnr_subset(official_metrics: Dict[str, object], official_manifest: List[Dict[str, str]], target_images: Sequence[str]) -> List[Dict[str, object]]:
    psnr_table = official_metrics["psnr"]["sample"]
    by_image: Dict[str, Tuple[int, Sequence[float]]] = {}
    for i, row in enumerate(official_manifest):
        image_id = Path(row["source_path"]).stem
        by_image[image_id] = (i, psnr_table[i])
    out = []
    for image_id in target_images:
        if image_id not in by_image:
            continue
        image_i, vals = by_image[image_id]
        for run_i, psnr in enumerate(vals):
            out.append(
                {
                    "image_id": image_id,
                    "official_image_index": image_i,
                    "run_index": run_i,
                    "final_psnr": float(psnr),
                    "bad25": int(float(psnr) < 25.0),
                    "bad20": int(float(psnr) < 20.0),
                }
            )
    return out


def summarize_psnr(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_id"])].append(float(row["final_psnr"]))
    out: Dict[str, Dict[str, float]] = {}
    for image_id, vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        out[image_id] = {
            "best": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "bad25_count": int(np.sum(arr < 25.0)),
            "bad20_count": int(np.sum(arr < 20.0)),
        }
    return out


def sample_png_stats(path: Path) -> Dict[str, object]:
    with Image.open(path) as img:
        arr = np.asarray(img)
        return {
            "path": str(path),
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "dtype": str(arr.dtype),
            "pixel_min": int(arr.min()),
            "pixel_max": int(arr.max()),
        }


def render_a55_markdown(
    outdir: Path,
    a5_summary: Dict[str, Dict[str, float]],
    official_summary: Dict[str, Dict[str, float]],
    target_images: Sequence[str],
    a5_sample: Dict[str, object],
    official_sample: Dict[str, object],
    a5_config: Dict[str, object],
) -> None:
    rows = []
    for image_id in target_images:
        a5 = a5_summary.get(image_id, {})
        off = official_summary.get(image_id, {})
        rows.append(
            "| {image} | {a5_best:.3f} | {off_best:.3f} | {a5_mean:.3f} | {off_mean:.3f} | {a5_min:.3f} | {off_min:.3f} | {a5_b25} | {off_b25} |".format(
                image=image_id,
                a5_best=to_float(a5.get("best")),
                off_best=to_float(off.get("best")),
                a5_mean=to_float(a5.get("mean")),
                off_mean=to_float(off.get("mean")),
                a5_min=to_float(a5.get("min")),
                off_min=to_float(off.get("min")),
                a5_b25=int(a5.get("bad25_count", 0)),
                off_b25=int(off.get("bad25_count", 0)),
            )
        )

    text = [
        "# A5.5 Instrumentation Sanity",
        "",
        "## Scope",
        "",
        "This compares the instrumented hard-image A5 SITCOM runner against the existing official SITCOM `noise=0.05` FFHQ-25 run on the overlapping images `00005, 00013, 00027, 00028, 00034`.",
        "",
        "Exact run identity is **not** assumed comparable. The A5 pass used a 5-image subset with a different batch ordering than the official 25-image run, so run index `0..3` may not correspond to the same latent noise stream even though both use four SITCOM runs.",
        "",
        "## PSNR Summary",
        "",
        "| image_id | A5 best | official best | A5 mean | official mean | A5 min | official min | A5 bad25 | official bad25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Catastrophic Counts",
        "",
        f"- A5 total `bad25`: `{sum(int(v['bad25_count']) for v in a5_summary.values())}`",
        f"- Official total `bad25`: `{sum(int(v['bad25_count']) for v in official_summary.values())}`",
        f"- A5 total `bad20`: `{sum(int(v['bad20_count']) for v in a5_summary.values())}`",
        f"- Official total `bad20`: `{sum(int(v['bad20_count']) for v in official_summary.values())}`",
        "",
        "## Sample Format",
        "",
        f"- A5 sample: `{a5_sample['mode']}` `{a5_sample['width']}x{a5_sample['height']}` `{a5_sample['dtype']}` pixel range `{a5_sample['pixel_min']}..{a5_sample['pixel_max']}`",
        f"- Official sample: `{official_sample['mode']}` `{official_sample['width']}x{official_sample['height']}` `{official_sample['dtype']}` pixel range `{official_sample['pixel_min']}..{official_sample['pixel_max']}`",
        "",
        "## Residual Convention",
        "",
        "The instrumented A5 residuals were computed in SITCOM's own operator space, not in NP's posthoc image-selection space:",
        "",
        f"- `oversample = {a5_config.get('oversample')}`",
        "- `x0hat` and `x0y` residuals use the same centered Fourier-magnitude operator as SITCOM.",
        "- The noisy measurement is the one produced by the SITCOM operator path for the A5 run.",
        "- Residuals are computed before PNG export, so the saved-image `uint8` conversion does not affect the detector features.",
        "",
        "## Takeaway",
        "",
        "The instrumentation appears format-consistent with official SITCOM outputs, but it should be treated as a subset reproducibility pass rather than an exact run-by-run reproduction. The best/mean/min PSNR profile and catastrophic-failure counts are the meaningful sanity targets here.",
        "",
    ]
    write_text(outdir / "A5_5_instrumentation_sanity.md", "\n".join(text))


def render_a6_summary(
    outdir: Path,
    auc_rows: List[Dict[str, object]],
    best_detectors: Dict[str, Dict[str, object]],
    failure_rows: List[Dict[str, object]],
    detector_rows: List[Dict[str, object]],
) -> None:
    lines = [
        "# A6 Trajectory Risk Screen",
        "",
        "## Setup",
        "",
        f"- Image-runs analyzed: `{len(detector_rows)}`",
        f"- Images: `{','.join(sorted({str(r['image_id']) for r in detector_rows}))}`",
        f"- Labels: `bad25 = final_psnr < 25`, `bad20 = final_psnr < 20`",
        f"- Windows: `{', '.join(name for name, _ in WINDOWS)}`",
        "",
        "## Best Detectors",
        "",
    ]
    for label_name in ("bad25", "bad20"):
        det = best_detectors.get(label_name)
        if not det:
            continue
        miss_count = sum(1 for r in failure_rows if r["label"] == label_name and r["policy_type"] == "leave_one_image_out_threshold" and r["error_type"] == "false_negative")
        fp_count = sum(1 for r in failure_rows if r["label"] == label_name and r["policy_type"] == "leave_one_image_out_threshold" and r["error_type"] == "false_positive")
        lines.extend(
            [
                f"### {label_name}",
                "",
                f"- Detector: `{det['detector_id']}`",
                f"- AUROC: `{to_float(det['auroc']):.3f}`",
                f"- AUPRC: `{to_float(det['auprc']):.3f}`",
                f"- Global best balanced accuracy: `{to_float(det['best_balanced_accuracy']):.3f}` at `{det['best_threshold_direction']}` threshold `{to_float(det['best_threshold']):.6g}`",
                f"- Leave-one-image-out mean balanced accuracy: `{to_float(det['loo_mean_balanced_accuracy']):.3f}`",
                f"- Leave-one-image-out totals: `TP={int(det['loo_tp'])}, FP={int(det['loo_fp'])}, TN={int(det['loo_tn'])}, FN={int(det['loo_fn'])}`",
                f"- Leave-one-image-out failure cases: `false_negatives={miss_count}`, `false_positives={fp_count}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "Detectors are offline screens only. They use early-window trajectory features and simple inter-run comparisons, but they do not use final PSNR as an input feature.",
            "",
            "Artifacts:",
            "",
            "- `detector_auc_by_feature_window.csv`: one row per detector and label with AUROC, AUPRC, best threshold, and leave-one-image-out summary.",
            "- `threshold_policy_summary.csv`: threshold sweep counts for each detector and label.",
            "- `detector_failure_cases.csv`: per-image false positives and false negatives for the best global and leave-one-image-out policies.",
            "",
        ]
    )
    write_text(outdir / "SUMMARY.md", "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a5_dir", required=True)
    ap.add_argument("--a6_outdir", required=True)
    ap.add_argument("--a55_outdir", required=True)
    ap.add_argument(
        "--official_dir",
        default="/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/sitcom_official/sitcom_ffhq25_s4_noise005",
    )
    args = ap.parse_args()

    a5_dir = Path(args.a5_dir)
    a6_outdir = Path(args.a6_outdir)
    a55_outdir = Path(args.a55_outdir)
    official_dir = Path(args.official_dir)

    step_rows = read_csv(a5_dir / "trajectory_step_metrics.csv")
    run_level_rows = read_csv(a5_dir / "run_level_summary.csv")
    run_labels = load_run_level_labels(run_level_rows)
    add_interrun_features(step_rows)
    detector_rows = build_detector_rows(step_rows, run_labels)
    auc_rows, threshold_rows, loo_store = score_detectors(detector_rows)
    best_detectors = choose_best_detectors(auc_rows)
    failure_rows = collect_failure_cases(detector_rows, best_detectors, loo_store)

    write_csv(a6_outdir / "detector_auc_by_feature_window.csv", auc_rows)
    write_csv(a6_outdir / "threshold_policy_summary.csv", threshold_rows)
    write_csv(a6_outdir / "detector_failure_cases.csv", failure_rows)
    render_a6_summary(a6_outdir, auc_rows, best_detectors, failure_rows, detector_rows)

    official_metrics = load_json(official_dir / "results" / "sitcom_ffhq25_s4_noise005" / "metrics.json")
    official_manifest = read_csv(official_dir / "sitcom_images" / "manifest.csv")
    target_images = sorted({str(r["image_id"]) for r in detector_rows})
    official_rows = official_psnr_subset(official_metrics, official_manifest, target_images)
    a5_rows = [
        {
            "image_id": str(r["image_id"]),
            "run_index": int(r["run_index"]),
            "final_psnr": to_float(r["final_psnr"]),
            "bad25": int(r["bad25"]),
            "bad20": int(r["bad20"]),
        }
        for r in detector_rows
    ]
    a5_summary = summarize_psnr(a5_rows)
    official_summary = summarize_psnr(official_rows)

    a5_sample = sample_png_stats(next((a5_dir / "samples").glob("*.png")))
    official_sample = sample_png_stats(next((official_dir / "results" / "sitcom_ffhq25_s4_noise005" / "samples").glob("*.png")))
    a5_config = load_json(a5_dir / "config.json")
    render_a55_markdown(a55_outdir, a5_summary, official_summary, sorted(set(target_images)), a5_sample, official_sample, a5_config)


if __name__ == "__main__":
    main()
