#!/usr/bin/env python3
"""A10 constrained controller policy selection from A8/A9 outputs."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import multimode
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


WINDOW_NAME = "first50pct"
AGGREGATES = ("mean", "slope", "last_in_window")
THRESHOLD_POLICIES = ("balanced_train", "conservative_zero_fp_train")
TRAIN_LABELS = ("bad25", "bad20")
FP_BUDGETS = (0, 2, 5, 10, 15)
REPLACEMENT_BUDGETS = (10, 20, 30, 40)
RECALL_TARGETS = (0.5, 0.7, 0.8)


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
    xs = [v for v in vals if math.isfinite(v)]
    return float(np.mean(xs)) if xs else math.nan


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


def make_run_key(row: Dict[str, str]) -> Tuple[str, int]:
    return str(row["image_id"]), int(row["run_index"])


def load_np_fallbacks(np_rows: List[Dict[str, str]], noise: float, image_ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
    candidates = []
    targets = set(image_ids)
    for row in np_rows:
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
        rows = [r for r in candidates if r["image_id"] == image_id]
        if not rows:
            raise ValueError(f"No NP rows for {image_id} at noise {noise}")
        selected = min(
            rows,
            key=lambda r: (
                r["selector_post_winner_lf_mse_mean"],
                -r["psnr"],
                str(r["config_tag"]),
                int(r["seed"]),
            ),
        )
        out[image_id] = {
            "np_selected_psnr": selected["psnr"],
            "np_selected_config_tag": selected["config_tag"],
            "np_selected_seed": selected["seed"],
            "np_selected_selector_post_lf_mse": selected["selector_post_winner_lf_mse_mean"],
        }
    return out


def add_interrun_features(step_rows: List[Dict[str, str]]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["step"]))].append(row)

    bases = ("x0y_full_residual_normed", "x0y_lowfreq_residual_normed")
    for rows in grouped.values():
        for base in bases:
            vals = np.asarray([to_float(r[base]) for r in rows], dtype=float)
            if vals.size == 0 or np.any(~np.isfinite(vals)):
                continue
            median = float(np.median(vals))
            order = np.argsort(vals, kind="mergesort")
            ranks = np.empty(vals.size, dtype=float)
            ranks[order] = np.arange(1, vals.size + 1, dtype=float)
            for i, row in enumerate(rows):
                v = float(vals[i])
                row[f"{base}__interrun_rank"] = float(ranks[i])
                row[f"{base}__interrun_minus_median"] = v - median
                row[f"{base}__interrun_div_median"] = v / median if abs(median) > 1e-12 else math.nan


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
        return {"mean": math.nan, "slope": math.nan, "last_in_window": math.nan}
    return {
        "mean": float(np.mean(xs)),
        "slope": slope_or_nan(xs),
        "last_in_window": float(xs[-1]),
    }


def build_detector_table(step_rows: List[Dict[str, str]], run_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["run_index"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["step"]))

    run_meta = {make_run_key(r): r for r in run_rows}
    features = [
        "x0y_full_residual_normed__interrun_rank",
        "x0y_lowfreq_residual_normed__interrun_rank",
        "x0y_full_residual_normed__interrun_div_median",
        "x0y_lowfreq_residual_normed__interrun_div_median",
        "x0y_full_residual_normed__interrun_minus_median",
        "x0y_lowfreq_residual_normed__interrun_minus_median",
    ]

    out_rows: List[Dict[str, object]] = []
    for run_key in sorted(grouped):
        rows = grouped[run_key]
        n_steps = len(rows)
        meta = run_meta[run_key]
        final_psnr = to_float(meta["final_psnr"])
        out: Dict[str, object] = {
            "image_id": run_key[0],
            "run_index": run_key[1],
            "final_psnr": final_psnr,
            "bad25": int(final_psnr < 25.0),
            "bad20": int(final_psnr < 20.0),
            "num_steps": n_steps,
        }
        k = max(1, int(math.ceil(n_steps * 0.50)))
        window_rows = rows[:k]
        for feature in features:
            vals = [to_float(r.get(feature, math.nan)) for r in window_rows]
            for agg_name, agg_val in aggregate_window(vals).items():
                out[f"{feature}__{WINDOW_NAME}__{agg_name}"] = agg_val
        out_rows.append(out)
    return out_rows


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
    return 0.5 * (counts["tp"] / pos + counts["tn"] / neg)


def precision(counts: Dict[str, int]) -> float:
    denom = counts["tp"] + counts["fp"]
    return counts["tp"] / denom if denom else math.nan


def recall(counts: Dict[str, int]) -> float:
    denom = counts["tp"] + counts["fn"]
    return counts["tp"] / denom if denom else math.nan


def evaluate_threshold_candidates(scores: np.ndarray, y_true: np.ndarray) -> List[Dict[str, object]]:
    unique_thresholds = sorted({float(x) for x in scores if math.isfinite(float(x))})
    rows: List[Dict[str, object]] = []
    if not unique_thresholds:
        return rows
    directions = {
        "high_is_risky": lambda arr, thr: arr >= thr,
        "low_is_risky": lambda arr, thr: arr <= thr,
    }
    for direction, pred_fn in directions.items():
        for threshold in unique_thresholds:
            pred = pred_fn(scores, threshold).astype(int)
            counts = confusion_counts(y_true, pred)
            rows.append(
                {
                    "direction": direction,
                    "threshold": threshold,
                    "tp": counts["tp"],
                    "fp": counts["fp"],
                    "tn": counts["tn"],
                    "fn": counts["fn"],
                    "balanced_accuracy": balanced_accuracy(counts),
                    "precision": precision(counts),
                    "recall": recall(counts),
                    "num_flagged": int(np.sum(pred)),
                }
            )
    return rows


def select_balanced_threshold(candidate_rows: List[Dict[str, object]]) -> Dict[str, object]:
    return max(
        candidate_rows,
        key=lambda r: (
            to_float(r["balanced_accuracy"]),
            -int(r["fp"]),
            -int(r["fn"]),
            -int(r["num_flagged"]),
            -to_float(r["threshold"]) if r["direction"] == "high_is_risky" else to_float(r["threshold"]),
        ),
    )


def select_conservative_threshold(candidate_rows: List[Dict[str, object]]) -> Dict[str, object]:
    zero_fp = [r for r in candidate_rows if int(r["fp"]) == 0]
    if not zero_fp:
        return select_balanced_threshold(candidate_rows)
    return max(
        zero_fp,
        key=lambda r: (
            to_float(r["recall"]),
            to_float(r["balanced_accuracy"]),
            -int(r["num_flagged"]),
            -to_float(r["threshold"]) if r["direction"] == "high_is_risky" else to_float(r["threshold"]),
        ),
    )


def apply_threshold(scores: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    if direction == "high_is_risky":
        return (scores >= threshold).astype(int)
    if direction == "low_is_risky":
        return (scores <= threshold).astype(int)
    raise ValueError(direction)


def candidate_policy_specs() -> List[Dict[str, object]]:
    singles = [
        "x0y_full_residual_normed__interrun_rank__first50pct__mean",
        "x0y_full_residual_normed__interrun_rank__first50pct__slope",
        "x0y_full_residual_normed__interrun_rank__first50pct__last_in_window",
        "x0y_lowfreq_residual_normed__interrun_rank__first50pct__mean",
        "x0y_lowfreq_residual_normed__interrun_rank__first50pct__slope",
        "x0y_lowfreq_residual_normed__interrun_rank__first50pct__last_in_window",
        "x0y_full_residual_normed__interrun_div_median__first50pct__mean",
        "x0y_full_residual_normed__interrun_div_median__first50pct__slope",
        "x0y_full_residual_normed__interrun_div_median__first50pct__last_in_window",
        "x0y_lowfreq_residual_normed__interrun_div_median__first50pct__mean",
        "x0y_lowfreq_residual_normed__interrun_div_median__first50pct__slope",
        "x0y_lowfreq_residual_normed__interrun_div_median__first50pct__last_in_window",
        "x0y_full_residual_normed__interrun_minus_median__first50pct__mean",
        "x0y_full_residual_normed__interrun_minus_median__first50pct__slope",
        "x0y_full_residual_normed__interrun_minus_median__first50pct__last_in_window",
        "x0y_lowfreq_residual_normed__interrun_minus_median__first50pct__mean",
        "x0y_lowfreq_residual_normed__interrun_minus_median__first50pct__slope",
        "x0y_lowfreq_residual_normed__interrun_minus_median__first50pct__last_in_window",
    ]
    out: List[Dict[str, object]] = []
    for det in singles:
        out.append(
            {
                "policy_name": det,
                "policy_family": "single_detector",
                "combine_mode": "single",
                "components": [det],
                "description": det,
            }
        )

    combos = [
        (
            "full_rank_slope_or_div_last",
            "or",
            [
                "x0y_full_residual_normed__interrun_rank__first50pct__slope",
                "x0y_full_residual_normed__interrun_div_median__first50pct__last_in_window",
            ],
        ),
        (
            "full_rank_slope_and_div_last",
            "and",
            [
                "x0y_full_residual_normed__interrun_rank__first50pct__slope",
                "x0y_full_residual_normed__interrun_div_median__first50pct__last_in_window",
            ],
        ),
        (
            "full_rank_slope_or_rank_last",
            "or",
            [
                "x0y_full_residual_normed__interrun_rank__first50pct__slope",
                "x0y_full_residual_normed__interrun_rank__first50pct__last_in_window",
            ],
        ),
        (
            "full_rank_slope_and_rank_last",
            "and",
            [
                "x0y_full_residual_normed__interrun_rank__first50pct__slope",
                "x0y_full_residual_normed__interrun_rank__first50pct__last_in_window",
            ],
        ),
        (
            "low_rank_slope_or_div_last",
            "or",
            [
                "x0y_lowfreq_residual_normed__interrun_rank__first50pct__slope",
                "x0y_lowfreq_residual_normed__interrun_div_median__first50pct__last_in_window",
            ],
        ),
        (
            "low_rank_slope_and_div_last",
            "and",
            [
                "x0y_lowfreq_residual_normed__interrun_rank__first50pct__slope",
                "x0y_lowfreq_residual_normed__interrun_div_median__first50pct__last_in_window",
            ],
        ),
        (
            "low_rank_slope_or_rank_last",
            "or",
            [
                "x0y_lowfreq_residual_normed__interrun_rank__first50pct__slope",
                "x0y_lowfreq_residual_normed__interrun_rank__first50pct__last_in_window",
            ],
        ),
        (
            "low_rank_slope_and_rank_last",
            "and",
            [
                "x0y_lowfreq_residual_normed__interrun_rank__first50pct__slope",
                "x0y_lowfreq_residual_normed__interrun_rank__first50pct__last_in_window",
            ],
        ),
        (
            "full_rank_slope_or_minus_last",
            "or",
            [
                "x0y_full_residual_normed__interrun_rank__first50pct__slope",
                "x0y_full_residual_normed__interrun_minus_median__first50pct__last_in_window",
            ],
        ),
        (
            "full_rank_slope_and_minus_last",
            "and",
            [
                "x0y_full_residual_normed__interrun_rank__first50pct__slope",
                "x0y_full_residual_normed__interrun_minus_median__first50pct__last_in_window",
            ],
        ),
    ]
    for name, mode, components in combos:
        out.append(
            {
                "policy_name": name,
                "policy_family": "combo_detector",
                "combine_mode": mode,
                "components": components,
                "description": f"{components[0]} {mode.upper()} {components[1]}",
            }
        )
    return out


def select_threshold_for_policy(
    train_rows: List[Dict[str, object]],
    detector_id: str,
    train_label: str,
    threshold_policy: str,
) -> Dict[str, object]:
    scores = np.asarray([to_float(r[detector_id]) for r in train_rows], dtype=float)
    labels = np.asarray([int(r[train_label]) for r in train_rows], dtype=int)
    candidates = evaluate_threshold_candidates(scores, labels)
    if not candidates:
        raise ValueError(f"No threshold candidates for {detector_id}")
    if threshold_policy == "balanced_train":
        return select_balanced_threshold(candidates)
    if threshold_policy == "conservative_zero_fp_train":
        return select_conservative_threshold(candidates)
    raise ValueError(threshold_policy)


def combine_preds(mode: str, preds: List[np.ndarray]) -> np.ndarray:
    if len(preds) == 1:
        return preds[0]
    if mode == "and":
        out = preds[0].copy()
        for pred in preds[1:]:
            out = np.logical_and(out == 1, pred == 1).astype(int)
        return out
    if mode == "or":
        out = preds[0].copy()
        for pred in preds[1:]:
            out = np.logical_or(out == 1, pred == 1).astype(int)
        return out
    raise ValueError(mode)


def evaluate_policy_cv(
    detector_rows: List[Dict[str, object]],
    policy_spec: Dict[str, object],
    train_label: str,
    threshold_policy: str,
) -> Dict[str, object]:
    image_ids = sorted({str(r["image_id"]) for r in detector_rows})
    per_run_pred: Dict[Tuple[str, int], Dict[str, object]] = {}
    fold_rows: List[Dict[str, object]] = []
    components = list(policy_spec["components"])
    for heldout_image in image_ids:
        train_rows = [r for r in detector_rows if str(r["image_id"]) != heldout_image]
        test_rows = [r for r in detector_rows if str(r["image_id"]) == heldout_image]
        if not train_rows or not test_rows:
            continue

        component_thresholds = []
        component_preds = []
        for det in components:
            choice = select_threshold_for_policy(train_rows, det, train_label, threshold_policy)
            component_thresholds.append({"detector_id": det, **choice})
            test_scores = np.asarray([to_float(r[det]) for r in test_rows], dtype=float)
            component_preds.append(apply_threshold(test_scores, str(choice["direction"]), float(choice["threshold"])))

        pred = combine_preds(str(policy_spec["combine_mode"]), component_preds)
        y_true = np.asarray([int(r[train_label]) for r in test_rows], dtype=int)
        counts = confusion_counts(y_true, pred)

        fold_rows.append(
            {
                "heldout_image": heldout_image,
                "test_tp": counts["tp"],
                "test_fp": counts["fp"],
                "test_tn": counts["tn"],
                "test_fn": counts["fn"],
                "test_balanced_accuracy": balanced_accuracy(counts),
                "test_recall": recall(counts),
                "test_num_flagged": int(np.sum(pred)),
                "num_test_runs": len(test_rows),
                "component_thresholds": component_thresholds,
            }
        )

        for i, row in enumerate(test_rows):
            key = (str(row["image_id"]), int(row["run_index"]))
            per_run_pred[key] = {
                "pred": int(pred[i]),
                "component_thresholds": component_thresholds,
                "component_values": {det: to_float(row[det]) for det in components},
            }

    global_counts = {
        "tp": sum(int(r["test_tp"]) for r in fold_rows),
        "fp": sum(int(r["test_fp"]) for r in fold_rows),
        "tn": sum(int(r["test_tn"]) for r in fold_rows),
        "fn": sum(int(r["test_fn"]) for r in fold_rows),
    }
    return {
        "fold_rows": fold_rows,
        "per_run_pred": per_run_pred,
        "global_tp": global_counts["tp"],
        "global_fp": global_counts["fp"],
        "global_tn": global_counts["tn"],
        "global_fn": global_counts["fn"],
        "global_balanced_accuracy": balanced_accuracy(global_counts),
        "global_precision": precision(global_counts),
        "global_recall": recall(global_counts),
        "mean_fold_balanced_accuracy": mean_or_nan(to_float(r["test_balanced_accuracy"]) for r in fold_rows),
        "mean_fold_recall": mean_or_nan(to_float(r["test_recall"]) for r in fold_rows),
        "mean_test_num_flagged": mean_or_nan(to_float(r["test_num_flagged"]) for r in fold_rows),
        "mean_component_thresholds": {
            det: mean_or_nan(
                to_float(comp["threshold"])
                for fold in fold_rows
                for comp in fold["component_thresholds"]
                if comp["detector_id"] == det
            )
            for det in components
        },
        "direction_modes": {
            det: multimode(
                [
                    str(comp["direction"])
                    for fold in fold_rows
                    for comp in fold["component_thresholds"]
                    if comp["detector_id"] == det
                ]
            )[0]
            for det in components
        },
    }


def build_run_records(
    policy_spec: Dict[str, object],
    train_label: str,
    threshold_policy: str,
    detector_rows: List[Dict[str, object]],
    cv_result: Dict[str, object],
    np_fallbacks: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    run_records: List[Dict[str, object]] = []
    replaced_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    image_rows: List[Dict[str, object]] = []

    per_run_pred = cv_result["per_run_pred"]
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in detector_rows:
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        key = (image_id, run_index)
        pred_meta = per_run_pred[key]
        flagged = bool(pred_meta["pred"])
        sitcom_psnr = to_float(row["final_psnr"])
        fb = np_fallbacks[image_id]
        policy_psnr = float(fb["np_selected_psnr"]) if flagged else sitcom_psnr
        delta = policy_psnr - sitcom_psnr
        replaced = flagged
        record = {
            "policy_name": str(policy_spec["policy_name"]),
            "policy_family": str(policy_spec["policy_family"]),
            "combine_mode": str(policy_spec["combine_mode"]),
            "description": str(policy_spec["description"]),
            "train_label": train_label,
            "threshold_policy": threshold_policy,
            "image_id": image_id,
            "run_index": run_index,
            "sitcom_final_psnr": sitcom_psnr,
            "policy_final_psnr": policy_psnr,
            "delta_vs_sitcom": delta,
            "sitcom_bad25": sitcom_psnr < 25.0,
            "sitcom_bad20": sitcom_psnr < 20.0,
            "policy_bad25": policy_psnr < 25.0,
            "policy_bad20": policy_psnr < 20.0,
            "was_replaced": replaced,
            "replacement_source": "np_selected" if replaced else "sitcom",
            "false_positive_replacement": replaced and sitcom_psnr >= 25.0,
            "true_positive_replacement": replaced and sitcom_psnr < 25.0,
            "false_negative_remaining_bad25": (not replaced) and sitcom_psnr < 25.0,
            "false_negative_remaining_bad20": (not replaced) and sitcom_psnr < 20.0,
            "component_values": pred_meta["component_values"],
        }
        run_records.append(record)
        by_image[image_id].append(record)

        if replaced:
            replaced_rows.append(
                {
                    "policy_name": str(policy_spec["policy_name"]),
                    "policy_family": str(policy_spec["policy_family"]),
                    "combine_mode": str(policy_spec["combine_mode"]),
                    "train_label": train_label,
                    "threshold_policy": threshold_policy,
                    "image_id": image_id,
                    "run_index": run_index,
                    "sitcom_final_psnr": sitcom_psnr,
                    "replacement_psnr": policy_psnr,
                    "delta_vs_sitcom": delta,
                    "replacement_source": "np_selected",
                    "replacement_detail": f"{fb['np_selected_config_tag']} seed={fb['np_selected_seed']}",
                    "false_positive_replacement": sitcom_psnr >= 25.0,
                    "component_values": pred_meta["component_values"],
                    "component_thresholds": pred_meta["component_thresholds"],
                }
            )
        if replaced and sitcom_psnr >= 25.0:
            failure_rows.append(
                {
                    "policy_name": str(policy_spec["policy_name"]),
                    "train_label": train_label,
                    "threshold_policy": threshold_policy,
                    "image_id": image_id,
                    "run_index": run_index,
                    "case_type": "false_positive_replacement",
                    "sitcom_final_psnr": sitcom_psnr,
                    "policy_final_psnr": policy_psnr,
                    "delta_vs_sitcom": delta,
                    "component_values": pred_meta["component_values"],
                }
            )
        if (not replaced) and sitcom_psnr < 25.0:
            failure_rows.append(
                {
                    "policy_name": str(policy_spec["policy_name"]),
                    "train_label": train_label,
                    "threshold_policy": threshold_policy,
                    "image_id": image_id,
                    "run_index": run_index,
                    "case_type": "remaining_bad25_miss",
                    "sitcom_final_psnr": sitcom_psnr,
                    "policy_final_psnr": policy_psnr,
                    "delta_vs_sitcom": delta,
                    "component_values": pred_meta["component_values"],
                }
            )

    for image_id, records in sorted(by_image.items()):
        finals = [to_float(r["policy_final_psnr"]) for r in records]
        deltas = [to_float(r["delta_vs_sitcom"]) for r in records]
        image_rows.append(
            {
                "policy_name": str(policy_spec["policy_name"]),
                "train_label": train_label,
                "threshold_policy": threshold_policy,
                "image_id": image_id,
                "num_runs": len(records),
                "num_replacements": sum(int(bool(r["was_replaced"])) for r in records),
                "any_replacement": any(bool(r["was_replaced"]) for r in records),
                "avg_policy_psnr": mean_or_nan(finals),
                "best_of4_policy_psnr": max(finals) if finals else math.nan,
                "min_policy_psnr": min(finals) if finals else math.nan,
                "num_bad25_remaining": sum(int(bool(r["policy_bad25"])) for r in records),
                "num_bad20_remaining": sum(int(bool(r["policy_bad20"])) for r in records),
                "worst_delta_on_image": min(deltas) if deltas else math.nan,
            }
        )

    return run_records, replaced_rows, failure_rows, image_rows


def summarize_policy(
    policy_spec: Dict[str, object],
    train_label: str,
    threshold_policy: str,
    cv_result: Dict[str, object],
    run_records: List[Dict[str, object]],
    image_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    finals = [to_float(r["policy_final_psnr"]) for r in run_records]
    false_positive_deltas = [
        to_float(r["delta_vs_sitcom"])
        for r in run_records
        if bool(r["false_positive_replacement"])
    ]
    remaining_bad = [
        to_float(r["sitcom_final_psnr"])
        for r in run_records
        if bool(r["false_negative_remaining_bad25"])
    ]
    return {
        "policy_name": str(policy_spec["policy_name"]),
        "policy_family": str(policy_spec["policy_family"]),
        "combine_mode": str(policy_spec["combine_mode"]),
        "description": str(policy_spec["description"]),
        "train_label": train_label,
        "threshold_policy": threshold_policy,
        "components": " | ".join(str(c) for c in policy_spec["components"]),
        "component_mean_thresholds": " | ".join(
            f"{det}:{to_float(cv_result['mean_component_thresholds'][det]):.6g}"
            for det in policy_spec["components"]
        ),
        "component_direction_modes": " | ".join(
            f"{det}:{cv_result['direction_modes'][det]}"
            for det in policy_spec["components"]
        ),
        "run_mean_psnr": mean_or_nan(finals),
        "run_min_psnr": min(finals) if finals else math.nan,
        "image_best_of4_mean_psnr": mean_or_nan(to_float(r["best_of4_policy_psnr"]) for r in image_rows),
        "image_best_of4_min_psnr": min((to_float(r["best_of4_policy_psnr"]) for r in image_rows), default=math.nan),
        "bad25_count": sum(int(bool(r["policy_bad25"])) for r in run_records),
        "bad20_count": sum(int(bool(r["policy_bad20"])) for r in run_records),
        "true_positive_replacements": sum(int(bool(r["true_positive_replacement"])) for r in run_records),
        "false_positive_replacements": sum(int(bool(r["false_positive_replacement"])) for r in run_records),
        "false_negative_remaining_bad_runs": sum(int(bool(r["false_negative_remaining_bad25"])) for r in run_records),
        "num_images_with_any_replacement": sum(int(bool(r["any_replacement"])) for r in image_rows),
        "avg_replacements_per_image": mean_or_nan(to_float(r["num_replacements"]) for r in image_rows),
        "total_replacements": sum(int(bool(r["was_replaced"])) for r in run_records),
        "worst_false_positive_psnr_loss": min(false_positive_deltas) if false_positive_deltas else math.nan,
        "worst_remaining_miss": min(remaining_bad) if remaining_bad else math.nan,
        "cv_global_balanced_accuracy": cv_result["global_balanced_accuracy"],
        "cv_global_precision": cv_result["global_precision"],
        "cv_global_recall": cv_result["global_recall"],
        "cv_global_tp": cv_result["global_tp"],
        "cv_global_fp": cv_result["global_fp"],
        "cv_global_tn": cv_result["global_tn"],
        "cv_global_fn": cv_result["global_fn"],
        "cv_mean_fold_balanced_accuracy": cv_result["mean_fold_balanced_accuracy"],
        "cv_mean_fold_recall": cv_result["mean_fold_recall"],
        "cv_mean_test_num_flagged": cv_result["mean_test_num_flagged"],
    }


def baseline_policy_summary(
    name: str,
    family: str,
    combine_mode: str,
    description: str,
    run_rows: List[Dict[str, str]],
    np_fallbacks: Dict[str, Dict[str, object]],
    mode: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    run_records: List[Dict[str, object]] = []
    replaced_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        sitcom_psnr = to_float(row["final_psnr"])
        fb = np_fallbacks[image_id]
        replace = False
        if mode == "sitcom_only":
            replace = False
        elif mode == "oracle_bad25":
            replace = sitcom_psnr < 25.0
        elif mode == "replace_all":
            replace = True
        else:
            raise ValueError(mode)
        policy_psnr = float(fb["np_selected_psnr"]) if replace else sitcom_psnr
        delta = policy_psnr - sitcom_psnr
        rec = {
            "policy_name": name,
            "policy_family": family,
            "combine_mode": combine_mode,
            "description": description,
            "train_label": "diagnostic",
            "threshold_policy": "diagnostic",
            "image_id": image_id,
            "run_index": run_index,
            "sitcom_final_psnr": sitcom_psnr,
            "policy_final_psnr": policy_psnr,
            "delta_vs_sitcom": delta,
            "sitcom_bad25": sitcom_psnr < 25.0,
            "sitcom_bad20": sitcom_psnr < 20.0,
            "policy_bad25": policy_psnr < 25.0,
            "policy_bad20": policy_psnr < 20.0,
            "was_replaced": replace,
            "replacement_source": "np_selected" if replace else "sitcom",
            "false_positive_replacement": replace and sitcom_psnr >= 25.0,
            "true_positive_replacement": replace and sitcom_psnr < 25.0,
            "false_negative_remaining_bad25": (not replace) and sitcom_psnr < 25.0,
        }
        run_records.append(rec)
        by_image[image_id].append(rec)
        if replace:
            replaced_rows.append(
                {
                    "policy_name": name,
                    "policy_family": family,
                    "combine_mode": combine_mode,
                    "train_label": "diagnostic",
                    "threshold_policy": "diagnostic",
                    "image_id": image_id,
                    "run_index": run_index,
                    "sitcom_final_psnr": sitcom_psnr,
                    "replacement_psnr": policy_psnr,
                    "delta_vs_sitcom": delta,
                    "replacement_source": "np_selected",
                    "replacement_detail": f"{fb['np_selected_config_tag']} seed={fb['np_selected_seed']}",
                    "false_positive_replacement": sitcom_psnr >= 25.0,
                }
            )
        if replace and sitcom_psnr >= 25.0:
            failure_rows.append(
                {
                    "policy_name": name,
                    "train_label": "diagnostic",
                    "threshold_policy": "diagnostic",
                    "image_id": image_id,
                    "run_index": run_index,
                    "case_type": "false_positive_replacement",
                    "sitcom_final_psnr": sitcom_psnr,
                    "policy_final_psnr": policy_psnr,
                    "delta_vs_sitcom": delta,
                }
            )
        if (not replace) and sitcom_psnr < 25.0:
            failure_rows.append(
                {
                    "policy_name": name,
                    "train_label": "diagnostic",
                    "threshold_policy": "diagnostic",
                    "image_id": image_id,
                    "run_index": run_index,
                    "case_type": "remaining_bad25_miss",
                    "sitcom_final_psnr": sitcom_psnr,
                    "policy_final_psnr": policy_psnr,
                    "delta_vs_sitcom": delta,
                }
            )

    image_rows: List[Dict[str, object]] = []
    for image_id, records in sorted(by_image.items()):
        finals = [to_float(r["policy_final_psnr"]) for r in records]
        deltas = [to_float(r["delta_vs_sitcom"]) for r in records]
        image_rows.append(
            {
                "policy_name": name,
                "train_label": "diagnostic",
                "threshold_policy": "diagnostic",
                "image_id": image_id,
                "num_runs": len(records),
                "num_replacements": sum(int(bool(r["was_replaced"])) for r in records),
                "any_replacement": any(bool(r["was_replaced"]) for r in records),
                "avg_policy_psnr": mean_or_nan(finals),
                "best_of4_policy_psnr": max(finals) if finals else math.nan,
                "min_policy_psnr": min(finals) if finals else math.nan,
                "num_bad25_remaining": sum(int(bool(r["policy_bad25"])) for r in records),
                "num_bad20_remaining": sum(int(bool(r["policy_bad20"])) for r in records),
                "worst_delta_on_image": min(deltas) if deltas else math.nan,
            }
        )

    summary = {
        "policy_name": name,
        "policy_family": family,
        "combine_mode": combine_mode,
        "description": description,
        "train_label": "diagnostic",
        "threshold_policy": "diagnostic",
        "components": "",
        "component_mean_thresholds": "",
        "component_direction_modes": "",
        "run_mean_psnr": mean_or_nan(to_float(r["policy_final_psnr"]) for r in run_records),
        "run_min_psnr": min((to_float(r["policy_final_psnr"]) for r in run_records), default=math.nan),
        "image_best_of4_mean_psnr": mean_or_nan(to_float(r["best_of4_policy_psnr"]) for r in image_rows),
        "image_best_of4_min_psnr": min((to_float(r["best_of4_policy_psnr"]) for r in image_rows), default=math.nan),
        "bad25_count": sum(int(bool(r["policy_bad25"])) for r in run_records),
        "bad20_count": sum(int(bool(r["policy_bad20"])) for r in run_records),
        "true_positive_replacements": sum(int(bool(r["true_positive_replacement"])) for r in run_records),
        "false_positive_replacements": sum(int(bool(r["false_positive_replacement"])) for r in run_records),
        "false_negative_remaining_bad_runs": sum(int(bool(r["false_negative_remaining_bad25"])) for r in run_records),
        "num_images_with_any_replacement": sum(int(bool(r["any_replacement"])) for r in image_rows),
        "avg_replacements_per_image": mean_or_nan(to_float(r["num_replacements"]) for r in image_rows),
        "total_replacements": sum(int(bool(r["was_replaced"])) for r in run_records),
        "worst_false_positive_psnr_loss": min((to_float(r["delta_vs_sitcom"]) for r in run_records if bool(r["false_positive_replacement"])), default=math.nan),
        "worst_remaining_miss": min((to_float(r["sitcom_final_psnr"]) for r in run_records if bool(r["false_negative_remaining_bad25"])), default=math.nan),
        "cv_global_balanced_accuracy": math.nan,
        "cv_global_precision": math.nan,
        "cv_global_recall": math.nan,
        "cv_global_tp": math.nan,
        "cv_global_fp": math.nan,
        "cv_global_tn": math.nan,
        "cv_global_fn": math.nan,
        "cv_mean_fold_balanced_accuracy": math.nan,
        "cv_mean_fold_recall": math.nan,
        "cv_mean_test_num_flagged": math.nan,
    }
    return summary, run_records, replaced_rows, failure_rows, image_rows


def mark_constraint_flags(summary_rows: List[Dict[str, object]]) -> None:
    for row in summary_rows:
        fp = int(to_float(row["false_positive_replacements"])) if math.isfinite(to_float(row["false_positive_replacements"])) else 0
        total = int(to_float(row["total_replacements"])) if math.isfinite(to_float(row["total_replacements"])) else 0
        rec = to_float(row["cv_global_recall"])
        for budget in FP_BUDGETS:
            row[f"meets_fp_budget_le_{budget}"] = fp <= budget
        for budget in REPLACEMENT_BUDGETS:
            row[f"meets_total_replacements_le_{budget}"] = total <= budget
        for target in RECALL_TARGETS:
            tag = str(target).replace(".", "p")
            row[f"meets_bad25_recall_ge_{tag}"] = math.isfinite(rec) and rec >= target


def constraint_key(fp_budget: int, replacement_budget: int, recall_target: float) -> str:
    return f"fp<={fp_budget}|repl<={replacement_budget}|recall>={recall_target:.1f}"


def select_constraint_winners(summary_rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    winners: Dict[str, Dict[str, object]] = {}
    candidates = [
        r for r in summary_rows
        if r["policy_family"] not in {"diagnostic_oracle", "diagnostic_replace_all", "diagnostic_baseline"}
        and str(r["train_label"]) == "bad25"
    ]
    for fp_budget in FP_BUDGETS:
        for replacement_budget in REPLACEMENT_BUDGETS:
            for recall_target in RECALL_TARGETS:
                feasible = [
                    r for r in candidates
                    if int(to_float(r["false_positive_replacements"])) <= fp_budget
                    and int(to_float(r["total_replacements"])) <= replacement_budget
                    and math.isfinite(to_float(r["cv_global_recall"]))
                    and to_float(r["cv_global_recall"]) >= recall_target
                ]
                if not feasible:
                    continue
                best = max(
                    feasible,
                    key=lambda r: (
                        -int(to_float(r["bad25_count"])),
                        to_float(r["run_mean_psnr"]),
                        to_float(r["run_min_psnr"]),
                        -int(to_float(r["false_positive_replacements"])),
                        -int(to_float(r["total_replacements"])),
                    ),
                )
                winners[constraint_key(fp_budget, replacement_budget, recall_target)] = best
    return winners


def build_summary_md(summary_rows: List[Dict[str, object]], winners: Dict[str, Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# A10 Constrained Controller Policy")
    lines.append("")
    lines.append("This analysis uses only the existing A8 trajectories and A9 CV setting. No new SITCOM jobs were run.")
    lines.append("")
    lines.append("## Diagnostic baselines")
    lines.append("")
    for name in ("sitcom_only", "oracle_risk_np_selected", "replace_all_np_selected"):
        row = next((r for r in summary_rows if r["policy_name"] == name), None)
        if row is None:
            continue
        lines.append(
            f"- `{name}`: run mean/min `{to_float(row['run_mean_psnr']):.3f}` / "
            f"`{to_float(row['run_min_psnr']):.3f}`, bad25/bad20 "
            f"`{int(to_float(row['bad25_count']))}` / `{int(to_float(row['bad20_count']))}`."
        )
    lines.append("")
    lines.append("## Practical held-out policies")
    lines.append("")
    if not winners:
        lines.append("- No non-diagnostic held-out policy satisfied the requested budgets and recall targets simultaneously.")
    else:
        shown = 0
        for key, row in winners.items():
            lines.append(
                f"- `{key}`: `{row['policy_name']}` "
                f"({row['threshold_policy']}, {row['description']}) with run mean/min "
                f"`{to_float(row['run_mean_psnr']):.3f}` / `{to_float(row['run_min_psnr']):.3f}`, "
                f"bad25/bad20 `{int(to_float(row['bad25_count']))}` / `{int(to_float(row['bad20_count']))}`, "
                f"TP/FP replacements `{int(to_float(row['true_positive_replacements']))}` / "
                f"`{int(to_float(row['false_positive_replacements']))}`."
            )
            shown += 1
            if shown >= 8:
                break
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The main question is whether a held-out CV controller exists with a reasonable FP and replacement budget.")
    lines.append("- `oracle_risk_np_selected` remains a diagnostic upper bound only.")
    lines.append("- `replace_all_np_selected` is intentionally included as a degenerate not-acceptable reference.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--a8_dir", type=Path, required=True)
    p.add_argument("--a9_dir", type=Path, required=True)
    p.add_argument("--np_run_level", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--noise", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a8_dir = args.a8_dir
    a9_dir = args.a9_dir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    step_rows = read_csv(a8_dir / "trajectory_step_metrics.csv")
    run_rows = read_csv(a8_dir / "run_level_summary.csv")
    np_rows = read_csv(args.np_run_level)
    a9_summary_rows = read_csv(a9_dir / "detector_cv_summary.csv")

    add_interrun_features(step_rows)
    detector_rows = build_detector_table(step_rows, run_rows)
    image_ids = sorted({str(r["image_id"]) for r in detector_rows})
    np_fallbacks = load_np_fallbacks(np_rows, args.noise, image_ids)

    summary_rows: List[Dict[str, object]] = []
    image_rows_all: List[Dict[str, object]] = []
    replaced_rows_all: List[Dict[str, object]] = []
    failure_rows_all: List[Dict[str, object]] = []

    # Diagnostic baselines.
    for name, family, description, mode in (
        ("sitcom_only", "diagnostic_baseline", "No fallback; raw SITCOM final outputs.", "sitcom_only"),
        ("oracle_risk_np_selected", "diagnostic_oracle", "Diagnostic upper bound: replace exactly bad25 SITCOM runs with NP selected.", "oracle_bad25"),
        ("replace_all_np_selected", "diagnostic_replace_all", "Degenerate replace-all policy; not operationally acceptable.", "replace_all"),
    ):
        summary, _, replaced, failures, image_rows = baseline_policy_summary(
            name,
            family,
            "baseline",
            description,
            run_rows,
            np_fallbacks,
            mode,
        )
        summary_rows.append(summary)
        replaced_rows_all.extend(replaced)
        failure_rows_all.extend(failures)
        image_rows_all.extend(image_rows)

    specs = candidate_policy_specs()
    for spec in specs:
        for train_label in TRAIN_LABELS:
            for threshold_policy in THRESHOLD_POLICIES:
                cv_result = evaluate_policy_cv(detector_rows, spec, train_label, threshold_policy)
                run_records, replaced_rows, failure_rows, image_rows = build_run_records(
                    spec,
                    train_label,
                    threshold_policy,
                    detector_rows,
                    cv_result,
                    np_fallbacks,
                )
                summary = summarize_policy(spec, train_label, threshold_policy, cv_result, run_records, image_rows)
                summary_rows.append(summary)
                replaced_rows_all.extend(replaced_rows)
                failure_rows_all.extend(failure_rows)
                image_rows_all.extend(image_rows)

    # Mark A9 diagnostic references on matching policies.
    def top_a9(train_label: str, threshold_policy: str) -> str:
        rows = [r for r in a9_summary_rows if r["label"] == train_label and r["threshold_policy"] == threshold_policy]
        rows.sort(key=lambda r: (to_float(r["global_balanced_accuracy"]), to_float(r["global_recall"])), reverse=True)
        return rows[0]["detector_id"] if rows else ""

    a9_best_balanced = top_a9("bad25", "balanced_train")
    a9_best_conservative = top_a9("bad25", "conservative_zero_fp_train")
    for row in summary_rows:
        row["is_a9_best_balanced_bad25_reference"] = (
            str(row["policy_name"]) == a9_best_balanced
            and str(row["train_label"]) == "bad25"
            and str(row["threshold_policy"]) == "balanced_train"
        )
        row["is_a9_best_conservative_bad25_reference"] = (
            str(row["policy_name"]) == a9_best_conservative
            and str(row["train_label"]) == "bad25"
            and str(row["threshold_policy"]) == "conservative_zero_fp_train"
        )
        row["acceptable_operational_policy"] = row["policy_family"] not in {
            "diagnostic_oracle",
            "diagnostic_replace_all",
        }

    mark_constraint_flags(summary_rows)
    winners = select_constraint_winners(summary_rows)
    selected_map: Dict[str, List[str]] = defaultdict(list)
    for key, row in winners.items():
        selected_map[str(row["policy_name"]) + "|" + str(row["train_label"]) + "|" + str(row["threshold_policy"])].append(key)
    for row in summary_rows:
        marker = str(row["policy_name"]) + "|" + str(row["train_label"]) + "|" + str(row["threshold_policy"])
        row["selected_constraint_keys"] = " ; ".join(selected_map.get(marker, []))
        row["selected_for_any_constraint"] = marker in selected_map

    summary_rows.sort(
        key=lambda r: (
            str(r["policy_family"]),
            str(r["train_label"]),
            str(r["threshold_policy"]),
            -to_float(r["run_mean_psnr"]),
            str(r["policy_name"]),
        )
    )
    image_rows_all.sort(key=lambda r: (str(r["policy_name"]), str(r["train_label"]), str(r["threshold_policy"]), str(r["image_id"])))
    replaced_rows_all.sort(key=lambda r: (str(r["policy_name"]), str(r["train_label"]), str(r["threshold_policy"]), str(r["image_id"]), int(r["run_index"])))
    failure_rows_all.sort(key=lambda r: (str(r["policy_name"]), str(r["train_label"]), str(r["threshold_policy"]), str(r["case_type"]), str(r["image_id"]), int(r["run_index"])))

    write_csv(outdir / "constrained_policy_summary.csv", summary_rows)
    write_csv(outdir / "constrained_policy_image_level.csv", image_rows_all)
    write_csv(outdir / "constrained_policy_replaced_runs.csv", replaced_rows_all)
    write_csv(outdir / "constrained_policy_failure_cases.csv", failure_rows_all)
    write_text(outdir / "SUMMARY.md", build_summary_md(summary_rows, winners))


if __name__ == "__main__":
    main()
