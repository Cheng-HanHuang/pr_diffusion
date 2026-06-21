#!/usr/bin/env python3
"""A9 cross-validated relative-feature detector design from A8 trajectories."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import multimode
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


WINDOWS: List[Tuple[str, float]] = [
    ("first10pct", 0.10),
    ("first20pct", 0.20),
    ("first30pct", 0.30),
    ("first50pct", 0.50),
]

RELATIVE_BASES: List[str] = [
    "x0y_full_residual_normed",
    "x0y_lowfreq_residual_normed",
    "correction_norm",
    "x0hat_x0y_disagreement",
]

RELATIVE_SUFFIXES: List[str] = [
    "__interrun_rank",
    "__interrun_minus_median",
    "__interrun_div_median",
]

AGGREGATES: List[str] = ["max", "mean", "last_in_window", "slope"]

FROZEN_A7_DETECTOR_ID = "correction_norm__first10pct__max"
FROZEN_A7_DIRECTION = "low_is_risky"
FROZEN_A7_THRESHOLD = 0.315417


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

    for rows in grouped.values():
        for base in RELATIVE_BASES:
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
                if abs(median) > 1e-12:
                    row[f"{base}__interrun_div_median"] = v / median
                else:
                    row[f"{base}__interrun_div_median"] = math.nan


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
        "last_in_window": float(xs[-1]),
        "slope": slope_or_nan(xs),
    }


def build_detector_table(step_rows: List[Dict[str, str]], run_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["run_index"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["step"]))

    run_meta = {make_run_key(r): r for r in run_rows}
    feature_names = [f"{base}{suffix}" for base in RELATIVE_BASES for suffix in RELATIVE_SUFFIXES]

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
        for window_name, frac in WINDOWS:
            k = max(1, int(math.ceil(n_steps * frac)))
            window_rows = rows[:k]
            for feature_name in feature_names:
                vals = [to_float(r.get(feature_name, math.nan)) for r in window_rows]
                for agg_name, agg_val in aggregate_window(vals).items():
                    out[f"{feature_name}__{window_name}__{agg_name}"] = agg_val
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


def summarize_image_level_policy(rows: List[Dict[str, object]]) -> Tuple[float, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_id"])].append(to_float(row["policy_final_psnr"]))
    bests = [max(vals) for vals in grouped.values() if vals]
    return mean_or_nan(bests), (min(bests) if bests else math.nan)


def simulate_policy(
    policy_name: str,
    policy_kind: str,
    rows: List[Dict[str, object]],
    flags: Dict[Tuple[str, int], bool],
    np_fallbacks: Dict[str, Dict[str, object]],
    notes: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    run_records: List[Dict[str, object]] = []
    replaced_rows: List[Dict[str, object]] = []
    for row in rows:
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        sitcom_psnr = to_float(row["final_psnr"])
        flagged = bool(flags.get((image_id, run_index), False))
        policy_psnr = sitcom_psnr
        source = "sitcom"
        detail = ""
        replaced = False
        if flagged and policy_kind != "sitcom_only":
            fb = np_fallbacks[image_id]
            policy_psnr = float(fb["np_selected_psnr"])
            source = "np_selected"
            detail = f"{fb['np_selected_config_tag']} seed={fb['np_selected_seed']}"
            replaced = True

        record = {
            "policy_name": policy_name,
            "policy_kind": policy_kind,
            "notes": notes,
            "image_id": image_id,
            "run_index": run_index,
            "sitcom_final_psnr": sitcom_psnr,
            "policy_final_psnr": policy_psnr,
            "delta_vs_sitcom": policy_psnr - sitcom_psnr,
            "sitcom_bad_below25": sitcom_psnr < 25.0,
            "sitcom_bad_below20": sitcom_psnr < 20.0,
            "policy_bad_below25": policy_psnr < 25.0,
            "policy_bad_below20": policy_psnr < 20.0,
            "was_flagged": flagged,
            "was_replaced": replaced,
            "replacement_source": source,
            "replacement_detail": detail,
            "false_positive_replacement": replaced and sitcom_psnr >= 25.0,
            "true_positive_replacement": replaced and sitcom_psnr < 25.0,
            "false_negative_remaining_bad25": (not replaced) and sitcom_psnr < 25.0,
        }
        run_records.append(record)
        if replaced:
            replaced_rows.append(
                {
                    "policy_name": policy_name,
                    "policy_kind": policy_kind,
                    "image_id": image_id,
                    "run_index": run_index,
                    "sitcom_final_psnr": sitcom_psnr,
                    "replacement_psnr": policy_psnr,
                    "delta_vs_sitcom": policy_psnr - sitcom_psnr,
                    "replacement_source": source,
                    "replacement_detail": detail,
                    "false_positive_replacement": sitcom_psnr >= 25.0,
                    "true_positive_replacement": sitcom_psnr < 25.0,
                }
            )
    return run_records, replaced_rows


def summarize_policy_rows(
    policy_name: str,
    policy_kind: str,
    notes: str,
    run_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    final_psnrs = [to_float(r["policy_final_psnr"]) for r in run_rows]
    image_best_mean, image_best_min = summarize_image_level_policy(run_rows)
    return {
        "policy_name": policy_name,
        "policy_kind": policy_kind,
        "notes": notes,
        "run_mean_psnr": mean_or_nan(final_psnrs),
        "run_min_psnr": min(final_psnrs) if final_psnrs else math.nan,
        "run_bad25_count": sum(int(bool(r["policy_bad_below25"])) for r in run_rows),
        "run_bad20_count": sum(int(bool(r["policy_bad_below20"])) for r in run_rows),
        "num_replaced": sum(int(bool(r["was_replaced"])) for r in run_rows),
        "false_positive_replacements": sum(int(bool(r["false_positive_replacement"])) for r in run_rows),
        "true_positive_replacements": sum(int(bool(r["true_positive_replacement"])) for r in run_rows),
        "false_negative_remaining_bad25": sum(int(bool(r["false_negative_remaining_bad25"])) for r in run_rows),
        "image_best_of4_mean_psnr": image_best_mean,
        "image_best_of4_min_psnr": image_best_min,
    }


def build_summary_md(
    detector_rows: List[Dict[str, object]],
    policy_rows: List[Dict[str, object]],
    failure_rows: List[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# A9 Relative Detector CV")
    lines.append("")
    lines.append("This analysis uses only the existing A8 trajectory logs. No new SITCOM jobs were run.")
    lines.append("")
    lines.append("## Diagnostic baselines")
    lines.append("")
    lines.append(f"- Frozen A7 threshold: `{FROZEN_A7_DETECTOR_ID}` with `{FROZEN_A7_DIRECTION}` and threshold `{FROZEN_A7_THRESHOLD}`.")
    lines.append("- On A8 this frozen detector failed as a useful controller baseline; see the A8 folder for the full posthoc comparison.")
    lines.append("- The A8 posthoc threshold sweep is diagnostic only and should not be interpreted as held-out generalization evidence.")
    lines.append("")
    lines.append("## Main evidence: leave-one-image-out CV")
    lines.append("")
    for label in ("bad25", "bad20"):
        for threshold_policy in ("balanced_train", "conservative_zero_fp_train"):
            matches = [
                r for r in detector_rows
                if r["label"] == label and r["threshold_policy"] == threshold_policy
            ]
            if not matches:
                continue
            best = max(matches, key=lambda r: (to_float(r["global_balanced_accuracy"]), to_float(r["global_recall"])))
            lines.append(
                f"- Best `{label}` detector under `{threshold_policy}`: "
                f"`{best['detector_id']}` with global balanced accuracy "
                f"`{to_float(best['global_balanced_accuracy']):.3f}` and recall "
                f"`{to_float(best['global_recall']):.3f}`."
            )
            pol = next(
                (
                    p for p in policy_rows
                    if p["label"] == label
                    and p["threshold_policy"] == threshold_policy
                    and p["detector_id"] == best["detector_id"]
                    and p["policy_kind"] == "cv_np_selected_fallback"
                ),
                None,
            )
            if pol is not None:
                lines.append(
                    f"  NP-selected fallback gives run mean/min `{to_float(pol['run_mean_psnr']):.3f}` / "
                    f"`{to_float(pol['run_min_psnr']):.3f}` with bad25/bad20 counts "
                    f"`{int(pol['run_bad25_count'])}` / `{int(pol['run_bad20_count'])}`."
                )
    lines.append("")
    lines.append("## Failure cases")
    lines.append("")
    if failure_rows:
        lines.append(f"- Recorded `{len(failure_rows)}` false-positive / false-negative cases for the best CV detectors.")
    else:
        lines.append("- No failure cases were recorded for the selected summary detectors.")
    lines.append("")
    lines.append("## Policy separation")
    lines.append("")
    lines.append("- `sitcom_only` is the baseline.")
    lines.append("- `cv_np_selected_fallback` is the executable held-out detector result.")
    lines.append("- `oracle_risk_np_selected_diagnostic` is an upper bound using perfect risk labels and is diagnostic only.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--a8_dir", type=Path, required=True)
    p.add_argument("--np_run_level", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--noise", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a8_dir = args.a8_dir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    step_rows = read_csv(a8_dir / "trajectory_step_metrics.csv")
    run_rows = read_csv(a8_dir / "run_level_summary.csv")
    np_rows = read_csv(args.np_run_level)

    add_interrun_features(step_rows)
    detector_table = build_detector_table(step_rows, run_rows)
    image_ids = sorted({str(r["image_id"]) for r in detector_table})
    np_fallbacks = load_np_fallbacks(np_rows, args.noise, image_ids)

    detector_ids = sorted(
        key
        for key in detector_table[0].keys()
        if key not in {"image_id", "run_index", "final_psnr", "bad25", "bad20", "num_steps"}
    )

    detector_cv_summary: List[Dict[str, object]] = []
    detector_cv_image_level: List[Dict[str, object]] = []
    detector_cv_failure_cases: List[Dict[str, object]] = []
    controller_cv_policy_summary: List[Dict[str, object]] = []
    controller_cv_replaced_runs: List[Dict[str, object]] = []

    best_detector_by_label_policy: Dict[Tuple[str, str], Dict[str, object]] = {}

    # Fixed diagnostic baselines on the A8 run-level table.
    baseline_rows = [
        {
            "image_id": str(r["image_id"]),
            "run_index": int(r["run_index"]),
            "final_psnr": to_float(r["final_psnr"]),
        }
        for r in run_rows
    ]
    baseline_flags = {(r["image_id"], r["run_index"]): False for r in baseline_rows}
    baseline_policy_rows, baseline_replaced = simulate_policy(
        "sitcom_only",
        "sitcom_only",
        baseline_rows,
        baseline_flags,
        np_fallbacks,
        "No fallback; raw SITCOM final outputs.",
    )
    controller_cv_policy_summary.append(
        {
            "label": "diagnostic",
            "threshold_policy": "diagnostic",
            "detector_id": "",
            "direction_mode": "",
            "mean_threshold": math.nan,
            **summarize_policy_rows(
                "sitcom_only",
                "sitcom_only",
                "No fallback; raw SITCOM final outputs.",
                baseline_policy_rows,
            ),
        }
    )
    controller_cv_replaced_runs.extend(baseline_replaced)

    oracle_flags = {
        (r["image_id"], r["run_index"]): to_float(r["final_psnr"]) < 25.0
        for r in baseline_rows
    }
    oracle_policy_rows, oracle_replaced = simulate_policy(
        "oracle_risk_np_selected_diagnostic",
        "oracle_risk_np_selected_diagnostic",
        baseline_rows,
        oracle_flags,
        np_fallbacks,
        "Diagnostic upper bound: replace exactly the bad25 SITCOM runs with NP selected.",
    )
    controller_cv_policy_summary.append(
        {
            "label": "bad25",
            "threshold_policy": "oracle_diagnostic",
            "detector_id": "",
            "direction_mode": "",
            "mean_threshold": math.nan,
            **summarize_policy_rows(
                "oracle_risk_np_selected_diagnostic",
                "oracle_risk_np_selected_diagnostic",
                "Diagnostic upper bound: replace exactly the bad25 SITCOM runs with NP selected.",
                oracle_policy_rows,
            ),
        }
    )
    controller_cv_replaced_runs.extend(oracle_replaced)

    for label in ("bad25", "bad20"):
        label_rows = [
            r for r in detector_table
            if math.isfinite(to_float(r["final_psnr"]))
        ]
        for detector_id in detector_ids:
            rows = [r for r in label_rows if math.isfinite(to_float(r.get(detector_id, math.nan)))]
            if not rows:
                continue

            fold_records: Dict[str, List[Dict[str, object]]] = {
                "balanced_train": [],
                "conservative_zero_fp_train": [],
            }
            heldout_predictions: Dict[str, Dict[Tuple[str, int], Dict[str, object]]] = {
                "balanced_train": {},
                "conservative_zero_fp_train": {},
            }

            for heldout_image in image_ids:
                train_rows = [r for r in rows if str(r["image_id"]) != heldout_image]
                test_rows = [r for r in rows if str(r["image_id"]) == heldout_image]
                if not train_rows or not test_rows:
                    continue

                train_scores = np.asarray([to_float(r[detector_id]) for r in train_rows], dtype=float)
                train_labels = np.asarray([int(r[label]) for r in train_rows], dtype=int)
                candidates = evaluate_threshold_candidates(train_scores, train_labels)
                if not candidates:
                    continue

                selected = {
                    "balanced_train": select_balanced_threshold(candidates),
                    "conservative_zero_fp_train": select_conservative_threshold(candidates),
                }

                for threshold_policy, choice in selected.items():
                    test_scores = np.asarray([to_float(r[detector_id]) for r in test_rows], dtype=float)
                    test_labels = np.asarray([int(r[label]) for r in test_rows], dtype=int)
                    pred = apply_threshold(test_scores, str(choice["direction"]), float(choice["threshold"]))
                    counts = confusion_counts(test_labels, pred)
                    rec = {
                        "detector_id": detector_id,
                        "label": label,
                        "threshold_policy": threshold_policy,
                        "heldout_image": heldout_image,
                        "threshold_direction": choice["direction"],
                        "threshold": choice["threshold"],
                        "train_tp": choice["tp"],
                        "train_fp": choice["fp"],
                        "train_tn": choice["tn"],
                        "train_fn": choice["fn"],
                        "train_balanced_accuracy": choice["balanced_accuracy"],
                        "train_recall": choice["recall"],
                        "train_num_flagged": choice["num_flagged"],
                        "test_tp": counts["tp"],
                        "test_fp": counts["fp"],
                        "test_tn": counts["tn"],
                        "test_fn": counts["fn"],
                        "test_balanced_accuracy": balanced_accuracy(counts),
                        "test_precision": precision(counts),
                        "test_recall": recall(counts),
                        "test_num_flagged": int(np.sum(pred)),
                        "num_test_runs": len(test_rows),
                    }
                    fold_records[threshold_policy].append(rec)
                    for row, score, p in zip(test_rows, test_scores, pred):
                        heldout_predictions[threshold_policy][(str(row["image_id"]), int(row["run_index"]))] = {
                            "pred": int(p),
                            "score": float(score),
                            "threshold": float(choice["threshold"]),
                            "direction": str(choice["direction"]),
                        }

            for threshold_policy, fold_rows in fold_records.items():
                if not fold_rows:
                    continue
                detector_cv_image_level.extend(fold_rows)
                global_counts = {
                    "tp": sum(int(r["test_tp"]) for r in fold_rows),
                    "fp": sum(int(r["test_fp"]) for r in fold_rows),
                    "tn": sum(int(r["test_tn"]) for r in fold_rows),
                    "fn": sum(int(r["test_fn"]) for r in fold_rows),
                }
                direction_mode = multimode([str(r["threshold_direction"]) for r in fold_rows])[0]
                summary_row = {
                    "detector_id": detector_id,
                    "label": label,
                    "threshold_policy": threshold_policy,
                    "direction_mode": direction_mode,
                    "mean_threshold": mean_or_nan(to_float(r["threshold"]) for r in fold_rows),
                    "median_threshold": float(np.median([to_float(r["threshold"]) for r in fold_rows])),
                    "num_images": len(fold_rows),
                    "num_runs": sum(int(r["num_test_runs"]) for r in fold_rows),
                    "global_tp": global_counts["tp"],
                    "global_fp": global_counts["fp"],
                    "global_tn": global_counts["tn"],
                    "global_fn": global_counts["fn"],
                    "global_balanced_accuracy": balanced_accuracy(global_counts),
                    "global_precision": precision(global_counts),
                    "global_recall": recall(global_counts),
                    "mean_image_balanced_accuracy": mean_or_nan(to_float(r["test_balanced_accuracy"]) for r in fold_rows),
                    "mean_train_balanced_accuracy": mean_or_nan(to_float(r["train_balanced_accuracy"]) for r in fold_rows),
                    "mean_train_recall": mean_or_nan(to_float(r["train_recall"]) for r in fold_rows),
                    "mean_test_num_flagged": mean_or_nan(to_float(r["test_num_flagged"]) for r in fold_rows),
                }
                detector_cv_summary.append(summary_row)

                key = (label, threshold_policy)
                current_best = best_detector_by_label_policy.get(key)
                if current_best is None or (
                    to_float(summary_row["global_balanced_accuracy"]),
                    to_float(summary_row["global_recall"]),
                ) > (
                    to_float(current_best["global_balanced_accuracy"]),
                    to_float(current_best["global_recall"]),
                ):
                    best_detector_by_label_policy[key] = summary_row

                policy_flags = {k: bool(v["pred"]) for k, v in heldout_predictions[threshold_policy].items()}
                policy_rows, replaced_rows = simulate_policy(
                    f"{detector_id}__{label}__{threshold_policy}",
                    "cv_np_selected_fallback",
                    baseline_rows,
                    policy_flags,
                    np_fallbacks,
                    "Leave-one-image-out threshold selection with NP-selected fallback.",
                )
                controller_summary = summarize_policy_rows(
                    f"{detector_id}__{label}__{threshold_policy}",
                    "cv_np_selected_fallback",
                    "Leave-one-image-out threshold selection with NP-selected fallback.",
                    policy_rows,
                )
                controller_summary.update(
                    {
                        "label": label,
                        "threshold_policy": threshold_policy,
                        "detector_id": detector_id,
                        "direction_mode": direction_mode,
                        "mean_threshold": summary_row["mean_threshold"],
                    }
                )
                controller_cv_policy_summary.append(controller_summary)

                for row in replaced_rows:
                    key2 = (str(row["image_id"]), int(row["run_index"]))
                    pred_meta = heldout_predictions[threshold_policy].get(key2, {})
                    row.update(
                        {
                            "label": label,
                            "threshold_policy": threshold_policy,
                            "detector_id": detector_id,
                            "policy_kind": "cv_np_selected_fallback",
                            "detector_value": pred_meta.get("score", math.nan),
                            "threshold": pred_meta.get("threshold", math.nan),
                            "direction": pred_meta.get("direction", ""),
                        }
                    )
                controller_cv_replaced_runs.extend(replaced_rows)

    for (label, threshold_policy), best in best_detector_by_label_policy.items():
        detector_id = str(best["detector_id"])
        fold_rows = [
            r for r in detector_cv_image_level
            if r["label"] == label
            and r["threshold_policy"] == threshold_policy
            and r["detector_id"] == detector_id
        ]
        for row in detector_table:
            fold = next(
                (
                    fr for fr in fold_rows
                    if str(fr["heldout_image"]) == str(row["image_id"])
                ),
                None,
            )
            if fold is None:
                continue
            score = to_float(row.get(detector_id, math.nan))
            pred = apply_threshold(
                np.asarray([score], dtype=float),
                str(fold["threshold_direction"]),
                float(fold["threshold"]),
            )[0]
            bad = int(row[label]) == 1
            if int(pred) == 1 and not bad:
                detector_cv_failure_cases.append(
                    {
                        "label": label,
                        "threshold_policy": threshold_policy,
                        "detector_id": detector_id,
                        "image_id": row["image_id"],
                        "run_index": row["run_index"],
                        "case_type": "false_positive",
                        "final_psnr": row["final_psnr"],
                        "detector_value": score,
                        "threshold": fold["threshold"],
                        "direction": fold["threshold_direction"],
                        "np_selected_psnr": np_fallbacks[str(row["image_id"])]["np_selected_psnr"],
                        "fallback_delta": np_fallbacks[str(row["image_id"])]["np_selected_psnr"] - to_float(row["final_psnr"]),
                    }
                )
            if int(pred) == 0 and bad:
                detector_cv_failure_cases.append(
                    {
                        "label": label,
                        "threshold_policy": threshold_policy,
                        "detector_id": detector_id,
                        "image_id": row["image_id"],
                        "run_index": row["run_index"],
                        "case_type": "false_negative",
                        "final_psnr": row["final_psnr"],
                        "detector_value": score,
                        "threshold": fold["threshold"],
                        "direction": fold["threshold_direction"],
                        "np_selected_psnr": np_fallbacks[str(row["image_id"])]["np_selected_psnr"],
                        "fallback_delta": np_fallbacks[str(row["image_id"])]["np_selected_psnr"] - to_float(row["final_psnr"]),
                    }
                )

    detector_cv_summary.sort(key=lambda r: (str(r["label"]), str(r["threshold_policy"]), -to_float(r["global_balanced_accuracy"]), str(r["detector_id"])))
    detector_cv_image_level.sort(key=lambda r: (str(r["label"]), str(r["threshold_policy"]), str(r["detector_id"]), str(r["heldout_image"])))
    detector_cv_failure_cases.sort(key=lambda r: (str(r["label"]), str(r["threshold_policy"]), str(r["detector_id"]), str(r["case_type"]), str(r["image_id"]), int(r["run_index"])))
    controller_cv_policy_summary.sort(key=lambda r: (str(r["label"]), str(r["threshold_policy"]), str(r["policy_kind"]), -to_float(r["run_mean_psnr"]), str(r["detector_id"])))
    controller_cv_replaced_runs.sort(key=lambda r: (str(r.get("label", "")), str(r.get("threshold_policy", "")), str(r["detector_id"]) if "detector_id" in r else "", str(r["image_id"]), int(r["run_index"])))

    write_csv(outdir / "detector_cv_summary.csv", detector_cv_summary)
    write_csv(outdir / "detector_cv_image_level.csv", detector_cv_image_level)
    write_csv(outdir / "detector_cv_failure_cases.csv", detector_cv_failure_cases)
    write_csv(outdir / "controller_cv_policy_summary.csv", controller_cv_policy_summary)
    write_csv(outdir / "controller_cv_replaced_runs.csv", controller_cv_replaced_runs)
    write_text(outdir / "SUMMARY.md", build_summary_md(detector_cv_summary, controller_cv_policy_summary, detector_cv_failure_cases))


if __name__ == "__main__":
    main()
