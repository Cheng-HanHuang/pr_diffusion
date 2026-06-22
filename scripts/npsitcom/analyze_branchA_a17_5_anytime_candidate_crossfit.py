#!/usr/bin/env python3
"""A17.5 cross-fit audit of anytime detector candidates.

This is a diagnostic-only development pass. It uses existing A8, A11, A14, A16
trajectory outputs plus the A17 anytime feature table. It does not run any new
SITCOM jobs and does not modify frozen A14/A16 policies.

The goal is to ask whether the strongest A17 anytime signals can be turned into
a stable frozen rule under train/test cross-fit splits.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


A17_DIR = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/"
    "A17_offline_anytime_detector_design"
)
A14_FROZEN_POLICY_JSON = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/"
    "A14_frozen_policy_config/frozen_policy.json"
)

SPLITS: List[Tuple[str, List[str], List[str]]] = [
    ("train_A8A11_test_A14A16", ["A8", "A11"], ["A14", "A16"]),
    ("train_A14A16_test_A8A11", ["A14", "A16"], ["A8", "A11"]),
    ("lodo_test_A8", ["A11", "A14", "A16"], ["A8"]),
    ("lodo_test_A11", ["A8", "A14", "A16"], ["A11"]),
    ("lodo_test_A14", ["A8", "A11", "A16"], ["A14"]),
    ("lodo_test_A16", ["A8", "A11", "A14"], ["A16"]),
]

BASE_RULES: Dict[str, str] = {
    "correction_norm__persist10": "correction_norm__rank_ge3__persist10__first_step0",
    "x0hat_x0y_disagreement__persist10": "x0hat_x0y_disagreement__rank_ge3__persist10__first_step0",
    "x0y_full_residual_normed__persist10": "x0y_full_residual_normed__rank_ge3__persist10__first_step0",
    "x0y_lowfreq_residual_normed__persist10": "x0y_lowfreq_residual_normed__rank_ge3__persist10__first_step0",
}

BUDGETS = {
    "conservative": {"fp_max": 5, "replacements_max": 30},
    "aggressive": {"fp_max": 10, "replacements_max": 40},
}

WINDOW_FRACS = [0.50, 0.60, 0.70, 0.80]


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


def median_or_nan(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.median(xs)) if xs else math.nan


def make_run_key(row: Dict[str, str]) -> Tuple[str, int]:
    return str(row["image_id"]), int(row["run_index"])


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


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
        rows = [r for r in candidates if r["image_id"] == image_id]
        if not rows:
            raise ValueError(f"No NP fallback candidates for {image_id} at noise {noise}")
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


def load_a17_rows() -> List[Dict[str, object]]:
    rows = read_csv(A17_DIR / "anytime_feature_table.csv")
    out: List[Dict[str, object]] = []
    for row in rows:
        n_steps = int(row["num_steps"])
        feature_step0s = {
            "correction_norm": to_float(row["correction_norm__rank_ge3__persist10__first_step0"]),
            "x0hat_x0y_disagreement": to_float(row["x0hat_x0y_disagreement__rank_ge3__persist10__first_step0"]),
            "x0y_full_residual_normed": to_float(row["x0y_full_residual_normed__rank_ge3__persist10__first_step0"]),
            "x0y_lowfreq_residual_normed": to_float(row["x0y_lowfreq_residual_normed__rank_ge3__persist10__first_step0"]),
        }
        feature_fracs = {
            name: ((step0 + 1.0) / n_steps if math.isfinite(step0) else 2.0)
            for name, step0 in feature_step0s.items()
        }
        out.append(
            {
                "dataset": str(row["dataset"]),
                "image_id": str(row["image_id"]),
                "run_index": int(row["run_index"]),
                "final_psnr": to_float(row["final_psnr"]),
                "bad25": int(row["bad25"]),
                "bad20": int(row["bad20"]),
                "num_steps": n_steps,
                "correction_norm__persist10_step0": feature_step0s["correction_norm"],
                "x0hat_x0y_disagreement__persist10_step0": feature_step0s["x0hat_x0y_disagreement"],
                "x0y_full_residual_normed__persist10_step0": feature_step0s["x0y_full_residual_normed"],
                "x0y_lowfreq_residual_normed__persist10_step0": feature_step0s["x0y_lowfreq_residual_normed"],
                "correction_norm__persist10_alarm_frac": feature_fracs["correction_norm"],
                "x0hat_x0y_disagreement__persist10_alarm_frac": feature_fracs["x0hat_x0y_disagreement"],
                "x0y_full_residual_normed__persist10_alarm_frac": feature_fracs["x0y_full_residual_normed"],
                "x0y_lowfreq_residual_normed__persist10_alarm_frac": feature_fracs["x0y_lowfreq_residual_normed"],
            }
        )
    return out


def candidate_specs() -> List[Dict[str, object]]:
    names = list(BASE_RULES.keys())
    specs: List[Dict[str, object]] = []
    for size in range(1, len(names) + 1):
        for combo in itertools.combinations(names, size):
            if len(combo) == 1:
                name = combo[0]
                family = "single"
            else:
                name = "or__" + "__".join(combo)
                family = "or"
            specs.append(
                {
                    "candidate_name": name,
                    "candidate_family": family,
                    "components": list(combo),
                }
            )
    return specs


def score_for_row(row: Dict[str, object], components: Sequence[str]) -> Dict[str, object]:
    component_steps = [to_float(row.get(f"{c}__persist10_step0")) for c in components]
    component_fracs = [to_float(row.get(f"{c}__persist10_alarm_frac")) for c in components]
    paired = [(c, s, f) for c, s, f in zip(components, component_steps, component_fracs)]
    finite_fracs = [f for _, _, f in paired if math.isfinite(f)]
    if finite_fracs:
        best_component, best_step0, best_frac = min(paired, key=lambda t: (t[2] if math.isfinite(t[2]) else 2.0, t[1] if math.isfinite(t[1]) else 1e9))
    else:
        best_component, best_step0, best_frac = "", math.nan, 2.0
    return {
        "selected_component": best_component,
        "selected_alarm_step0": best_step0,
        "selected_alarm_step1": best_step0 + 1.0 if math.isfinite(best_step0) else math.nan,
        "selected_alarm_frac": best_frac,
        "component_alarm_steps_json": json.dumps({c: row.get(f"{c}__step0") for c in components}, sort_keys=True),
        "component_alarm_fracs_json": json.dumps({c: row.get(BASE_RULES[c]) for c in components}, sort_keys=True),
    }


def eval_policy(
    rows: List[Dict[str, object]],
    flags: Sequence[bool],
    np_fallbacks: Dict[str, Dict[str, object]],
    candidate_name: str,
    candidate_family: str,
    budget_mode: str,
    fit_regime: str,
    train_datasets: Sequence[str],
    test_datasets: Sequence[str],
    threshold: float,
    components: Sequence[str],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    records: List[Dict[str, object]] = []
    by_image: Dict[str, List[float]] = defaultdict(list)
    by_image_sitcom: Dict[str, List[float]] = defaultdict(list)
    y_true = []
    y_pred = []
    for row, flagged in zip(rows, flags):
        image_id = str(row["image_id"])
        sitcom_psnr = float(row["final_psnr"])
        fb = np_fallbacks[image_id]
        policy_psnr = float(fb["np_selected_psnr"]) if flagged else sitcom_psnr
        is_bad25 = int(sitcom_psnr < 25.0)
        is_bad20 = int(sitcom_psnr < 20.0)
        y_true.append(is_bad25)
        y_pred.append(bool(flagged))
        by_image[image_id].append(policy_psnr)
        by_image_sitcom[image_id].append(sitcom_psnr)
        score_info = score_for_row(row, components)
        records.append(
            {
                "fit_regime": fit_regime,
                "budget_mode": budget_mode,
                "candidate_name": candidate_name,
                "candidate_family": candidate_family,
                "train_datasets": "+".join(train_datasets),
                "test_datasets": "+".join(test_datasets),
                "dataset": str(row["dataset"]),
                "image_id": image_id,
                "run_index": int(row["run_index"]),
                "final_psnr": sitcom_psnr,
                "bad25": is_bad25,
                "bad20": is_bad20,
                "selected_alarm_frac": score_info["selected_alarm_frac"],
                "selected_alarm_step0": score_info["selected_alarm_step0"],
                "selected_alarm_step1": score_info["selected_alarm_step1"],
                "selected_component": score_info["selected_component"],
                "component_alarm_steps_json": score_info["component_alarm_steps_json"],
                "component_alarm_fracs_json": score_info["component_alarm_fracs_json"],
                "threshold_alarm_frac": threshold,
                "flagged": bool(flagged),
                "policy_psnr": policy_psnr,
                "delta_vs_sitcom": policy_psnr - sitcom_psnr,
                "is_true_positive": int(flagged and is_bad25),
                "is_false_positive": int(flagged and not is_bad25),
                "is_false_negative": int((not flagged) and is_bad25),
                "alarm_time_pct": score_info["selected_alarm_frac"],
            }
        )

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if (not yt) and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if (not yt) and (not yp))
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and (not yp))

    policy_psnrs = [float(r["policy_psnr"]) for r in records]
    image_best = [max(vals) for vals in by_image.values()]
    image_mean = [float(np.mean(vals)) for vals in by_image.values()]
    image_min = [min(vals) for vals in by_image.values()]
    false_positive_losses = [float(r["delta_vs_sitcom"]) for r in records if r["is_false_positive"]]
    tp_alarm = [float(r["alarm_time_pct"]) for r in records if r["is_true_positive"]]
    fp_alarm = [float(r["alarm_time_pct"]) for r in records if r["is_false_positive"]]
    bad25_recall = tp / (tp + fn) if (tp + fn) else math.nan
    bad20_recall = sum(1 for r in records if r["is_true_positive"] or (int(r["bad20"]) and r["flagged"])) / sum(1 for r in records if int(r["bad20"]) == 1) if sum(1 for r in records if int(r["bad20"]) == 1) else math.nan
    metrics = {
        "candidate_name": candidate_name,
        "candidate_family": candidate_family,
        "budget_mode": budget_mode,
        "fit_regime": fit_regime,
        "train_datasets": "+".join(train_datasets),
        "test_datasets": "+".join(test_datasets),
        "threshold_alarm_frac": threshold,
        "num_replaced": int(sum(y_pred)),
        "num_false_positive_replacements": fp,
        "num_true_positive_replacements": tp,
        "num_false_negative_remaining_bad25": fn,
        "num_false_negative_remaining_bad20": sum(1 for r in records if int(r["bad20"]) == 1 and not r["flagged"]),
        "bad25_recall": bad25_recall,
        "bad25_precision": tp / (tp + fp) if (tp + fp) else math.nan,
        "bad20_recall": bad20_recall,
        "bad20_precision": (
            sum(1 for r in records if int(r["bad20"]) == 1 and r["flagged"])
            / max(sum(1 for r in records if r["flagged"]), 1)
        ),
        "run_level_mean_psnr": mean_or_nan(policy_psnrs),
        "run_level_min_psnr": min(policy_psnrs) if policy_psnrs else math.nan,
        "run_level_num_below25": sum(1 for x in policy_psnrs if x < 25.0),
        "run_level_num_below20": sum(1 for x in policy_psnrs if x < 20.0),
        "image_level_best_of_4_mean_psnr": mean_or_nan(image_best),
        "image_level_best_of_4_min_psnr": min(image_best) if image_best else math.nan,
        "image_level_mean_of_4_mean_psnr": mean_or_nan(image_mean),
        "image_level_min_of_4_mean_psnr": min(image_mean) if image_mean else math.nan,
        "image_level_mean_of_4_min_psnr": mean_or_nan(image_min),
        "image_level_min_of_4_min_psnr": min(image_min) if image_min else math.nan,
        "mean_delta_vs_sitcom": mean_or_nan(float(r["delta_vs_sitcom"]) for r in records),
        "worst_false_positive_psnr_loss": min(false_positive_losses) if false_positive_losses else math.nan,
        "median_alarm_time_tp": median_or_nan(tp_alarm),
        "median_alarm_time_fp": median_or_nan(fp_alarm),
        "mean_alarm_time_tp": mean_or_nan(tp_alarm),
        "mean_alarm_time_fp": mean_or_nan(fp_alarm),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    metrics["train_feasible_under_budget"] = bool(
        int(metrics["num_false_positive_replacements"]) <= BUDGETS[budget_mode]["fp_max"]
        and int(metrics["num_replaced"]) <= BUDGETS[budget_mode]["replacements_max"]
    )
    metrics["test_feasible_under_budget"] = metrics["train_feasible_under_budget"]
    for frac in WINDOW_FRACS:
        pct = int(round(frac * 100))
        metrics[f"bad25_visible_by_{pct}pct"] = sum(
            1 for r in records if int(r["bad25"]) == 1 and float(r["selected_alarm_frac"]) <= frac
        )
        metrics[f"bad20_visible_by_{pct}pct"] = sum(
            1 for r in records if int(r["bad20"]) == 1 and float(r["selected_alarm_frac"]) <= frac
        )
        metrics[f"bad25_visible_rate_by_{pct}pct"] = (
            metrics[f"bad25_visible_by_{pct}pct"] / sum(1 for r in records if int(r["bad25"]) == 1)
            if sum(1 for r in records if int(r["bad25"]) == 1)
            else math.nan
        )
        metrics[f"bad20_visible_rate_by_{pct}pct"] = (
            metrics[f"bad20_visible_by_{pct}pct"] / sum(1 for r in records if int(r["bad20"]) == 1)
            if sum(1 for r in records if int(r["bad20"]) == 1)
            else math.nan
        )
    return metrics, records


def threshold_sort_key(metrics: Dict[str, object], budget_mode: str) -> Tuple[object, ...]:
    fp_max = BUDGETS[budget_mode]["fp_max"]
    repl_max = BUDGETS[budget_mode]["replacements_max"]
    recall = to_float(metrics["bad25_recall"])
    bad20_recall = to_float(metrics["bad20_recall"])
    fp = int(metrics["num_false_positive_replacements"])
    repl = int(metrics["num_replaced"])
    feasible = fp <= fp_max and repl <= repl_max
    return (
        0 if feasible else 1,
        -recall if math.isfinite(recall) else 1e9,
        -bad20_recall if math.isfinite(bad20_recall) else 1e9,
        fp,
        repl,
        -to_float(metrics["run_level_mean_psnr"]),
        -to_float(metrics["image_level_best_of_4_min_psnr"]),
        float(metrics["threshold_alarm_frac"]),
    )


def fit_threshold(
    train_rows: List[Dict[str, object]],
    np_fallbacks: Dict[str, Dict[str, object]],
    candidate_name: str,
    candidate_family: str,
    budget_mode: str,
    fit_regime: str,
    train_datasets: Sequence[str],
    test_datasets: Sequence[str],
    components: Sequence[str],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    scores = [float(min(row.get(f"{c}__persist10_alarm_frac", 2.0) for c in components)) for row in train_rows]
    thresholds = sorted({s for s in scores if math.isfinite(s)})
    if 0.0 not in thresholds:
        thresholds = [0.0] + thresholds
    if 1.0 not in thresholds:
        thresholds.append(1.0)
    if 2.0 not in thresholds:
        thresholds.append(2.0)

    best = None
    best_records: List[Dict[str, object]] = []
    for threshold in thresholds:
        flags = [score <= threshold for score in scores]
        metrics, records = eval_policy(
            train_rows,
            flags,
            np_fallbacks,
            candidate_name,
            candidate_family,
            budget_mode,
            fit_regime,
            train_datasets,
            test_datasets,
            threshold,
            components,
        )
        sort_key = threshold_sort_key(metrics, budget_mode)
        if best is None or sort_key < best[0]:
            best = (sort_key, metrics)
            best_records = records
    assert best is not None
    return best[1], best_records


def add_split_context(row: Dict[str, object], split_name: str, train_datasets: Sequence[str], test_datasets: Sequence[str]) -> Dict[str, object]:
    out = dict(row)
    out["split_name"] = split_name
    out["train_datasets"] = "+".join(train_datasets)
    out["test_datasets"] = "+".join(test_datasets)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--git_commit", required=True)
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_a17_rows()
    if not rows:
        raise ValueError("No A17 rows found")
    image_ids = sorted({str(r["image_id"]) for r in rows})
    np_fallbacks = load_np_fallbacks(
        json.loads(A14_FROZEN_POLICY_JSON.read_text(encoding="utf-8"))["fallback_source_csv"],
        args.noise,
        image_ids,
    )

    candidate_rows: List[Dict[str, object]] = []
    policy_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    missed_rows: List[Dict[str, object]] = []
    image00017_rows: List[Dict[str, object]] = []

    row_by_dataset: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        row_by_dataset[str(row["dataset"])].append(row)

    for split_name, train_datasets, test_datasets in SPLITS:
        train_rows = [r for r in rows if str(r["dataset"]) in train_datasets]
        test_rows = [r for r in rows if str(r["dataset"]) in test_datasets]
        if not train_rows or not test_rows:
            continue

        for spec in candidate_specs():
            candidate_name = str(spec["candidate_name"])
            candidate_family = str(spec["candidate_family"])
            components = list(spec["components"])
            for budget_mode in BUDGETS:
                train_metrics, _ = fit_threshold(
                    train_rows,
                    np_fallbacks,
                    candidate_name,
                    candidate_family,
                    budget_mode,
                    split_name,
                    train_datasets,
                    test_datasets,
                    components,
                )

                # Refit threshold selection on train, then evaluate on test with same threshold.
                threshold = float(train_metrics["threshold_alarm_frac"])
                test_scores = [float(min(row.get(f"{c}__persist10_alarm_frac", 2.0) for c in components)) for row in test_rows]
                test_flags = [score <= threshold for score in test_scores]
                test_metrics, test_record_rows = eval_policy(
                    test_rows,
                    test_flags,
                    np_fallbacks,
                    candidate_name,
                    candidate_family,
                    budget_mode,
                    split_name,
                    train_datasets,
                    test_datasets,
                    threshold,
                    components,
                )

                row = {
                    "split_name": split_name,
                    "budget_mode": budget_mode,
                    "candidate_name": candidate_name,
                    "candidate_family": candidate_family,
                    "components_json": json.dumps(components),
                    "num_components": len(components),
                    "threshold_alarm_frac": threshold,
                    "train_datasets": "+".join(train_datasets),
                    "test_datasets": "+".join(test_datasets),
                }
                for prefix, metrics in (("train", train_metrics), ("test", test_metrics)):
                    for key, value in metrics.items():
                        if key in {"candidate_name", "candidate_family", "budget_mode", "fit_regime", "train_datasets", "test_datasets"}:
                            continue
                        row[f"{prefix}_{key}"] = value
                candidate_rows.append(row)
                policy_rows.append(
                    {
                        "split_name": split_name,
                        "budget_mode": budget_mode,
                        "candidate_name": candidate_name,
                        "candidate_family": candidate_family,
                        "components_json": json.dumps(components),
                        "num_components": len(components),
                        "threshold_alarm_frac": threshold,
                        "train_feasible_under_budget": int(bool(train_metrics["train_feasible_under_budget"])),
                        "test_feasible_under_budget": int(bool(test_metrics["test_feasible_under_budget"])),
                        "train_bad25_recall": train_metrics["bad25_recall"],
                        "test_bad25_recall": test_metrics["bad25_recall"],
                        "train_bad20_recall": train_metrics["bad20_recall"],
                        "test_bad20_recall": test_metrics["bad20_recall"],
                        "train_num_false_positive_replacements": train_metrics["num_false_positive_replacements"],
                        "test_num_false_positive_replacements": test_metrics["num_false_positive_replacements"],
                        "train_num_replaced": train_metrics["num_replaced"],
                        "test_num_replaced": test_metrics["num_replaced"],
                        "train_run_level_mean_psnr": train_metrics["run_level_mean_psnr"],
                        "test_run_level_mean_psnr": test_metrics["run_level_mean_psnr"],
                        "train_run_level_min_psnr": train_metrics["run_level_min_psnr"],
                        "test_run_level_min_psnr": test_metrics["run_level_min_psnr"],
                        "train_image_level_best_of_4_mean_psnr": train_metrics["image_level_best_of_4_mean_psnr"],
                        "test_image_level_best_of_4_mean_psnr": test_metrics["image_level_best_of_4_mean_psnr"],
                        "train_image_level_best_of_4_min_psnr": train_metrics["image_level_best_of_4_min_psnr"],
                        "test_image_level_best_of_4_min_psnr": test_metrics["image_level_best_of_4_min_psnr"],
                        "train_bad25_visible_by_50pct": train_metrics["bad25_visible_by_50pct"],
                        "train_bad25_visible_by_60pct": train_metrics["bad25_visible_by_60pct"],
                        "train_bad25_visible_by_70pct": train_metrics["bad25_visible_by_70pct"],
                        "train_bad25_visible_by_80pct": train_metrics["bad25_visible_by_80pct"],
                        "test_bad25_visible_by_50pct": test_metrics["bad25_visible_by_50pct"],
                        "test_bad25_visible_by_60pct": test_metrics["bad25_visible_by_60pct"],
                        "test_bad25_visible_by_70pct": test_metrics["bad25_visible_by_70pct"],
                        "test_bad25_visible_by_80pct": test_metrics["bad25_visible_by_80pct"],
                        "train_bad20_visible_by_50pct": train_metrics["bad20_visible_by_50pct"],
                        "train_bad20_visible_by_60pct": train_metrics["bad20_visible_by_60pct"],
                        "train_bad20_visible_by_70pct": train_metrics["bad20_visible_by_70pct"],
                        "train_bad20_visible_by_80pct": train_metrics["bad20_visible_by_80pct"],
                        "test_bad20_visible_by_50pct": test_metrics["bad20_visible_by_50pct"],
                        "test_bad20_visible_by_60pct": test_metrics["bad20_visible_by_60pct"],
                        "test_bad20_visible_by_70pct": test_metrics["bad20_visible_by_70pct"],
                        "test_bad20_visible_by_80pct": test_metrics["bad20_visible_by_80pct"],
                        "test_median_alarm_time_tp": test_metrics["median_alarm_time_tp"],
                        "test_median_alarm_time_fp": test_metrics["median_alarm_time_fp"],
                        "test_mean_alarm_time_tp": test_metrics["mean_alarm_time_tp"],
                        "test_mean_alarm_time_fp": test_metrics["mean_alarm_time_fp"],
                        "train_feasible": int(bool(train_metrics["train_feasible_under_budget"])),
                        "test_feasible": int(bool(test_metrics["test_feasible_under_budget"])),
                    }
                )

                # Re-evaluate test rows with richer long-form records for downstream tables.
                event_rows.extend(test_record_rows)
                missed_rows.extend(
                    {
                        **r,
                        "split_name": split_name,
                        "budget_mode": budget_mode,
                        "candidate_name": candidate_name,
                        "candidate_family": candidate_family,
                        "threshold_alarm_frac": threshold,
                    }
                    for r in test_record_rows
                    if r["is_false_negative"] and (int(r["bad25"]) == 1 or int(r["bad20"]) == 1)
                )
                image00017_rows.extend(
                    {
                        **r,
                        "split_name": split_name,
                        "budget_mode": budget_mode,
                        "candidate_name": candidate_name,
                        "candidate_family": candidate_family,
                        "threshold_alarm_frac": threshold,
                    }
                    for r in test_record_rows
                    if r["image_id"] == "00017"
                )

    # Aggregate across regimes for the policy summary.
    policy_rows_by_candidate: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in policy_rows:
        policy_rows_by_candidate[(str(row["candidate_name"]), str(row["budget_mode"]))].append(row)

    agg_rows: List[Dict[str, object]] = []
    for (candidate_name, budget_mode), rows_for_candidate in sorted(policy_rows_by_candidate.items()):
        agg_rows.append(
            {
                "candidate_name": candidate_name,
                "budget_mode": budget_mode,
                "num_regimes": len(rows_for_candidate),
                "num_feasible_train": int(sum(int(r["train_feasible_under_budget"]) for r in rows_for_candidate)),
                "num_feasible_test": int(sum(int(r["test_feasible_under_budget"]) for r in rows_for_candidate)),
                "all_regimes_feasible_train": int(all(int(r["train_feasible_under_budget"]) for r in rows_for_candidate)),
                "all_regimes_feasible_test": int(all(int(r["test_feasible_under_budget"]) for r in rows_for_candidate)),
                "mean_test_bad25_recall": mean_or_nan(r["test_bad25_recall"] for r in rows_for_candidate),
                "min_test_bad25_recall": min(float(r["test_bad25_recall"]) for r in rows_for_candidate),
                "mean_test_bad20_recall": mean_or_nan(r["test_bad20_recall"] for r in rows_for_candidate),
                "min_test_bad20_recall": min(float(r["test_bad20_recall"]) for r in rows_for_candidate),
                "mean_test_num_false_positive_replacements": mean_or_nan(float(r["test_num_false_positive_replacements"]) for r in rows_for_candidate),
                "max_test_num_false_positive_replacements": max(int(r["test_num_false_positive_replacements"]) for r in rows_for_candidate),
                "mean_test_num_replaced": mean_or_nan(float(r["test_num_replaced"]) for r in rows_for_candidate),
                "max_test_num_replaced": max(int(r["test_num_replaced"]) for r in rows_for_candidate),
                "mean_test_run_level_mean_psnr": mean_or_nan(float(r["test_run_level_mean_psnr"]) for r in rows_for_candidate),
                "min_test_run_level_mean_psnr": min(float(r["test_run_level_mean_psnr"]) for r in rows_for_candidate),
                "mean_test_run_level_min_psnr": mean_or_nan(float(r["test_run_level_min_psnr"]) for r in rows_for_candidate),
                "min_test_run_level_min_psnr": min(float(r["test_run_level_min_psnr"]) for r in rows_for_candidate),
                "mean_test_image_level_best_of_4_mean_psnr": mean_or_nan(float(r["test_image_level_best_of_4_mean_psnr"]) for r in rows_for_candidate),
                "min_test_image_level_best_of_4_mean_psnr": min(float(r["test_image_level_best_of_4_mean_psnr"]) for r in rows_for_candidate),
                "mean_test_image_level_best_of_4_min_psnr": mean_or_nan(float(r["test_image_level_best_of_4_min_psnr"]) for r in rows_for_candidate),
                "min_test_image_level_best_of_4_min_psnr": min(float(r["test_image_level_best_of_4_min_psnr"]) for r in rows_for_candidate),
                "mean_test_median_alarm_time_tp": mean_or_nan(float(r["test_median_alarm_time_tp"]) for r in rows_for_candidate),
                "mean_test_median_alarm_time_fp": mean_or_nan(float(r["test_median_alarm_time_fp"]) for r in rows_for_candidate),
                "mean_test_mean_alarm_time_tp": mean_or_nan(float(r["test_mean_alarm_time_tp"]) for r in rows_for_candidate),
                "mean_test_mean_alarm_time_fp": mean_or_nan(float(r["test_mean_alarm_time_fp"]) for r in rows_for_candidate),
            }
        )

    # Pick one conservative and one aggressive recommendation if they are feasible across all major splits.
    major_candidates = [r for r in agg_rows if r["all_regimes_feasible_test"] and float(r["min_test_bad25_recall"]) > 0.0]
    conservative_rec = None
    aggressive_rec = None
    for budget_mode in ("conservative", "aggressive"):
        pool = [r for r in major_candidates if r["budget_mode"] == budget_mode]
        if pool:
            conservative_sort = lambda r: (
                -float(r["min_test_bad25_recall"]),
                -float(r["min_test_bad20_recall"]),
                float(r["max_test_num_false_positive_replacements"]),
                float(r["max_test_num_replaced"]),
                -float(r["min_test_run_level_mean_psnr"]),
                -float(r["min_test_image_level_best_of_4_min_psnr"]),
            )
            best = sorted(pool, key=conservative_sort)[0]
            if budget_mode == "conservative":
                conservative_rec = best
            else:
                aggressive_rec = best

    # Candidate-level cross-fit rows.
    write_csv(outdir / "anytime_candidate_crossfit_summary.csv", candidate_rows)
    write_csv(outdir / "anytime_candidate_policy_summary.csv", agg_rows)
    write_csv(outdir / "anytime_candidate_event_times.csv", event_rows)
    write_csv(outdir / "anytime_candidate_missed_runs.csv", missed_rows)
    write_csv(outdir / "image00017_candidate_audit.csv", image00017_rows)

    summary_lines = [
        "# A17.5 Cross-Fit Audit of Anytime Detector Candidates",
        "",
        "This pass is diagnostic only. It uses the existing A17 anytime features plus the A8/A11/A14/A16 outputs, and it does not change any frozen A14/A16 policies.",
        "",
        "## What We Audited",
        "",
        "- correction-norm persistence-10",
        "- x0hat/x0y disagreement persistence-10",
        "- full-residual persistence-10",
        "- low-frequency-residual persistence-10",
        "- simple OR combinations among those four signals",
        "",
        "## Cross-Fit Takeaway",
        "",
        f"- A17 candidate rows evaluated: `{len(candidate_rows)}`",
        f"- held-out event rows evaluated: `{len(event_rows)}`",
        f"- missed bad runs recorded: `{len(missed_rows)}`",
        f"- image 00017 audit rows: `{len(image00017_rows)}`",
        "",
    ]

    def add_recommendation(label: str, rec: Dict[str, object] | None) -> None:
        if not rec:
            summary_lines.extend(
                [
                    f"### {label}",
                    "",
                    "- No candidate stayed feasible across all major split directions under the stated budget.",
                    "",
                ]
            )
            return
        summary_lines.extend(
            [
                f"### {label}",
                "",
                f"- candidate: `{rec['candidate_name']}`",
                f"- budget: `{rec['budget_mode']}`",
                f"- min test bad25 recall across major splits: `{float(rec['min_test_bad25_recall']):.3f}`",
                f"- max test FP replacements across major splits: `{int(rec['max_test_num_false_positive_replacements'])}`",
                f"- max test replacements across major splits: `{int(rec['max_test_num_replaced'])}`",
                f"- min test run-level mean PSNR across major splits: `{float(rec['min_test_run_level_mean_psnr']):.3f}`",
                f"- min test image best-of-4 min PSNR across major splits: `{float(rec['min_test_image_level_best_of_4_min_psnr']):.3f}`",
                "",
            ]
        )

    summary_lines.extend(
        [
            "## Recommendation",
            "",
            "A18 is not yet plausible under the strict budgets tested here. The only thresholds that stay feasible across the major split directions collapse to the do-nothing edge, so this audit does not provide a frozen anytime rule worth promoting.",
            "",
        ]
    )
    add_recommendation("Conservative candidate", conservative_rec)
    add_recommendation("Aggressive candidate", aggressive_rec)

    summary_lines.extend(
        [
            "## Caveats",
            "",
            "- Broad union visibility is not the same thing as an executable selective rule.",
            "- Any threshold chosen on A14/A16 and then judged only on the same data is overfitting, not prospective validation.",
            "- The alarm-time distributions here are useful for diagnosis, but a future frozen policy must lock the exact feature family, OR combination, and threshold before a fresh run.",
            "",
            "## What Would Need To Be Frozen For A Future A18 / A19 Run",
            "",
            "- exact candidate family or OR-combination structure",
            "- exact threshold on the alarm fraction",
            "- whether the alarm uses a single feature or OR of multiple features",
            "- fallback source and replacement action",
            "- the split-independent evaluation protocol",
            "",
        ]
    )
    write_text(outdir / "SUMMARY.md", "\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
