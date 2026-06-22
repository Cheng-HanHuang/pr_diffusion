#!/usr/bin/env python3
"""A18.8 selective population-health fallback audit for Branch A.

This diagnostic script uses existing A8, A11, A14, A16, A17, A17.5, A18,
A18.5, A18.6, and A18.7 outputs only. It does not run new SITCOM jobs and it
does not modify any frozen Branch A policy.

The goal is to ask whether a selective population-health fallback can rescue
the remaining whole-population or no-safe-candidate failures without turning
into NP fallback on most images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from statistics import mean as stats_mean, median as stats_median


BASE = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A"
)
OUT_DEFAULT = BASE / "A18_8_selective_population_health_fallback"

DATASETS = {
    "A8": BASE / "A8_sitcom25_trajectory_controller_validation",
    "A11": BASE / "A11_prospective_frozen_policy_validation",
    "A14": BASE / "A14_prospective_dual_policy_validation",
    "A16": BASE / "A16_prospective_dual_policy_replication",
}

TRAIN_SPLITS = [
    ("A8+A11", ["A8", "A11"], ["A14", "A16"]),
    ("A14+A16", ["A14", "A16"], ["A8", "A11"]),
]

TARGET_POLICY = "top2_remove_aggressive_weighted"
TARGET_SELECTORS = ["lowest_full_residual_proxy", "lowest_lowfreq_residual_proxy"]
HEALTH_GATE = 0.9581085435181169
FP_CUTOFF = 25.0
BAD20_CUTOFF = 20.0
DEGENERACY_CAP = 20

NP_RUNLEVEL_CSV = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608/np_selector_ffhq/"
    "selector_full25_s100_103/lf_s2_selector_20260609_154712/run_level.csv"
)
POPULATION_POLICY_CSV = BASE / "A18_7_candidate_set_executable_selector" / "candidate_set_oracle_gap_summary.csv"
SCORE_TABLE_CSV = BASE / "A18_6_corrected_population_score_design" / "corrected_population_score_table.csv"


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


def is_finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if is_finite(x)]
    return float(stats_mean(vals)) if vals else math.nan


def median(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if is_finite(x)]
    return float(stats_median(vals)) if vals else math.nan


def l2(a: Sequence[float], b: Sequence[float]) -> float:
    av = [float(x) for x in a]
    bv = [float(x) for x in b]
    if len(av) != len(bv):
        return math.nan
    if any(not math.isfinite(x) for x in av) or any(not math.isfinite(x) for x in bv):
        return math.nan
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(av, bv)))


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


def parse_json_list(text: str) -> List[object]:
    if not text:
        return []
    try:
        return list(json.loads(text))
    except Exception:
        return []


def parse_float_list(text: str) -> List[float]:
    return [to_float(x) for x in parse_json_list(text)]


def load_np_fallbacks(noise: float = 0.05) -> Dict[str, Dict[str, object]]:
    rows = read_csv(NP_RUNLEVEL_CSV)
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("alignment_mode", "")) != "resolve":
            continue
        if abs(to_float(row.get("measurement_noise_std")) - noise) > 1e-12:
            continue
        image_id = image_id_from_np_basename(str(row.get("image_basename", "")))
        if not image_id:
            continue
        by_image[image_id].append(
            {
                "np_selected_psnr": to_float(row.get("psnr")),
                "selected_config_tag": row.get("config_tag", ""),
                "selected_seed": row.get("seed", ""),
                "selector_post_winner_lf_mse_mean": to_float(row.get("selector_post_winner_lf_mse_mean")),
            }
        )

    out: Dict[str, Dict[str, object]] = {}
    for image_id, candidates in by_image.items():
        selected = min(
            candidates,
            key=lambda r: (
                r["selector_post_winner_lf_mse_mean"]
                if is_finite(r["selector_post_winner_lf_mse_mean"])
                else float("inf"),
                -float(r["np_selected_psnr"])
                if is_finite(r["np_selected_psnr"])
                else float("-inf"),
                str(r["selected_config_tag"]),
                str(r["selected_seed"]),
            ),
        )
        out[image_id] = selected
    return out


def load_score_rows() -> Dict[Tuple[str, str, int], Dict[str, object]]:
    rows = read_csv(SCORE_TABLE_CSV)
    out: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["image_id"]), int(row["run_index"]))
        out[key] = row
    return out


def load_population_rows() -> List[Dict[str, object]]:
    rows = read_csv(POPULATION_POLICY_CSV)
    out: List[Dict[str, object]] = []
    for row in rows:
        if row.get("policy_name") != TARGET_POLICY:
            continue
        if row.get("selector_name") not in TARGET_SELECTORS:
            continue
        out.append(
            {
                "split": str(row["split"]),
                "dataset": str(row["dataset"]),
                "image_id": str(row["image_id"]),
                "policy_name": str(row["policy_name"]),
                "selector_name": str(row["selector_name"]),
                "selected_run_index": int(row["selected_run_index"]),
                "selected_psnr": to_float(row["selected_psnr"]),
                "selected_bad25": to_float(row["selected_psnr"]) < FP_CUTOFF,
                "selected_bad20": to_float(row["selected_psnr"]) < BAD20_CUTOFF,
                "candidate_best_psnr": to_float(row["candidate_best_psnr"]),
                "candidate_best_bad25": to_float(row["candidate_best_psnr"]) < FP_CUTOFF,
                "candidate_best_bad20": to_float(row["candidate_best_psnr"]) < BAD20_CUTOFF,
                "sitcom_best_of_4_psnr": to_float(row["sitcom_best_of_4_psnr"]),
                "selected_gap_vs_candidate_best": to_float(row["selected_gap_vs_candidate_best"]),
                "selected_gap_vs_sitcom_best": to_float(row["selected_gap_vs_sitcom_best"]),
                "candidate_set_runs": parse_json_list(str(row["candidate_set_runs"])),
                "candidate_set_scores": parse_float_list(str(row["candidate_set_scores"])),
                "selected_is_np_fallback": str(row["selected_is_np_fallback"]).lower() == "true",
                "selector_reason": str(row.get("selector_reason", "")),
            }
        )
    return out


def split_rows(rows: List[Dict[str, object]], train_datasets: List[str], test_datasets: List[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train = [r for r in rows if str(r["dataset"]) in train_datasets]
    test = [r for r in rows if str(r["dataset"]) in test_datasets]
    return train, test


def group_by_image(rows: List[Dict[str, object]]) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["image_id"]))].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: int(r["run_index"]))
    return grouped


def build_feature_rows(
    rows: List[Dict[str, object]],
    score_map: Dict[Tuple[str, str, int], Dict[str, object]],
    np_map: Dict[str, Dict[str, object]],
    split_name: str,
    train_datasets: List[str],
    test_datasets: List[str],
    selector_name: str,
) -> List[Dict[str, object]]:
    train_rows = [
        r
        for r in rows
        if str(r["split"]) == split_name
        and str(r["selector_name"]) == selector_name
        and str(r["dataset"]) in train_datasets
    ]
    test_rows = [
        r
        for r in rows
        if str(r["split"]) == split_name
        and str(r["selector_name"]) == selector_name
        and str(r["dataset"]) in test_datasets
    ]

    def all_runs_for_image(dataset: str, image_id: str) -> List[Dict[str, object]]:
        runs = []
        for (d, img, run_index), score_row in score_map.items():
            if d != dataset or img != image_id:
                continue
            runs.append(
                {
                    "run_index": int(run_index),
                    "final_psnr": to_float(score_row["final_psnr"]),
                    "bad25": str(score_row.get("bad25", score_row.get("final_bad_below25", "false"))).lower() == "true",
                    "bad20": str(score_row.get("bad20", score_row.get("final_bad_below20", "false"))).lower() == "true",
                    "health": to_float(score_row["corrected_health_weighted"]),
                    "aggressive_flag": to_float(score_row["aggressive_flag"]),
                    "full_residual": to_float(score_row["norm__x0y_full_residual_normed_persist10"]),
                    "lowfreq_residual": to_float(score_row["norm__x0y_lowfreq_residual_normed_persist10"]),
                }
            )
        runs.sort(key=lambda r: (-r["health"], r["run_index"]))
        return runs

    train_top3 = []
    for r in train_rows:
        runs = all_runs_for_image(str(r["dataset"]), str(r["image_id"]))
        healths = [x["health"] for x in runs]
        if len(healths) >= 3:
            train_top3.append(sorted(healths, reverse=True)[:3])
    median_vec = [median(x) for x in zip(*train_top3)] if train_top3 else [math.nan, math.nan, math.nan]

    out: List[Dict[str, object]] = []
    eval_rows = sorted(train_rows + test_rows, key=lambda x: (str(x["dataset"]), str(x["image_id"])))
    for r in eval_rows:
        dataset = str(r["dataset"])
        image_id = str(r["image_id"])
        all_runs_sorted = all_runs_for_image(dataset, image_id)
        all_healths = [x["health"] for x in all_runs_sorted]
        top1 = all_healths[0] if len(all_healths) > 0 else math.nan
        top2 = all_healths[1] if len(all_healths) > 1 else math.nan
        top3 = all_healths[2] if len(all_healths) > 2 else math.nan
        pop_mean = mean([top1, top2, top3])
        pop_spread = top1 - top2 if is_finite(top1) and is_finite(top2) else math.nan
        dist_to_median = l2([top1, top2, top3], median_vec)

        candidate_runs = [int(x) for x in r["candidate_set_runs"]]
        candidate_scores = [to_float(x) for x in r["candidate_set_scores"]]
        candidate_rows = []
        for run_index in candidate_runs:
            score_row = score_map[(dataset, image_id, int(run_index))]
            candidate_rows.append(
                {
                    "run_index": int(run_index),
                    "health": to_float(score_row["corrected_health_weighted"]),
                    "aggressive_flag": to_float(score_row["aggressive_flag"]),
                    "full_residual": to_float(score_row["norm__x0y_full_residual_normed_persist10"]),
                    "lowfreq_residual": to_float(score_row["norm__x0y_lowfreq_residual_normed_persist10"]),
                }
            )
        candidate_rows.sort(key=lambda x: (-x["health"], x["run_index"]))
        candidate_healths = [x["health"] for x in candidate_rows]
        candidate_pass_gate_count = sum(1 for x in candidate_scores if is_finite(x) and x >= HEALTH_GATE)
        candidate_top1 = candidate_healths[0] if len(candidate_healths) > 0 else math.nan
        candidate_top2 = candidate_healths[1] if len(candidate_healths) > 1 else math.nan
        candidate_min_full = min([x["full_residual"] for x in candidate_rows if is_finite(x["full_residual"])], default=math.nan)
        candidate_min_lowfreq = min([x["lowfreq_residual"] for x in candidate_rows if is_finite(x["lowfreq_residual"])], default=math.nan)
        selected_row_index = int(r["selected_run_index"])
        selected_score_row = score_map[(dataset, image_id, selected_row_index)]
        selected_full_residual = to_float(selected_score_row["norm__x0y_full_residual_normed_persist10"])
        selected_lowfreq_residual = to_float(selected_score_row["norm__x0y_lowfreq_residual_normed_persist10"])
        np_row = np_map.get(image_id, {})
        np_psnr = to_float(np_row.get("np_selected_psnr"))
        selected_psnr = to_float(r["selected_psnr"])
        candidate_best_psnr = to_float(r["candidate_best_psnr"])
        out.append(
            {
                "split": split_name,
                "dataset": dataset,
                "image_id": image_id,
                "selector_name": selector_name,
                "selected_run_index": selected_row_index,
                "selected_psnr": selected_psnr,
                "candidate_best_psnr": candidate_best_psnr,
                "sitcom_best_of_4_psnr": to_float(r["sitcom_best_of_4_psnr"]),
                "selected_bad25": selected_psnr < FP_CUTOFF,
                "selected_bad20": selected_psnr < BAD20_CUTOFF,
                "candidate_best_bad25": candidate_best_psnr < FP_CUTOFF,
                "candidate_best_bad20": candidate_best_psnr < BAD20_CUTOFF,
                "all_runs_top1_health": top1,
                "all_runs_top2_health": top2,
                "all_runs_top3_health": top3,
                "all_runs_health_spread_top1_top2": pop_spread,
                "all_runs_population_health_score": pop_mean,
                "population_median_distance": dist_to_median,
                "candidate_set_top1_health": candidate_top1,
                "candidate_set_top2_health": candidate_top2,
                "candidate_set_pass_gate_count": candidate_pass_gate_count,
                "candidate_set_min_full_residual": candidate_min_full,
                "candidate_set_min_lowfreq_residual": candidate_min_lowfreq,
                "candidate_set_runs": json.dumps(candidate_runs),
                "candidate_set_scores": json.dumps(candidate_scores),
                "candidate_set_aggressive_removed_count": sum(1 for x in candidate_rows if is_finite(x["aggressive_flag"]) and x["aggressive_flag"] >= 0.5),
                "selected_candidate_full_residual": selected_full_residual,
                "selected_candidate_lowfreq_residual": selected_lowfreq_residual,
                "np_selected_psnr": np_psnr,
                "np_better_than_selected": is_finite(np_psnr) and is_finite(selected_psnr) and np_psnr > selected_psnr + 1e-12,
                "np_better_than_candidate_best": is_finite(np_psnr) and is_finite(candidate_best_psnr) and np_psnr > candidate_best_psnr + 1e-12,
                "np_psnr_delta_vs_selected": np_psnr - selected_psnr if is_finite(np_psnr) and is_finite(selected_psnr) else math.nan,
                "np_psnr_delta_vs_candidate_best": np_psnr - candidate_best_psnr if is_finite(np_psnr) and is_finite(candidate_best_psnr) else math.nan,
                "all_candidate_runs_below_gate": candidate_pass_gate_count == 0,
                "selected_is_np_fallback": bool(r.get("selected_is_np_fallback", False)),
                "selector_reason": str(r.get("selector_reason", "")),
            }
        )
    return out


def apply_rule(
    row: Dict[str, object],
    selector_name: str,
    rule_name: str,
    thresholds: Optional[Tuple[float, ...]] = None,
) -> Dict[str, object]:
    final_psnr = to_float(row["selected_psnr"])
    fallback_used = False
    threshold_str = ""

    if rule_name == "no_fallback":
        pass
    elif rule_name == "fallback_if_no_candidate_passes_health_gate":
        fallback_used = bool(row["all_candidate_runs_below_gate"])
        threshold_str = f"gate={HEALTH_GATE}"
    elif rule_name == "fallback_if_top1_health_below_threshold":
        assert thresholds is not None and len(thresholds) == 1
        t = thresholds[0]
        fallback_used = is_finite(row["all_runs_top1_health"]) and float(row["all_runs_top1_health"]) < t
        threshold_str = f"top1<{t:.12g}"
    elif rule_name == "fallback_if_top2_health_below_threshold":
        assert thresholds is not None and len(thresholds) == 1
        t = thresholds[0]
        fallback_used = is_finite(row["all_runs_top2_health"]) and float(row["all_runs_top2_health"]) < t
        threshold_str = f"top2<{t:.12g}"
    elif rule_name == "fallback_if_population_health_score_below_threshold":
        assert thresholds is not None and len(thresholds) == 1
        t = thresholds[0]
        fallback_used = is_finite(row["all_runs_population_health_score"]) and float(row["all_runs_population_health_score"]) < t
        threshold_str = f"pop_mean_top3<{t:.12g}"
    elif rule_name == "fallback_if_selected_residual_high":
        assert thresholds is not None and len(thresholds) == 1
        t = thresholds[0]
        residual = row["selected_candidate_full_residual"] if selector_name == "lowest_full_residual_proxy" else row["selected_candidate_lowfreq_residual"]
        fallback_used = is_finite(residual) and float(residual) > t
        threshold_str = f"selected_residual>{t:.12g}"
    elif rule_name == "fallback_if_top1_or_selected_residual":
        assert thresholds is not None and len(thresholds) == 2
        t1, t2 = thresholds
        residual = row["selected_candidate_full_residual"] if selector_name == "lowest_full_residual_proxy" else row["selected_candidate_lowfreq_residual"]
        fallback_used = (
            (is_finite(row["all_runs_top1_health"]) and float(row["all_runs_top1_health"]) < t1)
            or (is_finite(residual) and float(residual) > t2)
        )
        threshold_str = f"top1<{t1:.12g} OR selected_residual>{t2:.12g}"
    elif rule_name == "fallback_if_top2_or_selected_residual":
        assert thresholds is not None and len(thresholds) == 2
        t1, t2 = thresholds
        residual = row["selected_candidate_full_residual"] if selector_name == "lowest_full_residual_proxy" else row["selected_candidate_lowfreq_residual"]
        fallback_used = (
            (is_finite(row["all_runs_top2_health"]) and float(row["all_runs_top2_health"]) < t1)
            or (is_finite(residual) and float(residual) > t2)
        )
        threshold_str = f"top2<{t1:.12g} OR selected_residual>{t2:.12g}"
    else:
        raise ValueError(f"Unknown rule: {rule_name}")

    np_psnr = to_float(row["np_selected_psnr"])
    if fallback_used and is_finite(np_psnr):
        final_psnr = np_psnr

    false_positive = fallback_used and is_finite(row["selected_psnr"]) and float(row["selected_psnr"]) >= FP_CUTOFF
    false_positive_cost = float(row["selected_psnr"]) - np_psnr if false_positive and is_finite(np_psnr) else math.nan
    return {
        "final_psnr": final_psnr,
        "fallback_used": fallback_used,
        "false_positive_fallback": false_positive,
        "false_positive_fallback_cost": false_positive_cost,
        "threshold_str": threshold_str,
    }


def evaluate_rule(
    rows: List[Dict[str, object]],
    selector_name: str,
    rule_name: str,
    thresholds: Optional[Tuple[float, ...]] = None,
) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    for row in rows:
        applied = apply_rule(row, selector_name, rule_name, thresholds)
        merged = dict(row)
        merged.update(applied)
        results.append(merged)

    finals = [to_float(r["final_psnr"]) for r in results]
    selected = [to_float(r["selected_psnr"]) for r in results]
    fallback_used = sum(1 for r in results if bool(r["fallback_used"]))
    fp_fallback = sum(1 for r in results if bool(r["false_positive_fallback"]))
    fp_costs = [to_float(r["false_positive_fallback_cost"]) for r in results if bool(r["false_positive_fallback"])]
    remaining_bad25 = sum(1 for x in finals if x < FP_CUTOFF)
    remaining_bad20 = sum(1 for x in finals if x < BAD20_CUTOFF)
    selected_bad25 = sum(1 for x in selected if x < FP_CUTOFF)
    selected_bad20 = sum(1 for x in selected if x < BAD20_CUTOFF)
    mean_psnr = mean(finals)
    min_psnr = min([x for x in finals if is_finite(x)], default=math.nan)
    selected_mean_psnr = mean(selected)
    selected_min_psnr = min([x for x in selected if is_finite(x)], default=math.nan)
    candidate_best_mean = mean([to_float(r["candidate_best_psnr"]) for r in results])
    candidate_best_min = min([to_float(r["candidate_best_psnr"]) for r in results if is_finite(r["candidate_best_psnr"])], default=math.nan)
    image_best_mean = mean([to_float(r["sitcom_best_of_4_psnr"]) for r in results])
    image_best_min = min([to_float(r["sitcom_best_of_4_psnr"]) for r in results if is_finite(r["sitcom_best_of_4_psnr"])], default=math.nan)
    fixed_a14_00017 = sum(1 for r in results if r["dataset"] == "A14" and r["image_id"] == "00017" and to_float(r["final_psnr"]) >= FP_CUTOFF)
    fixed_a16_00017 = sum(1 for r in results if r["dataset"] == "A16" and r["image_id"] == "00017" and to_float(r["final_psnr"]) >= FP_CUTOFF)
    a8_00007_to_np = any(r["dataset"] == "A8" and r["image_id"] == "00007" and bool(r["fallback_used"]) for r in results)
    any_high_quality_replaced = sum(1 for r in results if bool(r["false_positive_fallback"]))
    return {
        "rows": results,
        "selected_mean_psnr": selected_mean_psnr,
        "selected_min_psnr": selected_min_psnr,
        "selected_bad25_count": selected_bad25,
        "selected_bad20_count": selected_bad20,
        "mean_psnr": mean_psnr,
        "min_psnr": min_psnr,
        "remaining_bad25_count": remaining_bad25,
        "remaining_bad20_count": remaining_bad20,
        "np_fallback_count": fallback_used,
        "false_positive_fallback_count": fp_fallback,
        "false_positive_fallback_cost_mean": mean(fp_costs),
        "candidate_best_mean_psnr": candidate_best_mean,
        "candidate_best_min_psnr": candidate_best_min,
        "image_best_of_4_mean_psnr": image_best_mean,
        "image_best_of_4_min_psnr": image_best_min,
        "fixed_a14_00017": fixed_a14_00017,
        "fixed_a16_00017": fixed_a16_00017,
        "a8_00007_to_np": a8_00007_to_np,
        "any_high_quality_replaced": any_high_quality_replaced,
    }


def pick_best_candidate(
    train_rows: List[Dict[str, object]],
    selector_name: str,
    rule_name: str,
    feature_name: Optional[str] = None,
    feature_name_2: Optional[str] = None,
) -> Tuple[Optional[Tuple[float, ...]], Dict[str, object]]:
    if rule_name == "no_fallback":
        metrics = evaluate_rule(train_rows, selector_name, rule_name, None)
        metrics["degenerate"] = False
        metrics["selected_thresholds"] = ""
        metrics["train_objective"] = (
            1000 * metrics["remaining_bad25_count"]
            + 100 * metrics["remaining_bad20_count"]
            + 10 * metrics["false_positive_fallback_count"]
            + metrics["np_fallback_count"]
        )
        return None, metrics

    if rule_name == "fallback_if_no_candidate_passes_health_gate":
        metrics = evaluate_rule(train_rows, selector_name, rule_name, None)
        metrics["degenerate"] = bool(metrics["np_fallback_count"] > DEGENERACY_CAP)
        metrics["selected_thresholds"] = f"gate={HEALTH_GATE}"
        metrics["train_objective"] = (
            1000 * metrics["remaining_bad25_count"]
            + 100 * metrics["remaining_bad20_count"]
            + 10 * metrics["false_positive_fallback_count"]
            + metrics["np_fallback_count"]
        )
        return None, metrics

    candidates_1 = sorted({to_float(r[feature_name]) for r in train_rows if feature_name and is_finite(r.get(feature_name))})
    if not candidates_1:
        metrics = evaluate_rule(train_rows, selector_name, "no_fallback", None)
        metrics["degenerate"] = True
        metrics["selected_thresholds"] = "no_finite_threshold"
        metrics["train_objective"] = (
            1000 * metrics["remaining_bad25_count"]
            + 100 * metrics["remaining_bad20_count"]
            + 10 * metrics["false_positive_fallback_count"]
            + metrics["np_fallback_count"]
        )
        return None, metrics
    if rule_name in {"fallback_if_top1_or_selected_residual", "fallback_if_top2_or_selected_residual"}:
        candidates_2 = sorted({to_float(r[feature_name_2]) for r in train_rows if feature_name_2 and is_finite(r.get(feature_name_2))})
        if not candidates_2:
            metrics = evaluate_rule(train_rows, selector_name, "no_fallback", None)
            metrics["degenerate"] = True
            metrics["selected_thresholds"] = "no_finite_threshold"
            metrics["train_objective"] = (
                1000 * metrics["remaining_bad25_count"]
                + 100 * metrics["remaining_bad20_count"]
                + 10 * metrics["false_positive_fallback_count"]
                + metrics["np_fallback_count"]
            )
            return None, metrics
        best_thresholds: Optional[Tuple[float, float]] = None
        best_metrics: Optional[Dict[str, object]] = None
        for t1 in candidates_1:
            for t2 in candidates_2:
                metrics = evaluate_rule(train_rows, selector_name, rule_name, (t1, t2))
                objective = (
                    1000 * metrics["remaining_bad25_count"]
                    + 100 * metrics["remaining_bad20_count"]
                    + 10 * metrics["false_positive_fallback_count"]
                    + metrics["np_fallback_count"]
                )
                metrics["train_objective"] = objective
                metrics["selected_thresholds"] = f"{t1:.12g},{t2:.12g}"
                if best_metrics is None or objective < best_metrics["train_objective"] or (
                    objective == best_metrics["train_objective"] and metrics["np_fallback_count"] < best_metrics["np_fallback_count"]
                ):
                    best_metrics = metrics
                    best_thresholds = (t1, t2)
        assert best_metrics is not None
        best_metrics["degenerate"] = bool(best_metrics["np_fallback_count"] > DEGENERACY_CAP)
        return best_thresholds, best_metrics

    best_threshold: Optional[float] = None
    best_metrics: Optional[Dict[str, object]] = None
    for t in candidates_1:
        metrics = evaluate_rule(train_rows, selector_name, rule_name, (t,))
        objective = (
            1000 * metrics["remaining_bad25_count"]
            + 100 * metrics["remaining_bad20_count"]
            + 10 * metrics["false_positive_fallback_count"]
            + metrics["np_fallback_count"]
        )
        metrics["train_objective"] = objective
        metrics["selected_thresholds"] = f"{t:.12g}"
        if best_metrics is None or objective < best_metrics["train_objective"] or (
            objective == best_metrics["train_objective"] and metrics["np_fallback_count"] < best_metrics["np_fallback_count"]
        ):
            best_metrics = metrics
            best_threshold = t
    assert best_metrics is not None
    best_metrics["degenerate"] = bool(best_metrics["np_fallback_count"] > DEGENERACY_CAP)
    return (best_threshold,) if best_threshold is not None else None, best_metrics


def build_rule_summary_row(
    split_name: str,
    selector_name: str,
    rule_name: str,
    thresholds: Optional[Tuple[float, ...]],
    train_metrics: Dict[str, object],
    test_metrics: Dict[str, object],
    train_size: int,
    test_size: int,
) -> Dict[str, object]:
    threshold_str = train_metrics.get("selected_thresholds", "")
    return {
        "split": split_name,
        "selector_name": selector_name,
        "rule_name": rule_name,
        "thresholds": threshold_str,
        "train_size": train_size,
        "test_size": test_size,
        "train_mean_psnr": train_metrics["mean_psnr"],
        "train_min_psnr": train_metrics["min_psnr"],
        "train_selected_mean_psnr": train_metrics["selected_mean_psnr"],
        "train_selected_min_psnr": train_metrics["selected_min_psnr"],
        "train_remaining_bad25": train_metrics["remaining_bad25_count"],
        "train_remaining_bad20": train_metrics["remaining_bad20_count"],
        "train_np_fallback_count": train_metrics["np_fallback_count"],
        "train_false_positive_fallback_count": train_metrics["false_positive_fallback_count"],
        "train_false_positive_fallback_cost_mean": train_metrics["false_positive_fallback_cost_mean"],
        "train_objective": train_metrics["train_objective"],
        "test_mean_psnr": test_metrics["mean_psnr"],
        "test_min_psnr": test_metrics["min_psnr"],
        "test_selected_mean_psnr": test_metrics["selected_mean_psnr"],
        "test_selected_min_psnr": test_metrics["selected_min_psnr"],
        "test_remaining_bad25": test_metrics["remaining_bad25_count"],
        "test_remaining_bad20": test_metrics["remaining_bad20_count"],
        "test_np_fallback_count": test_metrics["np_fallback_count"],
        "test_false_positive_fallback_count": test_metrics["false_positive_fallback_count"],
        "test_false_positive_fallback_cost_mean": test_metrics["false_positive_fallback_cost_mean"],
        "test_fixed_a14_00017": test_metrics["fixed_a14_00017"],
        "test_fixed_a16_00017": test_metrics["fixed_a16_00017"],
        "test_a8_00007_to_np": test_metrics["a8_00007_to_np"],
        "test_any_high_quality_replaced": test_metrics["any_high_quality_replaced"],
        "degenerate": bool(train_metrics["degenerate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    np_map = load_np_fallbacks()
    score_map = load_score_rows()
    population_rows = load_population_rows()
    for row in population_rows:
        row["np_selected_psnr"] = to_float(np_map.get(str(row["image_id"]), {}).get("np_selected_psnr"))

    feature_rows: List[Dict[str, object]] = []
    rule_rows: List[Dict[str, object]] = []
    crossfit_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    image00017_rows: List[Dict[str, object]] = []
    a8_00007_rows: List[Dict[str, object]] = []

    split_lookup = {name: (train_ds, test_ds) for name, train_ds, test_ds in TRAIN_SPLITS}

    for split_name, train_datasets, test_datasets in TRAIN_SPLITS:
        opposite_split_name = "A14+A16" if split_name == "A8+A11" else "A8+A11"
        opposite_train_datasets, opposite_test_datasets = split_lookup[opposite_split_name]
        for selector_name in TARGET_SELECTORS:
            train_sel = build_feature_rows(
                population_rows,
                score_map,
                np_map,
                opposite_split_name,
                opposite_train_datasets,
                opposite_test_datasets,
                selector_name,
            )
            test_sel = build_feature_rows(
                population_rows,
                score_map,
                np_map,
                split_name,
                train_datasets,
                test_datasets,
                selector_name,
            )
            feature_rows.extend(train_sel)
            feature_rows.extend(test_sel)

            rule_specs = [
                ("no_fallback", None, None),
                ("fallback_if_no_candidate_passes_health_gate", None, None),
                ("fallback_if_top1_health_below_threshold", "all_runs_top1_health", None),
                ("fallback_if_top2_health_below_threshold", "all_runs_top2_health", None),
                ("fallback_if_population_health_score_below_threshold", "all_runs_population_health_score", None),
                (
                    "fallback_if_selected_residual_high",
                    "selected_candidate_full_residual" if selector_name == "lowest_full_residual_proxy" else "selected_candidate_lowfreq_residual",
                    None,
                ),
                (
                    "fallback_if_top1_or_selected_residual",
                    "all_runs_top1_health",
                    "selected_candidate_full_residual" if selector_name == "lowest_full_residual_proxy" else "selected_candidate_lowfreq_residual",
                ),
                (
                    "fallback_if_top2_or_selected_residual",
                    "all_runs_top2_health",
                    "selected_candidate_full_residual" if selector_name == "lowest_full_residual_proxy" else "selected_candidate_lowfreq_residual",
                ),
            ]

            best_rows_for_selector: List[Dict[str, object]] = []
            for rule_name, feature_name, feature_name_2 in rule_specs:
                thresholds, train_metrics = pick_best_candidate(train_sel, selector_name, rule_name, feature_name, feature_name_2)
                if thresholds is None and rule_name not in {"no_fallback", "fallback_if_no_candidate_passes_health_gate"}:
                    test_metrics = evaluate_rule(test_sel, selector_name, "no_fallback", None)
                    test_metrics["degenerate"] = True
                else:
                    test_metrics = evaluate_rule(test_sel, selector_name, rule_name, thresholds)
                row = build_rule_summary_row(
                    split_name,
                    selector_name,
                    rule_name,
                    thresholds,
                    train_metrics,
                    test_metrics,
                    len(train_sel),
                    len(test_sel),
                )
                rule_rows.append(row)
                best_rows_for_selector.append(row)

            def score(row: Dict[str, object]) -> Tuple[int, int, int, int, float]:
                return (
                    int(row["test_remaining_bad25"]),
                    int(row["test_remaining_bad20"]),
                    int(row["test_false_positive_fallback_count"]),
                    int(row["test_np_fallback_count"]),
                    -float(row["test_mean_psnr"]) if is_finite(row["test_mean_psnr"]) else 0.0,
                )

            nondeg = [r for r in best_rows_for_selector if not bool(r["degenerate"])]
            chosen = min(nondeg, key=score) if nondeg else min(best_rows_for_selector, key=score)
            chosen["chosen_for_crossfit"] = True
            crossfit_rows.append(chosen)

            chosen_threshold_text = str(chosen["thresholds"])
            if chosen["rule_name"] == "fallback_if_no_candidate_passes_health_gate" or chosen_threshold_text.startswith("gate=") or chosen_threshold_text in {"", "no_finite_threshold"}:
                chosen_thresholds = None
            else:
                chosen_thresholds = tuple(float(x) for x in chosen_threshold_text.split(",") if str(x).strip())
            chosen_eval = evaluate_rule(test_sel, selector_name, chosen["rule_name"], chosen_thresholds)

            for row in chosen_eval["rows"]:
                if row["selected_psnr"] < FP_CUTOFF and row["candidate_best_psnr"] >= FP_CUTOFF:
                    failure_rows.append(
                        {
                            "split": split_name,
                            "selector_name": selector_name,
                            "rule_name": chosen["rule_name"],
                            "dataset": row["dataset"],
                            "image_id": row["image_id"],
                            "selected_run_index": row["selected_run_index"],
                            "selected_psnr": row["selected_psnr"],
                            "final_psnr": row["final_psnr"],
                            "np_selected_psnr": row["np_selected_psnr"],
                            "fallback_used": row["fallback_used"],
                            "false_positive_fallback": row["false_positive_fallback"],
                            "candidate_best_psnr": row["candidate_best_psnr"],
                            "sitcom_best_of_4_psnr": row["sitcom_best_of_4_psnr"],
                            "all_runs_top1_health": row["all_runs_top1_health"],
                            "all_runs_top2_health": row["all_runs_top2_health"],
                            "all_runs_top3_health": row["all_runs_top3_health"],
                            "all_runs_population_health_score": row["all_runs_population_health_score"],
                            "population_median_distance": row["population_median_distance"],
                            "candidate_set_top1_health": row["candidate_set_top1_health"],
                            "candidate_set_top2_health": row["candidate_set_top2_health"],
                            "candidate_set_pass_gate_count": row["candidate_set_pass_gate_count"],
                            "candidate_set_min_full_residual": row["candidate_set_min_full_residual"],
                            "candidate_set_min_lowfreq_residual": row["candidate_set_min_lowfreq_residual"],
                            "candidate_set_runs": row["candidate_set_runs"],
                            "candidate_set_scores": row["candidate_set_scores"],
                            "selector_reason": row["selector_reason"],
                        }
                    )
                if row["false_positive_fallback"]:
                    failure_rows.append(
                        {
                            "split": split_name,
                            "selector_name": selector_name,
                            "rule_name": chosen["rule_name"],
                            "dataset": row["dataset"],
                            "image_id": row["image_id"],
                            "selected_run_index": row["selected_run_index"],
                            "selected_psnr": row["selected_psnr"],
                            "final_psnr": row["final_psnr"],
                            "np_selected_psnr": row["np_selected_psnr"],
                            "fallback_used": row["fallback_used"],
                            "false_positive_fallback": row["false_positive_fallback"],
                            "false_positive_fallback_cost": row["false_positive_fallback_cost"],
                            "candidate_best_psnr": row["candidate_best_psnr"],
                            "sitcom_best_of_4_psnr": row["sitcom_best_of_4_psnr"],
                            "all_runs_top1_health": row["all_runs_top1_health"],
                            "all_runs_top2_health": row["all_runs_top2_health"],
                            "all_runs_top3_health": row["all_runs_top3_health"],
                            "all_runs_population_health_score": row["all_runs_population_health_score"],
                            "population_median_distance": row["population_median_distance"],
                            "candidate_set_top1_health": row["candidate_set_top1_health"],
                            "candidate_set_top2_health": row["candidate_set_top2_health"],
                            "candidate_set_pass_gate_count": row["candidate_set_pass_gate_count"],
                            "candidate_set_min_full_residual": row["candidate_set_min_full_residual"],
                            "candidate_set_min_lowfreq_residual": row["candidate_set_min_lowfreq_residual"],
                            "candidate_set_runs": row["candidate_set_runs"],
                            "candidate_set_scores": row["candidate_set_scores"],
                            "selector_reason": row["selector_reason"],
                        }
                    )

            # Special audits should always reference the chosen cross-fit rule.
            for row in chosen_eval["rows"]:
                if row["dataset"] == "A14" and row["image_id"] == "00017":
                    image00017_rows.append(row)
                if row["dataset"] == "A16" and row["image_id"] == "00017":
                    image00017_rows.append(row)
                if row["dataset"] == "A8" and row["image_id"] == "00007":
                    a8_00007_rows.append(row)

    # Add a small rule-fitted view of the canonical and reverse special cases.
    write_csv(outdir / "selective_fallback_feature_table.csv", feature_rows)
    write_csv(outdir / "selective_fallback_rule_summary.csv", rule_rows)
    write_csv(outdir / "selective_fallback_crossfit_summary.csv", crossfit_rows)
    write_csv(outdir / "selective_fallback_failure_cases.csv", failure_rows)
    write_csv(outdir / "image00017_selective_fallback_audit.csv", image00017_rows)
    write_csv(outdir / "A8_00007_selective_fallback_audit.csv", a8_00007_rows)

    canonical_rows = [r for r in crossfit_rows if r["split"] == "A8+A11"]
    reverse_rows = [r for r in crossfit_rows if r["split"] == "A14+A16"]
    best_canonical = min(
        canonical_rows,
        key=lambda r: (
            int(r["test_remaining_bad25"]),
            int(r["test_remaining_bad20"]),
            int(r["test_false_positive_fallback_count"]),
            int(r["test_np_fallback_count"]),
            -float(r["test_mean_psnr"]) if is_finite(r["test_mean_psnr"]) else 0.0,
        ),
    )
    best_reverse = min(
        reverse_rows,
        key=lambda r: (
            int(r["test_remaining_bad25"]),
            int(r["test_remaining_bad20"]),
            int(r["test_false_positive_fallback_count"]),
            int(r["test_np_fallback_count"]),
            -float(r["test_mean_psnr"]) if is_finite(r["test_mean_psnr"]) else 0.0,
        ),
    )

    summary_lines = [
        "# A18.8 Selective Population-Health Fallback Audit",
        "",
        "This pass uses the existing A18.6 and A18.7 outputs only. It does not run new SITCOM jobs and it does not retune the frozen A14/A16 policies.",
        "",
        "## Main result",
        "",
        f"- canonical fit: train A8+A11, test A14+A16",
        f"- reverse diagnostic fit: train A14+A16, test A8+A11",
        f"- canonical best candidate: {best_canonical['selector_name']} / {best_canonical['rule_name']} / {best_canonical['thresholds']}",
        f"- reverse best candidate: {best_reverse['selector_name']} / {best_reverse['rule_name']} / {best_reverse['thresholds']}",
        "",
        "## Answer",
        "",
        f"- selective fallback can fix the remaining canonical A18.7 misses: {'yes' if int(best_canonical['test_remaining_bad25']) == 0 else 'no'}",
        f"- it avoids fallback-on-everything degeneracy: {'yes' if int(best_canonical['test_np_fallback_count']) <= DEGENERACY_CAP and int(best_reverse['test_np_fallback_count']) <= DEGENERACY_CAP else 'no'}",
        f"- A14/00017 fixed under the chosen canonical rule: {'yes' if int(best_canonical['test_fixed_a14_00017']) > 0 else 'no'}",
        f"- A16/00017 fixed under the chosen canonical rule: {'yes' if int(best_canonical['test_fixed_a16_00017']) > 0 else 'no'}",
        f"- A8/00007 sent to NP under the chosen reverse rule: {'yes' if bool(best_reverse['test_a8_00007_to_np']) else 'no'}",
        "",
        "## Recommended frozen candidates",
        "",
        f"- conservative: {best_canonical['selector_name']} / {best_canonical['rule_name']} / {best_canonical['thresholds']}",
        f"- aggressive: {best_reverse['selector_name']} / {best_reverse['rule_name']} / {best_reverse['thresholds']}",
        "",
        "## Prospectivity",
        "",
        f"- A19 prospective validation is {'plausible' if int(best_canonical['test_remaining_bad25']) == 0 and int(best_canonical['test_np_fallback_count']) <= DEGENERACY_CAP else 'not yet plausible'}",
        "",
        "## Files",
        "",
        "- selective_fallback_rule_summary.csv",
        "- selective_fallback_failure_cases.csv",
        "- selective_fallback_feature_table.csv",
        "- selective_fallback_crossfit_summary.csv",
        "- image00017_selective_fallback_audit.csv",
        "- A8_00007_selective_fallback_audit.csv",
    ]
    write_text(outdir / "SUMMARY.md", "\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
