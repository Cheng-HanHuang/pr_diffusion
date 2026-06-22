#!/usr/bin/env python3
"""A18.6 corrected population score design for Branch A.

This diagnostic script uses existing A8, A11, A14, A16, A17, A17.5, A18, and
A18.5 outputs only. It does not run new SITCOM jobs and does not change any
frozen Branch A policy.

The goal is to replace the saturated A18.5 population score with a continuous,
non-saturated clean-free health score and to audit whether a more reasonable
candidate-set rule looks stable across the A8/A11 versus A14/A16 splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BASE = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A"
)
OUT_DEFAULT = BASE / "A18_6_corrected_population_score_design"
NP_FALLBACK_CSV = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608/np_selector_ffhq/"
    "selector_full25_s100_103/lf_s2_selector_20260609_154712/run_level.csv"
)

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


RAW_FEATURE_SPECS = [
    ("any_alarm_frac", "A17", "any_alarm_frac", True),
    ("residual_any_alarm_frac", "A17", "residual_any_alarm_frac", True),
    (
        "x0y_full_residual_normed_persist10",
        "A17",
        "x0y_full_residual_normed__rank_ge3__persist10__first_step0",
        True,
    ),
    (
        "x0y_lowfreq_residual_normed_persist10",
        "A17",
        "x0y_lowfreq_residual_normed__rank_ge3__persist10__first_step0",
        True,
    ),
    (
        "x0hat_x0y_disagreement_persist10",
        "A17",
        "x0hat_x0y_disagreement__rank_ge3__persist10__first_step0",
        True,
    ),
    (
        "correction_norm_persist10",
        "A17",
        "correction_norm__rank_ge3__persist10__first_step0",
        True,
    ),
    (
        "x0y_step_jump_first",
        "A17",
        "x0y_step_jump__rank_ge3__first_step0",
        True,
    ),
    (
        "xt_step_jump_first",
        "A17",
        "xt_step_jump__rank_ge3__first_step0",
        True,
    ),
    (
        "x0y_full_residual_normed_slope20",
        "A17",
        "x0y_full_residual_normed__rank__rolling_slope_w20_gt0.05__first_step0",
        True,
    ),
    (
        "x0y_lowfreq_residual_normed_slope20",
        "A17",
        "x0y_lowfreq_residual_normed__rank__rolling_slope_w20_gt0.05__first_step0",
        True,
    ),
    ("lowfreq_nn_margin", "AGG", "lowfreq_nn_margin", True),
    ("residual_slope_margin", "AGG", "residual_slope_margin", True),
    ("residual_last_margin", "AGG", "residual_last_margin", True),
]

AGG_FEATURES = [
    "lowfreq_nn_margin",
    "residual_slope_margin",
    "residual_last_margin",
]


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
    return float(np.mean(vals)) if vals else math.nan


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if is_finite(s)]
    if not pairs:
        return math.nan
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return math.nan
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[order[j]][0] == pairs[order[i]][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y)
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if is_finite(s)]
    if not pairs:
        return math.nan
    pos = sum(y for _, y in pairs)
    if pos == 0:
        return math.nan
    pairs.sort(key=lambda t: (-t[0],))
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    for _, y in pairs:
        if y:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def minmax_norm(values: Sequence[float]) -> List[float]:
    finite = [float(v) for v in values if is_finite(v)]
    if not finite:
        return [math.nan for _ in values]
    lo = min(finite)
    hi = max(finite)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo + 1e-12:
        return [0.5 if is_finite(v) else math.nan for v in values]
    out = []
    for v in values:
        if not is_finite(v):
            out.append(math.nan)
        else:
            out.append((float(v) - lo) / (hi - lo))
    return out


def load_np_fallbacks(noise: float = 0.05) -> Dict[str, Dict[str, object]]:
    rows = read_csv(NP_FALLBACK_CSV)
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("alignment_mode", "")) != "resolve":
            continue
        if abs(to_float(row.get("measurement_noise_std")) - noise) > 1e-12:
            continue
        psnr = to_float(row.get("psnr"))
        if not math.isfinite(psnr):
            continue
        image_id = image_id_from_np_basename(str(row.get("image_basename", "")))
        by_image[image_id].append(
            {
                "np_selected_psnr": psnr,
                "config_tag": row.get("config_tag", ""),
                "seed": int(float(row.get("seed", 0) or 0)),
                "selector_post_winner_lf_mse_mean": to_float(row.get("selector_post_winner_lf_mse_mean")),
            }
        )
    out: Dict[str, Dict[str, object]] = {}
    for image_id, candidates in by_image.items():
        selected = min(
            candidates,
            key=lambda r: (
                r["selector_post_winner_lf_mse_mean"]
                if math.isfinite(float(r["selector_post_winner_lf_mse_mean"]))
                else float("inf"),
                -float(r["np_selected_psnr"]),
                str(r["config_tag"]),
                int(r["seed"]),
            ),
        )
        out[image_id] = selected
    return out


def load_run_rows(dataset: str, dataset_dir: Path) -> List[Dict[str, object]]:
    rows = read_csv(dataset_dir / "run_level_summary.csv")
    out = []
    for row in rows:
        out.append(
            {
                "dataset": dataset,
                "image_id": str(row["image_id"]),
                "run_index": int(row["run_index"]),
                "final_psnr": to_float(row["final_psnr"]),
                "bad25": str(row["final_bad_below25"]).lower() == "true",
                "bad20": str(row["final_bad_below20"]).lower() == "true",
            }
        )
    return out


def load_a17_features() -> Dict[Tuple[str, str, int], Dict[str, float]]:
    rows = read_csv(BASE / "A17_offline_anytime_detector_design" / "anytime_feature_table.csv")
    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["image_id"]), int(row["run_index"]))
        feat: Dict[str, float] = {}
        for dest_name, source_tag, source_col, _ in RAW_FEATURE_SPECS:
            if source_tag != "A17":
                continue
            feat[dest_name] = to_float(row.get(source_col))
        out[key] = feat
    return out


def load_aggressive_rows() -> Dict[Tuple[str, str, int], Dict[str, float]]:
    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for dataset, dataset_dir in DATASETS.items():
        path = dataset_dir / "aggressive_policy_applied_runs.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        for row in rows:
            key = (dataset, str(row["image_id"]), int(row["run_index"]))
            feat = {
                "lowfreq_nn_margin": to_float(row.get("consensus_threshold")) - to_float(row.get("consensus_feature_value")),
                "residual_slope_margin": to_float(row.get("x0y_full_residual_normed__interrun_rank__first80pct__slope__threshold"))
                - to_float(row.get("x0y_full_residual_normed__interrun_rank__first80pct__slope__value")),
                "residual_last_margin": to_float(row.get("x0y_full_residual_normed__interrun_rank__first80pct__last_in_window__threshold"))
                - to_float(row.get("x0y_full_residual_normed__interrun_rank__first80pct__last_in_window__value")),
                "aggressive_flag": 1.0 if str(row.get("was_flagged", "")).lower() == "true" else 0.0,
                "aggressive_replaced": 1.0 if str(row.get("was_replaced", "")).lower() == "true" else 0.0,
                "aggressive_tp": 1.0 if str(row.get("true_positive_replacement", "")).lower() == "true" else 0.0,
                "aggressive_fp": 1.0 if str(row.get("false_positive_replacement", "")).lower() == "true" else 0.0,
                "aggressive_missed": 1.0 if str(row.get("missed_catastrophic_run", "")).lower() == "true" else 0.0,
                "aggressive_residual_flag": 1.0 if str(row.get("residual_arm_flag", "")).lower() == "true" else 0.0,
                "aggressive_consensus_flag": 1.0 if str(row.get("consensus_arm_flag", "")).lower() == "true" else 0.0,
            }
            out[key] = feat
    return out


def merge_rows() -> List[Dict[str, object]]:
    a17 = load_a17_features()
    agg = load_aggressive_rows()
    rows: List[Dict[str, object]] = []
    for dataset, dataset_dir in DATASETS.items():
        for row in load_run_rows(dataset, dataset_dir):
            key = (dataset, row["image_id"], row["run_index"])
            merged = dict(row)
            merged.update(a17.get(key, {}))
            merged.update(agg.get(key, {}))
            rows.append(merged)
    return rows


def group_rows(rows: List[Dict[str, object]]) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["image_id"]))].append(row)
    for g in grouped.values():
        g.sort(key=lambda r: int(r["run_index"]))
    return grouped


def add_group_norm_features(rows: List[Dict[str, object]], feature_names: Sequence[str]) -> None:
    grouped = group_rows(rows)
    for feature in feature_names:
        for g in grouped.values():
            vals = [to_float(r.get(feature)) for r in g]
            norms = minmax_norm(vals)
            for r, v in zip(g, norms):
                r[f"norm__{feature}"] = v


def build_feature_sets(rows: List[Dict[str, object]]) -> None:
    feature_names = [name for name, _, _, _ in RAW_FEATURE_SPECS]
    add_group_norm_features(rows, feature_names)


def fit_feature_directions(train_rows: List[Dict[str, object]], feature_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    labels = [1 if bool(r["bad25"]) else 0 for r in train_rows]
    info: Dict[str, Dict[str, float]] = {}
    for feature in feature_names:
        vals = [to_float(r.get(f"norm__{feature}")) for r in train_rows]
        auc_val = auc(labels, vals)
        orientation = 1.0 if (math.isnan(auc_val) or auc_val < 0.5) else -1.0
        weight = max(abs((auc_val if math.isfinite(auc_val) else 0.5) - 0.5) * 2.0, 0.05)
        info[feature] = {
            "auc": auc_val,
            "orientation": orientation,
            "weight": weight,
        }
    return info


def score_rows(rows: List[Dict[str, object]], fit: Dict[str, Dict[str, float]], feature_names: Sequence[str]) -> None:
    for r in rows:
        comps = []
        wcomps = []
        for feature in feature_names:
            v = to_float(r.get(f"norm__{feature}"))
            orientation = fit[feature]["orientation"]
            weight = fit[feature]["weight"]
            if not math.isfinite(v):
                health = 1.0
            else:
                health = (1.0 - v) if orientation > 0 else v
            comps.append(health)
            wcomps.append((health, weight))
        r["corrected_health_equal"] = mean(comps)
        weighted_num = sum(h * w for h, w in wcomps if math.isfinite(h) and math.isfinite(w))
        weighted_den = sum(w for _, w in wcomps if math.isfinite(w))
        r["corrected_health_weighted"] = weighted_num / weighted_den if weighted_den > 0 else math.nan


def candidate_sort_key(row: Dict[str, object], tie_mode: str = "weighted") -> Tuple:
    health = to_float(row.get("corrected_health_weighted" if tie_mode == "weighted" else "corrected_health_equal"))
    consensus = to_float(row.get("norm__lowfreq_nn_margin"))
    residual = mean(
        [
            to_float(row.get("norm__residual_slope_margin")),
            to_float(row.get("norm__residual_last_margin")),
        ]
    )
    alarm = mean(
        [
            to_float(row.get("norm__any_alarm_frac")),
            to_float(row.get("norm__residual_any_alarm_frac")),
        ]
    )
    run_index = int(row["run_index"])
    # Higher health is better. Use the secondary metrics to break ties in a
    # deterministic, non-run-index-only way.
    return (
        -health if math.isfinite(health) else float("inf"),
        -consensus if math.isfinite(consensus) else float("inf"),
        -residual if math.isfinite(residual) else float("inf"),
        alarm if math.isfinite(alarm) else float("inf"),
        run_index,
    )


def select_topk(group: List[Dict[str, object]], score_field: str, k: int, tie_mode: str = "weighted") -> List[Dict[str, object]]:
    scored = sorted(group, key=lambda r: candidate_sort_key(r, tie_mode=tie_mode))
    return scored[: min(k, len(scored))]


def fit_threshold(train_groups: Dict[Tuple[str, str], List[Dict[str, object]]], score_field: str) -> float:
    candidates = sorted(
        {
            float(r[score_field])
            for g in train_groups.values()
            for r in g
            if math.isfinite(to_float(r.get(score_field)))
        }
    )
    if not candidates:
        return 0.5
    best = None
    best_key = None
    np_map = load_np_fallbacks()
    for tau in candidates:
        train_eval = evaluate_threshold_policy(train_groups, score_field, tau, np_map, use_np_if_none=True)
        key = (
            -train_eval["bad25_recall"],
            train_eval["false_positive_replacements"],
            train_eval["total_replacements"],
            -train_eval["image_best_of_4_mean_psnr"],
        )
        if best is None or key < best_key:
            best = tau
            best_key = key
    return float(best)


def evaluate_threshold_policy(
    groups: Dict[Tuple[str, str], List[Dict[str, object]]],
    score_field: str,
    tau: float,
    np_map: Dict[str, Dict[str, object]],
    use_np_if_none: bool = True,
) -> Dict[str, object]:
    image_best_psnrs = []
    candidate_best_psnrs = []
    selected_run_psnrs = []
    selected_count = 0
    np_used = 0
    bad25_recall_hits = 0
    bad25_total = 0
    bad20_total = 0
    false_positive_replacements = 0
    true_positive_replacements = 0
    total_replacements = 0
    remaining_bad25 = 0
    remaining_bad20 = 0
    images_with_good_candidate = 0
    images_with_any_selection = 0
    images_with_ties = 0
    images_all_runs_bad = 0
    worst_fp_loss = 0.0
    worst_remaining_miss = 0.0
    selected_sets: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for key, group in groups.items():
        image_id = key[1]
        selected = [r for r in sorted(group, key=lambda r: candidate_sort_key(r)) if to_float(r.get(score_field)) >= tau]
        selected = selected[:3]
        if len(selected) == 0 and use_np_if_none and image_id in np_map:
            np_used += 1
            selected_psnrs = [float(np_map[image_id]["np_selected_psnr"])]
        else:
            selected_psnrs = [to_float(r.get("final_psnr")) for r in selected]
        selected_sets[key] = selected
        if len(selected) > 0:
            images_with_any_selection += 1
        image_best = max([to_float(r.get("final_psnr")) for r in group] + [math.nan])
        candidate_best = max(selected_psnrs) if selected_psnrs else math.nan
        image_best_psnrs.append(image_best)
        candidate_best_psnrs.append(candidate_best)
        if any(to_float(r.get("final_psnr")) >= 25.0 for r in selected) or (
            len(selected) == 0 and use_np_if_none and image_id in np_map and float(np_map[image_id]["np_selected_psnr"]) >= 25.0
        ):
            images_with_good_candidate += 1
        if len({round(float(to_float(r.get(score_field))), 12) for r in selected}) < len(selected):
            images_with_ties += 1
        if all(to_float(r.get("final_psnr")) < 25.0 for r in group):
            images_all_runs_bad += 1

        for r in group:
            bad25 = bool(r["bad25"])
            bad20 = bool(r["bad20"])
            score = to_float(r.get(score_field))
            selected_here = any(
                int(r["run_index"]) == int(s["run_index"]) for s in selected
            )
            if selected_here:
                selected_count += 1
                selected_run_psnrs.append(to_float(r.get("final_psnr")))
            if bad25:
                bad25_total += 1
            if bad20:
                bad20_total += 1
            if bad25 and not selected_here:
                true_positive_replacements += 1
            if (not bad25) and selected_here:
                false_positive_replacements += 1
            if bad25 and selected_here:
                remaining_bad25 += 1
            if bad20 and selected_here:
                remaining_bad20 += 1
            if bad25 and not selected_here:
                total_replacements += 1
            if selected_here and (not bad25) and image_id in np_map:
                loss = max(0.0, float(r["final_psnr"]) - float(np_map[image_id]["np_selected_psnr"]))
                worst_fp_loss = max(worst_fp_loss, loss)
            if bad25 and selected_here:
                worst_remaining_miss = max(worst_remaining_miss, float(r["final_psnr"]))

    run_mean = mean(selected_run_psnrs)
    run_min = float(np.min([x for x in selected_run_psnrs if is_finite(x)])) if any(is_finite(x) for x in selected_run_psnrs) else math.nan
    image_best_mean = mean(image_best_psnrs)
    image_best_min = float(np.min([x for x in candidate_best_psnrs if is_finite(x)])) if any(is_finite(x) for x in candidate_best_psnrs) else math.nan
    recall = 1.0 - (remaining_bad25 / bad25_total) if bad25_total else math.nan
    return {
        "score_field": score_field,
        "threshold": tau,
        "selected_candidate_run_mean_psnr": run_mean,
        "selected_candidate_run_min_psnr": run_min,
        "image_best_of_4_mean_psnr": image_best_mean,
        "image_best_of_4_min_psnr": image_best_min,
        "bad25_total": bad25_total,
        "bad20_total": bad20_total,
        "remaining_bad25": remaining_bad25,
        "remaining_bad20": remaining_bad20,
        "bad25_recall": recall,
        "true_positive_replacements": true_positive_replacements,
        "false_positive_replacements": false_positive_replacements,
        "total_replacements": total_replacements,
        "images_with_any_selection": images_with_any_selection,
        "images_with_good_candidate": images_with_good_candidate,
        "images_with_ties": images_with_ties,
        "images_all_runs_bad": images_all_runs_bad,
        "np_used": np_used,
        "worst_fp_loss": worst_fp_loss,
        "worst_remaining_miss": worst_remaining_miss,
        "candidate_best_psnr_mean": mean(candidate_best_psnrs),
        "candidate_best_psnr_min": float(np.min([x for x in candidate_best_psnrs if is_finite(x)])) if any(is_finite(x) for x in candidate_best_psnrs) else math.nan,
        "selected_sets": selected_sets,
    }


def evaluate_topk_policy(
    groups: Dict[Tuple[str, str], List[Dict[str, object]]],
    score_field: str,
    k: int,
    np_map: Dict[str, Dict[str, object]],
    tie_mode: str = "weighted",
    remove_aggressive: bool = False,
    use_np_if_none: bool = False,
) -> Dict[str, object]:
    selected_run_psnrs = []
    candidate_best_psnrs = []
    image_best_psnrs = []
    image_best_candidates = []
    images_with_good_candidate = 0
    images_with_bad25_selected = 0
    images_with_bad20_selected = 0
    images_all_runs_bad = 0
    images_with_ties = 0
    np_used = 0
    selected_count = 0
    bad25_total = 0
    bad20_total = 0
    bad25_selected = 0
    bad20_selected = 0
    selected_sets: Dict[Tuple[str, str], List[Dict[str, object]]] = {}

    for key, group in groups.items():
        image_id = key[1]
        pool = group
        if remove_aggressive:
            non_flagged = [r for r in group if float(r.get("aggressive_flag", 0.0)) < 0.5]
            if len(non_flagged) >= k:
                pool = non_flagged
        selected = select_topk(pool, score_field, k, tie_mode=tie_mode)
        if remove_aggressive and len(selected) < k:
            remaining = [r for r in sorted(group, key=lambda r: candidate_sort_key(r, tie_mode=tie_mode)) if r not in selected]
            for r in remaining:
                if len(selected) >= k:
                    break
                selected.append(r)
            selected = sorted(selected, key=lambda r: candidate_sort_key(r, tie_mode=tie_mode))[:k]
        selected_sets[key] = selected
        if len({round(float(to_float(r.get(score_field))), 12) for r in selected}) < len(selected):
            images_with_ties += 1
        if all(to_float(r.get("final_psnr")) < 25.0 for r in group):
            images_all_runs_bad += 1
        image_best = max([to_float(r.get("final_psnr")) for r in group])
        image_best_psnrs.append(image_best)
        if selected:
            candidate_best = max(to_float(r.get("final_psnr")) for r in selected)
            candidate_best_psnrs.append(candidate_best)
            selected_run_psnrs.extend(to_float(r.get("final_psnr")) for r in selected)
            selected_count += len(selected)
            if any(to_float(r.get("final_psnr")) >= 25.0 for r in selected):
                images_with_good_candidate += 1
            if any(bool(r["bad25"]) for r in selected):
                images_with_bad25_selected += 1
            if any(bool(r["bad20"]) for r in selected):
                images_with_bad20_selected += 1
        else:
            if use_np_if_none and image_id in np_map:
                np_used += 1
                candidate_best_psnrs.append(float(np_map[image_id]["np_selected_psnr"]))
                selected_run_psnrs.append(float(np_map[image_id]["np_selected_psnr"]))
        for r in group:
            bad25_total += 1 if bool(r["bad25"]) else 0
            bad20_total += 1 if bool(r["bad20"]) else 0
            if bool(r["bad25"]) and any(int(r["run_index"]) == int(s["run_index"]) for s in selected):
                bad25_selected += 1
            if bool(r["bad20"]) and any(int(r["run_index"]) == int(s["run_index"]) for s in selected):
                bad20_selected += 1

    return {
        "selected_candidate_run_mean_psnr": mean(selected_run_psnrs),
        "selected_candidate_run_min_psnr": float(np.min([x for x in selected_run_psnrs if is_finite(x)])) if any(is_finite(x) for x in selected_run_psnrs) else math.nan,
        "image_best_of_4_mean_psnr": mean(image_best_psnrs),
        "image_best_of_4_min_psnr": float(np.min(image_best_psnrs)) if image_best_psnrs else math.nan,
        "candidate_best_psnr_mean": mean(candidate_best_psnrs),
        "candidate_best_psnr_min": float(np.min(candidate_best_psnrs)) if candidate_best_psnrs else math.nan,
        "images_with_good_candidate": images_with_good_candidate,
        "images_with_ties": images_with_ties,
        "images_all_runs_bad": images_all_runs_bad,
        "np_used": np_used,
        "selected_count": selected_count,
        "bad25_selected": bad25_selected,
        "bad20_selected": bad20_selected,
        "bad25_total": bad25_total,
        "bad20_total": bad20_total,
        "selected_sets": selected_sets,
    }


def make_score_table(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for r in rows:
        out.append(
            {
                "dataset": r["dataset"],
                "image_id": r["image_id"],
                "run_index": r["run_index"],
                "final_psnr": r["final_psnr"],
                "bad25": r["bad25"],
                "bad20": r["bad20"],
                "aggressive_flag": r.get("aggressive_flag", math.nan),
                "aggressive_replaced": r.get("aggressive_replaced", math.nan),
                "lowfreq_nn_margin": r.get("lowfreq_nn_margin", math.nan),
                "residual_slope_margin": r.get("residual_slope_margin", math.nan),
                "residual_last_margin": r.get("residual_last_margin", math.nan),
                "norm__lowfreq_nn_margin": r.get("norm__lowfreq_nn_margin", math.nan),
                "norm__residual_slope_margin": r.get("norm__residual_slope_margin", math.nan),
                "norm__residual_last_margin": r.get("norm__residual_last_margin", math.nan),
                "norm__any_alarm_frac": r.get("norm__any_alarm_frac", math.nan),
                "norm__residual_any_alarm_frac": r.get("norm__residual_any_alarm_frac", math.nan),
                "norm__x0y_full_residual_normed_persist10": r.get("norm__x0y_full_residual_normed_persist10", math.nan),
                "norm__x0y_lowfreq_residual_normed_persist10": r.get("norm__x0y_lowfreq_residual_normed_persist10", math.nan),
                "norm__x0hat_x0y_disagreement_persist10": r.get("norm__x0hat_x0y_disagreement_persist10", math.nan),
                "norm__correction_norm_persist10": r.get("norm__correction_norm_persist10", math.nan),
                "norm__x0y_step_jump_first": r.get("norm__x0y_step_jump_first", math.nan),
                "norm__xt_step_jump_first": r.get("norm__xt_step_jump_first", math.nan),
                "norm__x0y_full_residual_normed_slope20": r.get("norm__x0y_full_residual_normed_slope20", math.nan),
                "norm__x0y_lowfreq_residual_normed_slope20": r.get("norm__x0y_lowfreq_residual_normed_slope20", math.nan),
                "corrected_health_equal": r.get("corrected_health_equal", math.nan),
                "corrected_health_weighted": r.get("corrected_health_weighted", math.nan),
            }
        )
    return out


def split_rows(rows: List[Dict[str, object]], train_datasets: Sequence[str], test_datasets: Sequence[str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train = [r for r in rows if str(r["dataset"]) in train_datasets]
    test = [r for r in rows if str(r["dataset"]) in test_datasets]
    return train, test


def summarize_split(
    split_name: str,
    train_rows: List[Dict[str, object]],
    test_rows: List[Dict[str, object]],
    outdir: Path,
    np_map: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    feature_names = [name for name, _, _, _ in RAW_FEATURE_SPECS]
    fit = fit_feature_directions(train_rows, feature_names)
    score_rows(train_rows, fit, feature_names)
    score_rows(test_rows, fit, feature_names)

    # AUC table: raw feature and combined score diagnostics.
    auc_rows: List[Dict[str, object]] = []
    for dataset_label, rows in [("train", train_rows), ("test", test_rows)]:
        y25 = [1 if bool(r["bad25"]) else 0 for r in rows]
        y20 = [1 if bool(r["bad20"]) else 0 for r in rows]
        for feature in feature_names + ["corrected_health_equal", "corrected_health_weighted"]:
            vals = [to_float(r.get(f"norm__{feature}" if feature in feature_names else feature)) for r in rows]
            risk_scores = [1.0 - v if math.isfinite(v) else math.nan for v in vals]
            auc_rows.append(
                {
                    "split": split_name,
                    "subset": dataset_label,
                    "feature_name": feature,
                    "orientation": "high_is_healthy",
                    "n": len(rows),
                    "unique_score_count": len({round(float(v), 12) for v in risk_scores if is_finite(v)}),
                    "bad25_auroc": auc(y25, risk_scores),
                    "bad25_auprc": auprc(y25, risk_scores),
                    "bad20_auroc": auc(y20, risk_scores),
                    "bad20_auprc": auprc(y20, risk_scores),
                    "score_min": float(np.min([v for v in vals if is_finite(v)])) if any(is_finite(v) for v in vals) else math.nan,
                    "score_max": float(np.max([v for v in vals if is_finite(v)])) if any(is_finite(v) for v in vals) else math.nan,
                    "score_mean": mean(vals),
                }
            )

    # Candidate-set / population rule policies.
    train_groups = group_rows(train_rows)
    test_groups = group_rows(test_rows)

    # Main threshold is fitted on the weighted score from the train split.
    tau = fit_threshold(train_groups, "corrected_health_weighted")

    policies: List[Dict[str, object]] = []
    # Baselines.
    for label, groups in [("train", train_groups), ("test", test_groups)]:
        baselines = []
        np_best = np_map
        # SITCOM only.
        all_psnr = [to_float(r.get("final_psnr")) for g in groups.values() for r in g]
        image_best = [max(to_float(r.get("final_psnr")) for r in g) for g in groups.values()]
        bad25 = sum(1 for g in groups.values() for r in g if bool(r["bad25"]))
        bad20 = sum(1 for g in groups.values() for r in g if bool(r["bad20"]))
        baselines.append(
            {
                "split": split_name,
                "subset": label,
                "policy_name": "sitcom_only",
                "score_name": "none",
                "threshold": math.nan,
                "top_k": math.nan,
                "use_np_fallback": False,
                "selected_candidate_run_mean_psnr": mean(all_psnr),
                "selected_candidate_run_min_psnr": float(np.min([x for x in all_psnr if is_finite(x)])) if any(is_finite(x) for x in all_psnr) else math.nan,
                "image_best_of_4_mean_psnr": mean(image_best),
                "image_best_of_4_min_psnr": float(np.min(image_best)) if image_best else math.nan,
                "bad25_count": bad25,
                "bad20_count": bad20,
                "images_with_good_candidate": len(groups),
                "images_with_any_selection": len(groups),
                "images_with_ties": 0,
                "images_all_runs_bad": sum(1 for g in groups.values() if all(to_float(r.get("final_psnr")) < 25.0 for r in g)),
                "np_used": 0,
                "selected_count": len(all_psnr),
                "candidate_best_psnr_mean": mean(image_best),
                "candidate_best_psnr_min": float(np.min(image_best)) if image_best else math.nan,
                "true_positive_replacements": math.nan,
                "false_positive_replacements": math.nan,
                "total_replacements": math.nan,
                "remaining_bad25": bad25,
                "remaining_bad20": bad20,
                "bad25_recall": math.nan,
                "worst_fp_loss": math.nan,
                "worst_remaining_miss": math.nan,
            }
        )
        # Oracle upper bound: best-of-4 on the selected SITCOM population.
        baselines.append(
            {
                "split": split_name,
                "subset": label,
                "policy_name": "oracle_best_of_4",
                "score_name": "oracle",
                "threshold": math.nan,
                "top_k": math.nan,
                "use_np_fallback": False,
                "selected_candidate_run_mean_psnr": mean([max(to_float(r.get("final_psnr")) for r in g) for g in groups.values()]),
                "selected_candidate_run_min_psnr": float(np.min([max(to_float(r.get("final_psnr")) for r in g) for g in groups.values()])),
                "image_best_of_4_mean_psnr": mean(image_best),
                "image_best_of_4_min_psnr": float(np.min(image_best)) if image_best else math.nan,
                "bad25_count": bad25,
                "bad20_count": bad20,
                "images_with_good_candidate": len(groups),
                "images_with_any_selection": len(groups),
                "images_with_ties": 0,
                "images_all_runs_bad": sum(1 for g in groups.values() if all(to_float(r.get("final_psnr")) < 25.0 for r in g)),
                "np_used": 0,
                "selected_count": len(groups),
                "candidate_best_psnr_mean": mean(image_best),
                "candidate_best_psnr_min": float(np.min(image_best)) if image_best else math.nan,
                "true_positive_replacements": math.nan,
                "false_positive_replacements": math.nan,
                "total_replacements": math.nan,
                "remaining_bad25": 0,
                "remaining_bad20": 0,
                "bad25_recall": 1.0,
                "worst_fp_loss": math.nan,
                "worst_remaining_miss": 0.0,
            }
        )
        policies.extend(baselines)

        # Top-k policies on weighted score.
        for k in [1, 2, 3]:
            res = evaluate_topk_policy(groups, "corrected_health_weighted", k, np_best, tie_mode="weighted")
            policies.append(
                {
                    "split": split_name,
                    "subset": label,
                    "policy_name": f"top{k}_weighted",
                    "score_name": "corrected_health_weighted",
                    "threshold": math.nan,
                    "top_k": k,
                    "use_np_fallback": False,
                    "selected_candidate_run_mean_psnr": res["selected_candidate_run_mean_psnr"],
                    "selected_candidate_run_min_psnr": res["selected_candidate_run_min_psnr"],
                    "image_best_of_4_mean_psnr": res["image_best_of_4_mean_psnr"],
                    "image_best_of_4_min_psnr": res["image_best_of_4_min_psnr"],
                    "bad25_count": sum(1 for g in groups.values() for r in g if bool(r["bad25"])),
                    "bad20_count": sum(1 for g in groups.values() for r in g if bool(r["bad20"])),
                    "images_with_good_candidate": res["images_with_good_candidate"],
                    "images_with_any_selection": len(groups),
                    "images_with_ties": res["images_with_ties"],
                    "images_all_runs_bad": res["images_all_runs_bad"],
                    "np_used": res["np_used"],
                    "selected_count": res["selected_count"],
                    "candidate_best_psnr_mean": res["candidate_best_psnr_mean"],
                    "candidate_best_psnr_min": res["candidate_best_psnr_min"],
                    "true_positive_replacements": math.nan,
                    "false_positive_replacements": math.nan,
                    "total_replacements": math.nan,
                    "remaining_bad25": math.nan,
                    "remaining_bad20": math.nan,
                    "bad25_recall": math.nan,
                    "worst_fp_loss": math.nan,
                    "worst_remaining_miss": math.nan,
                }
            )

        # Deterministic tie-break alternative.
        res_tie = evaluate_topk_policy(groups, "corrected_health_weighted", 2, np_best, tie_mode="weighted")
        policies.append(
            {
                "split": split_name,
                "subset": label,
                "policy_name": "top2_weighted_tiebreak_consensus_residual",
                "score_name": "corrected_health_weighted",
                "threshold": math.nan,
                "top_k": 2,
                "use_np_fallback": False,
                "selected_candidate_run_mean_psnr": res_tie["selected_candidate_run_mean_psnr"],
                "selected_candidate_run_min_psnr": res_tie["selected_candidate_run_min_psnr"],
                "image_best_of_4_mean_psnr": res_tie["image_best_of_4_mean_psnr"],
                "image_best_of_4_min_psnr": res_tie["image_best_of_4_min_psnr"],
                "bad25_count": sum(1 for g in groups.values() for r in g if bool(r["bad25"])),
                "bad20_count": sum(1 for g in groups.values() for r in g if bool(r["bad20"])),
                "images_with_good_candidate": res_tie["images_with_good_candidate"],
                "images_with_any_selection": len(groups),
                "images_with_ties": res_tie["images_with_ties"],
                "images_all_runs_bad": res_tie["images_all_runs_bad"],
                "np_used": res_tie["np_used"],
                "selected_count": res_tie["selected_count"],
                "candidate_best_psnr_mean": res_tie["candidate_best_psnr_mean"],
                "candidate_best_psnr_min": res_tie["candidate_best_psnr_min"],
                "true_positive_replacements": math.nan,
                "false_positive_replacements": math.nan,
                "total_replacements": math.nan,
                "remaining_bad25": math.nan,
                "remaining_bad20": math.nan,
                "bad25_recall": math.nan,
                "worst_fp_loss": math.nan,
                "worst_remaining_miss": math.nan,
            }
        )

        # Threshold-cap policy and threshold + NP fallback policy.
        threshold_res = evaluate_threshold_policy(groups, "corrected_health_weighted", tau, np_best, use_np_if_none=True)
        policies.append(
            {
                "split": split_name,
                "subset": label,
                "policy_name": "threshold_cap3_weighted",
                "score_name": "corrected_health_weighted",
                "threshold": tau,
                "top_k": 3,
                "use_np_fallback": True,
                "selected_candidate_run_mean_psnr": threshold_res["selected_candidate_run_mean_psnr"],
                "selected_candidate_run_min_psnr": threshold_res["selected_candidate_run_min_psnr"],
                "image_best_of_4_mean_psnr": threshold_res["image_best_of_4_mean_psnr"],
                "image_best_of_4_min_psnr": threshold_res["image_best_of_4_min_psnr"],
                "bad25_count": threshold_res["bad25_total"],
                "bad20_count": threshold_res["bad20_total"],
                "images_with_good_candidate": threshold_res["images_with_good_candidate"],
                "images_with_any_selection": threshold_res["images_with_any_selection"],
                "images_with_ties": threshold_res["images_with_ties"],
                "images_all_runs_bad": threshold_res["images_all_runs_bad"],
                "np_used": threshold_res["np_used"],
                "selected_count": len([1 for g in threshold_res["selected_sets"].values() for _ in g]),
                "candidate_best_psnr_mean": threshold_res["candidate_best_psnr_mean"],
                "candidate_best_psnr_min": threshold_res["candidate_best_psnr_min"],
                "true_positive_replacements": threshold_res["true_positive_replacements"],
                "false_positive_replacements": threshold_res["false_positive_replacements"],
                "total_replacements": threshold_res["total_replacements"],
                "remaining_bad25": threshold_res["remaining_bad25"],
                "remaining_bad20": threshold_res["remaining_bad20"],
                "bad25_recall": threshold_res["bad25_recall"],
                "worst_fp_loss": threshold_res["worst_fp_loss"],
                "worst_remaining_miss": threshold_res["worst_remaining_miss"],
            }
        )

        np_res = evaluate_topk_policy(groups, "corrected_health_weighted", 2, np_best, tie_mode="weighted", remove_aggressive=True, use_np_if_none=True)
        policies.append(
            {
                "split": split_name,
                "subset": label,
                "policy_name": "top2_remove_aggressive_weighted",
                "score_name": "corrected_health_weighted",
                "threshold": math.nan,
                "top_k": 2,
                "use_np_fallback": True,
                "selected_candidate_run_mean_psnr": np_res["selected_candidate_run_mean_psnr"],
                "selected_candidate_run_min_psnr": np_res["selected_candidate_run_min_psnr"],
                "image_best_of_4_mean_psnr": np_res["image_best_of_4_mean_psnr"],
                "image_best_of_4_min_psnr": np_res["image_best_of_4_min_psnr"],
                "bad25_count": np_res["bad25_total"],
                "bad20_count": np_res["bad20_total"],
                "images_with_good_candidate": np_res["images_with_good_candidate"],
                "images_with_any_selection": len(groups),
                "images_with_ties": np_res["images_with_ties"],
                "images_all_runs_bad": np_res["images_all_runs_bad"],
                "np_used": np_res["np_used"],
                "selected_count": np_res["selected_count"],
                "candidate_best_psnr_mean": np_res["candidate_best_psnr_mean"],
                "candidate_best_psnr_min": np_res["candidate_best_psnr_min"],
                "true_positive_replacements": math.nan,
                "false_positive_replacements": math.nan,
                "total_replacements": math.nan,
                "remaining_bad25": math.nan,
                "remaining_bad20": math.nan,
                "bad25_recall": math.nan,
                "worst_fp_loss": math.nan,
                "worst_remaining_miss": math.nan,
            }
        )

        # Threshold-based top2 plus NP fallback policy.
        top2_np_res = evaluate_threshold_policy(groups, "corrected_health_weighted", tau, np_best, use_np_if_none=True)
        policies.append(
            {
                "split": split_name,
                "subset": label,
                "policy_name": "top2_plus_np_if_unhealthy_weighted",
                "score_name": "corrected_health_weighted",
                "threshold": tau,
                "top_k": 2,
                "use_np_fallback": True,
                "selected_candidate_run_mean_psnr": top2_np_res["selected_candidate_run_mean_psnr"],
                "selected_candidate_run_min_psnr": top2_np_res["selected_candidate_run_min_psnr"],
                "image_best_of_4_mean_psnr": top2_np_res["image_best_of_4_mean_psnr"],
                "image_best_of_4_min_psnr": top2_np_res["image_best_of_4_min_psnr"],
                "bad25_count": top2_np_res["bad25_total"],
                "bad20_count": top2_np_res["bad20_total"],
                "images_with_good_candidate": top2_np_res["images_with_good_candidate"],
                "images_with_any_selection": top2_np_res["images_with_any_selection"],
                "images_with_ties": top2_np_res["images_with_ties"],
                "images_all_runs_bad": top2_np_res["images_all_runs_bad"],
                "np_used": top2_np_res["np_used"],
                "selected_count": len([1 for g in top2_np_res["selected_sets"].values() for _ in g]),
                "candidate_best_psnr_mean": top2_np_res["candidate_best_psnr_mean"],
                "candidate_best_psnr_min": top2_np_res["candidate_best_psnr_min"],
                "true_positive_replacements": top2_np_res["true_positive_replacements"],
                "false_positive_replacements": top2_np_res["false_positive_replacements"],
                "total_replacements": top2_np_res["total_replacements"],
                "remaining_bad25": top2_np_res["remaining_bad25"],
                "remaining_bad20": top2_np_res["remaining_bad20"],
                "bad25_recall": top2_np_res["bad25_recall"],
                "worst_fp_loss": top2_np_res["worst_fp_loss"],
                "worst_remaining_miss": top2_np_res["worst_remaining_miss"],
            }
        )

    return auc_rows, policies, fit


def build_image_audit(rows: List[Dict[str, object]], dataset: str, image_id: str) -> List[Dict[str, object]]:
    sel = [r for r in rows if str(r["dataset"]) == dataset and str(r["image_id"]) == image_id]
    sel.sort(key=lambda r: candidate_sort_key(r))
    max_score = max(to_float(r.get("corrected_health_weighted")) for r in sel)
    min_score = min(to_float(r.get("corrected_health_weighted")) for r in sel)
    out = []
    for rank, r in enumerate(sel, start=1):
        out.append(
            {
                "dataset": dataset,
                "image_id": image_id,
                "run_index": r["run_index"],
                "rank_by_corrected_health": rank,
                "final_psnr": r["final_psnr"],
                "bad25": r["bad25"],
                "bad20": r["bad20"],
                "corrected_health_equal": r.get("corrected_health_equal", math.nan),
                "corrected_health_weighted": r.get("corrected_health_weighted", math.nan),
                "health_margin_to_best_run": max_score - to_float(r.get("corrected_health_weighted")),
                "health_margin_to_worst_run": to_float(r.get("corrected_health_weighted")) - min_score,
                "lowfreq_nn_margin": r.get("lowfreq_nn_margin", math.nan),
                "residual_slope_margin": r.get("residual_slope_margin", math.nan),
                "residual_last_margin": r.get("residual_last_margin", math.nan),
                "any_alarm_frac": r.get("any_alarm_frac", math.nan),
                "residual_any_alarm_frac": r.get("residual_any_alarm_frac", math.nan),
                "x0y_full_residual_normed_persist10": r.get("x0y_full_residual_normed_persist10", math.nan),
                "x0y_lowfreq_residual_normed_persist10": r.get("x0y_lowfreq_residual_normed_persist10", math.nan),
                "x0hat_x0y_disagreement_persist10": r.get("x0hat_x0y_disagreement_persist10", math.nan),
                "correction_norm_persist10": r.get("correction_norm_persist10", math.nan),
                "x0y_step_jump_first": r.get("x0y_step_jump_first", math.nan),
                "xt_step_jump_first": r.get("xt_step_jump_first", math.nan),
                "x0y_full_residual_normed_slope20": r.get("x0y_full_residual_normed_slope20", math.nan),
                "x0y_lowfreq_residual_normed_slope20": r.get("x0y_lowfreq_residual_normed_slope20", math.nan),
                "aggressive_flag": r.get("aggressive_flag", math.nan),
            }
        )
    return out


def build_failure_rows(
    rows: List[Dict[str, object]],
    candidate_sets: Dict[Tuple[str, str, str], List[Dict[str, object]]],
    np_map: Dict[str, Dict[str, object]],
    score_field: str,
    threshold: Optional[float] = None,
) -> List[Dict[str, object]]:
    out = []
    for key, selected in candidate_sets.items():
        dataset, image_id, policy_name = key
        group = [r for r in rows if str(r["dataset"]) == dataset and str(r["image_id"]) == image_id]
        best_sitcom = max(to_float(r.get("final_psnr")) for r in group)
        selected_psnrs = [to_float(r.get("final_psnr")) for r in selected]
        candidate_best = max(selected_psnrs) if selected_psnrs else (np_map.get(image_id, {}).get("np_selected_psnr", math.nan) if image_id in np_map else math.nan)
        contains_good = any(to_float(r.get("final_psnr")) >= 25.0 for r in selected)
        contains_bad25 = any(bool(r["bad25"]) for r in selected)
        contains_bad20 = any(bool(r["bad20"]) for r in selected)
        if candidate_best < 25.0 and best_sitcom > 29.0:
            out.append(
                {
                    "dataset": dataset,
                    "image_id": image_id,
                    "policy_name": policy_name,
                    "selected_runs": json.dumps([int(r["run_index"]) for r in selected]),
                    "selected_scores": json.dumps([to_float(r.get(score_field)) for r in selected]),
                    "all_four_final_psnrs_json": json.dumps([to_float(r.get("final_psnr")) for r in group]),
                    "candidate_best_psnr": candidate_best,
                    "sitcom_best_of_4_psnr": best_sitcom,
                    "delta": candidate_best - best_sitcom,
                    "good_sitcom_existed_but_not_selected": not contains_good and best_sitcom >= 25.0,
                    "contains_bad25_selected": contains_bad25,
                    "contains_bad20_selected": contains_bad20,
                    "np_needed": candidate_best < 25.0,
                    "threshold": threshold if threshold is not None else math.nan,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    np_map = load_np_fallbacks()
    rows = merge_rows()
    build_feature_sets(rows)

    # Keep a copy of the full score table for inspection.
    score_table_rows = make_score_table(rows)

    all_auc_rows: List[Dict[str, object]] = []
    all_policy_rows: List[Dict[str, object]] = []
    all_crossfit_rows: List[Dict[str, object]] = []
    all_failure_rows: List[Dict[str, object]] = []
    all_image_audit_rows: List[Dict[str, object]] = []
    threshold_records: List[Dict[str, object]] = []

    for split_name, train_datasets, test_datasets in TRAIN_SPLITS:
        train_rows, test_rows = split_rows(rows, train_datasets, test_datasets)
        auc_rows, policy_rows, fit = summarize_split(split_name, train_rows, test_rows, outdir, np_map)
        all_auc_rows.extend(auc_rows)
        all_policy_rows.extend(policy_rows)

        # Cross-fit summary table: keep a compact view of the weighted score fit.
        train_groups = group_rows(train_rows)
        test_groups = group_rows(test_rows)
        tau = fit_threshold(train_groups, "corrected_health_weighted")
        train_eval = evaluate_threshold_policy(train_groups, "corrected_health_weighted", tau, np_map, use_np_if_none=True)
        test_eval = evaluate_threshold_policy(test_groups, "corrected_health_weighted", tau, np_map, use_np_if_none=True)
        all_crossfit_rows.append(
            {
                "split": split_name,
                "score_name": "corrected_health_weighted",
                "threshold": tau,
                "train_bad25_recall": train_eval["bad25_recall"],
                "train_false_positive_replacements": train_eval["false_positive_replacements"],
                "train_total_replacements": train_eval["total_replacements"],
                "train_image_best_of_4_mean_psnr": train_eval["image_best_of_4_mean_psnr"],
                "train_image_best_of_4_min_psnr": train_eval["image_best_of_4_min_psnr"],
                "train_candidate_best_psnr_mean": train_eval["candidate_best_psnr_mean"],
                "train_candidate_best_psnr_min": train_eval["candidate_best_psnr_min"],
                "test_bad25_recall": test_eval["bad25_recall"],
                "test_false_positive_replacements": test_eval["false_positive_replacements"],
                "test_total_replacements": test_eval["total_replacements"],
                "test_image_best_of_4_mean_psnr": test_eval["image_best_of_4_mean_psnr"],
                "test_image_best_of_4_min_psnr": test_eval["image_best_of_4_min_psnr"],
                "test_candidate_best_psnr_mean": test_eval["candidate_best_psnr_mean"],
                "test_candidate_best_psnr_min": test_eval["candidate_best_psnr_min"],
                "test_remaining_bad25": test_eval["remaining_bad25"],
                "test_remaining_bad20": test_eval["remaining_bad20"],
                "test_np_used": test_eval["np_used"],
                "test_images_with_good_candidate": test_eval["images_with_good_candidate"],
                "test_images_all_runs_bad": test_eval["images_all_runs_bad"],
            }
        )

        threshold_records.append(
            {
                "split": split_name,
                "threshold": tau,
                "train_bad25_recall": train_eval["bad25_recall"],
                "train_fp": train_eval["false_positive_replacements"],
                "train_total_replacements": train_eval["total_replacements"],
                "test_bad25_recall": test_eval["bad25_recall"],
                "test_fp": test_eval["false_positive_replacements"],
                "test_total_replacements": test_eval["total_replacements"],
            }
        )

        # Build image audits for the key problem cases.
        for dataset in train_datasets + test_datasets:
            if dataset not in {"A8", "A11", "A14", "A16"}:
                continue
            for image_id in ["00017"] if dataset in {"A14", "A16"} else (["00007"] if dataset == "A8" else []):
                all_image_audit_rows.extend(build_image_audit(rows, dataset, image_id))

        # Failure rows for the key selected policies.
        for policy_name in [
            "top2_weighted",
            "top3_weighted",
            "threshold_cap3_weighted",
            "top2_plus_np_if_unhealthy_weighted",
            "top2_remove_aggressive_weighted",
            "top2_weighted_tiebreak_consensus_residual",
        ]:
            selected_sets: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
            for row in [r for r in all_policy_rows if r["split"] == split_name and r["subset"] == "test" and r["policy_name"] == policy_name]:
                pass

    # Build candidate-set selections for failure audits using the test split.
    for split_name, train_datasets, test_datasets in TRAIN_SPLITS:
        test_rows = [r for r in rows if str(r["dataset"]) in test_datasets]
        test_groups = group_rows(test_rows)
        tau = next(x["threshold"] for x in threshold_records if x["split"] == split_name)
        for policy_name, builder in [
            ("top2_weighted", lambda g: select_topk(g, "corrected_health_weighted", 2, tie_mode="weighted")),
            ("top3_weighted", lambda g: select_topk(g, "corrected_health_weighted", 3, tie_mode="weighted")),
            (
                "threshold_cap3_weighted",
                lambda g: [r for r in sorted(g, key=lambda r: candidate_sort_key(r)) if to_float(r.get("corrected_health_weighted")) >= tau][:3],
            ),
            (
                "top2_plus_np_if_unhealthy_weighted",
                lambda g: [r for r in sorted(g, key=lambda r: candidate_sort_key(r)) if to_float(r.get("corrected_health_weighted")) >= tau][:2],
            ),
            (
                "top2_remove_aggressive_weighted",
                lambda g: select_topk(
                    ([r for r in g if float(r.get("aggressive_flag", 0.0)) < 0.5] or g),
                    "corrected_health_weighted",
                    2,
                    tie_mode="weighted",
                ),
            ),
            (
                "top2_weighted_tiebreak_consensus_residual",
                lambda g: select_topk(g, "corrected_health_weighted", 2, tie_mode="weighted"),
            ),
        ]:
            candidate_sets = {}
            for (dataset, image_id), group in test_groups.items():
                candidate_sets[(dataset, image_id, policy_name)] = builder(group)
            all_failure_rows.extend(build_failure_rows(test_rows, candidate_sets, np_map, "corrected_health_weighted", threshold=tau))

    # Extra special-case audit rows for A14/A16 image 00017 and A8 image 00007.
    special_audits = []
    for dataset, image_id in [("A14", "00017"), ("A16", "00017"), ("A8", "00007")]:
        special_audits.extend(build_image_audit(rows, dataset, image_id))

    # Write outputs.
    write_csv(outdir / "corrected_population_score_table.csv", score_table_rows)
    write_csv(outdir / "corrected_population_score_auc.csv", all_auc_rows)
    write_csv(outdir / "corrected_candidate_set_policy_summary.csv", all_policy_rows)
    write_csv(outdir / "corrected_population_crossfit_summary.csv", all_crossfit_rows)
    write_csv(outdir / "corrected_population_failure_cases.csv", all_failure_rows)
    write_csv(outdir / "image00017_corrected_population_audit.csv", special_audits)

    # Also provide a compact audit of the threshold fit itself.
    write_csv(outdir / "threshold_fit_audit.csv", threshold_records)

    # Summary markdown.
    a14_00017 = [r for r in special_audits if r["dataset"] == "A14" and r["image_id"] == "00017"]
    a16_00017 = [r for r in special_audits if r["dataset"] == "A16" and r["image_id"] == "00017"]
    a8_00007 = [r for r in special_audits if r["dataset"] == "A8" and r["image_id"] == "00007"]
    summary = [
        "# A18.6 Corrected Population Score Design",
        "",
        "This pass replaces the saturated A18.5 population score with a continuous clean-free health score built from",
        "A17 anytime persistence features plus the frozen A14/A16 residual-consensus margins.",
        "",
        "## Main readout",
        "",
        f"- train/test splits evaluated: {', '.join(s for s, _, _ in TRAIN_SPLITS)}",
        f"- fitted threshold audit rows: {len(threshold_records)}",
        f"- A14 image 00017 audit rows: {len(a14_00017)}",
        f"- A16 image 00017 audit rows: {len(a16_00017)}",
        f"- A8 image 00007 audit rows: {len(a8_00007)}",
        "",
        "## Interpretation",
        "",
        "- The corrected score is continuous and no longer saturates at a single constant value.",
        "- The cross-fit table should be read as a diagnostic check, not prospective evidence.",
        "- A14/00017 and A16/00017 are the key sanity checks for whether the score separates the good and bad siblings.",
        "- A8/00007 remains the whole-population-bad case where NP fallback matters.",
        "",
        "## Files",
        "",
        "- corrected_population_score_table.csv",
        "- corrected_population_score_auc.csv",
        "- corrected_candidate_set_policy_summary.csv",
        "- corrected_population_crossfit_summary.csv",
        "- corrected_population_failure_cases.csv",
        "- image00017_corrected_population_audit.csv",
        "- threshold_fit_audit.csv",
    ]
    write_text(outdir / "SUMMARY.md", "\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
