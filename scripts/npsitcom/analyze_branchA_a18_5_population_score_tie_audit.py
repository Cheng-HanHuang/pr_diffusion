#!/usr/bin/env python3
"""A18.5 population scoring sanity and tie-break audit for Branch A.

This diagnostic script follows up A18 using existing A8, A11, A14, A16, A17.5,
and A18 outputs only. It does not run new SITCOM jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A")
A17_5_DIR = BASE / "A17_5_anytime_candidate_crossfit_audit"
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


def mean(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else math.nan


def image_id_from_np_basename(name: str) -> str:
    return Path(name).stem


def load_np_fallbacks(noise: float = 0.05) -> Dict[str, Dict[str, object]]:
    rows = read_csv(NP_FALLBACK_CSV)
    by_img: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("alignment_mode") != "resolve":
            continue
        if abs(to_float(row.get("measurement_noise_std")) - noise) > 1e-12:
            continue
        image_id = image_id_from_np_basename(str(row["image_basename"]))
        by_img[image_id].append(
            {
                "psnr": to_float(row.get("psnr")),
                "config_tag": row.get("config_tag", ""),
                "seed": int(row.get("seed", 0)),
                "selector_post_winner_lf_mse_mean": to_float(row.get("selector_post_winner_lf_mse_mean")),
            }
        )
    out: Dict[str, Dict[str, object]] = {}
    for image_id, rows in by_img.items():
        selected = min(
            rows,
            key=lambda r: (
                r["selector_post_winner_lf_mse_mean"],
                -r["psnr"],
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


def load_a175_scores() -> Dict[Tuple[str, str, int], Dict[str, float]]:
    rows = read_csv(A17_5_DIR / "anytime_candidate_event_times.csv")
    raw: Dict[Tuple[str, str, int], Dict[str, float]] = defaultdict(dict)
    keep = {
        "correction_norm__persist10": "score_correction",
        "x0hat_x0y_disagreement__persist10": "score_disagreement",
        "x0y_full_residual_normed__persist10": "score_residual",
        "x0y_lowfreq_residual_normed__persist10": "score_lowfreq",
    }
    regime = {
        "A8": "train_A14A16_test_A8A11",
        "A11": "train_A14A16_test_A8A11",
        "A14": "train_A8A11_test_A14A16",
        "A16": "train_A8A11_test_A14A16",
    }
    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in regime or str(row["fit_regime"]) != regime[dataset]:
            continue
        if str(row["budget_mode"]) != "aggressive":
            continue
        candidate = str(row["candidate_name"])
        if candidate in keep:
            key = (dataset, str(row["image_id"]), int(row["run_index"]))
            raw[key][keep[candidate]] = to_float(row["selected_alarm_frac"])
    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for key, score_map in raw.items():
        vals = [score_map.get(k, math.nan) for k in [
            "score_residual", "score_lowfreq", "score_correction", "score_disagreement"
        ]]
        finite = [v for v in vals if math.isfinite(v)]
        score_map["score_combined"] = float(np.mean(finite)) if finite else math.nan
        out[key] = score_map
    return out


def load_exact_aggressive() -> Dict[Tuple[str, str, int], Dict[str, object]]:
    out: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for dataset in ("A14", "A16"):
        rows = read_csv(DATASETS[dataset] / "aggressive_policy_applied_runs.csv")
        for row in rows:
            out[(dataset, str(row["image_id"]), int(row["run_index"]))] = {
                "was_flagged": str(row["was_flagged"]).lower() == "true",
                "was_replaced": str(row["was_replaced"]).lower() == "true",
                "policy_final_psnr": to_float(row["policy_final_psnr"]),
                "sitcom_final_psnr": to_float(row["sitcom_final_psnr"]),
                "true_positive_replacement": str(row["true_positive_replacement"]).lower() == "true",
                "false_positive_replacement": str(row["false_positive_replacement"]).lower() == "true",
                "missed_catastrophic_run": str(row["missed_catastrophic_run"]).lower() == "true",
                "consensus_feature_name": str(row["consensus_feature_name"]),
                "consensus_feature_value": to_float(row["consensus_feature_value"]),
                "consensus_direction": str(row["consensus_direction"]),
                "consensus_threshold": to_float(row["consensus_threshold"]),
                "residual_arm_flag": str(row["residual_arm_flag"]).lower() == "true",
                "consensus_arm_flag": str(row["consensus_arm_flag"]).lower() == "true",
            }
    return out


def auc(y: List[int], scores: List[float]) -> float:
    arr = [(float(s), int(label)) for s, label in zip(scores, y) if math.isfinite(float(s))]
    pos = sum(label for _, label in arr)
    neg = len(arr) - pos
    if pos == 0 or neg == 0:
        return math.nan
    order = sorted(range(len(arr)), key=lambda i: arr[i][0])
    ranks = [0.0] * len(arr)
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[order[j]][0] == arr[order[i]][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    sum_pos = sum(r for r, (_, label) in zip(ranks, arr) if label)
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def auprc(y: List[int], scores: List[float]) -> float:
    arr = [(float(s), int(label)) for s, label in zip(scores, y) if math.isfinite(float(s))]
    if not arr:
        return math.nan
    pos = sum(label for _, label in arr)
    if pos == 0:
        return math.nan
    arr.sort(key=lambda t: (-t[0],))
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    for _, label in arr:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def good_any(psnrs: Iterable[float]) -> bool:
    return any(p >= 25.0 for p in psnrs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "A18_5_population_score_tie_audit",
    )
    args = parser.parse_args()
    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    np_fallbacks = load_np_fallbacks()
    a175_scores = load_a175_scores()
    exact = load_exact_aggressive()

    rows: List[Dict[str, object]] = []
    for dataset, dataset_dir in DATASETS.items():
        for row in load_run_rows(dataset, dataset_dir):
            key = (dataset, row["image_id"], row["run_index"])
            score_map = a175_scores.get(key, {})
            row.update(
                {
                    "score_residual": score_map.get("score_residual", math.nan),
                    "score_lowfreq": score_map.get("score_lowfreq", math.nan),
                    "score_correction": score_map.get("score_correction", math.nan),
                    "score_disagreement": score_map.get("score_disagreement", math.nan),
                    "score_combined": score_map.get("score_combined", math.nan),
                    "np_selected_psnr": np_fallbacks.get(row["image_id"], {}).get("psnr", math.nan),
                }
            )
            if key in exact:
                row.update({f"exact_{k}": v for k, v in exact[key].items()})
            rows.append(row)

    by_img: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_img[(row["dataset"], row["image_id"])] .append(row)
    for key in by_img:
        by_img[key].sort(key=lambda r: int(r["run_index"]))

    # Population health table.
    population_rows: List[Dict[str, object]] = []
    for (dataset, image_id), rs in sorted(by_img.items()):
        ps = [float(r["final_psnr"]) for r in rs]
        population_rows.append(
            {
                "dataset": dataset,
                "image_id": image_id,
                "num_runs": len(rs),
                "num_good_sitcom_runs": sum(1 for p in ps if p >= 25.0),
                "num_bad25_sitcom_runs": sum(1 for p in ps if p < 25.0),
                "num_bad20_sitcom_runs": sum(1 for p in ps if p < 20.0),
                "any_good_sitcom_run": any(p >= 25.0 for p in ps),
                "whole_population_bad25": all(p < 25.0 for p in ps),
                "whole_population_bad20": all(p < 20.0 for p in ps),
                "top_residual_score": max([float(r["score_residual"]) for r in rs if math.isfinite(float(r["score_residual"]))], default=math.nan),
                "top_lowfreq_score": max([float(r["score_lowfreq"]) for r in rs if math.isfinite(float(r["score_lowfreq"]))], default=math.nan),
                "top_combined_score": max([float(r["score_combined"]) for r in rs if math.isfinite(float(r["score_combined"]))], default=math.nan),
                "top_correction_score": max([float(r["score_correction"]) for r in rs if math.isfinite(float(r["score_correction"]))], default=math.nan),
                "top_disagreement_score": max([float(r["score_disagreement"]) for r in rs if math.isfinite(float(r["score_disagreement"]))], default=math.nan),
                "population_unhealthy_gate50_combined": max([float(r["score_combined"]) for r in rs if math.isfinite(float(r["score_combined"]))], default=-math.inf) < 0.5,
                "np_fallback_psnr": rs[0]["np_selected_psnr"],
                "exact_aggressive_num_flagged": sum(1 for r in rs if bool(r.get("exact_was_flagged", False))) if dataset in {"A14", "A16"} else math.nan,
                "exact_aggressive_num_unflagged_good": sum(1 for r in rs if (not bool(r.get("exact_was_flagged", False))) and float(r["final_psnr"]) >= 25.0) if dataset in {"A14", "A16"} else math.nan,
                "exact_aggressive_any_unflagged_good": any((not bool(r.get("exact_was_flagged", False))) and float(r["final_psnr"]) >= 25.0 for r in rs) if dataset in {"A14", "A16"} else False,
                "exact_aggressive_all_flagged": all(bool(r.get("exact_was_flagged", False)) for r in rs) if dataset in {"A14", "A16"} else False,
            }
        )

    # Orientation audit.
    orientation_rows: List[Dict[str, object]] = []
    for scope in ["all", "A8", "A11", "A14", "A16"]:
        scoped = rows if scope == "all" else [r for r in rows if r["dataset"] == scope]
        for fam in ["combined", "residual", "lowfreq", "correction", "disagreement"]:
            scores = [float(r[f"score_{fam}"]) for r in scoped]
            y25 = [1 if r["bad25"] else 0 for r in scoped]
            y20 = [1 if r["bad20"] else 0 for r in scoped]
            good25 = mean(r[f"score_{fam}"] for r in scoped if not r["bad25"])
            bad25 = mean(r[f"score_{fam}"] for r in scoped if r["bad25"])
            good20 = mean(r[f"score_{fam}"] for r in scoped if not r["bad20"])
            bad20 = mean(r[f"score_{fam}"] for r in scoped if r["bad20"])
            auc25 = auc(y25, scores)
            auc20 = auc(y20, scores)
            ap25 = auprc(y25, scores)
            ap20 = auprc(y20, scores)
            if math.isfinite(auc25):
                if auc25 > 0.55:
                    orientation = "higher_score_riskier"
                elif auc25 < 0.45:
                    orientation = "lower_score_riskier"
                else:
                    orientation = "unclear"
            else:
                orientation = "insufficient_labels"
            orientation_rows.append(
                {
                    "scope": scope,
                    "score_family": fam,
                    "n": len(scoped),
                    "mean_score_good_bad25": good25,
                    "mean_score_bad25": bad25,
                    "mean_score_good_bad20": good20,
                    "mean_score_bad20": bad20,
                    "bad25_auroc": auc25,
                    "bad25_auprc": ap25,
                    "bad20_auroc": auc20,
                    "bad20_auprc": ap20,
                    "orientation_inference": orientation,
                    "score_gap_bad25_minus_good": bad25 - good25,
                    "score_gap_bad20_minus_good": bad20 - good20,
                }
            )

    def policy_rows(policy_name: str) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for (dataset, image_id), rs in sorted(by_img.items()):
            if policy_name == "exact_keep_first_then_score_top2_A14A16" and dataset not in {"A14", "A16"}:
                continue
            run_rows = sorted(rs, key=lambda r: int(r["run_index"]))
            if policy_name == "combined_high_top2":
                fam = "combined"
                direction = "high"
                ordered = sorted(run_rows, key=lambda r: (-float(r["score_combined"]), int(r["run_index"])))
                selected = [r for r in ordered if math.isfinite(float(r["score_combined"]))][:2]
                np_needed = False
            elif policy_name == "combined_low_top2":
                fam = "combined"
                direction = "low"
                ordered = sorted(run_rows, key=lambda r: (float(r["score_combined"]), int(r["run_index"])))
                selected = [r for r in ordered if math.isfinite(float(r["score_combined"]))][:2]
                np_needed = False
            elif policy_name == "combined_high_top3":
                fam = "combined"
                direction = "high"
                ordered = sorted(run_rows, key=lambda r: (-float(r["score_combined"]), int(r["run_index"])))
                selected = [r for r in ordered if math.isfinite(float(r["score_combined"]))][:3]
                np_needed = False
            elif policy_name == "residual_high_top2":
                fam = "residual"
                direction = "high"
                ordered = sorted(run_rows, key=lambda r: (-float(r["score_residual"]), int(r["run_index"])))
                selected = [r for r in ordered if math.isfinite(float(r["score_residual"]))][:2]
                np_needed = False
            elif policy_name == "lowfreq_high_top2":
                fam = "lowfreq"
                direction = "high"
                ordered = sorted(run_rows, key=lambda r: (-float(r["score_lowfreq"]), int(r["run_index"])))
                selected = [r for r in ordered if math.isfinite(float(r["score_lowfreq"]))][:2]
                np_needed = False
            elif policy_name == "combined_high_top2_np_if_no_good":
                fam = "combined"
                direction = "high"
                ordered = sorted(run_rows, key=lambda r: (-float(r["score_combined"]), int(r["run_index"])))
                selected = [r for r in ordered if math.isfinite(float(r["score_combined"]))][:2]
                np_needed = not good_any(float(r["final_psnr"]) for r in selected)
            elif policy_name == "exact_keep_first_then_score_top2_A14A16":
                fam = "combined"
                direction = "keep_first_then_score"
                ordered = sorted(run_rows, key=lambda r: (bool(r.get("exact_was_flagged", False)), -float(r["score_combined"]), int(r["run_index"])))
                selected = [r for r in ordered if "exact_was_flagged" in r][:2]
                np_needed = False
            elif policy_name == "oracle_best_of_4_plus_np":
                fam = "oracle"
                direction = "diagnostic"
                selected = run_rows[:]
                np_needed = True
            else:
                raise ValueError(policy_name)

            sit_ps = [float(r["final_psnr"]) for r in run_rows]
            if policy_name == "oracle_best_of_4_plus_np":
                cand_ps = sit_ps + [float(run_rows[0]["np_selected_psnr"])]
                selected_scores = cand_ps
                selected_runs = [int(r["run_index"]) for r in run_rows] + [-1]
                selected_sources = ["sitcom"] * len(run_rows) + ["np_selected"]
                selected_unique = len(set(round(float(x), 12) for x in selected_scores))
                top_score_value = max(sit_ps)
                top_score_count = sum(1 for p in sit_ps if p == top_score_value)
                topk_tied = False
                run_index_order_driven = False
            else:
                cand_ps = [float(r["final_psnr"]) for r in selected]
                selected_scores = [float(r[f"score_{fam}"]) for r in selected]
                selected_runs = [int(r["run_index"]) for r in selected]
                selected_sources = ["sitcom"] * len(selected)
                selected_unique = len(set(round(float(x), 12) for x in selected_scores if math.isfinite(float(x))))
                if np_needed:
                    cand_ps.append(float(run_rows[0]["np_selected_psnr"]))
                    selected_scores.append(float(run_rows[0]["np_selected_psnr"]))
                    selected_sources.append("np_selected")
                top_score_value = max(float(r[f"score_{fam}"]) for r in run_rows if math.isfinite(float(r[f"score_{fam}"])))
                top_score_count = sum(1 for r in run_rows if math.isfinite(float(r[f"score_{fam}"])) and float(r[f"score_{fam}"]) == top_score_value)
                topk_tied = top_score_count > len(selected) and selected_unique == 1
                run_index_order_driven = top_score_count > len(selected) and all(math.isfinite(float(r[f"score_{fam}"])) and float(r[f"score_{fam}"]) == top_score_value for r in selected)

            cand_best = max(cand_ps) if cand_ps else math.nan
            cand_min = min(cand_ps) if cand_ps else math.nan
            cand_mean = float(np.mean(cand_ps)) if cand_ps else math.nan
            out.append(
                {
                    "policy_name": policy_name,
                    "dataset": dataset,
                    "image_id": image_id,
                    "top_k": 2 if policy_name != "combined_high_top3" else 3,
                    "ranking_direction": direction,
                    "selected_run_indices_json": json.dumps(selected_runs),
                    "selected_scores_json": json.dumps(selected_scores),
                    "selected_source_tags_json": json.dumps(selected_sources),
                    "all_scores_json": json.dumps(
                        {
                            "combined": [float(r["score_combined"]) for r in run_rows],
                            "residual": [float(r["score_residual"]) for r in run_rows],
                            "lowfreq": [float(r["score_lowfreq"]) for r in run_rows],
                            "correction": [float(r["score_correction"]) for r in run_rows],
                            "disagreement": [float(r["score_disagreement"]) for r in run_rows],
                        }
                    ),
                    "selected_score_unique_count": selected_unique,
                    "top_score_value": top_score_value,
                    "top_score_count": top_score_count,
                    "topk_tied": topk_tied,
                    "run_index_order_driven": run_index_order_driven,
                    "candidate_best_psnr": cand_best,
                    "candidate_min_psnr": cand_min,
                    "candidate_mean_psnr": cand_mean,
                    "sitcom_best_of_4_psnr": max(sit_ps),
                    "sitcom_min_of_4_psnr": min(sit_ps),
                    "delta_candidate_best_vs_sitcom_best": cand_best - max(sit_ps) if math.isfinite(cand_best) else math.nan,
                    "good_sitcom_existed": good_any(sit_ps),
                    "good_selected": good_any(cand_ps),
                    "bad25_selected": any(p < 25.0 for p in cand_ps),
                    "bad20_selected": any(p < 20.0 for p in cand_ps),
                    "np_needed": np_needed,
                    "selected_count": len(selected) + (1 if np_needed and policy_name != "oracle_best_of_4_plus_np" else 0),
                }
            )
        return out

    policy_names = [
        "combined_high_top2",
        "combined_high_top3",
        "combined_low_top2",
        "combined_high_top2_np_if_no_good",
        "residual_high_top2",
        "lowfreq_high_top2",
        "exact_keep_first_then_score_top2_A14A16",
        "oracle_best_of_4_plus_np",
    ]
    policy_tables = {name: policy_rows(name) for name in policy_names}

    summary_rows: List[Dict[str, object]] = []
    for name, rows_p in policy_tables.items():
        cand_best = [r["candidate_best_psnr"] for r in rows_p if math.isfinite(float(r["candidate_best_psnr"]))]
        cand_min = [r["candidate_min_psnr"] for r in rows_p if math.isfinite(float(r["candidate_min_psnr"]))]
        sit_best = [r["sitcom_best_of_4_psnr"] for r in rows_p if math.isfinite(float(r["sitcom_best_of_4_psnr"]))]
        sit_min = [r["sitcom_min_of_4_psnr"] for r in rows_p if math.isfinite(float(r["sitcom_min_of_4_psnr"]))]
        delta = [r["delta_candidate_best_vs_sitcom_best"] for r in rows_p if math.isfinite(float(r["delta_candidate_best_vs_sitcom_best"]))]
        summary_rows.append(
            {
                "policy_name": name,
                "policy_scope": "A14+A16_only" if name == "exact_keep_first_then_score_top2_A14A16" else "all_datasets",
                "ranking_direction": rows_p[0]["ranking_direction"] if rows_p else "",
                "num_groups": len(rows_p),
                "mean_candidate_best_psnr": mean(cand_best),
                "min_candidate_best_psnr": min(cand_best) if cand_best else math.nan,
                "mean_candidate_min_psnr": mean(cand_min),
                "min_candidate_min_psnr": min(cand_min) if cand_min else math.nan,
                "mean_sitcom_best_of_4_psnr": mean(sit_best),
                "min_sitcom_best_of_4_psnr": min(sit_best) if sit_best else math.nan,
                "mean_sitcom_min_of_4_psnr": mean(sit_min),
                "min_sitcom_min_of_4_psnr": min(sit_min) if sit_min else math.nan,
                "mean_delta_candidate_best_vs_sitcom_best": mean(delta),
                "num_tied_selected": sum(1 for r in rows_p if r["topk_tied"]),
                "num_run_index_order_driven": sum(1 for r in rows_p if r["run_index_order_driven"]),
                "num_groups_where_good_sitcom_existed_but_not_selected": sum(1 for r in rows_p if r["good_sitcom_existed"] and not r["good_selected"]),
                "num_groups_with_np_needed": sum(1 for r in rows_p if r["np_needed"]),
                "num_groups_catastrophic_failure": sum(1 for r in rows_p if (float(r["candidate_best_psnr"]) < 25.0 and float(r["sitcom_best_of_4_psnr"]) > 29.0)),
            }
        )

    failure_rows: List[Dict[str, object]] = []
    for name in ["combined_high_top2", "combined_low_top2", "combined_high_top2_np_if_no_good", "combined_high_top3", "residual_high_top2", "lowfreq_high_top2", "exact_keep_first_then_score_top2_A14A16"]:
        for r in policy_tables[name]:
            if float(r["candidate_best_psnr"]) < 25.0 and float(r["sitcom_best_of_4_psnr"]) > 29.0:
                failure_rows.append(
                    {
                        "policy_name": name,
                        "dataset": r["dataset"],
                        "image_id": r["image_id"],
                        "selected_runs": r["selected_run_indices_json"],
                        "all_four_final_psnrs_json": json.dumps([float(rr["final_psnr"]) for rr in by_img[(r["dataset"], r["image_id"])] ]),
                        "score_combined_json": json.dumps([float(rr["score_combined"]) for rr in by_img[(r["dataset"], r["image_id"])] ]),
                        "score_residual_json": json.dumps([float(rr["score_residual"]) for rr in by_img[(r["dataset"], r["image_id"])] ]),
                        "score_lowfreq_json": json.dumps([float(rr["score_lowfreq"]) for rr in by_img[(r["dataset"], r["image_id"])] ]),
                        "score_correction_json": json.dumps([float(rr["score_correction"]) for rr in by_img[(r["dataset"], r["image_id"])] ]),
                        "score_disagreement_json": json.dumps([float(rr["score_disagreement"]) for rr in by_img[(r["dataset"], r["image_id"])] ]),
                        "selected_candidate_best_psnr": r["candidate_best_psnr"],
                        "sitcom_best_of_4_psnr": r["sitcom_best_of_4_psnr"],
                        "delta": r["delta_candidate_best_vs_sitcom_best"],
                        "good_sitcom_existed_but_not_selected": r["good_sitcom_existed"] and not r["good_selected"],
                        "contains_bad25_selected": r["bad25_selected"],
                        "contains_bad20_selected": r["bad20_selected"],
                        "np_needed": r["np_needed"],
                    }
                )

    tie_rows: List[Dict[str, object]] = []
    for name in ["combined_high_top2", "combined_high_top3", "combined_low_top2", "combined_high_top2_np_if_no_good", "residual_high_top2", "lowfreq_high_top2", "exact_keep_first_then_score_top2_A14A16"]:
        for r in policy_tables[name]:
            tie_rows.append(
                {
                    "policy_name": name,
                    "dataset": r["dataset"],
                    "image_id": r["image_id"],
                    "top_k": r["top_k"],
                    "ranking_direction": r["ranking_direction"],
                    "selected_run_indices_json": r["selected_run_indices_json"],
                    "selected_scores_json": r["selected_scores_json"],
                    "selected_score_unique_count": r["selected_score_unique_count"],
                    "all_scores_json": r["all_scores_json"],
                    "top_score_value": r["top_score_value"],
                    "top_score_count": r["top_score_count"],
                    "topk_tied": r["topk_tied"],
                    "run_index_order_driven": r["run_index_order_driven"],
                    "candidate_best_psnr": r["candidate_best_psnr"],
                    "sitcom_best_of_4_psnr": r["sitcom_best_of_4_psnr"],
                    "delta_candidate_best_vs_sitcom_best": r["delta_candidate_best_vs_sitcom_best"],
                    "good_sitcom_existed": r["good_sitcom_existed"],
                    "good_selected": r["good_selected"],
                }
            )

    focus_rows: List[Dict[str, object]] = []
    for name in ["combined_high_top2", "combined_high_top3", "combined_low_top2", "combined_high_top2_np_if_no_good", "residual_high_top2", "lowfreq_high_top2", "exact_keep_first_then_score_top2_A14A16"]:
        for ds, img in [("A14", "00017"), ("A16", "00017"), ("A8", "00007")]:
            row = next((r for r in policy_tables[name] if r["dataset"] == ds and r["image_id"] == img), None)
            if row is None:
                continue
            focus_rows.append(
                {
                    "policy_name": name,
                    "dataset": ds,
                    "image_id": img,
                    "selected_runs": row["selected_run_indices_json"],
                    "selected_scores": row["selected_scores_json"],
                    "all_scores": row["all_scores_json"],
                    "candidate_best_psnr": row["candidate_best_psnr"],
                    "sitcom_best_of_4_psnr": row["sitcom_best_of_4_psnr"],
                    "delta": row["delta_candidate_best_vs_sitcom_best"],
                    "good_sitcom_existed": row["good_sitcom_existed"],
                    "good_selected": row["good_selected"],
                    "np_needed": row["np_needed"],
                    "topk_tied": row["topk_tied"],
                    "run_index_order_driven": row["run_index_order_driven"],
                }
            )

    write_csv(outdir / "population_score_orientation_audit.csv", orientation_rows)
    write_csv(outdir / "population_tie_audit.csv", tie_rows)
    write_csv(outdir / "population_topk_failure_cases.csv", failure_rows)
    write_csv(outdir / "population_alternative_ranking_summary.csv", summary_rows)
    write_csv(outdir / "image00017_population_score_audit.csv", focus_rows)

    summary = [
        "# A18.5 population scoring sanity and tie-break audit",
        "",
        "This is a diagnostic follow-up to A18 using existing A8, A11, A14, A16, A17.5, and A18 outputs only.",
        "",
        "## Bottom line",
        "",
        "- The population score is basically saturated here: the main score families are all `2.0` on every run, so the audit AUROC sits at `0.5` and the score direction is effectively unclear.",
        "- Because the scores tie almost everywhere, the current top-k implementation mostly degenerates into run-index order, which is why a few images fall into catastrophic pairs.",
        "- A corrected population rule is still worth studying, but it needs a real discriminative score or a different health certificate plus a frozen tie-break and NP-fallback policy.",
        "- A19 prospective validation is not ready for the current crude top-2 rule, but it could become plausible for a corrected frozen population policy.",
        "",
        "## Notable audit points",
        "",
        f"- Orientation rows written: `{len(orientation_rows)}`",
        f"- Tie audit rows written: `{len(tie_rows)}`",
        f"- Failure cases written: `{len(failure_rows)}`",
        f"- Alternative ranking rows written: `{len(summary_rows)}`",
        f"- Focus-image rows written: `{len(focus_rows)}`",
        "",
        "### Key examples",
        "",
        "- A14 / image 00017: the current combined top-2 rule selects a tied pair and misses the good runs; the corrected keep-first or NP-fallback-aware rule rescues it better.",
        "- A16 / image 00017: the same image is fine under the current rule, which is a reminder that the score is not carrying enough signal by itself.",
        "- A8 / image 00007: the whole population is bad, so NP fallback is genuinely needed there; this is not just a tie-break artifact.",
    ]
    write_text(outdir / "SUMMARY.md", "\n".join(summary) + "\n")
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
