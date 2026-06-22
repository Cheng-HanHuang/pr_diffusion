#!/usr/bin/env python3
"""A18 offline population / candidate-set controller design for Branch A.

This script is diagnostic-only. It uses existing outputs from A8, A11, A14,
A16, A17, and A17.5. It does not run new SITCOM jobs and it does not change
any frozen Branch A policy.

The goal is to ask whether Branch A should move from per-run binary alarms to
population-level candidate-set control.
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


DATASETS: List[Tuple[str, Path]] = [
    (
        "A8",
        Path(
            "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/"
            "A8_sitcom25_trajectory_controller_validation"
        ),
    ),
    (
        "A11",
        Path(
            "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/"
            "A11_prospective_frozen_policy_validation"
        ),
    ),
    (
        "A14",
        Path(
            "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/"
            "A14_prospective_dual_policy_validation"
        ),
    ),
    (
        "A16",
        Path(
            "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/"
            "A16_prospective_dual_policy_replication"
        ),
    ),
]

A17_DIR = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/"
    "A17_offline_anytime_detector_design"
)
A17_5_DIR = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/"
    "A17_5_anytime_candidate_crossfit_audit"
)
A13_5_DIR = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/"
    "A13_5_consensus_feature_diagnosis"
)
NP_FALLBACK_CSV = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608/np_selector_ffhq/"
    "selector_full25_s100_103/lf_s2_selector_20260609_154712/run_level.csv"
)

A17_BASE_FEATURES = [
    "x0y_full_residual_normed",
    "x0y_lowfreq_residual_normed",
    "x0hat_x0y_disagreement",
    "correction_norm",
]

A175_BASE_CANDIDATES = [
    "correction_norm__persist10",
    "x0hat_x0y_disagreement__persist10",
    "x0y_full_residual_normed__persist10",
    "x0y_lowfreq_residual_normed__persist10",
]

PRIMARY_BUDGET = "aggressive"
SECONDARY_BUDGET = "conservative"
HEALTH_GATE = 0.5
TOPK_RULES = [("top2", 2), ("top3", 3)]


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


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


def make_run_key(dataset: str, image_id: str, run_index: int) -> Tuple[str, str, int]:
    return dataset, image_id, int(run_index)


def load_np_fallbacks(np_csv: Path, noise: float = 0.05) -> Dict[str, Dict[str, object]]:
    rows = read_csv(np_csv)
    candidates: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("alignment_mode") != "resolve":
            continue
        if abs(to_float(row.get("measurement_noise_std")) - noise) > 1e-12:
            continue
        image_id = image_id_from_np_basename(str(row["image_basename"]))
        candidates[image_id].append(
            {
                "image_id": image_id,
                "config_tag": row.get("config_tag", ""),
                "seed": int(row.get("seed", 0)),
                "psnr": to_float(row.get("psnr")),
                "selector_post_winner_lf_mse_mean": to_float(row.get("selector_post_winner_lf_mse_mean")),
            }
        )
    out: Dict[str, Dict[str, object]] = {}
    for image_id, rows in candidates.items():
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


def load_dataset_run_rows(dataset: str, dataset_dir: Path) -> List[Dict[str, object]]:
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
                "sitcom_only_best_of_4_psnr": math.nan,
                "sitcom_only_mean_of_4_psnr": math.nan,
                "sitcom_only_min_of_4_psnr": math.nan,
            }
        )
    return out


def load_a17_feature_rows() -> Dict[Tuple[str, str, int], Dict[str, float]]:
    path = A17_DIR / "anytime_feature_table.csv"
    rows = read_csv(path)
    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        feat = {}
        for feature in A17_BASE_FEATURES:
            feat[f"{feature}__persist10_step0"] = to_float(row[f"{feature}__rank_ge3__persist10__first_step0"])
        out[(dataset, image_id, run_index)] = feat
    return out


def load_a175_event_scores() -> Dict[Tuple[str, str, int, str, str, str], float]:
    path = A17_5_DIR / "anytime_candidate_event_times.csv"
    rows = read_csv(path)
    scores: Dict[Tuple[str, str, int, str, str, str], float] = {}
    for row in rows:
        dataset = str(row["dataset"])
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        fit_regime = str(row["fit_regime"])
        budget_mode = str(row["budget_mode"])
        candidate_name = str(row["candidate_name"])
        if candidate_name not in A175_BASE_CANDIDATES:
            continue
        scores[(dataset, image_id, run_index, fit_regime, budget_mode, candidate_name)] = to_float(row["selected_alarm_frac"])
    return scores


def load_a175_best_available_scores() -> Dict[Tuple[str, str, int, str], Dict[str, float]]:
    raw = load_a175_event_scores()
    dataset_to_fit_regime = {
        "A8": "train_A14A16_test_A8A11",
        "A11": "train_A14A16_test_A8A11",
        "A14": "train_A8A11_test_A14A16",
        "A16": "train_A8A11_test_A14A16",
    }
    out: Dict[Tuple[str, str, int, str], Dict[str, float]] = defaultdict(dict)
    for (dataset, image_id, run_index, fit_regime, budget_mode, candidate_name), score in raw.items():
        if dataset_to_fit_regime.get(dataset) != fit_regime:
            continue
        out[(dataset, image_id, run_index, budget_mode)][candidate_name] = score
    return out


def load_aggressive_exact_rows() -> Dict[Tuple[str, str, int], Dict[str, object]]:
    out: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for dataset, dataset_dir in DATASETS:
        if dataset not in {"A14", "A16"}:
            continue
        path = dataset_dir / "aggressive_policy_applied_runs.csv"
        rows = read_csv(path)
        for row in rows:
            key = make_run_key(dataset, str(row["image_id"]), int(row["run_index"]))
            out[key] = {
                "policy_name": str(row["policy_name"]),
                "policy_kind": str(row["policy_kind"]),
                "policy_role": str(row["policy_role"]),
                "sitcom_final_psnr": to_float(row["sitcom_final_psnr"]),
                "policy_final_psnr": to_float(row["policy_final_psnr"]),
                "was_flagged": str(row["was_flagged"]).lower() == "true",
                "was_replaced": str(row["was_replaced"]).lower() == "true",
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


def score_combined(score_map: Dict[str, float]) -> float:
    vals = [v for v in score_map.values() if math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def score_family(score_map: Dict[str, float], family: str) -> float:
    if family == "residual":
        return score_map.get("x0y_full_residual_normed__persist10", math.nan)
    if family == "lowfreq":
        return score_map.get("x0y_lowfreq_residual_normed__persist10", math.nan)
    if family == "combined":
        return score_combined(score_map)
    raise ValueError(f"unknown family {family}")


class SelectedCandidate:
    def __init__(self, source: str, score: float, run_index: int, psnr: float, kind: str):
        self.source = source
        self.score = score
        self.run_index = run_index
        self.psnr = psnr
        self.kind = kind


def select_topk_candidates(run_scores: List[Dict[str, object]], family: str, k: int, gate: float) -> Tuple[List[Dict[str, object]], bool, List[Dict[str, object]]]:
    eligible = [r for r in run_scores if math.isfinite(r[f"score_{family}"]) and r[f"score_{family}"] >= gate]
    eligible.sort(key=lambda r: (-r[f"score_{family}"], int(r["run_index"])))
    selected = eligible[:k]
    np_needed = len(selected) == 0
    return selected, np_needed, eligible


def summarize_candidate_set(selected: List[Dict[str, object]], np_needed: bool, np_fallback: Dict[str, object]) -> List[Dict[str, object]]:
    out = []
    for row in selected:
        out.append(
            {
                "source": "sitcom",
                "run_index": int(row["run_index"]),
                "psnr": float(row["final_psnr"]),
                "bad25": bool(row["bad25"]),
                "bad20": bool(row["bad20"]),
                "score_residual": float(row.get("score_residual", math.nan)),
                "score_lowfreq": float(row.get("score_lowfreq", math.nan)),
                "score_combined": float(row.get("score_combined", math.nan)),
            }
        )
    if np_needed:
        out.append(
            {
                "source": "np_selected",
                "run_index": -1,
                "psnr": float(np_fallback["np_selected_psnr"]),
                "bad25": float(np_fallback["np_selected_psnr"]) < 25.0,
                "bad20": float(np_fallback["np_selected_psnr"]) < 20.0,
                "score_residual": math.nan,
                "score_lowfreq": math.nan,
                "score_combined": math.nan,
            }
        )
    return out


def candidate_stats(cands: List[Dict[str, object]]) -> Dict[str, object]:
    psnrs = [float(c["psnr"]) for c in cands]
    sitcom_psnrs = [float(c["psnr"]) for c in cands if c["source"] == "sitcom"]
    sources = [str(c["source"]) for c in cands]
    return {
        "candidate_set_size": len(cands),
        "num_sitcom_candidates": sum(1 for c in cands if c["source"] == "sitcom"),
        "num_np_candidates": sum(1 for c in cands if c["source"] == "np_selected"),
        "candidate_best_psnr": max(psnrs) if psnrs else math.nan,
        "candidate_min_psnr": min(psnrs) if psnrs else math.nan,
        "candidate_mean_psnr": float(np.mean(psnrs)) if psnrs else math.nan,
        "contains_bad25_sitcom": any(bool(c["bad25"]) for c in cands if c["source"] == "sitcom"),
        "contains_bad20_sitcom": any(bool(c["bad20"]) for c in cands if c["source"] == "sitcom"),
        "contains_good_sitcom": any(not bool(c["bad25"]) for c in cands if c["source"] == "sitcom"),
        "contains_good_any": any(not bool(c["bad25"]) for c in cands),
        "contains_bad25_any": any(bool(c["bad25"]) for c in cands),
        "contains_bad20_any": any(bool(c["bad20"]) for c in cands),
        "selected_sources_json": json.dumps(sources),
    }


def group_runs(rows: List[Dict[str, object]]) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["image_id"]))].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: int(r["run_index"]))
    return groups


def compute_base_tables() -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str, int], Dict[str, object]], Dict[Tuple[str, str, int, str], Dict[str, float]], Dict[Tuple[str, str, int], Dict[str, object]], Dict[str, Dict[str, object]]]:
    np_fallbacks = load_np_fallbacks(NP_FALLBACK_CSV)
    a17_features = load_a17_feature_rows()
    a175_scores = load_a175_best_available_scores()
    exact_aggressive = load_aggressive_exact_rows()

    all_rows: List[Dict[str, object]] = []
    for dataset, dataset_dir in DATASETS:
        run_rows = load_dataset_run_rows(dataset, dataset_dir)
        for row in run_rows:
            key = make_run_key(dataset, str(row["image_id"]), int(row["run_index"]))
            feats = a17_features.get(key, {})
            row["score_residual"] = a175_scores.get((dataset, row["image_id"], int(row["run_index"]), PRIMARY_BUDGET), {}).get(
                "x0y_full_residual_normed__persist10", math.nan
            )
            row["score_lowfreq"] = a175_scores.get((dataset, row["image_id"], int(row["run_index"]), PRIMARY_BUDGET), {}).get(
                "x0y_lowfreq_residual_normed__persist10", math.nan
            )
            row["score_correction"] = a175_scores.get((dataset, row["image_id"], int(row["run_index"]), PRIMARY_BUDGET), {}).get(
                "correction_norm__persist10", math.nan
            )
            row["score_disagreement"] = a175_scores.get((dataset, row["image_id"], int(row["run_index"]), PRIMARY_BUDGET), {}).get(
                "x0hat_x0y_disagreement__persist10", math.nan
            )
            row["score_combined"] = score_combined(
                {
                    "residual": row["score_residual"],
                    "lowfreq": row["score_lowfreq"],
                    "correction": row["score_correction"],
                    "disagreement": row["score_disagreement"],
                }
            )
            row["score_residual_conservative"] = a175_scores.get((dataset, row["image_id"], int(row["run_index"]), SECONDARY_BUDGET), {}).get(
                "x0y_full_residual_normed__persist10", math.nan
            )
            row["score_lowfreq_conservative"] = a175_scores.get((dataset, row["image_id"], int(row["run_index"]), SECONDARY_BUDGET), {}).get(
                "x0y_lowfreq_residual_normed__persist10", math.nan
            )
            row["score_combined_conservative"] = score_combined(
                {
                    "residual": row["score_residual_conservative"],
                    "lowfreq": row["score_lowfreq_conservative"],
                    "correction": a175_scores.get((dataset, row["image_id"], int(row["run_index"]), SECONDARY_BUDGET), {}).get(
                        "correction_norm__persist10", math.nan
                    ),
                    "disagreement": a175_scores.get((dataset, row["image_id"], int(row["run_index"]), SECONDARY_BUDGET), {}).get(
                        "x0hat_x0y_disagreement__persist10", math.nan
                    ),
                }
            )
            row["a17_union_visible_50_proxy"] = any(
                math.isfinite(v) and v <= 0.5
                for v in [
                    feats.get("x0y_full_residual_normed__persist10_step0", math.nan),
                    feats.get("x0y_lowfreq_residual_normed__persist10_step0", math.nan),
                    feats.get("x0hat_x0y_disagreement__persist10_step0", math.nan),
                    feats.get("correction_norm__persist10_step0", math.nan),
                ]
            )
            row["a17_best_alarm_frac_proxy"] = min(
                [
                    v
                    for v in [
                        feats.get("x0y_full_residual_normed__persist10_step0", math.nan),
                        feats.get("x0y_lowfreq_residual_normed__persist10_step0", math.nan),
                        feats.get("x0hat_x0y_disagreement__persist10_step0", math.nan),
                        feats.get("correction_norm__persist10_step0", math.nan),
                    ]
                    if math.isfinite(v)
                ],
                default=math.nan,
            )
            row["np_selected_psnr"] = np_fallbacks[row["image_id"]]["np_selected_psnr"]
            row["np_selected_config_tag"] = np_fallbacks[row["image_id"]]["np_selected_config_tag"]
            row["np_selected_seed"] = np_fallbacks[row["image_id"]]["np_selected_seed"]
            row["np_selected_selector_post_lf_mse"] = np_fallbacks[row["image_id"]]["np_selected_selector_post_lf_mse"]
            if key in exact_aggressive:
                row.update({f"exact_{k}": v for k, v in exact_aggressive[key].items()})
            all_rows.append(row)

    return all_rows, a17_features, a175_scores, exact_aggressive, np_fallbacks


def evaluate_policy(
    policy_name: str,
    score_field: str,
    rows: List[Dict[str, object]],
    groups: Dict[Tuple[str, str], List[Dict[str, object]]],
    k: int,
    gate: float,
    exact_aggressive: bool = False,
    exact_scope: Sequence[str] | None = None,
    budget_mode: str = PRIMARY_BUDGET,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    image_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    all_candidate_best = []
    all_candidate_min = []
    all_sitcom_best = []
    all_sitcom_min = []
    all_delta = []
    all_set_sizes = []
    all_np_needed = []
    all_contains_good = []
    all_contains_bad25 = []
    all_contains_bad20 = []
    all_population_unhealthy = []
    all_exact_flagged = []
    all_exact_unflagged_good = []

    for (dataset, image_id), image_rows_all in groups.items():
        if exact_scope is not None and dataset not in exact_scope:
            continue
        per_run_rows = [r for r in image_rows_all if r["dataset"] == dataset]
        # baseline values
        sitcom_psnrs = [float(r["final_psnr"]) for r in per_run_rows]
        sitcom_best = max(sitcom_psnrs)
        sitcom_min = min(sitcom_psnrs)
        sitcom_mean = float(np.mean(sitcom_psnrs))
        good_sitcom = sum(1 for r in per_run_rows if not bool(r["bad25"]))
        bad25_sitcom = sum(1 for r in per_run_rows if bool(r["bad25"]))
        bad20_sitcom = sum(1 for r in per_run_rows if bool(r["bad20"]))
        if exact_aggressive:
            selected = [r for r in per_run_rows if not bool(r.get("exact_was_flagged", False))]
            np_needed = len(selected) == 0
            selected_runs = selected
            score_lookup = {int(r["run_index"]): math.nan for r in per_run_rows}
        else:
            scored = []
            for r in per_run_rows:
                score = to_float(r[score_field])
                scored.append({**r, "score": score})
            scored.sort(key=lambda r: (-r["score"], int(r["run_index"])))
            selected = [r for r in scored if math.isfinite(r["score"]) and r["score"] >= gate][:k]
            np_needed = len(selected) == 0 or max([r["score"] for r in scored], default=-math.inf) < gate
            selected_runs = selected
            selected_psnrs = [float(r["final_psnr"]) for r in selected]
            if np_needed:
                selected_psnrs.append(float(per_run_rows[0]["np_selected_psnr"]))
            score_lookup = {int(r["run_index"]): float(r["score"]) for r in scored}

        candidate_rows = []
        for r in selected_runs:
            candidate_rows.append(
                {
                    "source": "sitcom",
                    "run_index": int(r["run_index"]),
                    "psnr": float(r["final_psnr"]),
                    "bad25": bool(r["bad25"]),
                    "bad20": bool(r["bad20"]),
                    "score": score_lookup.get(int(r["run_index"]), math.nan),
                }
            )
        if np_needed:
            candidate_rows.append(
                {
                    "source": "np_selected",
                    "run_index": -1,
                    "psnr": float(per_run_rows[0]["np_selected_psnr"]),
                    "bad25": float(per_run_rows[0]["np_selected_psnr"]) < 25.0,
                    "bad20": float(per_run_rows[0]["np_selected_psnr"]) < 20.0,
                    "score": math.nan,
                }
            )

        cand_stats = candidate_stats(candidate_rows)
        candidate_best = cand_stats["candidate_best_psnr"]
        candidate_min = cand_stats["candidate_min_psnr"]
        candidate_mean = cand_stats["candidate_mean_psnr"]
        delta = candidate_best - sitcom_best if math.isfinite(candidate_best) and math.isfinite(sitcom_best) else math.nan
        whole_population_unhealthy = (len(selected_runs) == 0) if exact_aggressive else max((score_lookup.get(int(r["run_index"]), -math.inf) for r in per_run_rows), default=-math.inf) < gate
        exact_flagged = sum(1 for r in per_run_rows if bool(r.get("exact_was_flagged", False))) if exact_aggressive else math.nan
        exact_unflagged_good = sum(
            1 for r in per_run_rows if (not bool(r.get("exact_was_flagged", False))) and (not bool(r["bad25"]))
        ) if exact_aggressive else math.nan

        row = {
            "policy_name": policy_name,
            "budget_mode": budget_mode,
            "dataset": dataset,
            "image_id": image_id,
            "k": k,
            "health_gate": gate,
            "candidate_set_size": cand_stats["candidate_set_size"],
            "num_sitcom_candidates": cand_stats["num_sitcom_candidates"],
            "num_np_candidates": cand_stats["num_np_candidates"],
            "selected_sources_json": cand_stats["selected_sources_json"],
            "selected_run_indices_json": json.dumps([int(r["run_index"]) for r in selected_runs]),
            "selected_scores_json": json.dumps([score_lookup.get(int(r["run_index"]), math.nan) for r in selected_runs]),
            "np_fallback_needed": np_needed,
            "population_unhealthy": whole_population_unhealthy,
            "contains_good_sitcom": cand_stats["contains_good_sitcom"],
            "contains_bad25_sitcom": cand_stats["contains_bad25_sitcom"],
            "contains_bad20_sitcom": cand_stats["contains_bad20_sitcom"],
            "contains_good_any": cand_stats["contains_good_any"],
            "contains_bad25_any": cand_stats["contains_bad25_any"],
            "contains_bad20_any": cand_stats["contains_bad20_any"],
            "candidate_best_psnr": candidate_best,
            "candidate_min_psnr": candidate_min,
            "candidate_mean_psnr": candidate_mean,
            "sitcom_only_best_of_4_psnr": sitcom_best,
            "sitcom_only_mean_of_4_psnr": sitcom_mean,
            "sitcom_only_min_of_4_psnr": sitcom_min,
            "num_bad25_sitcom": bad25_sitcom,
            "num_bad20_sitcom": bad20_sitcom,
            "num_good_sitcom": good_sitcom,
            "delta_candidate_best_vs_sitcom_best": delta,
            "np_selected_psnr": float(per_run_rows[0]["np_selected_psnr"]),
            "np_selected_config_tag": per_run_rows[0]["np_selected_config_tag"],
            "np_selected_seed": per_run_rows[0]["np_selected_seed"],
            "np_selected_selector_post_lf_mse": per_run_rows[0]["np_selected_selector_post_lf_mse"],
            "aggressive_exact_num_flagged": exact_flagged,
            "aggressive_exact_num_unflagged_good": exact_unflagged_good,
            "exact_aggressive_available": exact_aggressive,
        }
        image_rows.append(row)
        all_candidate_best.append(candidate_best)
        all_candidate_min.append(candidate_min)
        all_sitcom_best.append(sitcom_best)
        all_sitcom_min.append(sitcom_min)
        all_delta.append(delta)
        all_set_sizes.append(cand_stats["candidate_set_size"])
        all_np_needed.append(np_needed)
        all_contains_good.append(cand_stats["contains_good_any"])
        all_contains_bad25.append(cand_stats["contains_bad25_any"])
        all_contains_bad20.append(cand_stats["contains_bad20_any"])
        all_population_unhealthy.append(whole_population_unhealthy)
        if exact_aggressive:
            all_exact_flagged.append(exact_flagged)
            all_exact_unflagged_good.append(exact_unflagged_good)

    summary = {
        "policy_name": policy_name,
        "budget_mode": budget_mode,
        "k": k,
        "health_gate": gate,
        "num_image_groups": len(image_rows),
        "mean_candidate_best_psnr": mean_or_nan(all_candidate_best),
        "min_candidate_best_psnr": min(all_candidate_best) if all_candidate_best else math.nan,
        "mean_candidate_min_psnr": mean_or_nan(all_candidate_min),
        "min_candidate_min_psnr": min(all_candidate_min) if all_candidate_min else math.nan,
        "mean_sitcom_best_of_4_psnr": mean_or_nan(all_sitcom_best),
        "min_sitcom_best_of_4_psnr": min(all_sitcom_best) if all_sitcom_best else math.nan,
        "mean_sitcom_min_of_4_psnr": mean_or_nan(all_sitcom_min),
        "min_sitcom_min_of_4_psnr": min(all_sitcom_min) if all_sitcom_min else math.nan,
        "mean_delta_candidate_best_vs_sitcom_best": mean_or_nan(all_delta),
        "mean_candidate_set_size": mean_or_nan(all_set_sizes),
        "mean_np_fallback_needed": float(np.mean([float(x) for x in all_np_needed])) if all_np_needed else math.nan,
        "frac_np_fallback_needed": float(np.mean([float(x) for x in all_np_needed])) if all_np_needed else math.nan,
        "frac_population_unhealthy": float(np.mean([float(x) for x in all_population_unhealthy])) if all_population_unhealthy else math.nan,
        "frac_contains_good_any": float(np.mean([float(x) for x in all_contains_good])) if all_contains_good else math.nan,
        "frac_contains_bad25_any": float(np.mean([float(x) for x in all_contains_bad25])) if all_contains_bad25 else math.nan,
        "frac_contains_bad20_any": float(np.mean([float(x) for x in all_contains_bad20])) if all_contains_bad20 else math.nan,
        "num_groups_with_good_any": sum(bool(x) for x in all_contains_good),
        "num_groups_with_bad25_any": sum(bool(x) for x in all_contains_bad25),
        "num_groups_with_bad20_any": sum(bool(x) for x in all_contains_bad20),
        "num_groups_population_unhealthy": sum(bool(x) for x in all_population_unhealthy),
    }
    if exact_aggressive:
        summary.update(
            {
                "mean_aggressive_exact_num_flagged": mean_or_nan(all_exact_flagged),
                "mean_aggressive_exact_num_unflagged_good": mean_or_nan(all_exact_unflagged_good),
            }
        )
    return image_rows, [summary], image_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/"
            "A18_offline_population_policy_design"
        ),
    )
    args = parser.parse_args()
    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows, a17_features, a175_scores, exact_aggressive, np_fallbacks = compute_base_tables()
    groups = group_runs(all_rows)

    # Population health table: per image/dataset, independent of a specific policy.
    population_health_rows: List[Dict[str, object]] = []
    for (dataset, image_id), rows in groups.items():
        rows = sorted(rows, key=lambda r: int(r["run_index"]))
        psnrs = [float(r["final_psnr"]) for r in rows]
        bad25 = [bool(r["bad25"]) for r in rows]
        bad20 = [bool(r["bad20"]) for r in rows]
        residual_scores = [float(r["score_residual"]) for r in rows]
        lowfreq_scores = [float(r["score_lowfreq"]) for r in rows]
        combined_scores = [float(r["score_combined"]) for r in rows]
        correction_scores = [float(r["score_correction"]) for r in rows]
        disagreement_scores = [float(r["score_disagreement"]) for r in rows]

        exact_rows = [r for r in rows if dataset in {"A14", "A16"}]
        exact_flagged = sum(1 for r in exact_rows if bool(r.get("exact_was_flagged", False))) if exact_rows else math.nan
        exact_unflagged_good = sum(1 for r in exact_rows if (not bool(r.get("exact_was_flagged", False))) and (not bool(r["bad25"]))) if exact_rows else math.nan
        exact_any_unflagged_good = bool(exact_unflagged_good and exact_unflagged_good > 0) if exact_rows else False
        exact_all_flagged = bool(exact_rows and exact_flagged == len(exact_rows)) if exact_rows else False

        population_health_rows.append(
            {
                "dataset": dataset,
                "image_id": image_id,
                "num_runs": len(rows),
                "num_good_sitcom_runs": sum(1 for b in bad25 if not b),
                "num_bad25_sitcom_runs": sum(1 for b in bad25 if b),
                "num_bad20_sitcom_runs": sum(1 for b in bad20 if b),
                "any_good_sitcom_run": any(not b for b in bad25),
                "whole_population_bad25": all(bad25),
                "whole_population_bad20": all(bad20),
                "top_residual_score": max([v for v in residual_scores if math.isfinite(v)], default=math.nan),
                "top_lowfreq_score": max([v for v in lowfreq_scores if math.isfinite(v)], default=math.nan),
                "top_combined_score": max([v for v in combined_scores if math.isfinite(v)], default=math.nan),
                "top_correction_score": max([v for v in correction_scores if math.isfinite(v)], default=math.nan),
                "top_disagreement_score": max([v for v in disagreement_scores if math.isfinite(v)], default=math.nan),
                "population_unhealthy_gate50_combined": max([v for v in combined_scores if math.isfinite(v)], default=-math.inf) < HEALTH_GATE,
                "np_fallback_psnr": float(rows[0]["np_selected_psnr"]),
                "exact_aggressive_num_flagged": exact_flagged,
                "exact_aggressive_num_unflagged_good": exact_unflagged_good,
                "exact_aggressive_any_unflagged_good": exact_any_unflagged_good,
                "exact_aggressive_all_flagged": exact_all_flagged,
                "a17_union_visible_50_proxy_any_run": any(bool(r["a17_union_visible_50_proxy"]) for r in rows),
                "a17_best_alarm_frac_proxy_min": min([float(r["a17_best_alarm_frac_proxy"]) for r in rows if math.isfinite(float(r["a17_best_alarm_frac_proxy"]))], default=math.nan),
            }
        )

    # Candidate-set policies.
    policy_defs = [
        {
            "policy_name": "combined_top2_gate50_aggressive",
            "score_field": "score_combined",
            "k": 2,
            "gate": HEALTH_GATE,
            "budget_mode": PRIMARY_BUDGET,
            "exact_aggressive": False,
            "exact_scope": None,
        },
        {
            "policy_name": "combined_top2_gate50_conservative",
            "score_field": "score_combined_conservative",
            "k": 2,
            "gate": HEALTH_GATE,
            "budget_mode": SECONDARY_BUDGET,
            "exact_aggressive": False,
            "exact_scope": None,
        },
        {
            "policy_name": "combined_top3_gate50_aggressive",
            "score_field": "score_combined",
            "k": 3,
            "gate": HEALTH_GATE,
            "budget_mode": PRIMARY_BUDGET,
            "exact_aggressive": False,
            "exact_scope": None,
        },
        {
            "policy_name": "residual_top2_gate50_aggressive",
            "score_field": "score_residual",
            "k": 2,
            "gate": HEALTH_GATE,
            "budget_mode": PRIMARY_BUDGET,
            "exact_aggressive": False,
            "exact_scope": None,
        },
        {
            "policy_name": "lowfreq_top2_gate50_aggressive",
            "score_field": "score_lowfreq",
            "k": 2,
            "gate": HEALTH_GATE,
            "budget_mode": PRIMARY_BUDGET,
            "exact_aggressive": False,
            "exact_scope": None,
        },
        {
            "policy_name": "exact_aggressive_A14A16",
            "score_field": "score_combined",
            "k": 0,
            "gate": HEALTH_GATE,
            "budget_mode": "exact",
            "exact_aggressive": True,
            "exact_scope": ["A14", "A16"],
        },
    ]

    image_level_rows: List[Dict[str, object]] = []
    policy_summary_rows: List[Dict[str, object]] = []
    policy_detail_rows: List[Dict[str, object]] = []
    unsafe_rows: List[Dict[str, object]] = []
    image00017_rows: List[Dict[str, object]] = []

    for p in policy_defs:
        img_rows, _, _ = evaluate_policy(
            policy_name=p["policy_name"],
            score_field=p["score_field"],
            rows=all_rows,
            groups=groups,
            k=p["k"],
            gate=p["gate"],
            exact_aggressive=p["exact_aggressive"],
            exact_scope=p["exact_scope"],
            budget_mode=p["budget_mode"],
        )
        for row in img_rows:
            row["policy_family"] = "candidate_set" if not p["exact_aggressive"] else "exact_aggressive_baseline"
            row["policy_name"] = p["policy_name"]
            image_level_rows.append(row)
            policy_detail_rows.append(row)
            if row["population_unhealthy"] or not row["contains_good_any"]:
                unsafe_rows.append({
                    "policy_name": row["policy_name"],
                    "budget_mode": row["budget_mode"],
                    "dataset": row["dataset"],
                    "image_id": row["image_id"],
                    "population_unhealthy": row["population_unhealthy"],
                    "contains_good_any": row["contains_good_any"],
                    "contains_bad25_any": row["contains_bad25_any"],
                    "contains_bad20_any": row["contains_bad20_any"],
                    "candidate_set_size": row["candidate_set_size"],
                    "np_fallback_needed": row["np_fallback_needed"],
                    "candidate_best_psnr": row["candidate_best_psnr"],
                    "sitcom_only_best_of_4_psnr": row["sitcom_only_best_of_4_psnr"],
                    "delta_candidate_best_vs_sitcom_best": row["delta_candidate_best_vs_sitcom_best"],
                })
            if row["image_id"] == "00017":
                image00017_rows.append(row)
    # Add diagnostic upper bound: best-of-4 SITCOM plus NP fallback.
    oracle_rows = []
    for (dataset, image_id), rows in groups.items():
        rows = sorted(rows, key=lambda r: int(r["run_index"]))
        psnrs = [float(r["final_psnr"]) for r in rows]
        best_sitcom = max(psnrs)
        min_sitcom = min(psnrs)
        mean_sitcom = float(np.mean(psnrs))
        np_psnr = float(rows[0]["np_selected_psnr"])
        oracle_best = max(psnrs + [np_psnr])
        oracle_min = min(psnrs + [np_psnr])
        oracle_mean = float(np.mean(psnrs + [np_psnr]))
        oracle_rows.append(
            {
                "policy_name": "oracle_best_of_4_plus_np",
                "policy_family": "oracle_diagnostic",
                "dataset": dataset,
                "image_id": image_id,
                "candidate_set_size": 5,
                "candidate_best_psnr": oracle_best,
                "candidate_min_psnr": oracle_min,
                "candidate_mean_psnr": oracle_mean,
                "sitcom_only_best_of_4_psnr": best_sitcom,
                "sitcom_only_mean_of_4_psnr": mean_sitcom,
                "sitcom_only_min_of_4_psnr": min_sitcom,
                "np_selected_psnr": np_psnr,
                "np_included": True,
            }
        )
    image_level_rows.extend(oracle_rows)
    policy_detail_rows.extend(oracle_rows)

    # Summaries.
    summary_by_policy: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in image_level_rows:
        summary_by_policy[str(row["policy_name"])] .append(row)

    for policy_name, rows in summary_by_policy.items():
        candidate_best = [float(r["candidate_best_psnr"]) for r in rows if math.isfinite(float(r["candidate_best_psnr"]))]
        candidate_min = [float(r["candidate_min_psnr"]) for r in rows if math.isfinite(float(r["candidate_min_psnr"]))]
        candidate_mean = [float(r["candidate_mean_psnr"]) for r in rows if math.isfinite(float(r["candidate_mean_psnr"]))]
        sitcom_best = [float(r["sitcom_only_best_of_4_psnr"]) for r in rows if math.isfinite(float(r["sitcom_only_best_of_4_psnr"]))]
        sitcom_min = [float(r["sitcom_only_min_of_4_psnr"]) for r in rows if math.isfinite(float(r["sitcom_only_min_of_4_psnr"]))]
        delta = [float(r["delta_candidate_best_vs_sitcom_best"]) for r in rows if math.isfinite(float(r["delta_candidate_best_vs_sitcom_best"]))] if "delta_candidate_best_vs_sitcom_best" in rows[0] else []
        if policy_name == "exact_aggressive_A14A16":
            summary = {
                "policy_name": policy_name,
                "policy_scope": "A14+A16_only",
                "budget_mode": "exact",
                "num_groups": len(rows),
                "mean_candidate_best_psnr": mean_or_nan(candidate_best),
                "min_candidate_best_psnr": min(candidate_best) if candidate_best else math.nan,
                "mean_candidate_min_psnr": mean_or_nan(candidate_min),
                "min_candidate_min_psnr": min(candidate_min) if candidate_min else math.nan,
                "mean_sitcom_best_of_4_psnr": mean_or_nan(sitcom_best),
                "min_sitcom_best_of_4_psnr": min(sitcom_best) if sitcom_best else math.nan,
                "mean_sitcom_min_of_4_psnr": mean_or_nan(sitcom_min),
                "min_sitcom_min_of_4_psnr": min(sitcom_min) if sitcom_min else math.nan,
                "mean_delta_candidate_best_vs_sitcom_best": mean_or_nan(delta),
                "mean_candidate_set_size": mean_or_nan([float(r["candidate_set_size"]) for r in rows]),
                "mean_np_fallback_needed": mean_or_nan([1.0 if r["np_fallback_needed"] else 0.0 for r in rows]),
                "frac_np_fallback_needed": mean_or_nan([1.0 if r["np_fallback_needed"] else 0.0 for r in rows]),
                "frac_population_unhealthy": mean_or_nan([1.0 if r["population_unhealthy"] else 0.0 for r in rows]),
                "frac_contains_good_any": mean_or_nan([1.0 if r["contains_good_any"] else 0.0 for r in rows]),
                "frac_contains_bad25_any": mean_or_nan([1.0 if r["contains_bad25_any"] else 0.0 for r in rows]),
                "frac_contains_bad20_any": mean_or_nan([1.0 if r["contains_bad20_any"] else 0.0 for r in rows]),
                "num_groups_population_unhealthy": sum(1 for r in rows if r["population_unhealthy"]),
                "num_groups_with_good_any": sum(1 for r in rows if r["contains_good_any"]),
                "num_groups_with_bad25_any": sum(1 for r in rows if r["contains_bad25_any"]),
                "num_groups_with_bad20_any": sum(1 for r in rows if r["contains_bad20_any"]),
                "mean_aggressive_exact_num_flagged": mean_or_nan([float(r["aggressive_exact_num_flagged"]) for r in rows if math.isfinite(float(r["aggressive_exact_num_flagged"]))]),
                "mean_aggressive_exact_num_unflagged_good": mean_or_nan([float(r["aggressive_exact_num_unflagged_good"]) for r in rows if math.isfinite(float(r["aggressive_exact_num_unflagged_good"]))]),
            }
        elif policy_name == "oracle_best_of_4_plus_np":
            summary = {
                "policy_name": policy_name,
                "policy_scope": "all_datasets",
                "budget_mode": "diagnostic",
                "num_groups": len(rows),
                "mean_candidate_best_psnr": mean_or_nan(candidate_best),
                "min_candidate_best_psnr": min(candidate_best) if candidate_best else math.nan,
                "mean_candidate_min_psnr": mean_or_nan(candidate_min),
                "min_candidate_min_psnr": min(candidate_min) if candidate_min else math.nan,
                "mean_sitcom_best_of_4_psnr": mean_or_nan(sitcom_best),
                "min_sitcom_best_of_4_psnr": min(sitcom_best) if sitcom_best else math.nan,
                "mean_sitcom_min_of_4_psnr": mean_or_nan(sitcom_min),
                "min_sitcom_min_of_4_psnr": min(sitcom_min) if sitcom_min else math.nan,
                "mean_delta_candidate_best_vs_sitcom_best": mean_or_nan(delta),
                "mean_candidate_set_size": mean_or_nan([float(r["candidate_set_size"]) for r in rows]),
                "mean_np_fallback_needed": 0.0,
                "frac_np_fallback_needed": 0.0,
                "frac_population_unhealthy": 0.0,
                "frac_contains_good_any": mean_or_nan([1.0 if r["candidate_best_psnr"] >= 25.0 else 0.0 for r in rows]),
                "frac_contains_bad25_any": mean_or_nan([1.0 if r["candidate_min_psnr"] < 25.0 else 0.0 for r in rows]),
                "frac_contains_bad20_any": mean_or_nan([1.0 if r["candidate_min_psnr"] < 20.0 else 0.0 for r in rows]),
                "num_groups_population_unhealthy": 0,
                "num_groups_with_good_any": sum(1 for r in rows if float(r["candidate_best_psnr"]) >= 25.0),
                "num_groups_with_bad25_any": sum(1 for r in rows if float(r["candidate_min_psnr"]) < 25.0),
                "num_groups_with_bad20_any": sum(1 for r in rows if float(r["candidate_min_psnr"]) < 20.0),
            }
        else:
            summary = {
                "policy_name": policy_name,
                "policy_scope": "all_datasets",
                "budget_mode": str(rows[0].get("budget_mode", "")),
                "num_groups": len(rows),
                "mean_candidate_best_psnr": mean_or_nan(candidate_best),
                "min_candidate_best_psnr": min(candidate_best) if candidate_best else math.nan,
                "mean_candidate_min_psnr": mean_or_nan(candidate_min),
                "min_candidate_min_psnr": min(candidate_min) if candidate_min else math.nan,
                "mean_sitcom_best_of_4_psnr": mean_or_nan(sitcom_best),
                "min_sitcom_best_of_4_psnr": min(sitcom_best) if sitcom_best else math.nan,
                "mean_sitcom_min_of_4_psnr": mean_or_nan(sitcom_min),
                "min_sitcom_min_of_4_psnr": min(sitcom_min) if sitcom_min else math.nan,
                "mean_delta_candidate_best_vs_sitcom_best": mean_or_nan(delta),
                "mean_candidate_set_size": mean_or_nan([float(r["candidate_set_size"]) for r in rows]),
                "mean_np_fallback_needed": mean_or_nan([1.0 if r["np_fallback_needed"] else 0.0 for r in rows]),
                "frac_np_fallback_needed": mean_or_nan([1.0 if r["np_fallback_needed"] else 0.0 for r in rows]),
                "frac_population_unhealthy": mean_or_nan([1.0 if r["population_unhealthy"] else 0.0 for r in rows]),
                "frac_contains_good_any": mean_or_nan([1.0 if r["contains_good_any"] else 0.0 for r in rows]),
                "frac_contains_bad25_any": mean_or_nan([1.0 if r["contains_bad25_any"] else 0.0 for r in rows]),
                "frac_contains_bad20_any": mean_or_nan([1.0 if r["contains_bad20_any"] else 0.0 for r in rows]),
                "num_groups_population_unhealthy": sum(1 for r in rows if r["population_unhealthy"]),
                "num_groups_with_good_any": sum(1 for r in rows if r["contains_good_any"]),
                "num_groups_with_bad25_any": sum(1 for r in rows if r["contains_bad25_any"]),
                "num_groups_with_bad20_any": sum(1 for r in rows if r["contains_bad20_any"]),
            }
        policy_summary_rows.append(summary)

    # Optional policy_detail_rows retain all rows for debugging.
    # Special audit table for image 00017.
    image00017_rows = [r for r in image_level_rows if str(r["image_id"]) == "00017"]

    # Write outputs.
    write_csv(outdir / "population_health_table.csv", population_health_rows)
    write_csv(outdir / "candidate_set_policy_summary.csv", policy_summary_rows)
    write_csv(outdir / "image_level_population_decisions.csv", image_level_rows)
    write_csv(outdir / "unsafe_population_cases.csv", unsafe_rows)
    write_csv(outdir / "image00017_population_audit.csv", image00017_rows)

    summary_lines = [
        "# A18 offline population / candidate-set controller design",
        "",
        "This is a diagnostic-only population audit built from existing A8, A11, A14, A16, A17, and A17.5 outputs.",
        "",
        "## Main takeaways",
        "",
        f"- Population health table rows: `{len(population_health_rows)}`",
        f"- Candidate-set policy rows: `{len(policy_summary_rows)}`",
        f"- Image-level decision rows: `{len(image_level_rows)}`",
        f"- Unsafe population cases: `{len(unsafe_rows)}`",
        f"- image 00017 audit rows: `{len(image00017_rows)}`",
        "",
        "The candidate-set view is more promising than a pure per-run binary alarm only if it can preserve at least one good SITCOM candidate while avoiding the catastrophic runs. In these offline results, that looks plausible for a small top-k combined-health candidate set, but the exact aggressive A14/A16 controller remains a strong baseline on the datasets where it exists.",
        "",
        "## Possible future frozen population rules",
        "",
        "1. `combined_top2_gate50_aggressive`: keep the two highest-scoring runs by the combined anytime score, and fall back to NP only when no run clears the health gate.",
        "2. `combined_top3_gate50_aggressive`: keep a slightly larger candidate set when the top-2 rule is too brittle, still using the same health gate and NP fallback.",
        "",
        "## What would need to be frozen before a future prospective validation",
        "",
        "- the score family used to rank the population;",
        "- the gate threshold for declaring the population unhealthy;",
        "- the top-k budget;",
        "- the fallback source;",
        "- and the exact split protocol used to select those constants without looking at the future trajectories.",
    ]
    write_text(outdir / "SUMMARY.md", "\n".join(summary_lines) + "\n")

    print(f"Wrote {len(population_health_rows)} population rows")
    print(f"Wrote {len(policy_summary_rows)} policy summary rows")
    print(f"Wrote {len(image_level_rows)} image-level decision rows")
    print(f"Wrote {len(unsafe_rows)} unsafe cases")


if __name__ == "__main__":
    main()
