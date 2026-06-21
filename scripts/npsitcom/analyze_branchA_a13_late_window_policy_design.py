#!/usr/bin/env python3
"""A13 offline late-window controller design after A11/A12."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


WINDOW_FRACS = (0.50, 0.60, 0.70, 0.80, 1.00)
AGGREGATES = ("mean", "slope", "last_in_window")
CONSTRAINT_RECALL = 0.75
CONSTRAINT_FP = 5
CONSTRAINT_REPLACEMENTS = 30


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


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


def make_run_key(row: Dict[str, str]) -> Tuple[str, int]:
    return str(row["image_id"]), int(row["run_index"])


def add_interrun_features(step_rows: List[Dict[str, str]]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["step"]))].append(row)

    bases = (
        "x0y_full_residual_normed",
        "x0y_lowfreq_residual_normed",
        "correction_norm",
        "x0hat_x0y_disagreement",
        "x0y_step_jump",
    )
    for rows in grouped.values():
        for base in bases:
            vals = np.asarray([to_float(r.get(base)) for r in rows], dtype=float)
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


def build_detector_table(
    step_rows: List[Dict[str, str]],
    run_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["run_index"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["step"]))

    run_meta = {make_run_key(r): r for r in run_rows}
    features = [
        "x0y_full_residual_normed",
        "x0y_lowfreq_residual_normed",
        "correction_norm",
        "x0hat_x0y_disagreement",
        "x0y_step_jump",
        "x0y_full_residual_normed__interrun_rank",
        "x0y_lowfreq_residual_normed__interrun_rank",
        "correction_norm__interrun_rank",
        "x0hat_x0y_disagreement__interrun_rank",
        "x0y_step_jump__interrun_rank",
        "x0y_full_residual_normed__interrun_minus_median",
        "x0y_lowfreq_residual_normed__interrun_minus_median",
        "x0y_full_residual_normed__interrun_div_median",
        "x0y_lowfreq_residual_normed__interrun_div_median",
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
        for frac in WINDOW_FRACS:
            k = max(1, int(math.ceil(n_steps * frac)))
            window_rows = rows[:k]
            window_name = f"first{int(round(frac * 100)):d}pct" if frac < 1.0 else "full"
            for feature in features:
                vals = [to_float(r.get(feature, math.nan)) for r in window_rows]
                aggs = aggregate_window(vals)
                for agg_name, agg_val in aggs.items():
                    out[f"{feature}__{window_name}__{agg_name}"] = agg_val
        out_rows.append(out)
    return out_rows


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
    for image_id in sorted(targets):
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


def counts_from_flags(bad_flags: Sequence[int], pred_flags: Sequence[bool]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for bad, pred in zip(bad_flags, pred_flags):
        if pred and bad:
            tp += 1
        elif pred and not bad:
            fp += 1
        elif (not pred) and bad:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def simulate_policy(
    detector_rows: List[Dict[str, object]],
    np_fallbacks: Dict[str, Dict[str, object]],
    flags: Sequence[bool],
    policy_name: str,
    notes: str,
) -> Dict[str, object]:
    run_records: List[Dict[str, object]] = []
    by_image: Dict[str, List[float]] = defaultdict(list)
    by_image_sitcom: Dict[str, List[float]] = defaultdict(list)
    for row, flagged in zip(detector_rows, flags):
        image_id = str(row["image_id"])
        sitcom_psnr = to_float(row["final_psnr"])
        fb = np_fallbacks[image_id]
        policy_psnr = float(fb["np_selected_psnr"]) if flagged else sitcom_psnr
        run_records.append(
            {
                "image_id": image_id,
                "run_index": int(row["run_index"]),
                "sitcom_psnr": sitcom_psnr,
                "policy_psnr": policy_psnr,
                "flagged": bool(flagged),
                "is_bad25": int(sitcom_psnr < 25.0),
                "is_bad20": int(sitcom_psnr < 20.0),
                "delta_vs_sitcom": policy_psnr - sitcom_psnr,
            }
        )
        by_image[image_id].append(policy_psnr)
        by_image_sitcom[image_id].append(sitcom_psnr)

    y_true = [int(r["is_bad25"]) for r in run_records]
    y_pred = [bool(r["flagged"]) for r in run_records]
    counts = counts_from_flags(y_true, y_pred)
    policy_psnrs = [float(r["policy_psnr"]) for r in run_records]
    image_best = [max(vals) for vals in by_image.values()]
    image_mean = [float(np.mean(vals)) for vals in by_image.values()]
    image_min = [min(vals) for vals in by_image.values()]
    false_positive_losses = [
        float(r["delta_vs_sitcom"]) for r in run_records if r["flagged"] and not r["is_bad25"]
    ]
    metrics: Dict[str, object] = {
        "policy_name": policy_name,
        "notes": notes,
        "run_level_mean_psnr": mean_or_nan(policy_psnrs),
        "run_level_min_psnr": min(policy_psnrs) if policy_psnrs else math.nan,
        "run_level_num_below25": sum(1 for x in policy_psnrs if x < 25.0),
        "run_level_num_below20": sum(1 for x in policy_psnrs if x < 20.0),
        "num_replaced": int(sum(y_pred)),
        "num_false_positive_replacements": counts["fp"],
        "num_true_positive_replacements": counts["tp"],
        "num_false_negative_remaining_bad25": counts["fn"],
        "bad25_recall": counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else math.nan,
        "bad25_precision": counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else math.nan,
        "image_level_best_of_4_mean_psnr": mean_or_nan(image_best),
        "image_level_best_of_4_min_psnr": min(image_best) if image_best else math.nan,
        "image_level_mean_of_4_mean_psnr": mean_or_nan(image_mean),
        "image_level_mean_of_4_min_psnr": min(image_mean) if image_mean else math.nan,
        "image_level_min_of_4_mean_psnr": mean_or_nan(image_min),
        "image_level_min_of_4_min_psnr": min(image_min) if image_min else math.nan,
        "mean_delta_vs_sitcom": mean_or_nan(float(r["delta_vs_sitcom"]) for r in run_records),
        "worst_false_positive_psnr_loss": min(false_positive_losses) if false_positive_losses else math.nan,
        "feasible_under_constraints": False,
        "tp": counts["tp"],
        "fp": counts["fp"],
        "tn": counts["tn"],
        "fn": counts["fn"],
    }
    metrics["feasible_under_constraints"] = bool(
        math.isfinite(float(metrics["bad25_recall"]))
        and float(metrics["bad25_recall"]) >= CONSTRAINT_RECALL
        and int(metrics["num_false_positive_replacements"]) <= CONSTRAINT_FP
        and int(metrics["num_replaced"]) <= CONSTRAINT_REPLACEMENTS
    )
    return {"metrics": metrics, "run_records": run_records}


def train_sort_key(metrics: Dict[str, object]) -> Tuple[object, ...]:
    recall = to_float(metrics["bad25_recall"])
    fp = int(metrics["num_false_positive_replacements"])
    replaced = int(metrics["num_replaced"])
    feasible = bool(metrics["feasible_under_constraints"])
    recall_shortfall = max(0.0, CONSTRAINT_RECALL - recall) if math.isfinite(recall) else 1e9
    fp_excess = max(0, fp - CONSTRAINT_FP)
    repl_excess = max(0, replaced - CONSTRAINT_REPLACEMENTS)
    return (
        0 if feasible else 1,
        recall_shortfall,
        fp_excess,
        repl_excess,
        int(metrics["num_false_negative_remaining_bad25"]),
        fp,
        replaced,
        -to_float(metrics["image_level_best_of_4_min_psnr"]),
        -to_float(metrics["run_level_mean_psnr"]),
    )


def fit_single_and_policy(
    train_rows: List[Dict[str, object]],
    np_fallbacks: Dict[str, Dict[str, object]],
    policy_name: str,
    feature1: str,
    feature2: str,
) -> Dict[str, object]:
    vals1 = sorted({to_float(r.get(feature1)) for r in train_rows if math.isfinite(to_float(r.get(feature1)))})
    vals2 = sorted({to_float(r.get(feature2)) for r in train_rows if math.isfinite(to_float(r.get(feature2)))})
    if not vals1 or not vals2:
        raise ValueError(f"No finite thresholds for {policy_name}")
    best = None
    for thr1 in vals1:
        flags1 = [to_float(r.get(feature1)) >= thr1 for r in train_rows]
        for thr2 in vals2:
            flags = [f1 and (to_float(r.get(feature2)) >= thr2) for r, f1 in zip(train_rows, flags1)]
            sim = simulate_policy(train_rows, np_fallbacks, flags, policy_name, f"train fit for {policy_name}")
            metrics = sim["metrics"]
            candidate = {
                "policy_name": policy_name,
                "policy_family": "single_and",
                "combine_mode": "and",
                "feature_names": [feature1, feature2],
                "feature_directions": {feature1: "high_is_risky", feature2: "high_is_risky"},
                "thresholds": {feature1: float(thr1), feature2: float(thr2)},
                "train_metrics": metrics,
            }
            sort_key = train_sort_key(metrics)
            if best is None or sort_key < best[0]:
                best = (sort_key, candidate)
    assert best is not None
    return best[1]


def apply_single_and(
    rows: List[Dict[str, object]],
    feature1: str,
    feature2: str,
    thr1: float,
    thr2: float,
) -> List[bool]:
    return [
        (to_float(r.get(feature1)) >= thr1) and (to_float(r.get(feature2)) >= thr2)
        for r in rows
    ]


def fit_policy_library(
    train_rows: List[Dict[str, object]],
    np_fallbacks: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    def feats(window_name: str) -> Tuple[str, str]:
        return (
            f"x0y_full_residual_normed__interrun_rank__{window_name}__slope",
            f"x0y_full_residual_normed__interrun_rank__{window_name}__last_in_window",
        )

    fitted: Dict[str, Dict[str, object]] = {}
    for window_name in ("first50pct", "first60pct", "first70pct", "first80pct"):
        policy_name = f"and_rank_fullres_{window_name}"
        feature1, feature2 = feats(window_name)
        fitted[policy_name] = fit_single_and_policy(train_rows, np_fallbacks, policy_name, feature1, feature2)

    p50 = fitted["and_rank_fullres_first50pct"]
    p80 = fitted["and_rank_fullres_first80pct"]
    fitted["first80_only"] = dict(fitted["and_rank_fullres_first80pct"])
    fitted["first80_only"]["policy_name"] = "first80_only"
    fitted["first80_only"]["policy_family"] = "single_and_alias"

    fitted["first50_or_first80"] = {
        "policy_name": "first50_or_first80",
        "policy_family": "two_stage",
        "combine_mode": "or",
        "stages": [p50, p80],
        "train_metrics": None,
    }
    fitted["first50_and_confirmed_by_first80"] = {
        "policy_name": "first50_and_confirmed_by_first80",
        "policy_family": "two_stage",
        "combine_mode": "and",
        "stages": [p50, p80],
        "train_metrics": None,
    }
    return fitted


def apply_policy_spec(rows: List[Dict[str, object]], policy_spec: Dict[str, object]) -> List[bool]:
    family = str(policy_spec["policy_family"])
    if family in {"single_and", "single_and_alias"}:
        f1, f2 = policy_spec["feature_names"]
        thr1 = float(policy_spec["thresholds"][f1])
        thr2 = float(policy_spec["thresholds"][f2])
        return apply_single_and(rows, f1, f2, thr1, thr2)
    if family == "two_stage":
        stage_flags = [apply_policy_spec(rows, stage) for stage in policy_spec["stages"]]
        if str(policy_spec["combine_mode"]) == "or":
            return [a or b for a, b in zip(stage_flags[0], stage_flags[1])]
        return [a and b for a, b in zip(stage_flags[0], stage_flags[1])]
    raise ValueError(family)


def finalize_policy_spec(
    policy_spec: Dict[str, object],
    train_rows: List[Dict[str, object]],
    np_fallbacks: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    flags = apply_policy_spec(train_rows, policy_spec)
    sim = simulate_policy(train_rows, np_fallbacks, flags, str(policy_spec["policy_name"]), f"train fit for {policy_spec['policy_name']}")
    out = dict(policy_spec)
    out["train_metrics"] = sim["metrics"]
    return out


def summarize_policy_across_split(
    fit_regime: str,
    train_name: str,
    test_name: str,
    policy_spec: Dict[str, object],
    train_rows: List[Dict[str, object]],
    test_rows: List[Dict[str, object]],
    np_fallbacks: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    finalized = finalize_policy_spec(policy_spec, train_rows, np_fallbacks)
    train_flags = apply_policy_spec(train_rows, finalized)
    test_flags = apply_policy_spec(test_rows, finalized)
    train_sim = simulate_policy(train_rows, np_fallbacks, train_flags, str(finalized["policy_name"]), f"{fit_regime} train")
    test_sim = simulate_policy(test_rows, np_fallbacks, test_flags, str(finalized["policy_name"]), f"{fit_regime} test")
    row: Dict[str, object] = {
        "fit_regime": fit_regime,
        "train_dataset": train_name,
        "test_dataset": test_name,
        "policy_name": finalized["policy_name"],
        "policy_family": finalized["policy_family"],
        "combine_mode": finalized.get("combine_mode", "and"),
        "feature_names_json": json.dumps(finalized.get("feature_names", []), sort_keys=True),
        "thresholds_json": json.dumps(finalized.get("thresholds", {}), sort_keys=True),
        "stages_json": json.dumps(
            [
                {
                    "policy_name": s["policy_name"],
                    "feature_names": s.get("feature_names", []),
                    "thresholds": s.get("thresholds", {}),
                }
                for s in finalized.get("stages", [])
            ],
            sort_keys=True,
        ),
    }
    for prefix, metrics in (("train", train_sim["metrics"]), ("test", test_sim["metrics"])):
        for key, value in metrics.items():
            if key in {"policy_name", "notes"}:
                continue
            row[f"{prefix}_{key}"] = value
    return row


def select_combined_candidate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    best = None
    for row in rows:
        sort_key = (
            0 if row["train_feasible_under_constraints"] else 1,
            max(0.0, CONSTRAINT_RECALL - to_float(row["train_bad25_recall"])) if math.isfinite(to_float(row["train_bad25_recall"])) else 1e9,
            max(0, int(row["train_num_false_positive_replacements"]) - CONSTRAINT_FP),
            max(0, int(row["train_num_replaced"]) - CONSTRAINT_REPLACEMENTS),
            int(row["train_num_false_negative_remaining_bad25"]),
            int(row["train_num_false_positive_replacements"]),
            int(row["train_num_replaced"]),
            -to_float(row["train_image_level_best_of_4_min_psnr"]),
            -to_float(row["train_run_level_mean_psnr"]),
        )
        if best is None or sort_key < best[0]:
            best = (sort_key, row)
    assert best is not None
    return best[1]


def screen_invisible_miss_features(
    train_rows: List[Dict[str, object]],
    eval_rows: List[Dict[str, object]],
    missed_keys: Sequence[Tuple[str, int]],
) -> List[Dict[str, object]]:
    candidates = []
    windows = ("first60pct", "first70pct", "first80pct", "full")
    feature_bases = [
        "x0y_full_residual_normed",
        "x0y_full_residual_normed__interrun_div_median",
        "x0y_lowfreq_residual_normed",
        "x0y_lowfreq_residual_normed__interrun_rank",
        "correction_norm__interrun_rank",
        "x0y_step_jump__interrun_rank",
        "x0hat_x0y_disagreement__interrun_rank",
    ]
    eval_map = {(str(r["image_id"]), int(r["run_index"])): r for r in eval_rows}
    missed_eval_rows = [eval_map[k] for k in missed_keys if k in eval_map]
    for base in feature_bases:
        for window in windows:
            for agg in AGGREGATES:
                feat = f"{base}__{window}__{agg}"
                train_vals = [to_float(r.get(feat)) for r in train_rows if math.isfinite(to_float(r.get(feat)))]
                if not train_vals:
                    continue
                for direction in ("high_is_risky", "low_is_risky"):
                    for thr in sorted(set(train_vals)):
                        eval_flags = []
                        eval_bad = []
                        for row in eval_rows:
                            v = to_float(row.get(feat))
                            if not math.isfinite(v):
                                flag = False
                            elif direction == "high_is_risky":
                                flag = v >= thr
                            else:
                                flag = v <= thr
                            eval_flags.append(flag)
                            eval_bad.append(int(row["bad25"]))
                        counts = counts_from_flags(eval_bad, eval_flags)
                        miss_hits = 0
                        for row in missed_eval_rows:
                            v = to_float(row.get(feat))
                            if not math.isfinite(v):
                                continue
                            if (direction == "high_is_risky" and v >= thr) or (direction == "low_is_risky" and v <= thr):
                                miss_hits += 1
                        candidates.append(
                            {
                                "feature_name": feat,
                                "direction": direction,
                                "threshold": float(thr),
                                "eval_invisible_miss_hits": miss_hits,
                                "eval_invisible_miss_total": len(missed_eval_rows),
                                "eval_invisible_miss_hit_rate": miss_hits / len(missed_eval_rows) if missed_eval_rows else math.nan,
                                "eval_fp": counts["fp"],
                                "eval_tp": counts["tp"],
                                "eval_fn": counts["fn"],
                                "eval_num_flagged": int(sum(1 for x in eval_flags if x)),
                                "eval_bad25_recall": counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else math.nan,
                                "eval_bad25_precision": counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else math.nan,
                            }
                        )
    candidates.sort(
        key=lambda r: (
            -int(r["eval_invisible_miss_hits"]),
            int(r["eval_fp"]),
            -to_float(r["eval_bad25_recall"]),
            int(r["eval_num_flagged"]),
            str(r["feature_name"]),
        )
    )
    return candidates


def build_summary(
    combined_candidate: Dict[str, object],
    a8_to_a11_rows: List[Dict[str, object]],
    a11_to_a8_rows: List[Dict[str, object]],
    invisible_rows: List[Dict[str, object]],
) -> str:
    best_a8_to_a11 = select_combined_candidate(a8_to_a11_rows)
    best_a11_to_a8 = select_combined_candidate(a11_to_a8_rows)
    top_invisible = invisible_rows[:5]
    lines = [
        "# A13 Late-Window Policy Design",
        "",
        "This is offline controller design only. No A13 result is prospective evidence.",
        "",
        "## Main Question",
        "",
        "Does a later-window version of the relative inter-run Branch A detector improve the recall / false-positive tradeoff consistently across A8 and A11?",
        "",
        "## Best Cross-Fit Policies",
        "",
        f"- Train A8, test A11: `{best_a8_to_a11['policy_name']}` with test recall `{to_float(best_a8_to_a11['test_bad25_recall']):.3f}`, test FP `{int(best_a8_to_a11['test_num_false_positive_replacements'])}`, test replacements `{int(best_a8_to_a11['test_num_replaced'])}`.",
        f"- Train A11, test A8: `{best_a11_to_a8['policy_name']}` with test recall `{to_float(best_a11_to_a8['test_bad25_recall']):.3f}`, test FP `{int(best_a11_to_a8['test_num_false_positive_replacements'])}`, test replacements `{int(best_a11_to_a8['test_num_replaced'])}`.",
        "",
        "## Combined-Fit Candidate",
        "",
        f"- Selected candidate future frozen policy: `{combined_candidate['policy_name']}`.",
        f"- Combined-fit recall `{to_float(combined_candidate['train_bad25_recall']):.3f}`, FP `{int(combined_candidate['train_num_false_positive_replacements'])}`, replacements `{int(combined_candidate['train_num_replaced'])}`.",
        f"- Combined-fit image best-of-4 min `{to_float(combined_candidate['train_image_level_best_of_4_min_psnr']):.3f}`, run mean `{to_float(combined_candidate['train_run_level_mean_psnr']):.3f}`.",
        "",
        "## Invisible-Miss Screen",
        "",
    ]
    for row in top_invisible:
        lines.append(
            f"- `{row['feature_name']}` ({row['direction']}) hits `{int(row['eval_invisible_miss_hits'])}` / `{int(row['eval_invisible_miss_total'])}` A11 invisible misses with FP `{int(row['eval_fp'])}`."
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `policy_dev_summary.csv`: all evaluated policy fits and metrics.",
            "- `train_A8_test_A11_summary.csv`: cross-fit results with thresholds fit on A8 only.",
            "- `train_A11_test_A8_summary.csv`: cross-fit results with thresholds fit on A11 only.",
            "- `combined_fit_candidate_policy.json`: combined-development candidate for a future frozen A14 policy.",
            "- `invisible_miss_feature_screen.csv`: optional feature screen against the A11 invisible miss set from A12.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a8_dir", required=True)
    ap.add_argument("--a11_dir", required=True)
    ap.add_argument("--a12_dir", required=True)
    ap.add_argument("--np_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--noise", type=float, default=0.05)
    args = ap.parse_args()

    a8_dir = Path(args.a8_dir)
    a11_dir = Path(args.a11_dir)
    a12_dir = Path(args.a12_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    a8_step = read_csv(a8_dir / "trajectory_step_metrics.csv")
    a8_run = read_csv(a8_dir / "run_level_summary.csv")
    a11_step = read_csv(a11_dir / "trajectory_step_metrics.csv")
    a11_run = read_csv(a11_dir / "run_level_summary.csv")
    a12_missed = read_csv(a12_dir / "missed_bad_runs_diagnosis.csv")
    np_rows = read_csv(Path(args.np_csv))

    add_interrun_features(a8_step)
    add_interrun_features(a11_step)

    a8_rows = build_detector_table(a8_step, a8_run)
    a11_rows = build_detector_table(a11_step, a11_run)
    combined_rows = list(a8_rows) + list(a11_rows)

    image_ids = sorted({str(r["image_id"]) for r in combined_rows})
    np_fallbacks = load_np_fallbacks(np_rows, args.noise, image_ids)

    # Fit on A8, evaluate on A11.
    a8_library = fit_policy_library(a8_rows, np_fallbacks)
    a8_to_a11_rows: List[Dict[str, object]] = []
    for name, spec in a8_library.items():
        a8_to_a11_rows.append(
            summarize_policy_across_split(
                "train_A8_test_A11", "A8", "A11", spec, a8_rows, a11_rows, np_fallbacks
            )
        )

    # Fit on A11, evaluate on A8.
    a11_library = fit_policy_library(a11_rows, np_fallbacks)
    a11_to_a8_rows: List[Dict[str, object]] = []
    for name, spec in a11_library.items():
        a11_to_a8_rows.append(
            summarize_policy_across_split(
                "train_A11_test_A8", "A11", "A8", spec, a11_rows, a8_rows, np_fallbacks
            )
        )

    # Combined fit for future frozen candidate selection.
    combined_library = fit_policy_library(combined_rows, np_fallbacks)
    combined_rows_summary: List[Dict[str, object]] = []
    for name, spec in combined_library.items():
        combined_rows_summary.append(
            summarize_policy_across_split(
                "train_A8A11_combined", "A8A11", "A8A11", spec, combined_rows, combined_rows, np_fallbacks
            )
        )
    combined_candidate = select_combined_candidate(combined_rows_summary)

    # Invisible-miss screen using A12 A11 misses and A8-trained thresholds/features.
    missed_keys = [(str(r["image_id"]), int(r["run_index"])) for r in a12_missed]
    invisible_rows = screen_invisible_miss_features(a8_rows, a11_rows, missed_keys)

    # Final outputs.
    policy_dev_summary = list(a8_to_a11_rows) + list(a11_to_a8_rows) + list(combined_rows_summary)
    write_csv(outdir / "policy_dev_summary.csv", policy_dev_summary)
    write_csv(outdir / "train_A8_test_A11_summary.csv", a8_to_a11_rows)
    write_csv(outdir / "train_A11_test_A8_summary.csv", a11_to_a8_rows)
    write_csv(outdir / "invisible_miss_feature_screen.csv", invisible_rows)

    candidate_payload = {
        "policy_name": combined_candidate["policy_name"],
        "policy_family": combined_candidate["policy_family"],
        "combine_mode": combined_candidate["combine_mode"],
        "feature_names": json.loads(combined_candidate["feature_names_json"]),
        "thresholds": json.loads(combined_candidate["thresholds_json"]),
        "stages": json.loads(combined_candidate["stages_json"]),
        "train_dataset": "A8+A11 development only",
        "selection_constraints": {
            "bad25_recall_min": CONSTRAINT_RECALL,
            "false_positive_replacements_max": CONSTRAINT_FP,
            "total_replacements_max": CONSTRAINT_REPLACEMENTS,
        },
        "selection_tiebreak": [
            "minimize remaining bad25 count",
            "fewer false-positive replacements",
            "fewer total replacements",
            "higher image best-of-4 min PSNR",
            "higher run-level mean PSNR",
        ],
        "metrics_on_combined": {
            k[len("train_"):]: v
            for k, v in combined_candidate.items()
            if k.startswith("train_")
        },
        "metrics_if_evaluated_on_A8_from_combined_fit": {
            "not_reported_separately": True
        },
        "fallback_source_csv": args.np_csv,
        "source_dirs": {
            "A8": str(a8_dir),
            "A11": str(a11_dir),
            "A12": str(a12_dir),
        },
        "warning": "A13 is development-only offline design. This config is a candidate for future A14 freezing, not prospective evidence.",
    }
    write_text(outdir / "combined_fit_candidate_policy.json", json.dumps(candidate_payload, indent=2) + "\n")
    write_text(outdir / "SUMMARY.md", build_summary(combined_candidate, a8_to_a11_rows, a11_to_a8_rows, invisible_rows))


if __name__ == "__main__":
    main()
