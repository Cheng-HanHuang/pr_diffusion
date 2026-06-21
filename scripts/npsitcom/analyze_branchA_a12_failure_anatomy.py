#!/usr/bin/env python3
"""A12 failure anatomy and detector diagnosis for A11."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


WINDOWS = (0.50, 0.60, 0.70, 0.80, 1.00)


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


def to_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {'1', 'true', 'yes', 'y'}


def mean_or_nan(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else math.nan


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


def apply_threshold(value: float, direction: str, threshold: float) -> bool:
    if direction == "high_is_risky":
        return value >= threshold
    if direction == "low_is_risky":
        return value <= threshold
    raise ValueError(direction)


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
            raise ValueError(f"No NP fallback candidates for {image_id}")
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
            order = np.argsort(vals, kind="mergesort")
            ranks = np.empty(vals.size, dtype=float)
            ranks[order] = np.arange(1, vals.size + 1, dtype=float)
            for i, row in enumerate(rows):
                row[f"{base}__interrun_rank"] = float(ranks[i])


def build_window_feature_table(
    step_rows: List[Dict[str, str]],
    feature_names: Sequence[str],
    window_fracs: Sequence[float],
) -> Dict[Tuple[str, int], Dict[str, float]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["run_index"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["step"]))

    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for run_key, rows in grouped.items():
        feat_map: Dict[str, float] = {}
        n_steps = len(rows)
        for frac in window_fracs:
            k = max(1, int(math.ceil(n_steps * frac)))
            window_name = f"first{int(round(frac * 100)):d}pct" if frac < 1.0 else "full"
            window_rows = rows[:k]
            for feature in feature_names:
                raw_feature = feature.split("__first", 1)[0]
                agg_name = feature.rsplit("__", 1)[-1]
                vals = [to_float(r.get(raw_feature, math.nan)) for r in window_rows]
                feat_map[f"{raw_feature}__{window_name}__{agg_name}"] = aggregate_window(vals)[agg_name]
        out[run_key] = feat_map
    return out


def confusion_label(row: Dict[str, str]) -> str:
    flagged = str(row["was_flagged"]).lower() in {"true", "1"}
    bad25 = str(row["sitcom_bad_below25"]).lower() in {"true", "1"}
    if flagged and bad25:
        return "TP"
    if flagged and not bad25:
        return "FP"
    if (not flagged) and bad25:
        return "FN"
    return "TN"


def summarize_image_level_survival(image_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in image_rows:
        out[(str(row["policy_name"]), str(row["image_id"]))] = {
            "best_of_4_psnr": to_float(row["best_of_4_psnr"]),
            "mean_of_4_psnr": to_float(row["mean_of_4_psnr"]),
            "min_of_4_psnr": to_float(row["min_of_4_psnr"]),
            "num_runs_below25": to_float(row["num_runs_below25"]),
            "num_runs_below20": to_float(row["num_runs_below20"]),
        }
    return out


def build_confusion_run_table(
    frozen_rows: List[Dict[str, str]],
    np_fallbacks: Dict[str, Dict[str, object]],
    image_policy_stats: Dict[Tuple[str, str], Dict[str, float]],
    feature_names: Sequence[str],
    thresholds: Dict[str, float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in frozen_rows:
        image_id = str(row["image_id"])
        label = confusion_label(row)
        fb = np_fallbacks[image_id]
        out: Dict[str, object] = {
            "confusion_label": label,
            "image_id": image_id,
            "run_index": int(row["run_index"]),
            "final_sitcom_psnr": to_float(row["sitcom_final_psnr"]),
            "fallback_np_selected_psnr": float(fb["np_selected_psnr"]),
            "policy_output_psnr": to_float(row["policy_final_psnr"]),
            "replacement_detail": row["replacement_detail"],
            "was_flagged": row["was_flagged"],
            "was_replaced": row["was_replaced"],
            "psnr_delta_vs_sitcom": to_float(row["delta_vs_sitcom"]),
        }
        for feature in feature_names:
            value = to_float(row[f"{feature}__value"])
            threshold = float(thresholds[feature])
            out[f"{feature}__value"] = value
            out[f"{feature}__threshold"] = threshold
            out[f"{feature}__distance_to_threshold"] = value - threshold
            out[f"{feature}__flagged"] = row[f"{feature}__flagged"]
        sitcom_stats = image_policy_stats[("sitcom_only", image_id)]
        frozen_stats = image_policy_stats[("frozen_A11_policy", image_id)]
        out["sitcom_image_best_of4_psnr"] = sitcom_stats["best_of_4_psnr"]
        out["frozen_image_best_of4_psnr"] = frozen_stats["best_of_4_psnr"]
        out["image_best_of4_survived"] = abs(sitcom_stats["best_of_4_psnr"] - frozen_stats["best_of_4_psnr"]) < 1e-9
        rows.append(out)
    rows.sort(key=lambda r: (str(r["confusion_label"]), str(r["image_id"]), int(r["run_index"])))
    return rows


def build_missed_bad_runs_diagnosis(
    confusion_rows: List[Dict[str, object]],
    late_feature_map: Dict[Tuple[str, int], Dict[str, float]],
    feature_names: Sequence[str],
    thresholds: Dict[str, float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in confusion_rows:
        if str(row["confusion_label"]) != "FN":
            continue
        run_key = (str(row["image_id"]), int(row["run_index"]))
        late_feats = late_feature_map[run_key]
        out = dict(row)
        f1, f2 = feature_names
        f1_flag = to_bool(row[f"{f1}__flagged"])
        f2_flag = to_bool(row[f"{f2}__flagged"])
        if (not f1_flag) and (not f2_flag):
            out["missed_by_feature"] = "both"
        elif not f1_flag:
            out["missed_by_feature"] = "feature1_only"
        elif not f2_flag:
            out["missed_by_feature"] = "feature2_only"
        else:
            out["missed_by_feature"] = "none"

        first50_risky = f1_flag and f2_flag
        out["became_risky_after_first50pct"] = False
        for frac in WINDOWS[1:]:
            window_name = f"first{int(round(frac * 100)):d}pct" if frac < 1.0 else "full"
            f1_key = f"{f1.split('__first', 1)[0]}__{window_name}__{f1.rsplit('__', 1)[-1]}"
            f2_key = f"{f2.split('__first', 1)[0]}__{window_name}__{f2.rsplit('__', 1)[-1]}"
            f1_val = to_float(late_feats.get(f1_key, math.nan))
            f2_val = to_float(late_feats.get(f2_key, math.nan))
            f1_late_flag = apply_threshold(f1_val, "high_is_risky", float(thresholds[f1]))
            f2_late_flag = apply_threshold(f2_val, "high_is_risky", float(thresholds[f2]))
            out[f"{window_name}__feature1_value"] = f1_val
            out[f"{window_name}__feature2_value"] = f2_val
            out[f"{window_name}__feature1_flagged"] = f1_late_flag
            out[f"{window_name}__feature2_flagged"] = f2_late_flag
            out[f"{window_name}__both_flagged"] = f1_late_flag and f2_late_flag
            if (not first50_risky) and f1_late_flag and f2_late_flag:
                out["became_risky_after_first50pct"] = True
        rows.append(out)

    counts = Counter(str(r["image_id"]) for r in rows)
    for row in rows:
        row["miss_count_for_image"] = counts[str(row["image_id"])]
    rows.sort(key=lambda r: (-int(r["miss_count_for_image"]), str(r["image_id"]), int(r["run_index"])))
    return rows


def build_false_positive_diagnosis(confusion_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in confusion_rows:
        if str(row["confusion_label"]) != "FP":
            continue
        out = dict(row)
        out["psnr_loss_from_replacement"] = -to_float(row["psnr_delta_vs_sitcom"])
        f1 = "x0y_full_residual_normed__interrun_rank__first50pct__slope"
        f2 = "x0y_full_residual_normed__interrun_rank__first50pct__last_in_window"
        out["feature1_strongly_flagged"] = to_float(row[f"{f1}__distance_to_threshold"]) > 0.25
        out["feature2_strongly_flagged"] = to_float(row[f"{f2}__distance_to_threshold"]) > 1.0
        rows.append(out)
    rows.sort(key=lambda r: (-to_float(r["psnr_loss_from_replacement"]), str(r["image_id"]), int(r["run_index"])))
    return rows


def build_group_curve_summary(
    step_rows: List[Dict[str, str]],
    confusion_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    label_by_run = {
        (str(r["image_id"]), int(r["run_index"])): str(r["confusion_label"])
        for r in confusion_rows
    }
    out: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["run_index"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["step"]))

    per_group_step: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for run_key, rows in grouped.items():
        label = label_by_run[run_key]
        for row in rows:
            per_group_step[(label, int(row["step"]))].append(row)

    metrics = [
        ("x0y_full_residual_normed__interrun_rank", "x0y_full_residual_rank"),
        ("x0y_full_residual_normed", "x0y_full_residual_value"),
        ("correction_norm", "correction_norm"),
        ("x0hat_x0y_disagreement", "disagreement"),
        ("x0y_step_jump", "step_to_step_jump"),
    ]
    for (label, step), rows in sorted(per_group_step.items()):
        out_row: Dict[str, object] = {
            "confusion_label": label,
            "step": step,
            "sigma_mean": mean_or_nan(to_float(r["sigma"]) for r in rows),
            "num_runs": len(rows),
        }
        for key, short_name in metrics:
            out_row[f"{short_name}__mean"] = mean_or_nan(to_float(r.get(key, math.nan)) for r in rows)
            out_row[f"{short_name}__std"] = float(np.std([to_float(r.get(key, math.nan)) for r in rows if math.isfinite(to_float(r.get(key, math.nan)))], ddof=0)) if any(math.isfinite(to_float(r.get(key, math.nan))) for r in rows) else math.nan
        out.append(out_row)
    return out


def build_diagnostic_late_window_screen(
    confusion_rows: List[Dict[str, object]],
    late_feature_map: Dict[Tuple[str, int], Dict[str, float]],
    feature_names: Sequence[str],
    thresholds: Dict[str, float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for frac in WINDOWS:
        window_name = f"first{int(round(frac * 100)):d}pct" if frac < 1.0 else "full"
        tp = fp = tn = fn = 0
        for row in confusion_rows:
            run_key = (str(row["image_id"]), int(row["run_index"]))
            late_feats = late_feature_map[run_key]
            flags = []
            for feature in feature_names:
                raw_feature = feature.split("__first", 1)[0]
                agg_name = feature.rsplit("__", 1)[-1]
                key = f"{raw_feature}__{window_name}__{agg_name}"
                value = to_float(late_feats.get(key, math.nan))
                flags.append(apply_threshold(value, "high_is_risky", float(thresholds[feature])))
            pred = all(flags)
            bad25 = to_float(row["final_sitcom_psnr"]) < 25.0
            if pred and bad25:
                tp += 1
            elif pred and (not bad25):
                fp += 1
            elif (not pred) and bad25:
                fn += 1
            else:
                tn += 1
        recall = tp / (tp + fn) if (tp + fn) else math.nan
        precision = tp / (tp + fp) if (tp + fp) else math.nan
        rows.append(
            {
                "window_name": window_name,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "bad25_recall": recall,
                "precision": precision,
                "num_flagged": tp + fp,
                "diagnostic_only": True,
            }
        )
    return rows


def build_summary(
    confusion_rows: List[Dict[str, object]],
    missed_rows: List[Dict[str, object]],
    false_positive_rows: List[Dict[str, object]],
    late_screen_rows: List[Dict[str, object]],
) -> str:
    counts = Counter(str(r["confusion_label"]) for r in confusion_rows)
    miss_images = Counter(str(r["image_id"]) for r in missed_rows)
    fp_loss_mean = mean_or_nan(to_float(r["psnr_loss_from_replacement"]) for r in false_positive_rows)
    lines = [
        "# A12 A11 Failure Anatomy",
        "",
        "This analysis is offline only. It diagnoses the A11 prospective frozen-policy result without changing the policy or running new SITCOM jobs.",
        "",
        "## Confusion Counts",
        "",
        f"- TP: `{counts.get('TP', 0)}`",
        f"- FP: `{counts.get('FP', 0)}`",
        f"- FN: `{counts.get('FN', 0)}`",
        f"- TN: `{counts.get('TN', 0)}`",
        "",
        "## Missed Bad Runs",
        "",
        f"- Missed bad25 runs: `{len(missed_rows)}`",
        f"- Images with misses: `{', '.join(f'{image}:{count}' for image, count in sorted(miss_images.items()))}`",
        "",
        "## False Positive Replacements",
        "",
        f"- False positive replacements: `{len(false_positive_rows)}`",
        f"- Mean PSNR loss on false positives: `{fp_loss_mean:.3f}`" if math.isfinite(fp_loss_mean) else "- Mean PSNR loss on false positives: `nan`",
        "",
        "## Diagnostic Late Windows",
        "",
    ]
    for row in late_screen_rows:
        lines.append(
            f"- `{row['window_name']}`: recall `{to_float(row['bad25_recall']):.3f}`, "
            f"precision `{to_float(row['precision']):.3f}`, TP/FP/FN `{int(row['tp'])}` / `{int(row['fp'])}` / `{int(row['fn'])}`."
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `a11_confusion_run_table.csv`: per-run TP/FP/FN/TN table with feature distances to threshold.",
            "- `missed_bad_runs_diagnosis.csv`: missed-run anatomy, including late-window diagnostic flags.",
            "- `false_positive_diagnosis.csv`: false-positive replacement losses and threshold proximity.",
            "- `trajectory_group_curve_summary.csv`: mean per-step curves for TP/FP/FN/TN groups.",
            "- `diagnostic_late_window_screen.csv`: diagnostic-only later-window screening results.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a11_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--noise", type=float, default=0.05)
    args = ap.parse_args()

    a11_dir = Path(args.a11_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(a11_dir / "frozen_policy.json", encoding="utf-8") as f:
        frozen_cfg = json.load(f)

    frozen_rows = read_csv(a11_dir / "frozen_policy_applied_runs.csv")
    step_rows = read_csv(a11_dir / "trajectory_step_metrics.csv")
    run_rows = read_csv(a11_dir / "run_level_summary.csv")
    image_rows = read_csv(a11_dir / "controller_policy_image_level.csv")
    feature_names = [str(x) for x in frozen_cfg["feature_names"]]
    thresholds = {str(k): float(v) for k, v in frozen_cfg["thresholds"].items()}

    add_interrun_features(step_rows)
    late_feature_map = build_window_feature_table(step_rows, feature_names, WINDOWS)
    image_policy_stats = summarize_image_level_survival(image_rows)
    image_ids = sorted({str(r["image_id"]) for r in frozen_rows})
    np_fallbacks = load_np_fallbacks(
        read_csv(Path(frozen_cfg["fallback_source"]["source_csv"])),
        args.noise,
        image_ids,
    )

    confusion_rows = build_confusion_run_table(
        frozen_rows,
        np_fallbacks,
        image_policy_stats,
        feature_names,
        thresholds,
    )
    missed_rows = build_missed_bad_runs_diagnosis(
        confusion_rows,
        late_feature_map,
        feature_names,
        thresholds,
    )
    false_positive_rows = build_false_positive_diagnosis(confusion_rows)
    curve_rows = build_group_curve_summary(step_rows, confusion_rows)
    late_screen_rows = build_diagnostic_late_window_screen(
        confusion_rows,
        late_feature_map,
        feature_names,
        thresholds,
    )

    write_csv(outdir / "a11_confusion_run_table.csv", confusion_rows)
    write_csv(outdir / "missed_bad_runs_diagnosis.csv", missed_rows)
    write_csv(outdir / "false_positive_diagnosis.csv", false_positive_rows)
    write_csv(outdir / "trajectory_group_curve_summary.csv", curve_rows)
    write_csv(outdir / "diagnostic_late_window_screen.csv", late_screen_rows)
    write_text(outdir / "SUMMARY.md", build_summary(confusion_rows, missed_rows, false_positive_rows, late_screen_rows))


if __name__ == "__main__":
    main()
