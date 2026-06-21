#!/usr/bin/env python3
"""A11 prospective frozen-policy validation on a fresh SITCOM trajectory run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


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


def merge_chunk_csvs(outdir: Path, chunk_dirs: Sequence[Path]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    step_rows: List[Dict[str, str]] = []
    run_rows: List[Dict[str, str]] = []
    for chunk in chunk_dirs:
        step_rows.extend(read_csv(chunk / "trajectory_step_metrics.csv"))
        run_rows.extend(read_csv(chunk / "run_level_summary.csv"))
    step_rows.sort(key=lambda r: (str(r["image_id"]), int(r["run_index"]), int(r["step"])))
    run_rows.sort(key=lambda r: (str(r["image_id"]), int(r["run_index"])))
    write_csv(outdir / "trajectory_step_metrics.csv", step_rows)
    write_csv(outdir / "run_level_summary.csv", run_rows)
    return step_rows, run_rows


def copy_samples(outdir: Path, chunk_dirs: Sequence[Path]) -> None:
    sample_dir = outdir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunk_dirs:
        src_dir = chunk / "samples"
        if not src_dir.exists():
            continue
        for png in sorted(src_dir.glob("*.png")):
            dst = sample_dir / f"{chunk.name}_{png.name}"
            if not dst.exists():
                shutil.copy2(png, dst)


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


def build_detector_table(step_rows: List[Dict[str, str]], run_rows: List[Dict[str, str]], window_frac: float) -> List[Dict[str, object]]:
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

    window_name = f"first{int(round(window_frac * 100)):d}pct"
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
        k = max(1, int(math.ceil(n_steps * window_frac)))
        window_rows = rows[:k]
        for feature in features:
            vals = [to_float(r.get(feature, math.nan)) for r in window_rows]
            for agg_name, agg_val in aggregate_window(vals).items():
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


def apply_threshold(value: float, direction: str, threshold: float) -> bool:
    if direction == "high_is_risky":
        return value >= threshold
    if direction == "low_is_risky":
        return value <= threshold
    raise ValueError(direction)


def apply_frozen_policy(
    detector_rows: List[Dict[str, object]],
    feature_names: Sequence[str],
    directions: Dict[str, str],
    thresholds: Dict[str, float],
    np_fallbacks: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    run_records: List[Dict[str, object]] = []
    image_rows: List[Dict[str, object]] = []
    missed_rows: List[Dict[str, object]] = []
    false_positive_rows: List[Dict[str, object]] = []
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in detector_rows:
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        sitcom_psnr = to_float(row["final_psnr"])
        component_values = {feature: to_float(row.get(feature)) for feature in feature_names}
        component_flags = {
            feature: apply_threshold(component_values[feature], directions[feature], float(thresholds[feature]))
            for feature in feature_names
        }
        was_flagged = all(component_flags.values())
        fb = np_fallbacks[image_id]
        policy_psnr = float(fb["np_selected_psnr"]) if was_flagged else sitcom_psnr
        delta = policy_psnr - sitcom_psnr
        replaced = was_flagged
        record = {
            "policy_name": "frozen_A11_policy",
            "policy_kind": "prospective_frozen_validation",
            "image_id": image_id,
            "run_index": run_index,
            "sitcom_final_psnr": sitcom_psnr,
            "policy_final_psnr": policy_psnr,
            "delta_vs_sitcom": delta,
            "sitcom_bad_below25": sitcom_psnr < 25.0,
            "sitcom_bad_below20": sitcom_psnr < 20.0,
            "final_bad_below25": policy_psnr < 25.0,
            "final_bad_below20": policy_psnr < 20.0,
            "was_flagged": was_flagged,
            "was_replaced": replaced,
            "replacement_source": "np_selected" if replaced else "sitcom",
            "replacement_detail": f"{fb['np_selected_config_tag']} seed={fb['np_selected_seed']}" if replaced else "",
            "true_positive_replacement": replaced and sitcom_psnr < 25.0,
            "false_positive_replacement": replaced and sitcom_psnr >= 25.0,
            "false_negative_remaining_bad25": (not replaced) and sitcom_psnr < 25.0,
            "false_negative_remaining_bad20": (not replaced) and sitcom_psnr < 20.0,
        }
        for feature in feature_names:
            record[f"{feature}__value"] = component_values[feature]
            record[f"{feature}__direction"] = directions[feature]
            record[f"{feature}__threshold"] = thresholds[feature]
            record[f"{feature}__flagged"] = component_flags[feature]
        run_records.append(record)
        by_image[image_id].append(record)

        if (not replaced) and sitcom_psnr < 25.0:
            missed_rows.append(record.copy())
        if replaced and sitcom_psnr >= 25.0:
            false_positive_rows.append(record.copy())

    for image_id, rows in sorted(by_image.items()):
        vals = [to_float(r["policy_final_psnr"]) for r in rows]
        image_rows.append(
            {
                "policy_name": "frozen_A11_policy",
                "policy_kind": "prospective_frozen_validation",
                "image_id": image_id,
                "best_of_4_psnr": max(vals),
                "mean_of_4_psnr": mean_or_nan(vals),
                "min_of_4_psnr": min(vals),
                "num_runs_below25": sum(1 for v in vals if v < 25.0),
                "num_runs_below20": sum(1 for v in vals if v < 20.0),
                "replaced_run_indices": ",".join(str(int(r["run_index"])) for r in rows if bool(r["was_replaced"])),
            }
        )

    return run_records, image_rows, missed_rows, false_positive_rows


def baseline_policy(
    name: str,
    kind: str,
    detector_rows: List[Dict[str, object]],
    np_fallbacks: Dict[str, Dict[str, object]],
    mode: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    run_records: List[Dict[str, object]] = []
    image_rows: List[Dict[str, object]] = []
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in detector_rows:
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        sitcom_psnr = to_float(row["final_psnr"])
        if mode == "sitcom_only":
            replace = False
        elif mode == "replace_all":
            replace = True
        elif mode == "oracle_bad25":
            replace = sitcom_psnr < 25.0
        else:
            raise ValueError(mode)
        fb = np_fallbacks[image_id]
        policy_psnr = float(fb["np_selected_psnr"]) if replace else sitcom_psnr
        rec = {
            "policy_name": name,
            "policy_kind": kind,
            "image_id": image_id,
            "run_index": run_index,
            "sitcom_final_psnr": sitcom_psnr,
            "policy_final_psnr": policy_psnr,
            "delta_vs_sitcom": policy_psnr - sitcom_psnr,
            "sitcom_bad_below25": sitcom_psnr < 25.0,
            "sitcom_bad_below20": sitcom_psnr < 20.0,
            "final_bad_below25": policy_psnr < 25.0,
            "final_bad_below20": policy_psnr < 20.0,
            "was_replaced": replace,
            "replacement_source": "np_selected" if replace else "sitcom",
            "replacement_detail": f"{fb['np_selected_config_tag']} seed={fb['np_selected_seed']}" if replace else "",
            "true_positive_replacement": replace and sitcom_psnr < 25.0,
            "false_positive_replacement": replace and sitcom_psnr >= 25.0,
            "false_negative_remaining_bad25": (not replace) and sitcom_psnr < 25.0,
            "false_negative_remaining_bad20": (not replace) and sitcom_psnr < 20.0,
        }
        run_records.append(rec)
        by_image[image_id].append(rec)

    for image_id, rows in sorted(by_image.items()):
        vals = [to_float(r["policy_final_psnr"]) for r in rows]
        image_rows.append(
            {
                "policy_name": name,
                "policy_kind": kind,
                "image_id": image_id,
                "best_of_4_psnr": max(vals),
                "mean_of_4_psnr": mean_or_nan(vals),
                "min_of_4_psnr": min(vals),
                "num_runs_below25": sum(1 for v in vals if v < 25.0),
                "num_runs_below20": sum(1 for v in vals if v < 20.0),
                "replaced_run_indices": ",".join(str(int(r["run_index"])) for r in rows if bool(r["was_replaced"])),
            }
        )
    return run_records, image_rows


def summarize_policy(policy_rows: List[Dict[str, object]], image_rows: List[Dict[str, object]]) -> Dict[str, object]:
    psnrs = [to_float(r["policy_final_psnr"]) for r in policy_rows]
    fp_losses = [to_float(r["delta_vs_sitcom"]) for r in policy_rows if bool(r["false_positive_replacement"])]
    remaining_bad = [to_float(r["sitcom_final_psnr"]) for r in policy_rows if bool(r["false_negative_remaining_bad25"])]
    return {
        "policy_name": policy_rows[0]["policy_name"],
        "policy_kind": policy_rows[0]["policy_kind"],
        "run_level_mean_psnr": mean_or_nan(psnrs),
        "run_level_min_psnr": min(psnrs),
        "run_level_num_below25": sum(1 for v in psnrs if v < 25.0),
        "run_level_num_below20": sum(1 for v in psnrs if v < 20.0),
        "num_replaced": sum(1 for r in policy_rows if bool(r["was_replaced"])),
        "num_false_positive_replacements": sum(1 for r in policy_rows if bool(r["false_positive_replacement"])),
        "num_true_positive_replacements": sum(1 for r in policy_rows if bool(r["true_positive_replacement"])),
        "num_false_negative_remaining_bad25": sum(1 for r in policy_rows if bool(r["false_negative_remaining_bad25"])),
        "image_level_best_of_4_mean_psnr": mean_or_nan(to_float(r["best_of_4_psnr"]) for r in image_rows),
        "image_level_best_of_4_min_psnr": min(to_float(r["best_of_4_psnr"]) for r in image_rows),
        "worst_false_positive_psnr_loss": min(fp_losses) if fp_losses else math.nan,
        "worst_remaining_miss": min(remaining_bad) if remaining_bad else math.nan,
    }


def render_summary(
    outdir: Path,
    policy_summaries: List[Dict[str, object]],
    missed_rows: List[Dict[str, object]],
    false_positive_rows: List[Dict[str, object]],
    frozen_cfg: Dict[str, object],
) -> None:
    by_name = {str(r["policy_name"]): r for r in policy_summaries}
    lines = [
        "# A11 Prospective Frozen-Policy Validation",
        "",
        "This analysis uses a fresh SITCOM trajectory run and applies the frozen controller from `frozen_policy.json` without any retuning.",
        "",
        "## Frozen Policy",
        "",
        f"- Policy name: `{frozen_cfg['policy_name']}`",
        f"- Combine mode: `{frozen_cfg['combine_mode']}`",
        f"- Feature 1 threshold: `{frozen_cfg['feature_names'][0]}` `>= {to_float(frozen_cfg['thresholds'][frozen_cfg['feature_names'][0]]):.6g}`",
        f"- Feature 2 threshold: `{frozen_cfg['feature_names'][1]}` `>= {to_float(frozen_cfg['thresholds'][frozen_cfg['feature_names'][1]]):.6g}`",
        f"- Fallback source: `{frozen_cfg['fallback_source']['type']}`",
        "",
        "## Policy Summary",
        "",
        "| policy | kind | run mean | run min | below25 | below20 | image best-of-4 mean | image best-of-4 min | replaced | TP repl | FP repl | FN remain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "sitcom_only",
        "frozen_A11_policy",
        "replace_all_np_selected",
        "oracle_risk_np_selected",
    ]
    for name in order:
        row = by_name[name]
        lines.append(
            "| {policy} | {kind} | {run_mean:.3f} | {run_min:.3f} | {b25} | {b20} | {img_mean:.3f} | {img_min:.3f} | {repl} | {tp} | {fp} | {fn} |".format(
                policy=row["policy_name"],
                kind=row["policy_kind"],
                run_mean=to_float(row["run_level_mean_psnr"]),
                run_min=to_float(row["run_level_min_psnr"]),
                b25=int(to_float(row["run_level_num_below25"])),
                b20=int(to_float(row["run_level_num_below20"])),
                img_mean=to_float(row["image_level_best_of_4_mean_psnr"]),
                img_min=to_float(row["image_level_best_of_4_min_psnr"]),
                repl=int(to_float(row["num_replaced"])),
                tp=int(to_float(row["num_true_positive_replacements"])),
                fp=int(to_float(row["num_false_positive_replacements"])),
                fn=int(to_float(row["num_false_negative_remaining_bad25"])),
            )
        )
    lines.extend(
        [
            "",
            "## Prospective Result",
            "",
            "- `frozen_A11_policy` is the main prospective result.",
            "- `replace_all_np_selected` is a degenerate baseline and not an acceptable solver policy.",
            "- `oracle_risk_np_selected` is diagnostic only and not executable.",
            f"- Frozen-policy missed bad25 runs: `{len(missed_rows)}`",
            f"- Frozen-policy false-positive replacements: `{len(false_positive_rows)}`",
            "",
        ]
    )
    write_text(outdir / "SUMMARY.md", "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chunk_dirs", nargs="+", required=True)
    ap.add_argument("--frozen_policy_json", required=True)
    ap.add_argument("--noise", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_dirs = [Path(p) for p in args.chunk_dirs]

    with open(args.frozen_policy_json, encoding="utf-8") as f:
        frozen_cfg = json.load(f)
    shutil.copy2(args.frozen_policy_json, outdir / "frozen_policy.json")

    step_rows, run_rows = merge_chunk_csvs(outdir, chunk_dirs)
    copy_samples(outdir, chunk_dirs)
    add_interrun_features(step_rows)

    window_name = str(frozen_cfg["window_definition"])
    if window_name != "first50pct":
        raise ValueError(f"Unsupported frozen window {window_name}")
    detector_rows = build_detector_table(step_rows, run_rows, window_frac=0.50)

    image_ids = sorted({str(r["image_id"]) for r in run_rows})
    np_fallbacks = load_np_fallbacks(
        read_csv(Path(frozen_cfg["fallback_source"]["source_csv"])),
        args.noise,
        image_ids,
    )
    feature_names = list(frozen_cfg["feature_names"])
    directions = {str(k): str(v) for k, v in frozen_cfg["feature_directions"].items()}
    thresholds = {str(k): float(v) for k, v in frozen_cfg["thresholds"].items()}

    frozen_policy_rows, frozen_policy_image_rows, missed_rows, false_positive_rows = apply_frozen_policy(
        detector_rows,
        feature_names,
        directions,
        thresholds,
        np_fallbacks,
    )
    sitcom_rows, sitcom_image_rows = baseline_policy(
        "sitcom_only",
        "executable_baseline",
        detector_rows,
        np_fallbacks,
        mode="sitcom_only",
    )
    replace_all_rows, replace_all_image_rows = baseline_policy(
        "replace_all_np_selected",
        "degenerate_replace_all_baseline",
        detector_rows,
        np_fallbacks,
        mode="replace_all",
    )
    oracle_rows, oracle_image_rows = baseline_policy(
        "oracle_risk_np_selected",
        "diagnostic_oracle_risk",
        detector_rows,
        np_fallbacks,
        mode="oracle_bad25",
    )

    all_image_rows = sitcom_image_rows + frozen_policy_image_rows + replace_all_image_rows + oracle_image_rows
    policy_summaries = [
        summarize_policy(sitcom_rows, sitcom_image_rows),
        summarize_policy(frozen_policy_rows, frozen_policy_image_rows),
        summarize_policy(replace_all_rows, replace_all_image_rows),
        summarize_policy(oracle_rows, oracle_image_rows),
    ]

    write_csv(outdir / "frozen_policy_applied_runs.csv", frozen_policy_rows)
    write_csv(outdir / "controller_policy_summary.csv", policy_summaries)
    write_csv(outdir / "controller_policy_image_level.csv", all_image_rows)
    write_csv(outdir / "detector_missed_bad_runs.csv", missed_rows)
    write_csv(outdir / "detector_false_positive_runs.csv", false_positive_rows)
    render_summary(outdir, policy_summaries, missed_rows, false_positive_rows, frozen_cfg)


if __name__ == "__main__":
    main()
