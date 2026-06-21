#!/usr/bin/env python3
"""A8 full-split validation for the frozen conservative Branch A controller."""
from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FROZEN_DETECTOR_ID = "correction_norm__first10pct__max"
FROZEN_DIRECTION = "low_is_risky"
FROZEN_THRESHOLD = 0.315417


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
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def mean(vals: Iterable[float]) -> float:
    xs = list(vals)
    return sum(xs) / len(xs) if xs else math.nan


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


def flagged(value: float, direction: str, threshold: float) -> bool:
    if direction == "low_is_risky":
        return value <= threshold
    if direction == "high_is_risky":
        return value >= threshold
    raise ValueError(direction)


def merge_chunk_csvs(outdir: Path, chunk_dirs: List[Path]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
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


def copy_samples(outdir: Path, chunk_dirs: List[Path]) -> None:
    sample_dir = outdir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunk_dirs:
        for png in sorted((chunk / "samples").glob("*.png")):
            dst = sample_dir / f"{chunk.name}_{png.name}"
            if not dst.exists():
                shutil.copy2(png, dst)


def load_np_fallbacks(np_rows: List[Dict[str, str]], noise: float, target_images: List[str]) -> Dict[str, Dict[str, object]]:
    candidates = []
    for row in np_rows:
        if row.get("alignment_mode") != "resolve":
            continue
        if abs(to_float(row.get("measurement_noise_std")) - noise) > 1e-12:
            continue
        image_id = image_id_from_np_basename(str(row["image_basename"]))
        if image_id not in target_images:
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
    for image_id in target_images:
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


def detector_values(step_rows: List[Dict[str, str]], frac: float = 0.10) -> Dict[Tuple[str, int], float]:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    for row in step_rows:
        grouped.setdefault((str(row["image_id"]), int(row["run_index"])), []).append(row)
    out: Dict[Tuple[str, int], float] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda r: int(r["step"]))
        k = max(1, int(math.ceil(len(rows) * frac)))
        out[key] = max(to_float(r["correction_norm"]) for r in rows[:k])
    return out


def confusion(run_rows: List[Dict[str, str]], values: Dict[Tuple[str, int], float], threshold: float) -> Dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in run_rows:
        key = (str(row["image_id"]), int(row["run_index"]))
        bad25 = to_float(row["final_psnr"]) < 25.0
        pred = flagged(values[key], FROZEN_DIRECTION, threshold)
        if pred and bad25:
            counts["tp"] += 1
        elif pred and not bad25:
            counts["fp"] += 1
        elif not pred and bad25:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def balanced_accuracy(counts: Dict[str, int]) -> float:
    pos = counts["tp"] + counts["fn"]
    neg = counts["tn"] + counts["fp"]
    if pos == 0 or neg == 0:
        return math.nan
    return 0.5 * (counts["tp"] / pos + counts["tn"] / neg)


def threshold_sweep(run_rows: List[Dict[str, str]], values: Dict[Tuple[str, int], float]) -> List[Dict[str, object]]:
    base = FROZEN_THRESHOLD
    thresholds = sorted(
        {
            base - 0.05,
            base - 0.03,
            base - 0.02,
            base - 0.01,
            base,
            base + 0.01,
            base + 0.02,
            base + 0.03,
            base + 0.05,
        }
    )
    rows = []
    best_ba = -math.inf
    best_idx = -1
    for i, threshold in enumerate(thresholds):
        counts = confusion(run_rows, values, threshold)
        ba = balanced_accuracy(counts)
        if math.isfinite(ba) and ba > best_ba:
            best_ba = ba
            best_idx = i
        rows.append(
            {
                "detector_id": FROZEN_DETECTOR_ID,
                "direction": FROZEN_DIRECTION,
                "threshold": threshold,
                "is_frozen_from_A7": abs(threshold - base) < 1e-12,
                "tp": counts["tp"],
                "fp": counts["fp"],
                "tn": counts["tn"],
                "fn": counts["fn"],
                "balanced_accuracy": ba,
                "num_flagged": counts["tp"] + counts["fp"],
            }
        )
    if best_idx >= 0:
        rows[best_idx]["is_retrospective_best_in_sweep"] = True
    for i, row in enumerate(rows):
        row.setdefault("is_retrospective_best_in_sweep", False)
    return rows


def apply_policy(
    policy_name: str,
    policy_kind: str,
    notes: str,
    run_rows: List[Dict[str, str]],
    values: Dict[Tuple[str, int], float],
    np_fallbacks: Dict[str, Dict[str, object]],
    threshold: float | None,
    oracle_risk: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    out_rows: List[Dict[str, object]] = []
    repl_rows: List[Dict[str, object]] = []
    for row in run_rows:
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        key = (image_id, run_index)
        sitcom_psnr = to_float(row["final_psnr"])
        bad25 = sitcom_psnr < 25.0
        bad20 = sitcom_psnr < 20.0
        value = values.get(key, math.nan)
        was_flagged = False
        if oracle_risk:
            was_flagged = bad25
        elif threshold is not None:
            was_flagged = flagged(value, FROZEN_DIRECTION, threshold)

        policy_psnr = sitcom_psnr
        source = "sitcom"
        detail = ""
        replaced = False
        if was_flagged and policy_name != "sitcom_only":
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
            "detector_id": FROZEN_DETECTOR_ID if threshold is not None else "",
            "detector_value": value,
            "detector_direction": FROZEN_DIRECTION if threshold is not None else "",
            "detector_threshold": threshold if threshold is not None else "",
            "sitcom_bad_below25": bad25,
            "sitcom_bad_below20": bad20,
            "final_bad_below25": policy_psnr < 25.0,
            "final_bad_below20": policy_psnr < 20.0,
            "was_flagged": was_flagged,
            "was_replaced": replaced,
            "replacement_source": source,
            "replacement_detail": detail,
            "false_positive_replacement": replaced and not bad25,
            "true_positive_replacement": replaced and bad25,
            "false_negative_remaining_bad25": (not replaced) and bad25,
        }
        out_rows.append(record)
        if replaced:
            repl_rows.append(record.copy())
    return out_rows, repl_rows


def summarize_runs(policy_rows: List[Dict[str, object]]) -> Dict[str, object]:
    psnrs = [float(r["policy_final_psnr"]) for r in policy_rows]
    return {
        "run_level_mean_psnr": mean(psnrs),
        "run_level_min_psnr": min(psnrs),
        "run_level_num_below25": sum(1 for v in psnrs if v < 25.0),
        "run_level_num_below20": sum(1 for v in psnrs if v < 20.0),
        "num_replaced": sum(1 for r in policy_rows if bool(r["was_replaced"])),
        "num_false_positive_replacements": sum(1 for r in policy_rows if bool(r["false_positive_replacement"])),
        "num_true_positive_replacements": sum(1 for r in policy_rows if bool(r["true_positive_replacement"])),
        "num_false_negative_remaining_bad25": sum(1 for r in policy_rows if bool(r["false_negative_remaining_bad25"])),
        "mean_delta_vs_sitcom": mean(float(r["delta_vs_sitcom"]) for r in policy_rows),
        "mean_delta_on_replaced": mean(float(r["delta_vs_sitcom"]) for r in policy_rows if bool(r["was_replaced"])),
        "mean_delta_on_false_positive_replacements": mean(
            float(r["delta_vs_sitcom"]) for r in policy_rows if bool(r["false_positive_replacement"])
        ),
    }


def image_level(policy_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in policy_rows:
        grouped.setdefault(str(row["image_id"]), []).append(row)
    out = []
    for image_id, rows in sorted(grouped.items()):
        vals = [float(r["policy_final_psnr"]) for r in rows]
        out.append(
            {
                "policy_name": rows[0]["policy_name"],
                "policy_kind": rows[0]["policy_kind"],
                "image_id": image_id,
                "best_of_4_psnr": max(vals),
                "mean_of_4_psnr": mean(vals),
                "min_of_4_psnr": min(vals),
                "num_runs_below25": sum(1 for v in vals if v < 25.0),
                "num_runs_below20": sum(1 for v in vals if v < 20.0),
                "replaced_run_indices": ",".join(str(int(r["run_index"])) for r in rows if bool(r["was_replaced"])),
            }
        )
    return out


def summarize_policy(policy_rows: List[Dict[str, object]], image_rows: List[Dict[str, object]]) -> Dict[str, object]:
    base = summarize_runs(policy_rows)
    bests = [float(r["best_of_4_psnr"]) for r in image_rows]
    mins = [float(r["min_of_4_psnr"]) for r in image_rows]
    means = [float(r["mean_of_4_psnr"]) for r in image_rows]
    return {
        "policy_name": policy_rows[0]["policy_name"],
        "policy_kind": policy_rows[0]["policy_kind"],
        "notes": policy_rows[0]["notes"],
        **base,
        "image_level_best_of_4_mean_psnr": mean(bests),
        "image_level_best_of_4_min_psnr": min(bests),
        "image_level_mean_of_4_mean_psnr": mean(means),
        "image_level_mean_of_4_min_psnr": min(means),
        "image_level_min_of_4_mean_psnr": mean(mins),
        "image_level_min_of_4_min_psnr": min(mins),
    }


def detector_cases(run_rows: List[Dict[str, str]], values: Dict[Tuple[str, int], float], threshold: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    missed = []
    false_pos = []
    for row in run_rows:
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        value = values[(image_id, run_index)]
        pred = flagged(value, FROZEN_DIRECTION, threshold)
        psnr = to_float(row["final_psnr"])
        bad25 = psnr < 25.0
        base = {
            "image_id": image_id,
            "run_index": run_index,
            "final_psnr": psnr,
            "detector_id": FROZEN_DETECTOR_ID,
            "detector_value": value,
            "detector_direction": FROZEN_DIRECTION,
            "detector_threshold": threshold,
            "bad25": bad25,
            "bad20": psnr < 20.0,
        }
        if bad25 and not pred:
            missed.append(base)
        if (not bad25) and pred:
            false_pos.append(base)
    return missed, false_pos


def render_summary(
    outdir: Path,
    policy_summaries: List[Dict[str, object]],
    validation_rows: List[Dict[str, object]],
    missed: List[Dict[str, object]],
    false_pos: List[Dict[str, object]],
) -> None:
    frozen = next(r for r in validation_rows if bool(r["is_frozen_from_A7"]))
    best = next(r for r in validation_rows if bool(r["is_retrospective_best_in_sweep"]))
    lines = [
        "# A8 SITCOM25 Trajectory Controller Validation",
        "",
        "## Frozen Detector",
        "",
        f"- Detector: `{FROZEN_DETECTOR_ID}`",
        f"- Direction: `{FROZEN_DIRECTION}`",
        f"- Frozen threshold from A7: `{FROZEN_THRESHOLD:.6g}`",
        f"- Frozen confusion on A8: `TP={frozen['tp']}`, `FP={frozen['fp']}`, `TN={frozen['tn']}`, `FN={frozen['fn']}`",
        f"- Retrospective best in local sweep: threshold `{best['threshold']:.6g}` with `TP={best['tp']}`, `FP={best['fp']}`, `TN={best['tn']}`, `FN={best['fn']}`",
        "",
        "## Policy Summary",
        "",
        "| policy | kind | run mean | run min | below25 | below20 | image best-of-4 mean | image best-of-4 min | replaced | FP repl | FN remain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_summaries:
        lines.append(
            "| {policy} | {kind} | {run_mean:.3f} | {run_min:.3f} | {b25} | {b20} | {img_mean:.3f} | {img_min:.3f} | {repl} | {fp} | {fn} |".format(
                policy=row["policy_name"],
                kind=row["policy_kind"],
                run_mean=float(row["run_level_mean_psnr"]),
                run_min=float(row["run_level_min_psnr"]),
                b25=int(row["run_level_num_below25"]),
                b20=int(row["run_level_num_below20"]),
                img_mean=float(row["image_level_best_of_4_mean_psnr"]),
                img_min=float(row["image_level_best_of_4_min_psnr"]),
                repl=int(row["num_replaced"]),
                fp=int(row["num_false_positive_replacements"]),
                fn=int(row["num_false_negative_remaining_bad25"]),
            )
        )
    lines.extend(
        [
            "",
            "## Validation Answer",
            "",
            f"- Frozen A7 threshold missed `{len(missed)}` bad25 runs and false-flagged `{len(false_pos)}` good runs.",
            "- The frozen-threshold policy is the main validation result. The sweep is included only to show local sensitivity and must not be treated as solver evidence.",
            "- Oracle-risk rows are diagnostics only; they use final PSNR labels and are not executable.",
            "",
        ]
    )
    write_text(outdir / "SUMMARY.md", "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chunk_dirs", nargs="+", required=True)
    ap.add_argument("--np_run_level", required=True)
    ap.add_argument("--noise", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_dirs = [Path(p) for p in args.chunk_dirs]
    step_rows, run_rows = merge_chunk_csvs(outdir, chunk_dirs)
    copy_samples(outdir, chunk_dirs)
    values = detector_values(step_rows)
    target_images = sorted({str(r["image_id"]) for r in run_rows})
    np_fallbacks = load_np_fallbacks(read_csv(Path(args.np_run_level)), args.noise, target_images)
    validation_rows = threshold_sweep(run_rows, values)

    frozen_policy_rows, frozen_repl = apply_policy(
        "frozen_A7_conservative_to_np_selected",
        "frozen_validation_executable_style",
        "Frozen A7 conservative detector threshold with no-ground-truth NP selected fallback.",
        run_rows,
        values,
        np_fallbacks,
        FROZEN_THRESHOLD,
        oracle_risk=False,
    )
    sitcom_rows, _ = apply_policy(
        "sitcom_only",
        "executable_baseline",
        "Keep each SITCOM final output.",
        run_rows,
        values,
        np_fallbacks,
        threshold=None,
        oracle_risk=False,
    )
    best_sweep = next(r for r in validation_rows if bool(r["is_retrospective_best_in_sweep"]))
    tuned_rows, tuned_repl = apply_policy(
        "retrospective_best_threshold_to_np_selected",
        "posthoc_threshold_diagnostic",
        "Best threshold inside the local A8 sweep. Diagnostic only.",
        run_rows,
        values,
        np_fallbacks,
        float(best_sweep["threshold"]),
        oracle_risk=False,
    )
    oracle_rows, oracle_repl = apply_policy(
        "oracle_risk_to_np_selected",
        "diagnostic_oracle_risk",
        "Replace exactly bad SITCOM runs with NP selected fallback. Diagnostic only.",
        run_rows,
        values,
        np_fallbacks,
        threshold=None,
        oracle_risk=True,
    )

    all_run_rows = sitcom_rows + frozen_policy_rows + tuned_rows + oracle_rows
    image_rows = []
    policy_summaries = []
    for rows in (sitcom_rows, frozen_policy_rows, tuned_rows, oracle_rows):
        imgs = image_level(rows)
        image_rows.extend(imgs)
        policy_summaries.append(summarize_policy(rows, imgs))
    replaced_rows = frozen_repl + tuned_repl + oracle_repl
    missed, false_pos = detector_cases(run_rows, values, FROZEN_THRESHOLD)

    write_csv(outdir / "detector_validation_summary.csv", validation_rows)
    write_csv(outdir / "controller_policy_summary.csv", policy_summaries)
    write_csv(outdir / "controller_policy_run_level.csv", all_run_rows)
    write_csv(outdir / "controller_policy_image_level.csv", image_rows)
    write_csv(outdir / "replaced_runs.csv", replaced_rows)
    write_csv(outdir / "detector_missed_bad_runs.csv", missed)
    write_csv(outdir / "detector_false_positive_runs.csv", false_pos)
    render_summary(outdir, policy_summaries, validation_rows, missed, false_pos)


if __name__ == "__main__":
    main()
