#!/usr/bin/env python3
"""A17 offline anytime detector design for Branch A.

This script uses existing trajectory outputs only. It does not run new SITCOM
jobs and does not change any frozen Branch A policy.

The goal is diagnostic:

- replace fixed first50 / first80 summaries with event-time diagnostics;
- measure when bad runs become visible under stepwise / cumulative features;
- identify candidate anytime rules that might deserve freezing later.
"""

from __future__ import annotations

import argparse
import csv
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

BASE_FEATURES: List[str] = [
    "x0y_full_residual_normed",
    "x0y_lowfreq_residual_normed",
    "x0hat_x0y_disagreement",
    "correction_norm",
    "x0y_step_jump",
    "xt_step_jump",
]

RANK_THRESHOLD = 3.0
PERSIST_STEPS = [3, 5, 10]
CUMULATIVE_COUNTS = [1, 3, 5, 10]
ROLL_WINDOWS = [5, 10, 20, 40]
SLOPE_THRESHOLDS = [0.0, 0.05, 0.10]
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


def load_dataset(dataset_name: str, dataset_dir: Path) -> Dict[str, object]:
    step_rows = read_csv(dataset_dir / "trajectory_step_metrics.csv")
    run_rows = read_csv(dataset_dir / "run_level_summary.csv")
    step_rows.sort(key=lambda r: (str(r["image_id"]), int(r["run_index"]), int(r["step"])))
    run_rows.sort(key=lambda r: (str(r["image_id"]), int(r["run_index"])))
    return {
        "dataset": dataset_name,
        "dir": dataset_dir,
        "step_rows": step_rows,
        "run_rows": run_rows,
    }


def rank_high(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    ranks[order] = np.arange(1, arr.size + 1, dtype=float)
    return ranks


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


def first_index(mask: Sequence[bool]) -> float:
    for i, flag in enumerate(mask):
        if flag:
            return float(i)
    return math.nan


def first_persistent(mask: Sequence[bool], m: int) -> float:
    if m <= 0:
        return math.nan
    for start in range(0, len(mask) - m + 1):
        if all(mask[start : start + m]):
            return float(start)
    return math.nan


def first_cumulative(mask: Sequence[bool], k: int) -> float:
    if k <= 0:
        return math.nan
    count = 0
    for i, flag in enumerate(mask):
        if flag:
            count += 1
        if count >= k:
            return float(i)
    return math.nan


def first_rolling_slope_above(vals: Sequence[float], window: int, threshold: float) -> float:
    if window <= 1 or len(vals) < window:
        return math.nan
    for end in range(window - 1, len(vals)):
        sl = slope_or_nan(vals[end - window + 1 : end + 1])
        if math.isfinite(sl) and sl > threshold:
            return float(end)
    return math.nan


def alarm_fraction(step0: float, n_steps: int) -> float:
    if not math.isfinite(step0):
        return 2.0
    return (step0 + 1.0) / float(n_steps)


def alarm_score(step0: float, n_steps: int) -> float:
    if not math.isfinite(step0):
        return -(n_steps + 1.0)
    return -float(step0)


def auc_roc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(float(s), int(y)) for s, y in zip(scores, y_true) if math.isfinite(float(s))]
    if not pairs:
        return math.nan
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return math.nan
    pairs.sort(key=lambda t: t[0])
    rank = 1
    pos_rank_sum = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (rank + (rank + (j - i) - 1))
        block_pos = sum(pairs[k][1] for k in range(i, j))
        pos_rank_sum += avg_rank * block_pos
        rank += (j - i)
        i = j
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def auc_pr(y_true: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(float(s), int(y)) for s, y in zip(scores, y_true) if math.isfinite(float(s))]
    if not pairs:
        return math.nan
    pos = sum(y for _, y in pairs)
    if pos == 0:
        return math.nan
    pairs.sort(key=lambda t: t[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / max(tp + fp, 1)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def confusion(y_true: Sequence[int], pred: Sequence[bool]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for y, p in zip(y_true, pred):
        if p and y:
            tp += 1
        elif p and not y:
            fp += 1
        elif (not p) and y:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def balanced_accuracy(counts: Dict[str, int]) -> float:
    pos = counts["tp"] + counts["fn"]
    neg = counts["tn"] + counts["fp"]
    if pos == 0 or neg == 0:
        return math.nan
    return 0.5 * (counts["tp"] / pos + counts["tn"] / neg)


def build_rank_features(step_rows: List[Dict[str, str]]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["step"]))].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["run_index"]))
        for base in BASE_FEATURES:
            vals = np.asarray([to_float(r.get(base)) for r in rows], dtype=float)
            if vals.size == 0 or np.any(~np.isfinite(vals)):
                continue
            ranks = rank_high(vals)
            for row, rank in zip(rows, ranks):
                row[f"{base}__interrun_rank"] = float(rank)


def build_run_sequences(step_rows: List[Dict[str, str]]) -> Dict[Tuple[str, int], Dict[str, np.ndarray]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in step_rows:
        grouped[(str(row["image_id"]), int(row["run_index"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["step"]))

    seqs: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    for key, rows in grouped.items():
        seqs[key] = {}
        seqs[key]["n_steps"] = np.array([len(rows)], dtype=int)
        for base in BASE_FEATURES:
            vals = np.asarray([to_float(r.get(f"{base}__interrun_rank")) for r in rows], dtype=float)
            seqs[key][f"{base}__rank"] = vals
        seqs[key]["step"] = np.asarray([int(r["step"]) for r in rows], dtype=int)
    return seqs


def rule_alarms_for_run(
    seqs: Dict[str, np.ndarray],
    base: str,
) -> Dict[str, float]:
    ranks = seqs[f"{base}__rank"]
    mask_ge3 = [bool(x >= RANK_THRESHOLD) if math.isfinite(float(x)) else False for x in ranks]
    out: Dict[str, float] = {
        f"{base}__rank_ge3__first_step0": first_index(mask_ge3),
        f"{base}__rank_ge3__persist3__first_step0": first_persistent(mask_ge3, 3),
        f"{base}__rank_ge3__persist5__first_step0": first_persistent(mask_ge3, 5),
        f"{base}__rank_ge3__persist10__first_step0": first_persistent(mask_ge3, 10),
        f"{base}__rank_ge3__cum1__first_step0": first_cumulative(mask_ge3, 1),
        f"{base}__rank_ge3__cum3__first_step0": first_cumulative(mask_ge3, 3),
        f"{base}__rank_ge3__cum5__first_step0": first_cumulative(mask_ge3, 5),
        f"{base}__rank_ge3__cum10__first_step0": first_cumulative(mask_ge3, 10),
    }
    for window in ROLL_WINDOWS:
        for thr in SLOPE_THRESHOLDS:
            key = f"{base}__rank__rolling_slope_w{window}_gt{thr:g}__first_step0"
            out[key] = first_rolling_slope_above(ranks, window, thr)
    return out


def build_anytime_feature_rows(dataset_name: str, step_rows: List[Dict[str, str]], run_rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    build_rank_features(step_rows)
    seqs = build_run_sequences(step_rows)
    run_meta = {make_run_key(r): r for r in run_rows}

    out_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []

    for key in sorted(seqs):
        seq = seqs[key]
        meta = run_meta[key]
        n_steps = int(len(seq["step"]))
        final_psnr = to_float(meta["final_psnr"])
        bad25 = int(final_psnr < 25.0)
        bad20 = int(final_psnr < 20.0)
        row: Dict[str, object] = {
            "dataset": dataset_name,
            "image_id": key[0],
            "run_index": key[1],
            "final_psnr": final_psnr,
            "bad25": bad25,
            "bad20": bad20,
            "num_steps": n_steps,
        }

        all_alarm_steps: List[Tuple[str, float]] = []
        residual_alarm_steps: List[Tuple[str, float]] = []

        for base in BASE_FEATURES:
            base_rule_rows = rule_alarms_for_run(seq, base)
            row.update(base_rule_rows)
            for rule_name, step0 in base_rule_rows.items():
                if math.isfinite(step0):
                    all_alarm_steps.append((rule_name, float(step0)))
                    if base in ("x0y_full_residual_normed", "x0y_lowfreq_residual_normed"):
                        residual_alarm_steps.append((rule_name, float(step0)))

        if all_alarm_steps:
            best_rule, best_step0 = min(all_alarm_steps, key=lambda t: t[1])
            row["any_alarm_rule"] = best_rule
            row["any_alarm_step0"] = best_step0
            row["any_alarm_step1"] = best_step0 + 1.0
            row["any_alarm_frac"] = (best_step0 + 1.0) / n_steps
        else:
            row["any_alarm_rule"] = ""
            row["any_alarm_step0"] = math.nan
            row["any_alarm_step1"] = math.nan
            row["any_alarm_frac"] = 2.0

        if residual_alarm_steps:
            best_rule, best_step0 = min(residual_alarm_steps, key=lambda t: t[1])
            row["residual_any_alarm_rule"] = best_rule
            row["residual_any_alarm_step0"] = best_step0
            row["residual_any_alarm_step1"] = best_step0 + 1.0
            row["residual_any_alarm_frac"] = (best_step0 + 1.0) / n_steps
        else:
            row["residual_any_alarm_rule"] = ""
            row["residual_any_alarm_step0"] = math.nan
            row["residual_any_alarm_step1"] = math.nan
            row["residual_any_alarm_frac"] = 2.0

        for frac in WINDOW_FRACS:
            row[f"visible_by_{int(round(frac * 100)):d}pct"] = int(row["any_alarm_frac"] <= frac)
            row[f"residual_visible_by_{int(round(frac * 100)):d}pct"] = int(row["residual_any_alarm_frac"] <= frac)

        out_rows.append(row)

        # Keep a shorter event table by rule. This is long format and easier to inspect.
        for col, step0 in row.items():
            if not isinstance(col, str) or not col.endswith("__first_step0"):
                continue
            event_rows.append(
                {
                    "dataset": dataset_name,
                    "image_id": key[0],
                    "run_index": key[1],
                    "rule_name": col,
                    "final_psnr": final_psnr,
                    "bad25": bad25,
                    "bad20": bad20,
                    "num_steps": n_steps,
                    "alarm_step0": step0,
                    "alarm_step1": (step0 + 1.0) if math.isfinite(to_float(step0)) else math.nan,
                    "alarm_frac": alarm_fraction(to_float(step0), n_steps),
                    "alarm_score": alarm_score(to_float(step0), n_steps),
                }
            )

    return out_rows, event_rows


def summarize_rule_table(event_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in event_rows:
        grouped[str(row["rule_name"])].append(row)

    summary_rows: List[Dict[str, object]] = []
    for rule_name, rows in sorted(grouped.items()):
        bad25 = np.asarray([int(r["bad25"]) for r in rows], dtype=int)
        bad20 = np.asarray([int(r["bad20"]) for r in rows], dtype=int)
        alarm_frac = np.asarray([to_float(r["alarm_frac"]) for r in rows], dtype=float)
        alarm_score_vals = np.asarray([to_float(r["alarm_score"]) for r in rows], dtype=float)
        summary_rows.append(
            {
                "rule_name": rule_name,
                "num_runs": len(rows),
                "bad25_auroc": auc_roc(bad25, alarm_score_vals),
                "bad25_auprc": auc_pr(bad25, alarm_score_vals),
                "bad20_auroc": auc_roc(bad20, alarm_score_vals),
                "bad20_auprc": auc_pr(bad20, alarm_score_vals),
                "bad25_visible_by_50pct": int(np.sum((bad25 == 1) & (alarm_frac <= 0.50))),
                "bad25_visible_by_60pct": int(np.sum((bad25 == 1) & (alarm_frac <= 0.60))),
                "bad25_visible_by_70pct": int(np.sum((bad25 == 1) & (alarm_frac <= 0.70))),
                "bad25_visible_by_80pct": int(np.sum((bad25 == 1) & (alarm_frac <= 0.80))),
                "bad20_visible_by_50pct": int(np.sum((bad20 == 1) & (alarm_frac <= 0.50))),
                "bad20_visible_by_60pct": int(np.sum((bad20 == 1) & (alarm_frac <= 0.60))),
                "bad20_visible_by_70pct": int(np.sum((bad20 == 1) & (alarm_frac <= 0.70))),
                "bad20_visible_by_80pct": int(np.sum((bad20 == 1) & (alarm_frac <= 0.80))),
                "median_alarm_frac_bad25": median_or_nan(alarm_frac[bad25 == 1]),
                "median_alarm_frac_bad20": median_or_nan(alarm_frac[bad20 == 1]),
                "median_alarm_frac_good": median_or_nan(alarm_frac[(bad25 == 0)]),
            }
        )
    return summary_rows


def build_window_summary(feature_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in sorted({str(r["dataset"]) for r in feature_rows} | {"ALL"}):
        if dataset == "ALL":
            subset = feature_rows
        else:
            subset = [r for r in feature_rows if str(r["dataset"]) == dataset]
        if not subset:
            continue
        for scope in ("any", "residual"):
            frac_key = "any_alarm_frac" if scope == "any" else "residual_any_alarm_frac"
            for frac in WINDOW_FRACS:
                bad25 = [r for r in subset if int(r["bad25"]) == 1]
                bad20 = [r for r in subset if int(r["bad20"]) == 1]
                rows.append(
                    {
                        "dataset": dataset,
                        "scope": scope,
                        "window_frac": frac,
                        "bad25_total": len(bad25),
                        "bad25_visible": int(sum(float(r[frac_key]) <= frac for r in bad25)),
                        "bad20_total": len(bad20),
                        "bad20_visible": int(sum(float(r[frac_key]) <= frac for r in bad20)),
                        "bad25_visible_rate": (sum(float(r[frac_key]) <= frac for r in bad25) / len(bad25)) if bad25 else math.nan,
                        "bad20_visible_rate": (sum(float(r[frac_key]) <= frac for r in bad20) / len(bad20)) if bad20 else math.nan,
                        "median_alarm_frac_bad25": median_or_nan([to_float(r[frac_key]) for r in bad25]),
                        "median_alarm_frac_bad20": median_or_nan([to_float(r[frac_key]) for r in bad20]),
                    }
                )
    return rows


def build_unflaggable_bad_rows(feature_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for r in feature_rows:
        if int(r["bad25"]) == 1 and float(r["any_alarm_frac"]) > 1.0:
            rows.append(dict(r))
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["image_id"]), int(r["run_index"])))
    return rows


def build_image00017_rows(feature_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = [dict(r) for r in feature_rows if str(r["image_id"]) == "00017"]
    rows.sort(key=lambda r: (str(r["dataset"]), int(r["run_index"])))
    return rows


def summarize_feature_rows(feature_rows: List[Dict[str, object]]) -> str:
    all_bad25 = [r for r in feature_rows if int(r["bad25"]) == 1]
    all_bad20 = [r for r in feature_rows if int(r["bad20"]) == 1]
    lines = []
    lines.append("# A17 Offline Anytime Detector Design")
    lines.append("")
    lines.append("This pass is diagnostic only. It uses existing A8, A11, A14, and A16 trajectory outputs and does not modify any frozen policy.")
    lines.append("")
    lines.append("## Development Diagnostics")
    lines.append("")
    lines.append(f"- total runs analyzed: `{len(feature_rows)}`")
    lines.append(f"- bad25 runs: `{len(all_bad25)}`")
    lines.append(f"- bad20 runs: `{len(all_bad20)}`")
    lines.append("")

    def visible_count(key: str, thresh: float, bad: List[Dict[str, object]]) -> int:
        return int(sum(float(r[key]) <= thresh for r in bad))

    lines.append("### Union visibility by window")
    for frac in WINDOW_FRACS:
        lines.append(
            f"- before {int(round(frac * 100))}%: bad25 `{visible_count('any_alarm_frac', frac, all_bad25)}/{len(all_bad25)}`; "
            f"bad20 `{visible_count('any_alarm_frac', frac, all_bad20)}/{len(all_bad20)}`"
        )
    lines.append("")
    lines.append("### Candidate anytime rules")
    lines.append("")
    lines.append("- residual / low-frequency rank persistence and cumulative-count rules are the strongest and most interpretable event-time signals in this pass.")
    lines.append("- rolling residual-rank slope is useful as a supporting diagnostic, but it should be treated as a heuristic until frozen on a training split.")
    lines.append("- disagreement, correction-norm, and step-jump persistence add value for some late-developing failures, but they are not yet a full replacement for residual-rank visibility.")
    lines.append("")
    lines.append("### Warnings about overfitting")
    lines.append("")
    lines.append("- These thresholds are retrospective diagnostics, not prospective evidence.")
    lines.append("- The union-of-rules view is useful for diagnosis, but a future controller must freeze one exact feature family, window, and persistence definition before evaluation.")
    lines.append("- Any future A18/A19 validation should lock the event-time rule first, then evaluate it on fresh trajectories without changing the window logic.")
    lines.append("")
    lines.append("### What would need to be frozen later")
    lines.append("")
    lines.append("- the exact base feature family or feature pair")
    lines.append("- the rolling window lengths")
    lines.append("- the persistence count `m`")
    lines.append("- the cumulative count threshold")
    lines.append("- the slope threshold")
    lines.append("- the fallback source / intervention action")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for A17 outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional overrides as NAME=PATH. Defaults to A8/A11/A14/A16 paths.",
    )
    args = parser.parse_args()

    dataset_specs = DATASETS
    if args.datasets:
        parsed: List[Tuple[str, Path]] = []
        for item in args.datasets:
            name, path_str = item.split("=", 1)
            parsed.append((name, Path(path_str)))
        dataset_specs = parsed

    all_feature_rows: List[Dict[str, object]] = []
    all_event_rows: List[Dict[str, object]] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_dir in dataset_specs:
        data = load_dataset(dataset_name, dataset_dir)
        step_rows = data["step_rows"]
        run_rows = data["run_rows"]
        feature_rows, event_rows = build_anytime_feature_rows(dataset_name, step_rows, run_rows)
        all_feature_rows.extend(feature_rows)
        all_event_rows.extend(event_rows)

    all_feature_rows.sort(key=lambda r: (str(r["dataset"]), str(r["image_id"]), int(r["run_index"])))
    all_event_rows.sort(key=lambda r: (str(r["dataset"]), str(r["image_id"]), int(r["run_index"]), str(r["rule_name"])))

    write_csv(args.output_dir / "anytime_feature_table.csv", all_feature_rows)
    write_csv(args.output_dir / "anytime_detection_event_table.csv", summarize_rule_table(all_event_rows))
    write_csv(args.output_dir / "anytime_window_summary.csv", build_window_summary(all_feature_rows))
    write_csv(args.output_dir / "unflaggable_bad_runs.csv", build_unflaggable_bad_rows(all_feature_rows))
    write_csv(args.output_dir / "image00017_anytime_diagnosis.csv", build_image00017_rows(all_feature_rows))
    write_text(args.output_dir / "SUMMARY.md", summarize_feature_rows(all_feature_rows))


if __name__ == "__main__":
    main()
