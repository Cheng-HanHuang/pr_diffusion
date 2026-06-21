#!/usr/bin/env python3
"""A7 retrospective controller simulation for Branch A hard-image A5 subset."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


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


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else math.nan


def image_id_from_np_basename(image_basename: str) -> str:
    return Path(image_basename).stem


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
            raise ValueError(f"No NP rows for {image_id} at noise {noise}")
        selected = min(
            rows,
            key=lambda r: (
                r["selector_post_winner_lf_mse_mean"],
                -r["psnr"],
                str(r["config_tag"]),
                int(r["seed"]),
            ),
        )
        oracle = max(rows, key=lambda r: r["psnr"])
        out[image_id] = {
            "np_selected_psnr": selected["psnr"],
            "np_selected_config_tag": selected["config_tag"],
            "np_selected_seed": selected["seed"],
            "np_selected_selector_post_lf_mse": selected["selector_post_winner_lf_mse_mean"],
            "np_oracle_psnr": oracle["psnr"],
            "np_oracle_config_tag": oracle["config_tag"],
            "np_oracle_seed": oracle["seed"],
        }
    return out


def pick_detector(detector_rows: List[Dict[str, str]], detector_id: str, label: str) -> Dict[str, object]:
    matches = [r for r in detector_rows if r["detector_id"] == detector_id and r["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"Expected one detector row for {detector_id} {label}, got {len(matches)}")
    row = matches[0]
    return {
        "detector_id": detector_id,
        "label": label,
        "best_threshold_direction": row["best_threshold_direction"],
        "best_threshold": to_float(row["best_threshold"]),
        "auroc": to_float(row["auroc"]),
        "auprc": to_float(row["auprc"]),
        "best_balanced_accuracy": to_float(row["best_balanced_accuracy"]),
        "loo_mean_balanced_accuracy": to_float(row["loo_mean_balanced_accuracy"]),
        "best_fp": int(float(row["best_fp"])),
        "best_fn": int(float(row["best_fn"])),
    }


def add_interrun_step_features(step_rows: List[Dict[str, str]]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    for row in step_rows:
        grouped.setdefault((str(row["image_id"]), int(row["step"])), []).append(row)
    for _, rows in grouped.items():
        vals = sorted(((to_float(r["x0y_full_residual_normed"]), r) for r in rows), key=lambda x: x[0])
        for rank, (_, row) in enumerate(vals, start=1):
            row["x0y_full_residual_normed__interrun_rank_high"] = float(rank)


def aggregate_detector_from_steps(step_rows: List[Dict[str, str]], detector_id: str) -> Dict[Tuple[str, int], float]:
    parts = detector_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"Bad detector id: {detector_id}")
    feature = "__".join(parts[:-2])
    window = parts[-2]
    aggregate = parts[-1]
    frac_map = {
        "first10pct": 0.10,
        "first20pct": 0.20,
        "first30pct": 0.30,
        "first50pct": 0.50,
    }
    if window not in frac_map:
        raise ValueError(f"Unsupported window in A7: {window}")

    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    for row in step_rows:
        grouped.setdefault((str(row["image_id"]), int(row["run_index"])), []).append(row)

    out: Dict[Tuple[str, int], float] = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r["step"]))
        k = max(1, int(math.ceil(len(rows) * frac_map[window])))
        window_rows = rows[:k]
        vals = [to_float(r[feature]) for r in window_rows]
        if aggregate == "max":
            out[key] = max(vals)
        elif aggregate == "mean":
            out[key] = mean(vals)
        elif aggregate == "final_in_window":
            out[key] = vals[-1]
        else:
            raise ValueError(f"Unsupported aggregate in A7: {aggregate}")
    return out


def flagged_by_detector(value: float, direction: str, threshold: float) -> bool:
    if direction == "high_is_risky":
        return value >= threshold
    if direction == "low_is_risky":
        return value <= threshold
    raise ValueError(direction)


def evaluate_policy(
    policy_name: str,
    policy_kind: str,
    sits_rows: List[Dict[str, str]],
    np_fallbacks: Dict[str, Dict[str, object]],
    detector: Dict[str, object] | None,
    detector_values: Dict[Tuple[str, int], float] | None,
    replacement_mode: str,
    notes: str,
    oracle_risk: bool = False,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    run_rows: List[Dict[str, object]] = []
    replaced_rows: List[Dict[str, object]] = []

    for row in sits_rows:
        image_id = str(row["image_id"])
        run_index = int(row["run_index"])
        final_psnr = to_float(row["final_psnr"])
        bad25 = final_psnr < 25.0
        bad20 = final_psnr < 20.0

        flagged = False
        detector_value = math.nan
        if oracle_risk:
            flagged = bad25
        elif detector is not None and detector_values is not None:
            detector_value = detector_values[(image_id, run_index)]
            flagged = flagged_by_detector(
                detector_value,
                str(detector["best_threshold_direction"]),
                float(detector["best_threshold"]),
            )

        replacement_psnr = final_psnr
        replacement_source = "sitcom"
        replacement_detail = ""
        replaced = False
        if flagged and replacement_mode != "none":
            fb = np_fallbacks[image_id]
            replaced = True
            if replacement_mode == "np_selected":
                replacement_psnr = float(fb["np_selected_psnr"])
                replacement_source = "np_selected"
                replacement_detail = f"{fb['np_selected_config_tag']} seed={fb['np_selected_seed']}"
            elif replacement_mode == "np_oracle":
                replacement_psnr = float(fb["np_oracle_psnr"])
                replacement_source = "np_oracle_diagnostic"
                replacement_detail = f"{fb['np_oracle_config_tag']} seed={fb['np_oracle_seed']}"
            else:
                raise ValueError(replacement_mode)

        delta = replacement_psnr - final_psnr
        run_rows.append(
            {
                "policy_name": policy_name,
                "policy_kind": policy_kind,
                "notes": notes,
                "image_id": image_id,
                "run_index": run_index,
                "sitcom_final_psnr": final_psnr,
                "policy_final_psnr": replacement_psnr,
                "delta_vs_sitcom": delta,
                "final_bad_below25": replacement_psnr < 25.0,
                "final_bad_below20": replacement_psnr < 20.0,
                "sitcom_bad_below25": bad25,
                "sitcom_bad_below20": bad20,
                "was_flagged": flagged,
                "was_replaced": replaced,
                "replacement_source": replacement_source,
                "detector_id": detector["detector_id"] if detector else "",
                "detector_value": detector_value,
                "detector_direction": detector["best_threshold_direction"] if detector else "",
                "detector_threshold": detector["best_threshold"] if detector else "",
                "false_positive_replacement": replaced and not bad25,
                "true_positive_replacement": replaced and bad25,
                "false_negative_remaining_bad25": (not replaced) and bad25,
                "replacement_detail": replacement_detail,
            }
        )
        if replaced:
            replaced_rows.append(
                {
                    "policy_name": policy_name,
                    "policy_kind": policy_kind,
                    "image_id": image_id,
                    "run_index": run_index,
                    "sitcom_final_psnr": final_psnr,
                    "replacement_psnr": replacement_psnr,
                    "delta_vs_sitcom": delta,
                    "sitcom_bad_below25": bad25,
                    "sitcom_bad_below20": bad20,
                    "detector_id": detector["detector_id"] if detector else "",
                    "detector_value": detector_value,
                    "detector_direction": detector["best_threshold_direction"] if detector else "",
                    "detector_threshold": detector["best_threshold"] if detector else "",
                    "replacement_source": replacement_source,
                    "replacement_detail": replacement_detail,
                    "false_positive_replacement": not bad25,
                    "true_positive_replacement": bad25,
                }
            )
    return run_rows, replaced_rows


def summarize_run_level(policy_rows: List[Dict[str, object]]) -> Dict[str, object]:
    psnrs = [float(r["policy_final_psnr"]) for r in policy_rows]
    return {
        "run_level_mean_psnr": mean(psnrs),
        "run_level_min_psnr": min(psnrs),
        "run_level_num_below25": sum(1 for p in psnrs if p < 25.0),
        "run_level_num_below20": sum(1 for p in psnrs if p < 20.0),
        "num_replaced": sum(1 for r in policy_rows if bool(r["was_replaced"])),
        "num_false_positive_replacements": sum(1 for r in policy_rows if bool(r["false_positive_replacement"])),
        "num_true_positive_replacements": sum(1 for r in policy_rows if bool(r["true_positive_replacement"])),
        "num_false_negative_remaining_bad25": sum(1 for r in policy_rows if bool(r["false_negative_remaining_bad25"])),
        "mean_delta_vs_sitcom": mean([float(r["delta_vs_sitcom"]) for r in policy_rows]),
        "mean_delta_on_replaced": mean([float(r["delta_vs_sitcom"]) for r in policy_rows if bool(r["was_replaced"])]),
        "mean_delta_on_false_positive_replacements": mean(
            [float(r["delta_vs_sitcom"]) for r in policy_rows if bool(r["false_positive_replacement"])]
        ),
    }


def summarize_image_level(policy_name: str, policy_kind: str, policy_rows: List[Dict[str, object]], notes: str) -> List[Dict[str, object]]:
    by_image: Dict[str, List[Dict[str, object]]] = {}
    for row in policy_rows:
        by_image.setdefault(str(row["image_id"]), []).append(row)
    out = []
    for image_id, rows in sorted(by_image.items()):
        vals = [float(r["policy_final_psnr"]) for r in rows]
        out.append(
            {
                "policy_name": policy_name,
                "policy_kind": policy_kind,
                "notes": notes,
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


def summarize_policy(policy_name: str, policy_kind: str, notes: str, policy_rows: List[Dict[str, object]], image_rows: List[Dict[str, object]]) -> Dict[str, object]:
    run_summary = summarize_run_level(policy_rows)
    bests = [float(r["best_of_4_psnr"]) for r in image_rows]
    mins = [float(r["min_of_4_psnr"]) for r in image_rows]
    means = [float(r["mean_of_4_psnr"]) for r in image_rows]
    return {
        "policy_name": policy_name,
        "policy_kind": policy_kind,
        "notes": notes,
        **run_summary,
        "image_level_best_of_4_mean_psnr": mean(bests),
        "image_level_best_of_4_min_psnr": min(bests),
        "image_level_mean_of_4_mean_psnr": mean(means),
        "image_level_mean_of_4_min_psnr": min(means),
        "image_level_min_of_4_mean_psnr": mean(mins),
        "image_level_min_of_4_min_psnr": min(mins),
    }


def render_summary(
    outdir: Path,
    policy_summary_rows: List[Dict[str, object]],
    best_detector: Dict[str, object],
    conservative_detector: Dict[str, object],
) -> None:
    lines = [
        "# A7 Retrospective Controller Simulation",
        "",
        "This is an offline simulation on the A5 hard-image subset. It tests whether detector-triggered action can improve final reliability, while keeping executable and oracle-assisted policies separate.",
        "",
        "## Detector Choices",
        "",
        f"- Retrospective detector fallback: `{best_detector['detector_id']}` with `{best_detector['best_threshold_direction']}` threshold `{best_detector['best_threshold']:.6g}`.",
        f"- Conservative detector fallback: `{conservative_detector['detector_id']}` with `{conservative_detector['best_threshold_direction']}` threshold `{conservative_detector['best_threshold']:.6g}`.",
        "",
        "## Policy Summary",
        "",
        "| policy | kind | run mean | run min | below25 | below20 | image best-of-4 mean | image best-of-4 min | replaced | FP repl | FN remain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_summary_rows:
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
            "## Interpretation",
            "",
            "- Policies using `np_selected` are the executable-style fallbacks in this retrospective sense: the trigger and replacement source both avoid PSNR/oracle information at decision time.",
            "- Policies using `np_oracle` or oracle risk labels are diagnostics only. They estimate the upper bound of what better risk detection or better fallback quality could buy, but they are not executable solvers.",
            "- `false_positive_replacements` count good SITCOM runs that were replaced anyway. Their PSNR delta shows whether over-triggering hurts.",
            "- `false_negative_remaining_bad25` counts catastrophic SITCOM runs that the detector failed to catch, so those failures remain after the fallback policy.",
            "",
        ]
    )
    write_text(outdir / 'SUMMARY.md', '\n'.join(lines) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--a5_run_level', required=True)
    ap.add_argument('--a5_step_metrics', required=True)
    ap.add_argument('--a6_detector_csv', required=True)
    ap.add_argument('--np_run_level', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--noise', type=float, default=0.05)
    args = ap.parse_args()

    a5_rows = read_csv(Path(args.a5_run_level))
    step_rows = read_csv(Path(args.a5_step_metrics))
    detector_rows = read_csv(Path(args.a6_detector_csv))
    np_rows = read_csv(Path(args.np_run_level))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target_images = sorted({str(r['image_id']) for r in a5_rows})
    np_fallbacks = load_np_fallbacks(np_rows, args.noise, target_images)

    add_interrun_step_features(step_rows)
    best_detector = pick_detector(
        detector_rows,
        'x0y_full_residual_normed__interrun_rank_high__first10pct__mean',
        'bad25',
    )
    conservative_detector = pick_detector(
        detector_rows,
        'correction_norm__first10pct__max',
        'bad25',
    )
    detector_value_maps = {
        best_detector['detector_id']: aggregate_detector_from_steps(step_rows, str(best_detector['detector_id'])),
        conservative_detector['detector_id']: aggregate_detector_from_steps(step_rows, str(conservative_detector['detector_id'])),
    }

    policies = [
        {
            'policy_name': 'sitcom_only',
            'policy_kind': 'executable_baseline',
            'detector': None,
            'detector_values': None,
            'replacement_mode': 'none',
            'notes': 'Keep each SITCOM final output.',
            'oracle_risk': False,
        },
        {
            'policy_name': 'detector_best_to_np_selected',
            'policy_kind': 'retrospective_executable_style',
            'detector': best_detector,
            'detector_values': detector_value_maps[str(best_detector['detector_id'])],
            'replacement_mode': 'np_selected',
            'notes': 'Retrospectively tuned detector threshold, executable-style NP selected fallback.',
            'oracle_risk': False,
        },
        {
            'policy_name': 'detector_best_to_np_oracle',
            'policy_kind': 'diagnostic_oracle_fallback',
            'detector': best_detector,
            'detector_values': detector_value_maps[str(best_detector['detector_id'])],
            'replacement_mode': 'np_oracle',
            'notes': 'Same detector trigger, but NP oracle replacement. Diagnostic upper bound only.',
            'oracle_risk': False,
        },
        {
            'policy_name': 'detector_conservative_to_np_selected',
            'policy_kind': 'retrospective_executable_style_conservative',
            'detector': conservative_detector,
            'detector_values': detector_value_maps[str(conservative_detector['detector_id'])],
            'replacement_mode': 'np_selected',
            'notes': 'Zero-FP-ish conservative detector with NP selected fallback.',
            'oracle_risk': False,
        },
        {
            'policy_name': 'oracle_risk_to_np_selected',
            'policy_kind': 'diagnostic_oracle_risk',
            'detector': None,
            'detector_values': None,
            'replacement_mode': 'np_selected',
            'notes': 'Replace exactly the bad SITCOM runs using NP selected fallback. Diagnostic only.',
            'oracle_risk': True,
        },
        {
            'policy_name': 'oracle_risk_to_np_oracle',
            'policy_kind': 'diagnostic_oracle_risk_and_fallback',
            'detector': None,
            'detector_values': None,
            'replacement_mode': 'np_oracle',
            'notes': 'Replace exactly the bad SITCOM runs using NP oracle fallback. Diagnostic ceiling only.',
            'oracle_risk': True,
        },
    ]

    all_run_rows: List[Dict[str, object]] = []
    all_image_rows: List[Dict[str, object]] = []
    all_summary_rows: List[Dict[str, object]] = []
    all_replaced_rows: List[Dict[str, object]] = []

    for policy in policies:
        run_rows, replaced_rows = evaluate_policy(
            policy_name=str(policy['policy_name']),
            policy_kind=str(policy['policy_kind']),
            sits_rows=a5_rows,
            np_fallbacks=np_fallbacks,
            detector=policy['detector'],
            detector_values=policy['detector_values'],
            replacement_mode=str(policy['replacement_mode']),
            notes=str(policy['notes']),
            oracle_risk=bool(policy['oracle_risk']),
        )
        image_rows = summarize_image_level(str(policy['policy_name']), str(policy['policy_kind']), run_rows, str(policy['notes']))
        summary_row = summarize_policy(str(policy['policy_name']), str(policy['policy_kind']), str(policy['notes']), run_rows, image_rows)
        all_run_rows.extend(run_rows)
        all_image_rows.extend(image_rows)
        all_summary_rows.append(summary_row)
        all_replaced_rows.extend(replaced_rows)

    write_csv(outdir / 'controller_policy_run_level.csv', all_run_rows)
    write_csv(outdir / 'controller_policy_image_level.csv', all_image_rows)
    write_csv(outdir / 'controller_policy_summary.csv', all_summary_rows)
    write_csv(outdir / 'replaced_runs.csv', all_replaced_rows)
    render_summary(outdir, all_summary_rows, best_detector, conservative_detector)


if __name__ == '__main__':
    main()
