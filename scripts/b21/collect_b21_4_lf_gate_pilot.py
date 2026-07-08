#!/usr/bin/env python3
"""Collect B21.4 matched base-vs-LF pilot results.

This script is intentionally clean-free for selection: it gates LF by final exact
measurement loss only:

    select LF iff exact_operator_loss_lf < exact_operator_loss_base

PSNR is reported only as a diagnostic after selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_B19_BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
DEFAULT_OUTDIR = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_4_lf_gate_pilot")
DEFAULT_IMAGES = "00171,00480,00746,00971"
DEFAULT_SEEDS = "6800-6815"


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).replace(" ", ",").split(",") if x.strip()]


def parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in parse_csv_list(s):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def first_numeric(row: pd.Series, names: Sequence[str]) -> float:
    for name in names:
        if name in row.index:
            try:
                val = row[name]
                if pd.notna(val) and str(val) != "":
                    return float(val)
            except Exception:
                pass
    return math.nan


def first_str(row: pd.Series, names: Sequence[str]) -> str:
    for name in names:
        if name in row.index:
            val = row[name]
            if pd.notna(val):
                return str(val)
    return ""


def find_selector_csv(base: Path, image: str, meas_seed: int, seed: int, variant: str) -> Optional[Path]:
    # Prefer exact B20.12A-style names, but allow local naming variants.
    patterns = [
        f"B20_12A_{image}_ann400_diff5_{variant}_daps1S_meas{meas_seed}_runseed{seed}_exact_final_loss_selector.csv",
        f"B20_12A*{image}*{variant}*meas{meas_seed}*runseed{seed}*exact_final_loss_selector.csv",
        f"*{image}*{variant}*meas{meas_seed}*runseed{seed}*exact_final_loss_selector.csv",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(sorted(base.glob(pat)))
        if hits:
            break
    # Remove obvious non-matching variants: base should not match lf050 etc.
    filtered: List[Path] = []
    for p in hits:
        name = p.name
        if variant == "base":
            if any(v in name for v in ["lf010", "lf025", "lf050"]):
                continue
        else:
            if variant not in name:
                continue
        filtered.append(p)
    return filtered[0] if filtered else None


def read_selector(path: Optional[Path]) -> Dict[str, object]:
    if path is None or not path.exists():
        return {"exists": False, "csv_path": "", "error": "missing_csv"}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"exists": False, "csv_path": str(path), "error": f"read_error:{e}"}
    if df.empty:
        return {"exists": False, "csv_path": str(path), "error": "empty_csv"}
    row = df.iloc[0]
    psnr = first_numeric(row, [
        "psnr_recomputed_from_png",
        "selector_psnr_recomputed_from_png",
        "psnr",
        "psnr_metrics_json",
    ])
    exact = first_numeric(row, [
        "exact_operator_loss",
        "selector_exact_operator_loss",
        "operator_loss",
        "loss",
    ])
    sqrt_loss = first_numeric(row, [
        "sqrt_loss_over_y_norm",
        "selector_sqrt_loss_over_y_norm",
    ])
    sample_path = first_str(row, ["sample_path", "selected_sample_path", "image_path"])
    return {
        "exists": True,
        "csv_path": str(path),
        "error": "",
        "psnr": psnr,
        "exact_operator_loss": exact,
        "sqrt_loss_over_y_norm": sqrt_loss,
        "sample_path": sample_path,
        "columns": ",".join(df.columns),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    complete = [r for r in rows if r.get("complete")]
    selected = [r for r in complete if math.isfinite(float(r.get("selected_psnr", math.nan)))]
    n = len(complete)
    def good_count(key: str) -> int:
        return sum(1 for r in complete if float(r.get(key, math.nan)) >= 25.0)
    out: Dict[str, object] = {
        "expected_pairs": len(rows),
        "complete_pairs": n,
        "missing_pairs": len(rows) - n,
        "base_good25": good_count("base_psnr"),
        "lf_good25": good_count("lf_psnr"),
        "gated_good25": good_count("selected_psnr"),
        "accepted_lf": sum(1 for r in complete if int(r.get("select_lf", 0)) == 1),
        "rejected_lf": sum(1 for r in complete if int(r.get("select_lf", 0)) == 0),
        "rescue25_ungated_lf": sum(1 for r in complete if float(r.get("base_psnr", math.nan)) < 25.0 and float(r.get("lf_psnr", math.nan)) >= 25.0),
        "lost25_ungated_lf": sum(1 for r in complete if float(r.get("base_psnr", math.nan)) >= 25.0 and float(r.get("lf_psnr", math.nan)) < 25.0),
        "rescue25_gated": sum(1 for r in complete if float(r.get("base_psnr", math.nan)) < 25.0 and float(r.get("selected_psnr", math.nan)) >= 25.0),
        "lost25_gated": sum(1 for r in complete if float(r.get("base_psnr", math.nan)) >= 25.0 and float(r.get("selected_psnr", math.nan)) < 25.0),
    }
    if selected:
        vals = [float(r["selected_psnr"]) for r in selected]
        out.update({
            "selected_mean_psnr": sum(vals) / len(vals),
            "selected_min_psnr": min(vals),
            "selected_max_psnr": max(vals),
        })
    by_image: Dict[str, Dict[str, object]] = {}
    for img in sorted(set(str(r.get("image_id")) for r in complete)):
        sub = [r for r in complete if str(r.get("image_id")) == img]
        by_image[img] = {
            "n": len(sub),
            "base_good25": sum(1 for r in sub if float(r.get("base_psnr", math.nan)) >= 25.0),
            "lf_good25": sum(1 for r in sub if float(r.get("lf_psnr", math.nan)) >= 25.0),
            "gated_good25": sum(1 for r in sub if float(r.get("selected_psnr", math.nan)) >= 25.0),
            "accepted_lf": sum(1 for r in sub if int(r.get("select_lf", 0)) == 1),
            "rescue25_gated": sum(1 for r in sub if float(r.get("base_psnr", math.nan)) < 25.0 and float(r.get("selected_psnr", math.nan)) >= 25.0),
            "lost25_gated": sum(1 for r in sub if float(r.get("base_psnr", math.nan)) >= 25.0 and float(r.get("selected_psnr", math.nan)) < 25.0),
        }
    out["by_image"] = by_image
    return out


def render_report(rows: Sequence[Dict[str, object]], summary: Dict[str, object], outdir: Path) -> str:
    lines = [
        "# B21.4 LF exact-loss gate pilot",
        "",
        "Selector: choose LF iff `exact_operator_loss_lf < exact_operator_loss_base`.",
        "PSNR is diagnostic only.",
        "",
        "## Summary",
        "",
    ]
    for k, v in summary.items():
        if k == "by_image":
            continue
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## By image", "", "| image | n | base good25 | LF good25 | gated good25 | accepted LF | gated rescues | gated losts |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for img, s in summary.get("by_image", {}).items():
        lines.append(f"| `{img}` | {s['n']} | {s['base_good25']} | {s['lf_good25']} | {s['gated_good25']} | {s['accepted_lf']} | {s['rescue25_gated']} | {s['lost25_gated']} |")
    lines += ["", "## Rows", "", "| image | seed | base PSNR | LF PSNR | selected | selected PSNR | base loss | LF loss |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        if not r.get("complete"):
            continue
        lines.append(
            f"| `{r['image_id']}` | {r['seed']} | {float(r['base_psnr']):.3f} | {float(r['lf_psnr']):.3f} | `{r['selected_variant']}` | {float(r['selected_psnr']):.3f} | {float(r['base_exact_operator_loss']):.3f} | {float(r['lf_exact_operator_loss']):.3f} |"
        )
    lines += ["", "Artifacts:", "", "```text", str(outdir / "b21_4_lf_gate_pilot_pairs.csv"), str(outdir / "b21_4_lf_gate_pilot_summary.json"), "```", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b19_base", default=str(DEFAULT_B19_BASE))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--meas_seed", type=int, default=5001)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--lf_variant", default="lf050")
    ap.add_argument("--ann_steps", type=int, default=400)
    ap.add_argument("--diff_steps", type=int, default=5)
    ap.add_argument("--report_path", default="")
    args = ap.parse_args()

    b19_base = Path(args.b19_base)
    outdir = Path(args.outdir)
    images = parse_csv_list(args.images)
    seeds = parse_seeds(args.seeds)
    rows: List[Dict[str, object]] = []

    for image in images:
        for seed in seeds:
            base_path = find_selector_csv(b19_base, image, args.meas_seed, seed, "base")
            lf_path = find_selector_csv(b19_base, image, args.meas_seed, seed, args.lf_variant)
            br = read_selector(base_path)
            lr = read_selector(lf_path)
            complete = bool(br.get("exists") and lr.get("exists") and math.isfinite(float(br.get("exact_operator_loss", math.nan))) and math.isfinite(float(lr.get("exact_operator_loss", math.nan))))
            select_lf = int(complete and float(lr["exact_operator_loss"]) < float(br["exact_operator_loss"]))
            selected = lr if select_lf else br
            row: Dict[str, object] = {
                "image_id": image,
                "meas_seed": args.meas_seed,
                "seed": seed,
                "ann_steps": args.ann_steps,
                "diff_steps": args.diff_steps,
                "lf_variant": args.lf_variant,
                "complete": int(complete),
                "base_csv_path": br.get("csv_path", ""),
                "lf_csv_path": lr.get("csv_path", ""),
                "base_error": br.get("error", ""),
                "lf_error": lr.get("error", ""),
                "base_psnr": br.get("psnr", math.nan),
                "lf_psnr": lr.get("psnr", math.nan),
                "base_exact_operator_loss": br.get("exact_operator_loss", math.nan),
                "lf_exact_operator_loss": lr.get("exact_operator_loss", math.nan),
                "base_sqrt_loss_over_y_norm": br.get("sqrt_loss_over_y_norm", math.nan),
                "lf_sqrt_loss_over_y_norm": lr.get("sqrt_loss_over_y_norm", math.nan),
                "select_lf": select_lf,
                "selected_variant": args.lf_variant if select_lf else "base",
                "selected_psnr": selected.get("psnr", math.nan),
                "selected_exact_operator_loss": selected.get("exact_operator_loss", math.nan),
                "selected_sample_path": selected.get("sample_path", ""),
            }
            if complete:
                row["delta_lf_minus_base_psnr"] = float(row["lf_psnr"]) - float(row["base_psnr"])
                row["delta_lf_minus_base_exact_loss"] = float(row["lf_exact_operator_loss"]) - float(row["base_exact_operator_loss"])
            rows.append(row)

    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "b21_4_lf_gate_pilot_pairs.csv", rows)
    summary = summarize(rows)
    summary.update({
        "images": images,
        "seeds": seeds,
        "meas_seed": args.meas_seed,
        "lf_variant": args.lf_variant,
        "ann_steps": args.ann_steps,
        "diff_steps": args.diff_steps,
        "selector": "select_lf_if_exact_operator_loss_lf_lt_base",
    })
    (outdir / "b21_4_lf_gate_pilot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = render_report(rows, summary, outdir)
    if args.report_path:
        rp = Path(args.report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report)
        print(f"[write] {rp}")
    print(f"[write] {outdir / 'b21_4_lf_gate_pilot_pairs.csv'}")
    print(f"[write] {outdir / 'b21_4_lf_gate_pilot_summary.json'}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
