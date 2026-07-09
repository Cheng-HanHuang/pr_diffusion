#!/usr/bin/env python3
"""Apply a B21.4 LF margin gate to an existing matched-pair CSV.

The input is normally produced by collect_b21_4_lf_gate_pilot.py.  This script
recomputes the clean-free final selector with a margin:

    select LF iff exact_operator_loss_lf < exact_operator_loss_base - theta

PSNR is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_IN = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_4_lf_gate_pilot/b21_4_lf_gate_pilot_pairs.csv")
DEFAULT_OUTDIR = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_4_lf_gate_pilot")


def good25(x) -> pd.Series:
    return pd.Series(x).astype(float) >= 25.0


def summarize(df: pd.DataFrame, theta: float) -> Dict[str, object]:
    complete = df[df["complete"].astype(int) == 1].copy()
    complete["base_good25"] = complete["base_psnr"].astype(float) >= 25.0
    complete["lf_good25"] = complete["lf_psnr"].astype(float) >= 25.0
    complete["loss_delta_lf_minus_base"] = complete["lf_exact_operator_loss"].astype(float) - complete["base_exact_operator_loss"].astype(float)
    complete["select_lf_margin"] = complete["loss_delta_lf_minus_base"] < -float(theta)
    complete["selected_variant_margin"] = np.where(complete["select_lf_margin"], complete["lf_variant"], "base")
    complete["selected_psnr_margin"] = np.where(complete["select_lf_margin"], complete["lf_psnr"], complete["base_psnr"])
    complete["selected_exact_operator_loss_margin"] = np.where(
        complete["select_lf_margin"],
        complete["lf_exact_operator_loss"],
        complete["base_exact_operator_loss"],
    )
    complete["selected_good25_margin"] = complete["selected_psnr_margin"].astype(float) >= 25.0

    summary: Dict[str, object] = {
        "gate_theta": float(theta),
        "expected_pairs": int(len(df)),
        "complete_pairs": int(len(complete)),
        "missing_pairs": int(len(df) - len(complete)),
        "base_good25": int(complete["base_good25"].sum()),
        "lf_good25": int(complete["lf_good25"].sum()),
        "margin_gated_good25": int(complete["selected_good25_margin"].sum()),
        "accepted_lf_margin": int(complete["select_lf_margin"].sum()),
        "rejected_lf_margin": int((~complete["select_lf_margin"]).sum()),
        "rescue25_ungated_lf": int(((~complete["base_good25"]) & complete["lf_good25"]).sum()),
        "lost25_ungated_lf": int((complete["base_good25"] & (~complete["lf_good25"])).sum()),
        "rescue25_margin_gated": int(((~complete["base_good25"]) & complete["selected_good25_margin"]).sum()),
        "lost25_margin_gated": int((complete["base_good25"] & (~complete["selected_good25_margin"])).sum()),
        "selected_min_psnr_margin": float(complete["selected_psnr_margin"].min()) if len(complete) else math.nan,
        "selected_mean_psnr_margin": float(complete["selected_psnr_margin"].mean()) if len(complete) else math.nan,
        "selected_max_psnr_margin": float(complete["selected_psnr_margin"].max()) if len(complete) else math.nan,
    }

    by_image: Dict[str, object] = {}
    for image_id, sub in complete.groupby("image_id", sort=True):
        by_image[str(image_id).zfill(5)] = {
            "n": int(len(sub)),
            "base_good25": int(sub["base_good25"].sum()),
            "lf_good25": int(sub["lf_good25"].sum()),
            "margin_gated_good25": int(sub["selected_good25_margin"].sum()),
            "accepted_lf_margin": int(sub["select_lf_margin"].sum()),
            "rescue25_margin_gated": int(((~sub["base_good25"]) & sub["selected_good25_margin"]).sum()),
            "lost25_margin_gated": int((sub["base_good25"] & (~sub["selected_good25_margin"])).sum()),
            "selected_min_psnr_margin": float(sub["selected_psnr_margin"].min()),
            "selected_mean_psnr_margin": float(sub["selected_psnr_margin"].mean()),
        }
    summary["by_image"] = by_image
    return summary, complete


def render_report(summary: Dict[str, object], rows: pd.DataFrame, outdir: Path) -> str:
    theta = summary["gate_theta"]
    lines: List[str] = [
        "# B21.4 LF margin-gate report",
        "",
        f"Selector: choose LF iff `exact_operator_loss_lf < exact_operator_loss_base - {theta}`.",
        "PSNR is diagnostic only.",
        "",
        "## Summary",
        "",
    ]
    for k, v in summary.items():
        if k == "by_image":
            continue
        lines.append(f"- {k}: `{v}`")
    lines += [
        "",
        "## By image",
        "",
        "| image | n | base good25 | LF good25 | margin-gated good25 | accepted LF | margin rescues | margin losts | min selected PSNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for img, s in summary.get("by_image", {}).items():
        lines.append(
            f"| `{img}` | {s['n']} | {s['base_good25']} | {s['lf_good25']} | {s['margin_gated_good25']} | {s['accepted_lf_margin']} | {s['rescue25_margin_gated']} | {s['lost25_margin_gated']} | {s['selected_min_psnr_margin']:.3f} |"
        )
    lines += [
        "",
        "## Selected rows",
        "",
        "| image | seed | base PSNR | LF PSNR | selected | selected PSNR | loss delta LF-base |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for _, r in rows.sort_values(["image_id", "seed"]).iterrows():
        lines.append(
            f"| `{str(r['image_id']).zfill(5)}` | {int(r['seed'])} | {float(r['base_psnr']):.3f} | {float(r['lf_psnr']):.3f} | `{r['selected_variant_margin']}` | {float(r['selected_psnr_margin']):.3f} | {float(r['loss_delta_lf_minus_base']):.3f} |"
        )
    lines += [
        "",
        "Artifacts:",
        "",
        "```text",
        str(outdir / "b21_4_margin_gate_pairs.csv"),
        str(outdir / "b21_4_margin_gate_summary.json"),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pairs", default=str(DEFAULT_IN))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--report_path", default="")
    args = ap.parse_args()

    input_pairs = Path(args.input_pairs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_pairs)
    summary, rows = summarize(df, args.theta)
    summary["input_pairs"] = str(input_pairs)

    pairs_out = outdir / "b21_4_margin_gate_pairs.csv"
    summary_out = outdir / "b21_4_margin_gate_summary.json"
    rows.to_csv(pairs_out, index=False)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = render_report(summary, rows, outdir)
    if args.report_path:
        rp = Path(args.report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report, encoding="utf-8")
        print(f"[write] {rp}")
    print(f"[write] {pairs_out}")
    print(f"[write] {summary_out}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
