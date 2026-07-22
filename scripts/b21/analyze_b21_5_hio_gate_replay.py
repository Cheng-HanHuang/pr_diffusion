#!/usr/bin/env python3
"""Replay a clean-free exact-loss gate on the B21.5 HIO pilot.

This does not reinterpret HIO-at-step-200 as a replacement policy: that policy
failed.  It asks whether the cheap warm candidate is useful as an *additional*
arm beside the base candidate.

Frozen exploratory rule (chosen before this script is run):

    choose HIO iff exact_loss_hio < exact_loss_base - 0.7

The 0.7 margin is inherited from the frozen B21.4 LF gate rather than tuned on
this HIO panel.  PSNR is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SWEEP = [0.0, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]


def evaluate(df: pd.DataFrame, theta: float) -> dict[str, float | int]:
    choose = df["warm_exact_operator_loss"] < df["base_exact_operator_loss"] - theta
    base_good = df["base_psnr"] >= 25.0
    warm_good = df["warm_psnr"] >= 25.0
    selected_psnr = np.where(choose, df["warm_psnr"], df["base_psnr"])
    selected_good = selected_psnr >= 25.0
    return {
        "theta": float(theta),
        "n": int(len(df)),
        "accepted_hio": int(choose.sum()),
        "base_good25": int(base_good.sum()),
        "warm_good25": int(warm_good.sum()),
        "gated_good25": int(selected_good.sum()),
        "gated_net_vs_base": int(selected_good.sum() - base_good.sum()),
        "gated_rescues": int((choose & ~base_good & warm_good).sum()),
        "gated_harms": int((choose & base_good & ~warm_good).sum()),
        "selected_mean_psnr": float(np.mean(selected_psnr)),
        "selected_median_psnr": float(np.median(selected_psnr)),
        "selected_min_psnr": float(np.min(selected_psnr)),
    }


def grouped(df: pd.DataFrame, theta: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for image_id, sub in df.groupby("image_id", sort=True):
        row = {"group": str(image_id), **evaluate(sub, theta)}
        rows.append(row)
    heldout = df[df["image_id"] != "00046"]
    rows.append({"group": "HELDOUT4", **evaluate(heldout, theta)})
    rows.append({"group": "ALL", **evaluate(df, theta)})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    cols = list(columns)
    labels = [c.replace("_", " ") for c in cols]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for _, row in frame.iterrows():
        vals: list[str] = []
        for c in cols:
            value = row[c]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    df = pd.read_csv(args.input, dtype={"image_id": str})
    required = [
        "image_id",
        "case_id",
        "base_exact_operator_loss",
        "warm_exact_operator_loss",
        "base_psnr",
        "warm_psnr",
        "base_wall_seconds",
        "warm_total_wall_seconds",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    if len(df) != 40:
        raise ValueError(f"expected 40 rows, found {len(df)}")

    df["image_id"] = df["image_id"].map(lambda x: f"{int(x):05d}")
    for col in required[2:]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    theta = float(args.theta)
    choose = df["warm_exact_operator_loss"] < df["base_exact_operator_loss"] - theta
    df["select_hio"] = choose.astype(int)
    df["selected_candidate"] = np.where(choose, "hio_warm", "base_full")
    df["selected_exact_operator_loss"] = np.where(
        choose, df["warm_exact_operator_loss"], df["base_exact_operator_loss"]
    )
    df["selected_psnr"] = np.where(choose, df["warm_psnr"], df["base_psnr"])
    df["selected_good25"] = (df["selected_psnr"] >= 25.0).astype(int)
    df["gated_rescue"] = (
        choose & (df["base_psnr"] < 25.0) & (df["warm_psnr"] >= 25.0)
    ).astype(int)
    df["gated_harm"] = (
        choose & (df["base_psnr"] >= 25.0) & (df["warm_psnr"] < 25.0)
    ).astype(int)
    df["loss_delta_hio_minus_base"] = (
        df["warm_exact_operator_loss"] - df["base_exact_operator_loss"]
    )

    sweep = pd.DataFrame([evaluate(df, t) for t in SWEEP])
    by_group = grouped(df, theta)
    overall = evaluate(df, theta)
    heldout = evaluate(df[df["image_id"] != "00046"], theta)

    marginal_cost_ratio = float(df["warm_total_wall_seconds"].sum() / df["base_wall_seconds"].sum())
    total_portfolio_cost_ratio = 1.0 + marginal_cost_ratio
    exploratory_support = bool(
        overall["gated_net_vs_base"] >= 4
        and overall["gated_harms"] <= 1
        and heldout["gated_net_vs_base"] >= 0
        and marginal_cost_ratio <= 0.70
    )

    summary = {
        "status_of_hio_as_replacement": "rejected",
        "question_of_this_replay": "HIO as a gated auxiliary candidate",
        "frozen_theta": theta,
        "selection_rule": "choose HIO iff warm_exact_operator_loss < base_exact_operator_loss - theta",
        "clean_free_selection": True,
        "overall": overall,
        "heldout4": heldout,
        "marginal_hio_cost_over_base": marginal_cost_ratio,
        "base_plus_hio_total_cost_over_base": total_portfolio_cost_ratio,
        "exploratory_support_for_auxiliary_arm": exploratory_support,
        "support_gate": {
            "net_good25_at_least": 4,
            "gated_harms_at_most": 1,
            "heldout4_net_nonnegative": True,
            "marginal_cost_ratio_at_most": 0.70,
        },
        "next_if_supported": "run matched LF050 extension on these same 40 development cases; evaluate incremental HIO value beyond frozen base+LF gate",
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "hio_gate_replay_rows.csv"
    sweep_path = args.outdir / "hio_gate_threshold_sweep.csv"
    group_path = args.outdir / "hio_gate_summary_by_group.csv"
    summary_path = args.outdir / "hio_gate_replay_summary.json"
    df.to_csv(rows_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    by_group.to_csv(group_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = [
        "# B21.5 HIO auxiliary-arm gate replay",
        "",
        "The frozen HIO-at-step-200 replacement policy is rejected. This replay asks a different question: whether its cheap candidate is useful when retained only by a clean-free exact-loss gate.",
        "",
        f"Frozen rule: select HIO iff `loss_hio < loss_base - {theta}`. The margin is inherited from B21.4 rather than tuned on this panel.",
        "",
        "## Frozen-gate results",
        "",
        *markdown_table(by_group, ["group", "n", "base_good25", "warm_good25", "gated_good25", "gated_net_vs_base", "accepted_hio", "gated_rescues", "gated_harms", "selected_mean_psnr"]),
        "",
        "## Cost",
        "",
        f"- marginal HIO arm cost / base: `{marginal_cost_ratio:.6f}`",
        f"- base + HIO portfolio cost / base: `{total_portfolio_cost_ratio:.6f}`",
        "",
        "## Sensitivity audit (not a tuning authorization)",
        "",
        *markdown_table(sweep, ["theta", "accepted_hio", "gated_good25", "gated_net_vs_base", "gated_rescues", "gated_harms", "selected_mean_psnr"]),
        "",
        f"Exploratory support for testing HIO beyond frozen base+LF: **{exploratory_support}**",
        "",
        "A supported result does not validate the policy. It promotes only an LF-extension development experiment on the same 40 cases, followed by a new fresh validation if HIO adds clean-free rescues beyond base+LF.",
        "",
        f"Artifacts: `{args.outdir}`",
    ]
    args.report.write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[write]", rows_path)
    print("[write]", sweep_path)
    print("[write]", group_path)
    print("[write]", summary_path)
    print("[write]", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
