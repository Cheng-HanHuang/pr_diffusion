#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def read_metric(path: Path) -> tuple[float, float]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"empty metric CSV: {path}")
    row = df.iloc[0]
    loss = float(pd.to_numeric(row.get("exact_operator_loss"), errors="raise"))
    psnr = float(pd.to_numeric(row.get("psnr_recomputed_from_png"), errors="raise"))
    return loss, psnr


def read_lf_wall(path: Path) -> float:
    if not path.exists():
        return float("nan")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name, value = line.split("\t", 1)
        if name == "lf050":
            return float(value)
    return float("nan")


def summarize(group: pd.DataFrame, name: str) -> dict[str, object]:
    n = len(group)
    return {
        "group": name,
        "n": n,
        "base_good25": int(group.base_good25.sum()),
        "lf_good25": int(group.lf_good25.sum()),
        "hio_good25": int(group.warm_good25.sum()),
        "base_lf_gated_good25": int(group.base_lf_good25.sum()),
        "three_arm_gated_good25": int(group.three_arm_good25.sum()),
        "three_arm_net_vs_base_lf": int(group.three_arm_good25.sum() - group.base_lf_good25.sum()),
        "three_arm_net_vs_base": int(group.three_arm_good25.sum() - group.base_good25.sum()),
        "incremental_hio_rescues": int(group.incremental_hio_rescue.sum()),
        "incremental_hio_harms": int(group.incremental_hio_harm.sum()),
        "accepted_lf": int(group.select_lf.sum()),
        "accepted_hio_after_lf": int(group.select_hio_after_lf.sum()),
        "oracle_base_lf_good25": int(group.oracle_base_lf_good25.sum()),
        "oracle_three_arm_good25": int(group.oracle_three_arm_good25.sum()),
        "oracle_incremental_hio_beyond_base_lf": int(group.oracle_three_arm_good25.sum() - group.oracle_base_lf_good25.sum()),
        "mean_base_psnr": float(group.base_psnr.mean()),
        "mean_base_lf_selected_psnr": float(group.base_lf_selected_psnr.mean()),
        "mean_three_arm_selected_psnr": float(group.three_arm_selected_psnr.mean()),
        "mean_three_minus_base_lf_psnr": float((group.three_arm_selected_psnr - group.base_lf_selected_psnr).mean()),
        "median_three_minus_base_lf_psnr": float((group.three_arm_selected_psnr - group.base_lf_selected_psnr).median()),
        "mean_lf_wall_over_base": float(group.lf_wall_seconds.sum() / group.base_wall_seconds.sum()),
        "mean_hio_wall_over_base": float(group.warm_total_wall_seconds.sum() / group.base_wall_seconds.sum()),
        "three_arm_total_cost_over_base": float(
            (group.base_wall_seconds.sum() + group.lf_wall_seconds.sum() + group.warm_total_wall_seconds.sum())
            / group.base_wall_seconds.sum()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hio-rows", type=Path, required=True)
    parser.add_argument("--lf-out", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--tuned-image", default="00046")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    hio_rows = pd.read_csv(args.hio_rows, dtype={"image_id": str})
    hio_rows["image_id"] = hio_rows.image_id.map(lambda x: f"{int(x):05d}")
    rows: list[dict[str, object]] = []

    for rec in hio_rows.itertuples(index=False):
        image = f"{int(rec.image_id):05d}"
        case_id = int(rec.case_id)
        case_dir = args.lf_out / "cases" / f"{image}_case{case_id:02d}"
        lf_loss, lf_psnr = read_metric(case_dir / "metrics" / "lf050.csv")
        lf_wall = read_lf_wall(case_dir / "timings.tsv")

        base_loss = float(rec.base_exact_operator_loss)
        hio_loss = float(rec.warm_exact_operator_loss)
        base_psnr = float(rec.base_psnr)
        hio_psnr = float(rec.warm_psnr)
        theta = float(args.theta)

        select_lf = int(lf_loss < base_loss - theta)
        if select_lf:
            base_lf_variant = "lf050"
            base_lf_loss = lf_loss
            base_lf_psnr = lf_psnr
        else:
            base_lf_variant = "base"
            base_lf_loss = base_loss
            base_lf_psnr = base_psnr

        select_hio_after_lf = int(hio_loss < base_lf_loss - theta)
        if select_hio_after_lf:
            three_variant = "hio_warm"
            three_loss = hio_loss
            three_psnr = hio_psnr
        else:
            three_variant = base_lf_variant
            three_loss = base_lf_loss
            three_psnr = base_lf_psnr

        base_good = int(base_psnr >= 25.0)
        lf_good = int(lf_psnr >= 25.0)
        hio_good = int(hio_psnr >= 25.0)
        base_lf_good = int(base_lf_psnr >= 25.0)
        three_good = int(three_psnr >= 25.0)

        rows.append({
            "job_id": int(rec.job_id),
            "image_id": image,
            "case_id": case_id,
            "base_seed": int(rec.base_seed),
            "hio_seed": int(rec.hio_seed),
            "warm_noise_seed": int(rec.warm_noise_seed),
            "theta": theta,
            "base_exact_operator_loss": base_loss,
            "lf_exact_operator_loss": lf_loss,
            "hio_exact_operator_loss": hio_loss,
            "base_psnr": base_psnr,
            "lf_psnr": lf_psnr,
            "hio_psnr": hio_psnr,
            "base_good25": base_good,
            "lf_good25": lf_good,
            "warm_good25": hio_good,
            "select_lf": select_lf,
            "base_lf_selected_variant": base_lf_variant,
            "base_lf_selected_loss": base_lf_loss,
            "base_lf_selected_psnr": base_lf_psnr,
            "base_lf_good25": base_lf_good,
            "select_hio_after_lf": select_hio_after_lf,
            "three_arm_selected_variant": three_variant,
            "three_arm_selected_loss": three_loss,
            "three_arm_selected_psnr": three_psnr,
            "three_arm_good25": three_good,
            "incremental_hio_rescue": int((not base_lf_good) and three_good),
            "incremental_hio_harm": int(base_lf_good and (not three_good)),
            "oracle_base_lf_good25": int(base_good or lf_good),
            "oracle_three_arm_good25": int(base_good or lf_good or hio_good),
            "base_wall_seconds": float(rec.base_wall_seconds),
            "lf_wall_seconds": lf_wall,
            "warm_total_wall_seconds": float(rec.warm_total_wall_seconds),
            "lf_wall_over_base": lf_wall / float(rec.base_wall_seconds),
            "hio_wall_over_base": float(rec.warm_total_wall_seconds) / float(rec.base_wall_seconds),
        })

    df = pd.DataFrame(rows).sort_values(["image_id", "case_id"]).reset_index(drop=True)
    expected = len(hio_rows)
    complete = len(df)
    tuned = f"{int(args.tuned_image):05d}"

    summaries = [summarize(group, image) for image, group in df.groupby("image_id", sort=True)]
    heldout_df = df[df.image_id != tuned].copy()
    heldout = summarize(heldout_df, "HELDOUT4")
    overall = summarize(df, "ALL")
    summaries.extend([heldout, overall])
    summary_df = pd.DataFrame(summaries)

    support_gate = {
        "complete_40": complete == expected == 40,
        "incremental_net_at_least_2": overall["three_arm_net_vs_base_lf"] >= 2,
        "incremental_harms_at_most_1": overall["incremental_hio_harms"] <= 1,
        "heldout4_net_nonnegative": heldout["three_arm_net_vs_base_lf"] >= 0,
        "marginal_hio_cost_at_most_0_70": overall["mean_hio_wall_over_base"] <= 0.70,
    }
    support = all(support_gate.values())

    verdict = {
        "question": "Does HIO add clean-free value beyond the frozen base+LF050 margin gate?",
        "selection_rule": (
            "first select LF iff loss_lf < loss_base - theta; then select HIO iff "
            "loss_hio < loss_current - theta"
        ),
        "frozen_theta": float(args.theta),
        "expected_cases": expected,
        "complete_cases": complete,
        "tuned_image": tuned,
        "overall": overall,
        "heldout4": heldout,
        "support_gate": support_gate,
        "support_for_fresh_three_arm_validation": support,
        "next_if_supported": (
            "freeze this sequential three-arm gate and run fresh images/seeds; do not retune theta or HIO settings"
        ),
        "next_if_not_supported": (
            "retire HIO as a portfolio arm; retain base+LF only"
        ),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "three_arm_rows.csv"
    summary_path = args.outdir / "three_arm_summary_by_group.csv"
    verdict_path = args.outdir / "three_arm_verdict.json"
    df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    lines = [
        "# B21.5 base + LF050 + HIO three-arm development analysis",
        "",
        "Frozen clean-free sequential policy:",
        "",
        "1. select LF iff `loss_lf < loss_base - 0.7`;",
        "2. select HIO iff `loss_hio < loss_current - 0.7`.",
        "",
        "Ground truth is used only for offline PSNR and good25 diagnostics.",
        "",
        "## Summary",
        "",
        "| group | n | base | LF | HIO | base+LF gated | three-arm gated | HIO net beyond base+LF | HIO rescues | HIO harms | accepted LF | accepted HIO | oracle base+LF | oracle 3-arm | total cost/base |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {group} | {n} | {base_good25} | {lf_good25} | {hio_good25} | "
            "{base_lf_gated_good25} | {three_arm_gated_good25} | {three_arm_net_vs_base_lf:+d} | "
            "{incremental_hio_rescues} | {incremental_hio_harms} | {accepted_lf} | "
            "{accepted_hio_after_lf} | {oracle_base_lf_good25} | {oracle_three_arm_good25} | "
            "{three_arm_total_cost_over_base:.3f} |".format(**item)
        )
    lines += [
        "",
        "## Promotion gate",
        "",
        *[f"- {key}: `{value}`" for key, value in support_gate.items()],
        f"- support for fresh three-arm validation: **{support}**",
        "",
        "## Incremental HIO selections",
        "",
        "| image | case | base+LF variant | base+LF PSNR | HIO PSNR | selected | rescue | harm | current loss | HIO loss |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    selected = df[df.select_hio_after_lf == 1]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| `{row.image_id}` | {row.case_id} | `{row.base_lf_selected_variant}` | "
            f"{row.base_lf_selected_psnr:.3f} | {row.hio_psnr:.3f} | `hio_warm` | "
            f"{row.incremental_hio_rescue} | {row.incremental_hio_harm} | "
            f"{row.base_lf_selected_loss:.3f} | {row.hio_exact_operator_loss:.3f} |"
        )
    lines += [
        "",
        "Artifacts:",
        "",
        "```text",
        str(rows_path),
        str(summary_path),
        str(verdict_path),
        "```",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print("[write]", rows_path)
    print("[write]", summary_path)
    print("[write]", verdict_path)
    print("[write]", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
