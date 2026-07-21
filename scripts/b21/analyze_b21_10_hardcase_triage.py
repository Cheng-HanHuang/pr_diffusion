#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def binary_auc(scores: pd.Series, labels: pd.Series) -> float:
    pos = scores[labels == 1].to_numpy(dtype=float)
    neg = scores[labels == 0].to_numpy(dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += float((value > neg).sum()) + 0.5 * float((value == neg).sum())
    return wins / (len(pos) * len(neg))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.rows, dtype={"image_id": str})
    frame["image_id"] = frame.image_id.map(lambda x: f"{int(x):05d}")
    if len(frame) != 80:
        raise ValueError(f"expected 80 rows, got {len(frame)}")

    required = [
        "arm1_exact_operator_loss", "arm2_exact_operator_loss", "arm3_exact_operator_loss",
        "arm1_psnr", "arm2_psnr", "arm3_psnr",
        "fresh2_selected_loss", "fresh2_selected_psnr", "fresh2_selected_good25",
        "fresh3_selected_loss", "fresh3_selected_psnr", "fresh3_selected_good25",
        "fresh2_selected_variant", "fresh3_selected_variant",
        "fresh2_accepted_new_arm", "fresh3_accepted_new_arm",
        "fresh3_incremental_rescue", "fresh3_incremental_harm",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    numeric = frame[[column for column in required if column not in {
        "fresh2_selected_variant", "fresh3_selected_variant"
    }]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("nonfinite required values")

    frame["fresh2_failure"] = 1 - frame.fresh2_selected_good25.astype(int)
    frame["fresh3_failure"] = 1 - frame.fresh3_selected_good25.astype(int)
    frame["persistent_failure_after3"] = (
        (frame.arm1_psnr < 25.0)
        & (frame.arm2_psnr < 25.0)
        & (frame.arm3_psnr < 25.0)
    ).astype(int)
    frame["fresh2_best_raw_loss"] = frame[[
        "arm1_exact_operator_loss", "arm2_exact_operator_loss"
    ]].min(axis=1)
    frame["fresh2_loss_spread"] = (
        frame.arm1_exact_operator_loss - frame.arm2_exact_operator_loss
    ).abs()
    frame["fresh2_selected_loss_rank_pct"] = frame.fresh2_selected_loss.rank(
        method="average", pct=True
    )
    frame["fresh2_selected_psnr_offline"] = frame.fresh2_selected_psnr

    case_cols = [
        "image_id", "case_id",
        "arm1_exact_operator_loss", "arm2_exact_operator_loss", "arm3_exact_operator_loss",
        "arm1_psnr", "arm2_psnr", "arm3_psnr",
        "fresh2_selected_variant", "fresh2_selected_loss", "fresh2_selected_psnr",
        "fresh2_accepted_new_arm", "fresh2_failure",
        "fresh3_selected_variant", "fresh3_selected_loss", "fresh3_selected_psnr",
        "fresh3_accepted_new_arm", "fresh3_incremental_rescue", "fresh3_incremental_harm",
        "fresh3_failure", "persistent_failure_after3",
        "fresh2_loss_spread", "fresh2_selected_loss_rank_pct",
    ]
    case_rows = frame[case_cols].sort_values(
        ["fresh2_failure", "persistent_failure_after3", "fresh2_selected_loss"],
        ascending=[False, False, False],
    )

    image_rows: list[dict[str, object]] = []
    for image_id, group in frame.groupby("image_id", sort=True):
        image_rows.append({
            "image_id": image_id,
            "n": len(group),
            "fresh1_good25": int((group.arm1_psnr >= 25.0).sum()),
            "fresh2_selected_good25": int(group.fresh2_selected_good25.sum()),
            "fresh3_selected_good25": int(group.fresh3_selected_good25.sum()),
            "fresh2_failures": int(group.fresh2_failure.sum()),
            "fresh3_failures": int(group.fresh3_failure.sum()),
            "persistent_failures_after3": int(group.persistent_failure_after3.sum()),
            "third_restart_rescues": int(group.fresh3_incremental_rescue.sum()),
            "third_restart_accepts": int(group.fresh3_accepted_new_arm.sum()),
            "mean_fresh2_selected_loss": float(group.fresh2_selected_loss.mean()),
            "max_fresh2_selected_loss": float(group.fresh2_selected_loss.max()),
            "median_fresh2_selected_loss": float(group.fresh2_selected_loss.median()),
            "mean_fresh2_loss_spread": float(group.fresh2_loss_spread.mean()),
            "mean_fresh2_selected_psnr_offline": float(group.fresh2_selected_psnr.mean()),
            "min_fresh2_selected_psnr_offline": float(group.fresh2_selected_psnr.min()),
        })
    image_summary = pd.DataFrame(image_rows)
    image_summary["mean_loss_rank"] = image_summary.mean_fresh2_selected_loss.rank(
        method="min", ascending=False
    ).astype(int)
    image_summary["max_loss_rank"] = image_summary.max_fresh2_selected_loss.rank(
        method="min", ascending=False
    ).astype(int)
    image_summary = image_summary.sort_values(
        ["fresh2_failures", "persistent_failures_after3", "mean_fresh2_selected_loss"],
        ascending=[False, False, False],
    )

    thresholds = sorted(set(
        float(value) for value in np.quantile(
            frame.fresh2_selected_loss.to_numpy(dtype=float),
            np.linspace(0.0, 1.0, 21),
        )
    ))
    sweep_rows: list[dict[str, object]] = []
    total_failures = int(frame.fresh2_failure.sum())
    total_persistent = int(frame.persistent_failure_after3.sum())
    total_rescues = int(frame.fresh3_incremental_rescue.sum())
    for threshold in thresholds:
        flagged = frame.fresh2_selected_loss >= threshold
        flagged_n = int(flagged.sum())
        captured_failures = int(frame.loc[flagged, "fresh2_failure"].sum())
        captured_persistent = int(frame.loc[flagged, "persistent_failure_after3"].sum())
        captured_rescues = int(frame.loc[flagged, "fresh3_incremental_rescue"].sum())
        sweep_rows.append({
            "threshold": threshold,
            "flagged_cases": flagged_n,
            "flagged_fraction": flagged_n / len(frame),
            "captured_fresh2_failures": captured_failures,
            "fresh2_failure_recall": captured_failures / total_failures if total_failures else 1.0,
            "fresh2_failure_precision": captured_failures / flagged_n if flagged_n else float("nan"),
            "captured_persistent_failures": captured_persistent,
            "persistent_failure_recall": captured_persistent / total_persistent if total_persistent else 1.0,
            "captured_third_restart_rescues": captured_rescues,
            "third_restart_rescue_recall": captured_rescues / total_rescues if total_rescues else 1.0,
        })
    sweep = pd.DataFrame(sweep_rows)

    top_case_counts = [4, 8, 12, 16, 20]
    ranking = frame.sort_values("fresh2_selected_loss", ascending=False).reset_index(drop=True)
    top_rows = []
    for count in top_case_counts:
        subset = ranking.head(count)
        top_rows.append({
            "top_cases": count,
            "fraction": count / len(frame),
            "fresh2_failures_captured": int(subset.fresh2_failure.sum()),
            "fresh2_failure_recall": float(subset.fresh2_failure.sum() / total_failures) if total_failures else 1.0,
            "persistent_failures_captured": int(subset.persistent_failure_after3.sum()),
            "persistent_failure_recall": float(subset.persistent_failure_after3.sum() / total_persistent) if total_persistent else 1.0,
            "third_restart_rescues_captured": int(subset.fresh3_incremental_rescue.sum()),
        })
    top_capture = pd.DataFrame(top_rows)

    persistent_images = image_summary.loc[
        image_summary.persistent_failures_after3 > 0, "image_id"
    ].tolist()
    fresh2_failure_images = image_summary.loc[
        image_summary.fresh2_failures > 0, "image_id"
    ].tolist()

    verdict = {
        "question": "Can completed Fresh2 clean-free signals identify cases or images needing a non-restart fallback?",
        "status": "diagnostic_only_no_policy_frozen",
        "n_rows": len(frame),
        "independent_images": int(frame.image_id.nunique()),
        "fresh2_failures": total_failures,
        "fresh3_failures": int(frame.fresh3_failure.sum()),
        "third_restart_rescues": total_rescues,
        "third_restart_harms": int(frame.fresh3_incremental_harm.sum()),
        "persistent_failures_after3": total_persistent,
        "fresh2_failure_images": fresh2_failure_images,
        "persistent_hard_images": persistent_images,
        "fresh2_selected_loss_auc_for_case_failure": binary_auc(
            frame.fresh2_selected_loss, frame.fresh2_failure
        ),
        "fresh2_selected_loss_auc_for_persistent_failure": binary_auc(
            frame.fresh2_selected_loss, frame.persistent_failure_after3
        ),
        "highest_mean_loss_images": image_summary.sort_values(
            "mean_fresh2_selected_loss", ascending=False
        ).head(5).image_id.tolist(),
        "highest_max_loss_images": image_summary.sort_values(
            "max_fresh2_selected_loss", ascending=False
        ).head(5).image_id.tolist(),
        "recommended_next_step": (
            "Use this panel only as development. If persistent hard images rank near the top under clean-free loss, "
            "freeze a small targeted complementary-candidate pilot on those images; otherwise develop a better "
            "clean-free hard-case detector before spending GPU budget. Do not run more blind restarts."
        ),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    case_path = args.outdir / "hardcase_case_rows.csv"
    image_path = args.outdir / "hardcase_image_summary.csv"
    sweep_path = args.outdir / "hardcase_loss_threshold_sweep.csv"
    top_path = args.outdir / "hardcase_top_loss_capture.csv"
    verdict_path = args.outdir / "hardcase_triage_verdict.json"
    case_rows.to_csv(case_path, index=False)
    image_summary.to_csv(image_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    top_capture.to_csv(top_path, index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    lines = [
        "# B21.10 zero-GPU hard-case triage",
        "",
        "This audit is diagnostic only. It uses PSNR to label historical outcomes, but any future runtime trigger must use clean-free quantities only.",
        "",
        "## Outcome structure",
        "",
        f"- Fresh2 failures: `{total_failures}/80`",
        f"- Fresh3 failures: `{int(frame.fresh3_failure.sum())}/80`",
        f"- third-restart rescues / harms: `{total_rescues} / {int(frame.fresh3_incremental_harm.sum())}`",
        f"- persistent failures after all three arms: `{total_persistent}`",
        f"- failure images after Fresh2: `{', '.join(fresh2_failure_images)}`",
        f"- persistent hard images: `{', '.join(persistent_images)}`",
        "",
        "## Clean-free loss diagnostics",
        "",
        f"- case-level AUC for Fresh2 failure: `{verdict['fresh2_selected_loss_auc_for_case_failure']}`",
        f"- case-level AUC for persistent failure: `{verdict['fresh2_selected_loss_auc_for_persistent_failure']}`",
        f"- highest mean-loss images: `{', '.join(verdict['highest_mean_loss_images'])}`",
        f"- highest max-loss images: `{', '.join(verdict['highest_max_loss_images'])}`",
        "",
        "## Image summary",
        "",
        "| image | Fresh2 failures | Fresh3 failures | persistent | third rescue | mean loss rank | max loss rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in image_summary.itertuples(index=False):
        lines.append(
            f"| `{row.image_id}` | {row.fresh2_failures} | {row.fresh3_failures} | "
            f"{row.persistent_failures_after3} | {row.third_restart_rescues} | "
            f"{row.mean_loss_rank} | {row.max_loss_rank} |"
        )
    lines += [
        "",
        "No threshold or adaptive policy is selected by this audit. Its role is to decide whether a targeted complementary-candidate experiment is scientifically justified.",
        "",
        f"Artifacts: `{args.outdir}`",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"[write] {case_path}")
    print(f"[write] {image_path}")
    print(f"[write] {sweep_path}")
    print(f"[write] {top_path}")
    print(f"[write] {verdict_path}")
    print(f"[write] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
