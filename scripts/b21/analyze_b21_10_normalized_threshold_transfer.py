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


def selected_normalized_residual(root: Path, image_id: str, case_id: int, variant: str) -> float:
    metric_path = root / "cases" / f"{image_id}_case{case_id:02d}" / "metrics" / f"{variant}.csv"
    frame = pd.read_csv(metric_path)
    if frame.empty:
        raise ValueError(f"empty metric CSV: {metric_path}")
    if "sqrt_loss_over_y_norm" not in frame.columns:
        raise KeyError(f"missing sqrt_loss_over_y_norm: {metric_path}")
    value = float(pd.to_numeric(frame.iloc[0]["sqrt_loss_over_y_norm"], errors="raise"))
    if not np.isfinite(value):
        raise ValueError(f"nonfinite normalized residual: {metric_path}")
    return value


def load_panel(rows_path: Path, root: Path, panel: str) -> pd.DataFrame:
    frame = pd.read_csv(rows_path, dtype={"image_id": str})
    frame["image_id"] = frame.image_id.map(lambda x: f"{int(x):05d}")
    if len(frame) != 80:
        raise ValueError(f"expected 80 {panel} rows, got {len(frame)}")
    required = ["image_id", "case_id", "fresh2_selected_variant", "fresh2_selected_good25"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"missing {panel} columns: {missing}")
    frame["fresh2_failure"] = 1 - frame.fresh2_selected_good25.astype(int)
    frame["fresh2_selected_normalized_residual"] = [
        selected_normalized_residual(
            root.resolve(),
            f"{int(row.image_id):05d}",
            int(row.case_id),
            str(row.fresh2_selected_variant),
        )
        for row in frame.itertuples(index=False)
    ]
    frame["panel"] = panel
    return frame


def metrics(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    flagged = frame.fresh2_selected_normalized_residual >= threshold
    labels = frame.fresh2_failure.astype(int)
    tp = int((flagged & labels.eq(1)).sum())
    fp = int((flagged & labels.eq(0)).sum())
    fn = int((~flagged & labels.eq(1)).sum())
    tn = int((~flagged & labels.eq(0)).sum())
    flagged_n = int(flagged.sum())
    failures = int(labels.sum())
    return {
        "n": len(frame),
        "failures": failures,
        "flagged": flagged_n,
        "flagged_fraction": flagged_n / len(frame),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "recall": tp / failures if failures else 1.0,
        "precision": tp / flagged_n if flagged_n else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "auc_normalized_residual": binary_auc(frame.fresh2_selected_normalized_residual, labels),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-rows", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--validation-rows", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    development = load_panel(args.development_rows, args.development_root, "development")
    validation = load_panel(args.validation_rows, args.validation_root, "validation")

    dev_failures = development.loc[
        development.fresh2_failure.eq(1), "fresh2_selected_normalized_residual"
    ]
    if dev_failures.empty:
        raise ValueError("development panel has no Fresh2 failures")
    threshold = float(dev_failures.min())

    development["flagged_by_transferred_normalized_threshold"] = (
        development.fresh2_selected_normalized_residual >= threshold
    ).astype(int)
    validation["flagged_by_transferred_normalized_threshold"] = (
        validation.fresh2_selected_normalized_residual >= threshold
    ).astype(int)

    dev_metrics = metrics(development, threshold)
    val_metrics = metrics(validation, threshold)
    strong = bool(
        val_metrics["recall"] >= 0.80
        and val_metrics["precision"] >= 0.50
        and val_metrics["flagged_fraction"] <= 0.25
    )

    combined = pd.concat([development, validation], ignore_index=True)
    validation_ranking = validation.sort_values(
        "fresh2_selected_normalized_residual", ascending=False
    ).reset_index(drop=True)
    validation_ranking["validation_normalized_rank"] = np.arange(1, len(validation_ranking) + 1)
    validation_ranking["validation_normalized_rank_pct_desc"] = (
        validation_ranking.validation_normalized_rank / len(validation_ranking)
    )

    image_rows = []
    for panel, frame in [("development", development), ("validation", validation)]:
        for image_id, group in frame.groupby("image_id", sort=True):
            image_rows.append({
                "panel": panel,
                "image_id": image_id,
                "n": len(group),
                "fresh2_failures": int(group.fresh2_failure.sum()),
                "flagged_cases": int(group.flagged_by_transferred_normalized_threshold.sum()),
                "captured_failures": int(
                    (group.fresh2_failure.eq(1) & group.flagged_by_transferred_normalized_threshold.eq(1)).sum()
                ),
                "mean_selected_normalized_residual": float(group.fresh2_selected_normalized_residual.mean()),
                "max_selected_normalized_residual": float(group.fresh2_selected_normalized_residual.max()),
                "min_selected_normalized_residual": float(group.fresh2_selected_normalized_residual.min()),
            })
    image_summary = pd.DataFrame(image_rows)

    verdict = {
        "question": "Does a scale-normalized Fresh2 residual trigger learned on B21.8 transfer to B21.9?",
        "status": "retrospective_transfer_audit_no_runtime_policy_frozen",
        "threshold_rule": "minimum selected sqrt_loss_over_y_norm among development Fresh2 failures",
        "transferred_normalized_threshold": threshold,
        "development": dev_metrics,
        "validation": val_metrics,
        "strong_transfer_signal": strong,
        "validation_failure_images": sorted(
            validation.loc[validation.fresh2_failure.eq(1), "image_id"].unique().tolist()
        ),
        "validation_flagged_images": sorted(
            validation.loc[
                validation.flagged_by_transferred_normalized_threshold.eq(1), "image_id"
            ].unique().tolist()
        ),
        "next_if_strong": (
            "Freeze a small targeted complementary-candidate development pilot on normalized-threshold cases, "
            "with ordinary Fresh3 as a matched control; validate detector and fallback on another disjoint panel."
        ),
        "next_if_weak": (
            "Do not launch a fallback GPU pilot; exact-loss ranking is informative but no deployable absolute detector transfers."
        ),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    verdict_path = args.outdir / "normalized_threshold_transfer_verdict.json"
    rows_path = args.outdir / "normalized_threshold_transfer_rows.csv"
    ranking_path = args.outdir / "validation_normalized_ranking.csv"
    image_path = args.outdir / "normalized_threshold_image_summary.csv"
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    combined.to_csv(rows_path, index=False)
    validation_ranking.to_csv(ranking_path, index=False)
    image_summary.to_csv(image_path, index=False)

    lines = [
        "# B21.10 normalized-residual threshold transfer",
        "",
        "The threshold is learned from B21.8 development failures only and applied unchanged to B21.9.",
        "",
        f"- transferred normalized threshold: `{threshold}`",
        f"- development failures / flagged: `{dev_metrics['failures']} / {dev_metrics['flagged']}`",
        f"- development recall / precision: `{dev_metrics['recall']} / {dev_metrics['precision']}`",
        f"- validation failures / flagged: `{val_metrics['failures']} / {val_metrics['flagged']}`",
        f"- validation recall / precision: `{val_metrics['recall']} / {val_metrics['precision']}`",
        f"- validation flagged fraction: `{val_metrics['flagged_fraction']}`",
        f"- development / validation AUC: `{dev_metrics['auc_normalized_residual']} / {val_metrics['auc_normalized_residual']}`",
        f"- strong transfer signal: **{strong}**",
        "",
        "This remains retrospective and cannot validate an adaptive runtime policy.",
        "",
        f"Artifacts: `{args.outdir}`",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"[write] {verdict_path}")
    print(f"[write] {rows_path}")
    print(f"[write] {ranking_path}")
    print(f"[write] {image_path}")
    print(f"[write] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
