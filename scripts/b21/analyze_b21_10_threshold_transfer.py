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


def load_panel(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"image_id": str})
    frame["image_id"] = frame.image_id.map(lambda x: f"{int(x):05d}")
    required = ["fresh2_selected_loss", "fresh2_selected_good25"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{name}: missing columns {missing}")
    if len(frame) != 80:
        raise ValueError(f"{name}: expected 80 rows, got {len(frame)}")
    frame["fresh2_selected_loss"] = pd.to_numeric(
        frame.fresh2_selected_loss, errors="raise"
    )
    frame["fresh2_selected_good25"] = pd.to_numeric(
        frame.fresh2_selected_good25, errors="raise"
    ).astype(int)
    if not np.isfinite(frame.fresh2_selected_loss.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: nonfinite selected losses")
    frame["fresh2_failure"] = 1 - frame.fresh2_selected_good25
    frame["panel"] = name
    return frame


def threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    flagged = frame.fresh2_selected_loss >= threshold
    failures = frame.fresh2_failure.astype(bool)
    tp = int((flagged & failures).sum())
    fp = int((flagged & ~failures).sum())
    fn = int((~flagged & failures).sum())
    tn = int((~flagged & ~failures).sum())
    return {
        "n": len(frame),
        "failures": int(failures.sum()),
        "flagged": int(flagged.sum()),
        "flagged_fraction": float(flagged.mean()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else 1.0,
        "auc_selected_loss": binary_auc(
            frame.fresh2_selected_loss, frame.fresh2_failure
        ),
    }


def image_summary(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for image_id, group in frame.groupby("image_id", sort=True):
        flagged = group.fresh2_selected_loss >= threshold
        rows.append({
            "panel": str(group.panel.iloc[0]),
            "image_id": image_id,
            "n": len(group),
            "fresh2_failures": int(group.fresh2_failure.sum()),
            "flagged_cases": int(flagged.sum()),
            "captured_failures": int(group.loc[flagged, "fresh2_failure"].sum()),
            "mean_selected_loss": float(group.fresh2_selected_loss.mean()),
            "max_selected_loss": float(group.fresh2_selected_loss.max()),
            "min_selected_loss": float(group.fresh2_selected_loss.min()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-rows", type=Path, required=True)
    parser.add_argument("--validation-rows", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    development = load_panel(args.development_rows, "development")
    validation = load_panel(args.validation_rows, "validation")

    dev_failures = development.loc[
        development.fresh2_failure == 1, "fresh2_selected_loss"
    ]
    if dev_failures.empty:
        raise ValueError("development panel has no Fresh2 failures")

    # Frozen from development only: the largest absolute threshold that preserves
    # 100% recall of development Fresh2 failures.
    threshold = float(dev_failures.min())

    dev_metrics = threshold_metrics(development, threshold)
    val_metrics = threshold_metrics(validation, threshold)

    development = development.copy()
    validation = validation.copy()
    development["flagged_by_transferred_threshold"] = (
        development.fresh2_selected_loss >= threshold
    ).astype(int)
    validation["flagged_by_transferred_threshold"] = (
        validation.fresh2_selected_loss >= threshold
    ).astype(int)

    all_rows = pd.concat([development, validation], ignore_index=True)
    images = pd.concat(
        [image_summary(development, threshold), image_summary(validation, threshold)],
        ignore_index=True,
    )

    val_ranking = validation.sort_values(
        "fresh2_selected_loss", ascending=False
    ).reset_index(drop=True)
    val_ranking["validation_loss_rank"] = np.arange(1, len(val_ranking) + 1)
    val_ranking["validation_loss_rank_pct_desc"] = (
        val_ranking.validation_loss_rank / len(val_ranking)
    )

    result = {
        "question": "Does an absolute Fresh2-loss trigger derived only from the B21.8 development panel transfer to B21.9?",
        "status": "retrospective_transfer_audit_no_runtime_policy_frozen",
        "threshold_rule": "minimum Fresh2 selected loss among development Fresh2 failures",
        "transferred_absolute_threshold": threshold,
        "development": dev_metrics,
        "validation": val_metrics,
        "validation_failure_images": sorted(
            validation.loc[validation.fresh2_failure == 1, "image_id"].unique().tolist()
        ),
        "validation_flagged_images": sorted(
            validation.loc[
                validation.flagged_by_transferred_threshold == 1, "image_id"
            ].unique().tolist()
        ),
        "strong_transfer_signal": bool(
            val_metrics["recall"] >= 0.80
            and val_metrics["precision"] >= 0.50
            and val_metrics["flagged_fraction"] <= 0.25
        ),
        "next_if_strong": (
            "Freeze a small targeted complementary-candidate development pilot on transferred-threshold cases, "
            "with ordinary Fresh3 retained as a matched control. Do not claim an adaptive policy until a new panel validates both trigger and fallback."
        ),
        "next_if_weak": (
            "Do not launch a targeted fallback yet; develop a better clean-free detector using additional measurement-level features."
        ),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    result_path = args.outdir / "threshold_transfer_verdict.json"
    rows_path = args.outdir / "threshold_transfer_rows.csv"
    ranking_path = args.outdir / "validation_loss_ranking.csv"
    images_path = args.outdir / "threshold_transfer_image_summary.csv"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    all_rows.to_csv(rows_path, index=False)
    val_ranking.to_csv(ranking_path, index=False)
    images.to_csv(images_path, index=False)

    lines = [
        "# B21.10 cross-panel hard-case threshold transfer",
        "",
        "The threshold is derived from B21.8 development rows only. B21.9 labels are used only for retrospective transfer evaluation.",
        "",
        f"- transferred threshold: `{threshold}`",
        f"- development failures / flagged: `{dev_metrics['failures']} / {dev_metrics['flagged']}`",
        f"- development recall / precision: `{dev_metrics['recall']} / {dev_metrics['precision']}`",
        f"- validation failures / flagged: `{val_metrics['failures']} / {val_metrics['flagged']}`",
        f"- validation recall / precision: `{val_metrics['recall']} / {val_metrics['precision']}`",
        f"- validation flagged fraction: `{val_metrics['flagged_fraction']}`",
        f"- development / validation AUC: `{dev_metrics['auc_selected_loss']} / {val_metrics['auc_selected_loss']}`",
        f"- strong transfer signal: **{result['strong_transfer_signal']}**",
        "",
        "This remains a retrospective audit. A successful result justifies a targeted fallback pilot; it does not validate an adaptive runtime policy.",
        "",
        f"Artifacts: `{args.outdir}`",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[write] {result_path}")
    print(f"[write] {rows_path}")
    print(f"[write] {ranking_path}")
    print(f"[write] {images_path}")
    print(f"[write] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
