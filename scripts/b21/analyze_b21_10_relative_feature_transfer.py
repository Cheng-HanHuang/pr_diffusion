#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def binary_auc(scores: pd.Series, labels: pd.Series) -> float:
    pos = scores[labels == 1].to_numpy(dtype=float)
    neg = scores[labels == 0].to_numpy(dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += float((value > neg).sum()) + 0.5 * float((value == neg).sum())
    return wins / (len(pos) * len(neg))


def json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_png01(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def disagreement_features(first: Path, second: Path) -> tuple[float, float]:
    x = load_png01(first)
    y = load_png01(second)
    candidates = [y, np.rot90(y, 2, axes=(0, 1))]
    mse_values = [float(np.mean((x - candidate) ** 2)) for candidate in candidates]
    l1_values = [float(np.mean(np.abs(x - candidate))) for candidate in candidates]
    return float(np.sqrt(min(mse_values))), min(l1_values)


def normalize_panel(rows_path: Path, root: Path, panel: str) -> pd.DataFrame:
    frame = pd.read_csv(rows_path, dtype={"image_id": str})
    frame["image_id"] = frame.image_id.map(lambda x: f"{int(x):05d}")
    if len(frame) != 80:
        raise ValueError(f"expected 80 {panel} rows, got {len(frame)}")

    if panel == "development":
        required = [
            "image_id", "case_id", "base_exact_operator_loss",
            "extra1_exact_operator_loss", "fresh2_selected_good25",
            "fresh2_selected_variant",
        ]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"missing development columns: {missing}")
        frame["loss1"] = pd.to_numeric(frame.base_exact_operator_loss, errors="raise")
        frame["loss2"] = pd.to_numeric(frame.extra1_exact_operator_loss, errors="raise")
    else:
        required = [
            "image_id", "case_id", "arm1_exact_operator_loss",
            "arm2_exact_operator_loss", "fresh2_selected_good25",
            "fresh2_selected_variant",
        ]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"missing validation columns: {missing}")
        frame["loss1"] = pd.to_numeric(frame.arm1_exact_operator_loss, errors="raise")
        frame["loss2"] = pd.to_numeric(frame.arm2_exact_operator_loss, errors="raise")

    if not np.isfinite(frame[["loss1", "loss2"]].to_numpy(dtype=float)).all():
        raise ValueError(f"nonfinite losses in {panel}")

    frame["fresh2_failure"] = 1 - frame.fresh2_selected_good25.astype(int)
    frame["selected_raw_loss"] = np.where(
        frame.fresh2_selected_variant.astype(str).eq("base_extra"),
        frame.loss2,
        frame.loss1,
    )
    frame["loss_abs_gap"] = (frame.loss1 - frame.loss2).abs()
    denominator = np.minimum(frame.loss1.abs(), frame.loss2.abs()).clip(lower=1e-12)
    frame["loss_relative_gap"] = frame.loss_abs_gap / denominator
    frame["loss_ratio"] = np.maximum(frame.loss1.abs(), frame.loss2.abs()) / denominator
    frame["second_arm_accepted"] = frame.fresh2_selected_variant.astype(str).eq("base_extra").astype(int)

    rmse_values: list[float] = []
    l1_values: list[float] = []
    for row in frame.itertuples(index=False):
        case_dir = root.resolve() / "cases" / f"{row.image_id}_case{int(row.case_id):02d}"
        first = case_dir / "daps_results" / "base_full" / "samples" / "00000_run0000.png"
        second = case_dir / "daps_results" / "base_extra" / "samples" / "00000_run0000.png"
        rmse, l1 = disagreement_features(first, second)
        rmse_values.append(rmse)
        l1_values.append(l1)
    frame["candidate_disagreement_rmse_rot180"] = rmse_values
    frame["candidate_disagreement_l1_rot180"] = l1_values
    frame["panel"] = panel
    return frame


def threshold_metrics(frame: pd.DataFrame, feature: str, threshold: float) -> dict[str, object]:
    flagged = frame[feature] >= threshold
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
        "auc": binary_auc(frame[feature], labels),
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

    development = normalize_panel(args.development_rows, args.development_root, "development")
    validation = normalize_panel(args.validation_rows, args.validation_root, "validation")

    features = [
        "selected_raw_loss",
        "loss_abs_gap",
        "loss_relative_gap",
        "loss_ratio",
        "candidate_disagreement_rmse_rot180",
        "candidate_disagreement_l1_rot180",
    ]

    rows: list[dict[str, object]] = []
    for feature in features:
        dev_failures = development.loc[development.fresh2_failure.eq(1), feature]
        if dev_failures.empty:
            raise ValueError("development panel has no Fresh2 failures")
        threshold = float(dev_failures.min())
        dev = threshold_metrics(development, feature, threshold)
        val = threshold_metrics(validation, feature, threshold)
        rows.append({
            "feature": feature,
            "threshold_rule": "minimum development-failure value; higher flags harder",
            "transferred_threshold": threshold,
            **{f"development_{key}": value for key, value in dev.items()},
            **{f"validation_{key}": value for key, value in val.items()},
            "strong_validation_signal": bool(
                val["recall"] >= 0.80
                and val["precision"] >= 0.50
                and val["flagged_fraction"] <= 0.25
            ),
        })

    feature_table = pd.DataFrame(rows)
    relative_features = feature_table.loc[
        feature_table.feature.ne("selected_raw_loss")
    ].copy()
    relative_features = relative_features.sort_values(
        ["development_specificity", "development_auc"], ascending=[False, False]
    )
    best_feature = str(relative_features.iloc[0].feature)
    best_series = feature_table.loc[feature_table.feature.eq(best_feature)].iloc[0]
    best_row = {str(key): json_scalar(value) for key, value in best_series.items()}

    combined = pd.concat([development, validation], ignore_index=True)
    validation_ranking = validation.sort_values(
        best_feature, ascending=False
    ).reset_index(drop=True)
    validation_ranking["validation_rank"] = np.arange(1, len(validation_ranking) + 1)

    verdict = {
        "question": "Do within-measurement two-run features transfer as a clean-free Fresh2 hard-case detector?",
        "status": "retrospective_feature_screen_no_runtime_policy_frozen",
        "feature_selection_rule": "among relative features only, maximize development specificity at 100% development-failure recall; break ties by development AUC",
        "best_development_relative_feature": best_feature,
        "best_feature_result": best_row,
        "any_relative_feature_strong_on_validation": bool(
            relative_features.strong_validation_signal.any()
        ),
        "validation_failure_images": sorted(
            validation.loc[validation.fresh2_failure.eq(1), "image_id"].unique().tolist()
        ),
        "next_if_strong": (
            "Only then freeze a small targeted complementary-candidate development pilot, with ordinary Fresh3 as a matched control. "
            "Detector and fallback still require a new disjoint validation panel."
        ),
        "next_if_weak": (
            "Stop detector threshold mining on these panels. Retain Fresh2 as the fixed default and defer adaptive fallback until a genuinely new clean-free feature or model is proposed."
        ),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    feature_path = args.outdir / "relative_feature_transfer_table.csv"
    rows_path = args.outdir / "relative_feature_transfer_rows.csv"
    ranking_path = args.outdir / "validation_best_relative_feature_ranking.csv"
    verdict_path = args.outdir / "relative_feature_transfer_verdict.json"
    feature_table.to_csv(feature_path, index=False)
    combined.to_csv(rows_path, index=False)
    validation_ranking.to_csv(ranking_path, index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    lines = [
        "# B21.10 within-measurement relative-feature transfer",
        "",
        "This audit screens only features available after the two Fresh2 trajectories. Candidate disagreement is the minimum image-space discrepancy over identity and 180-degree rotation.",
        "",
        f"- best development-selected relative feature: `{best_feature}`",
        f"- any relative feature meeting the validation support rule: **{verdict['any_relative_feature_strong_on_validation']}**",
        "",
        "## Feature table",
        "",
        "| feature | dev AUC | dev flagged | val AUC | val recall | val precision | val flagged fraction | strong |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in feature_table.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.development_auc:.4f} | {int(row.development_flagged)}/80 | "
            f"{row.validation_auc:.4f} | {row.validation_recall:.4f} | "
            f"{row.validation_precision:.4f} | {row.validation_flagged_fraction:.4f} | "
            f"{bool(row.strong_validation_signal)} |"
        )
    lines += [
        "",
        "This is a retrospective feature screen, not a validated adaptive policy. No GPU fallback is authorized unless a relative feature meets the frozen validation support rule.",
        "",
        f"Artifacts: `{args.outdir}`",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"[write] {feature_path}")
    print(f"[write] {rows_path}")
    print(f"[write] {ranking_path}")
    print(f"[write] {verdict_path}")
    print(f"[write] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
