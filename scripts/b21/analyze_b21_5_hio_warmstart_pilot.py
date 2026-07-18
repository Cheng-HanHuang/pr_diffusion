#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_metric(path: Path) -> tuple[float, float]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"empty metric CSV: {path}")
    row = df.iloc[0]
    loss = float(pd.to_numeric(row.get("exact_operator_loss"), errors="raise"))
    psnr = float(pd.to_numeric(row.get("psnr_recomputed_from_png"), errors="raise"))
    return loss, psnr


def read_timings(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name, value = line.split("\t", 1)
        out[name] = float(value)
    return out


def exact_mcnemar_p(branch_only: int, base_only: int) -> float:
    n = branch_only + base_only
    if n == 0:
        return 1.0
    k = min(branch_only, base_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def sign_test_p(differences: Iterable[float]) -> float:
    values = [x for x in differences if abs(x) > 1e-12]
    if not values:
        return 1.0
    positives = sum(x > 0 for x in values)
    negatives = len(values) - positives
    k = min(positives, negatives)
    tail = sum(math.comb(len(values), i) for i in range(k + 1)) / (2 ** len(values))
    return min(1.0, 2.0 * tail)


def summarize(group: pd.DataFrame, name: str) -> dict[str, object]:
    n = len(group)
    base_good = int(group["base_good25"].sum())
    warm_good = int(group["warm_good25"].sum())
    warm_only = int(group["warm_only_win"].sum())
    base_only = int(group["base_only_win"].sum())
    both_good = int(((group.base_good25 == 1) & (group.warm_good25 == 1)).sum())
    both_bad = int(((group.base_good25 == 0) & (group.warm_good25 == 0)).sum())
    return {
        "group": name,
        "n_cases": n,
        "base_good25": base_good,
        "warm_good25": warm_good,
        "base_good_rate": base_good / n if n else float("nan"),
        "warm_good_rate": warm_good / n if n else float("nan"),
        "warm_minus_base_good_cases": warm_good - base_good,
        "warm_minus_base_good_rate": (warm_good - base_good) / n if n else float("nan"),
        "warm_only_wins": warm_only,
        "base_only_wins": base_only,
        "both_good": both_good,
        "both_bad": both_bad,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(warm_only, base_only),
        "mean_base_psnr": float(group.base_psnr.mean()),
        "mean_warm_psnr": float(group.warm_psnr.mean()),
        "mean_warm_minus_base_psnr": float(group.warm_minus_base_psnr.mean()),
        "median_warm_minus_base_psnr": float(group.warm_minus_base_psnr.median()),
        "psnr_wins": int((group.warm_minus_base_psnr > 0).sum()),
        "psnr_losses": int((group.warm_minus_base_psnr < 0).sum()),
        "psnr_ties": int((group.warm_minus_base_psnr == 0).sum()),
        "psnr_sign_test_two_sided_p": sign_test_p(group.warm_minus_base_psnr.tolist()),
        "mean_base_wall_seconds": float(group.base_wall_seconds.mean()),
        "mean_warm_total_wall_seconds": float(group.warm_total_wall_seconds.mean()),
        "warm_over_base_wall_ratio": float(group.warm_total_wall_seconds.sum() / group.base_wall_seconds.sum()),
        "mean_hio_raw_psnr": float(group.hio_raw_psnr.mean()),
        "mean_hio_relative_residual": float(group.hio_sqrt_loss_over_y_norm.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tuned-image", default="00046")
    parser.add_argument("--inject-step", type=int, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    out = args.out.resolve()
    manifest = pd.read_csv(args.manifest, sep="\t", dtype={"image_id": str})
    manifest["image_id"] = manifest.image_id.map(lambda x: f"{int(x):05d}")

    rows: list[dict[str, object]] = []
    for rec in manifest.itertuples(index=False):
        case_dir = out / "cases" / f"{rec.image_id}_case{int(rec.case_id):02d}"
        base_loss, base_psnr = read_metric(case_dir / "metrics" / "base_full.csv")
        warm_loss, warm_psnr = read_metric(case_dir / "metrics" / "hio_warm.csv")
        hio = json.loads((case_dir / "hio" / "hio_summary.json").read_text())
        timing = read_timings(case_dir / "timings.tsv")
        base_wall = timing.get("base_full", float("nan"))
        hio_wall = timing.get("hio_generate", float("nan"))
        warm_wall = timing.get("hio_warm", float("nan"))
        warm_total = hio_wall + warm_wall
        base_good = int(base_psnr >= 25.0)
        warm_good = int(warm_psnr >= 25.0)
        state_valid = bool(
            hio.get("warm_state_finite")
            and hio.get("inject_step") == args.inject_step
            and hio.get("warm_state_shape") == [1, 3, 256, 256]
            and float(hio.get("warm_state_min")) >= -1.00001
            and float(hio.get("warm_state_max")) <= 1.00001
        )
        rows.append({
            "job_id": int(rec.job_id),
            "image_id": rec.image_id,
            "case_id": int(rec.case_id),
            "gpu": str(rec.gpu),
            "base_seed": int(rec.base_seed),
            "hio_seed": int(rec.hio_seed),
            "warm_noise_seed": int(rec.warm_noise_seed),
            "base_exact_operator_loss": base_loss,
            "warm_exact_operator_loss": warm_loss,
            "base_psnr": base_psnr,
            "hio_raw_psnr": float("nan"),
            "warm_psnr": warm_psnr,
            "base_good25": base_good,
            "warm_good25": warm_good,
            "warm_only_win": int(warm_good and not base_good),
            "base_only_win": int(base_good and not warm_good),
            "warm_minus_base_psnr": warm_psnr - base_psnr,
            "warm_exact_loss_minus_base": warm_loss - base_loss,
            "state_valid": int(state_valid),
            "base_wall_seconds": base_wall,
            "hio_wall_seconds": hio_wall,
            "warm_daps_wall_seconds": warm_wall,
            "warm_total_wall_seconds": warm_total,
            "warm_over_base_wall_ratio": warm_total / base_wall,
            "hio_measurement_squared_loss": float(hio["hio_measurement_squared_loss"]),
            "hio_sqrt_loss_over_y_norm": float(hio["hio_sqrt_loss_over_y_norm"]),
            "hio_state_sha256": hio["warm_state_sha256"],
        })

    df = pd.DataFrame(rows).sort_values(["image_id", "case_id"]).reset_index(drop=True)

    # Raw-HIO PSNR is diagnostic only; compute from saved checker values when available.
    # The generator remains clean-free and never reads ground truth.
    try:
        from PIL import Image
        import numpy as np
        def direct_psnr(sample: Path, gt: Path) -> float:
            a_img = Image.open(sample).convert("RGB")
            b_img = Image.open(gt).convert("RGB").resize(a_img.size, Image.Resampling.LANCZOS)
            a = np.asarray(a_img, dtype=np.float32)
            b = np.asarray(b_img, dtype=np.float32)
            mse = float(np.mean((a - b) ** 2))
            return float("inf") if mse <= 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)
        for idx, row in df.iterrows():
            sample = out / "cases" / f"{row.image_id}_case{int(row.case_id):02d}" / "hio" / "hio_raw.png"
            gt = Path("/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024/00000") / f"{row.image_id}.png"
            df.loc[idx, "hio_raw_psnr"] = direct_psnr(sample, gt)
    except Exception as exc:
        print(f"[warning] raw-HIO PSNR diagnostic unavailable: {exc}")

    summaries = [summarize(g, image) for image, g in df.groupby("image_id", sort=True)]
    all_summary = summarize(df, "ALL")
    heldout = df[df.image_id != f"{int(args.tuned_image):05d}"].copy()
    heldout_summary = summarize(heldout, "HELDOUT4")
    summary_df = pd.DataFrame(summaries + [heldout_summary, all_summary])

    complete_gate = len(df) == len(manifest) and bool(df.state_valid.all())
    cost_gate = bool(all_summary["warm_over_base_wall_ratio"] <= 0.70)
    registered_quality_gate = bool(all_summary["warm_minus_base_good_rate"] >= 0.10)
    heldout_nonnegative_gate = bool(heldout_summary["warm_minus_base_good_cases"] >= 0)
    promote = complete_gate and cost_gate and registered_quality_gate and heldout_nonnegative_gate

    verdict = {
        "expected_cases": int(len(manifest)),
        "complete_cases": int(len(df)),
        "inject_step": args.inject_step,
        "tuned_image": f"{int(args.tuned_image):05d}",
        "complete_and_state_gate": complete_gate,
        "cost_gate": cost_gate,
        "cost_gate_ratio_at_most": 0.70,
        "registered_quality_gate": registered_quality_gate,
        "registered_quality_requires_pooled_good_rate_gain_at_least": 0.10,
        "heldout_nonnegative_gate": heldout_nonnegative_gate,
        "promote_to_broader_validation": promote,
        "overall": all_summary,
        "heldout4": heldout_summary,
        "by_image": {str(row["group"]): row for row in summaries},
    }

    case_path = out / "hio_warmstart_pilot_rows.csv"
    summary_path = out / "hio_warmstart_summary_by_image.csv"
    verdict_path = out / "hio_warmstart_pilot_verdict.json"
    report_path = repo / "docs/b21/b21_5_hio_warmstart_pilot.md"
    df.to_csv(case_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    lines = [
        "# B21.5 HIO warm-start five-image pilot",
        "",
        f"- cases: `{len(df)}`",
        f"- injection step: `{args.inject_step}`",
        f"- tuned/smoke image: `{verdict['tuned_image']}`",
        f"- promote to broader validation: **{promote}**",
        "",
        "## Any-good results",
        "",
        "| group | n | base good | warm good | net | warm-only | base-only | McNemar p | PSNR mean delta | PSNR median delta | cost ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries + [heldout_summary, all_summary]:
        lines.append(
            "| {group} | {n_cases} | {base_good25} | {warm_good25} | {warm_minus_base_good_cases:+d} | "
            "{warm_only_wins} | {base_only_wins} | {mcnemar_exact_two_sided_p:.4g} | "
            "{mean_warm_minus_base_psnr:+.3f} | {median_warm_minus_base_psnr:+.3f} | "
            "{warm_over_base_wall_ratio:.3f} |".format(**row)
        )
    lines += [
        "",
        "## Frozen gates",
        "",
        f"- complete, valid states: `{complete_gate}`",
        f"- pooled warm/base wall ratio <= 0.70: `{cost_gate}`",
        f"- pooled good25 gain >= 10 percentage points: `{registered_quality_gate}`",
        f"- held-out four-image net good25 nonnegative: `{heldout_nonnegative_gate}`",
        "",
        "The HIO generator uses only the locked measurement and known support. Ground truth is used only in this offline analyzer.",
        "",
        f"Artifacts: `{out}`",
    ]
    report_path.write_text("\n".join(lines) + "\n")

    print("[write]", case_path)
    print("[write]", summary_path)
    print("[write]", verdict_path)
    print("[write]", report_path)
    print(summary_df.to_string(index=False))
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
