#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--fresh-out", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs, sep="\t", dtype={"image_id": str})
    if len(pairs) != 8:
        raise ValueError(f"expected 8 calibration pairs, found {len(pairs)}")
    required = {
        "job_id", "image_id", "case_id", "gpu", "order",
        "seed", "base_seconds", "lf_seconds",
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not (pairs[["base_seconds", "lf_seconds"]] > 0).all().all():
        raise ValueError("nonpositive timing found")

    pairs["lf_over_base"] = pairs.lf_seconds / pairs.base_seconds
    pairs["base_over_lf"] = pairs.base_seconds / pairs.lf_seconds

    historical_rows = []
    for rec in pairs.itertuples(index=False):
        timing_path = (
            args.fresh_out
            / "cases"
            / f"{rec.image_id}_case{int(rec.case_id):02d}"
            / "timings.tsv"
        )
        timings: dict[str, float] = {}
        if timing_path.exists():
            for line in timing_path.read_text().splitlines():
                if not line.strip():
                    continue
                key, value = line.split("\t", 1)
                timings[key] = float(value)
        base = timings.get("base_full")
        lf = timings.get("lf050")
        extra = timings.get("base_extra")
        historical_rows.append({
            "job_id": int(rec.job_id),
            "image_id": rec.image_id,
            "case_id": int(rec.case_id),
            "historical_base_seconds": base,
            "historical_lf_seconds": lf,
            "historical_extra_seconds": extra,
            "historical_lf_over_base": lf / base if base and lf else float("nan"),
            "historical_extra_over_base": extra / base if base and extra else float("nan"),
        })
    historical = pd.DataFrame(historical_rows)

    mean_ratio = float(pairs.lf_over_base.mean())
    median_ratio = float(pairs.lf_over_base.median())
    min_ratio = float(pairs.lf_over_base.min())
    max_ratio = float(pairs.lf_over_base.max())
    base_first = pairs.loc[pairs.order == "base_first", "lf_over_base"]
    lf_first = pairs.loc[pairs.order == "lf_first", "lf_over_base"]

    paired_runtime_equivalent = bool(
        0.80 <= mean_ratio <= 1.25
        and 0.85 <= median_ratio <= 1.20
    )
    historical_extra_ratio = float(historical.historical_extra_over_base.mean())
    historical_lf_ratio = float(historical.historical_lf_over_base.mean())
    cross_run_anomaly = bool(
        paired_runtime_equivalent
        and historical_extra_ratio >= 1.50
    )

    verdict = {
        "n_pairs": len(pairs),
        "paired_mean_lf_over_base": mean_ratio,
        "paired_median_lf_over_base": median_ratio,
        "paired_min_lf_over_base": min_ratio,
        "paired_max_lf_over_base": max_ratio,
        "base_first_mean_lf_over_base": float(base_first.mean()),
        "lf_first_mean_lf_over_base": float(lf_first.mean()),
        "historical_mean_lf_over_base": historical_lf_ratio,
        "historical_mean_extra_over_base": historical_extra_ratio,
        "paired_runtime_equivalent": paired_runtime_equivalent,
        "cross_run_base_extra_slowdown_is_timing_anomaly": cross_run_anomaly,
        "quality_result": "Fresh2 70/80 versus Base+LF 63/80; fixed-work Fresh2 win",
        "interpretation_if_equivalent": (
            "Treat full base and LF arms as approximately equal wall-cost under matched conditions; "
            "retain Fresh2 as preferred second arm and discard historical cross-run wall ratios."
        ),
        "interpretation_if_not_equivalent": (
            "Do not claim equal wall-clock cost; inspect GPU contention, clocks, and launch environment "
            "before the next budget-policy experiment."
        ),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.outdir / "runtime_calibration_pairs.csv"
    hist_path = args.outdir / "runtime_calibration_historical.csv"
    verdict_path = args.outdir / "runtime_calibration_verdict.json"
    pairs.to_csv(pairs_path, index=False)
    historical.to_csv(hist_path, index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    lines = [
        "# B21.7 interleaved runtime calibration",
        "",
        f"- pairs: `{len(pairs)}`",
        f"- paired mean LF/base: `{mean_ratio:.6f}`",
        f"- paired median LF/base: `{median_ratio:.6f}`",
        f"- paired range: `[{min_ratio:.6f}, {max_ratio:.6f}]`",
        f"- base-first mean LF/base: `{float(base_first.mean()):.6f}`",
        f"- LF-first mean LF/base: `{float(lf_first.mean()):.6f}`",
        f"- historical mean LF/base: `{historical_lf_ratio:.6f}`",
        f"- historical mean base-extra/base: `{historical_extra_ratio:.6f}`",
        f"- paired runtime equivalent: **{paired_runtime_equivalent}**",
        f"- cross-run slowdown classified as timing anomaly: **{cross_run_anomaly}**",
        "",
        "## Pairs",
        "",
        "| job | image | case | gpu | order | base sec | LF sec | LF/base |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in pairs.itertuples(index=False):
        lines.append(
            f"| {int(row.job_id)} | `{row.image_id}` | {int(row.case_id)} | {row.gpu} | "
            f"{row.order} | {row.base_seconds:.1f} | {row.lf_seconds:.1f} | {row.lf_over_base:.4f} |"
        )
    lines += ["", f"Artifacts: `{args.outdir}`", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"[write] {pairs_path}")
    print(f"[write] {hist_path}")
    print(f"[write] {verdict_path}")
    print(f"[write] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
