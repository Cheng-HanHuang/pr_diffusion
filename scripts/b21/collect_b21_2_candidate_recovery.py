#!/usr/bin/env python3
"""Collect B21.2 candidate-recovery rerun outputs.

The launcher calls the existing B20.12A one-image final-only runner with fresh
run seeds.  This collector assembles the per-run exact-final selector CSVs into
a single machine-readable candidate table with sample paths for later selector-v2
prior scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def split_items(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def parse_arms(text: str) -> List[Tuple[str, int, int]]:
    out = []
    for item in split_items(text):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad arm spec {item!r}; expected variant:ann_steps:diff_steps")
        out.append((parts[0], int(parts[1]), int(parts[2])))
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def exact_csv_path(base: Path, image_id: str, variant: str, ann_steps: int, diff_steps: int, meas_seed: int, run_seed: int) -> Path:
    tag = f"ann{ann_steps}_diff{diff_steps}_{variant}"
    return base / f"B20_12A_{image_id}_{tag}_daps1S_meas{meas_seed}_runseed{run_seed}_exact_final_loss_selector.csv"


def sample_dir_path(repo: Path, image_id: str, variant: str, ann_steps: int, diff_steps: int, meas_seed: int, run_seed: int) -> Path:
    tag = f"ann{ann_steps}_diff{diff_steps}_{variant}"
    return repo / "external" / "daps" / "results" / f"b20_12a_{image_id}_{tag}_1S_meas{meas_seed}_runseed{run_seed}" / f"b20_12a_{tag}_{image_id}_meas{meas_seed}_runseed{run_seed}_1S" / "samples"


def maybe_float(x: object) -> float | str:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect B21.2 candidate recovery outputs")
    ap.add_argument("--repo", default="/egr/research-pac/huang248/pr_diffusion_b19_solver")
    ap.add_argument("--b19_base", default="/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
    ap.add_argument("--outdir", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_2_candidate_recovery")
    ap.add_argument("--images", default="00136 00154 00253 00480 00971")
    ap.add_argument("--seeds", default="6700 6701 6702 6703 6704 6705")
    ap.add_argument("--meas_seed", type=int, default=5001)
    ap.add_argument("--arms", default="base:400:5 lf025:350:5 lf050:400:5")
    ap.add_argument("--report_path", default="docs/b21/b21_2_candidate_recovery.md")
    args = ap.parse_args()

    repo = Path(args.repo)
    base = Path(args.b19_base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    images = [f"{int(x):05d}" for x in split_items(args.images)]
    seeds = [int(x) for x in split_items(args.seeds)]
    arms = parse_arms(args.arms)

    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    for image_id in images:
        for seed in seeds:
            for variant, ann, diff in arms:
                exact_csv = exact_csv_path(base, image_id, variant, ann, diff, args.meas_seed, seed)
                sample_dir = sample_dir_path(repo, image_id, variant, ann, diff, args.meas_seed, seed)
                expected = {
                    "image_id": image_id,
                    "meas_seed": args.meas_seed,
                    "run_seed": seed,
                    "variant": variant,
                    "ann_steps": ann,
                    "diff_steps": diff,
                    "exact_csv": str(exact_csv),
                    "sample_dir": str(sample_dir),
                    "sample_dir_exists": sample_dir.exists(),
                }
                if not exact_csv.exists():
                    miss = dict(expected)
                    miss["reason"] = "missing_exact_csv"
                    missing.append(miss)
                    continue
                try:
                    csv_rows = read_csv_rows(exact_csv)
                except Exception as exc:
                    miss = dict(expected)
                    miss["reason"] = f"read_error:{exc!r}"
                    missing.append(miss)
                    continue
                if not csv_rows:
                    miss = dict(expected)
                    miss["reason"] = "empty_exact_csv"
                    missing.append(miss)
                    continue
                for i, r in enumerate(csv_rows):
                    row = dict(expected)
                    row["exact_csv_row"] = i
                    for k, v in r.items():
                        row[f"selector_{k}"] = v
                    # Best-effort canonical columns.
                    sample_path = r.get("sample_path", r.get("path", r.get("recon_path", "")))
                    if not sample_path and sample_dir.exists():
                        # For NUM_RUNS=1 this is normally the candidate path.
                        p = sample_dir / "00000_run0000.png"
                        if p.exists():
                            sample_path = str(p)
                    row["sample_path"] = sample_path
                    row["sample_path_exists"] = bool(sample_path) and Path(sample_path).exists()
                    row["psnr"] = r.get("psnr", r.get("final_psnr", r.get("selected_psnr", "")))
                    row["exact_operator_loss"] = r.get("exact_operator_loss", r.get("loss", ""))
                    row["sqrt_loss_over_y_norm"] = r.get("sqrt_loss_over_y_norm", "")
                    rows.append(row)

    write_csv(outdir / "candidate_recovery_rows.csv", rows)
    write_csv(outdir / "missing_expected.csv", missing)
    by_image: Dict[str, int] = {}
    by_arm: Dict[str, int] = {}
    existing = 0
    for r in rows:
        by_image[str(r["image_id"])] = by_image.get(str(r["image_id"]), 0) + 1
        arm = f"{r['variant']}:ann{r['ann_steps']}:diff{r['diff_steps']}"
        by_arm[arm] = by_arm.get(arm, 0) + 1
        existing += 1 if r.get("sample_path_exists") else 0
    summary = {
        "expected_jobs": len(images) * len(seeds) * len(arms),
        "rows": len(rows),
        "missing_expected": len(missing),
        "sample_path_exists_rows": existing,
        "images": images,
        "seeds": seeds,
        "meas_seed": args.meas_seed,
        "arms": [{"variant": a, "ann_steps": b, "diff_steps": c} for a, b, c in arms],
        "rows_by_image": by_image,
        "rows_by_arm": by_arm,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = [
        "# B21.2 candidate recovery rerun",
        "",
        "Status: generated by `scripts/b21/collect_b21_2_candidate_recovery.py`.",
        "",
        "## Summary",
        "",
        f"- Expected jobs: `{summary['expected_jobs']}`",
        f"- Selector rows found: `{summary['rows']}`",
        f"- Missing expected jobs: `{summary['missing_expected']}`",
        f"- Rows with existing sample path: `{summary['sample_path_exists_rows']}`",
        "",
        "## Artifacts",
        "",
        "```text",
        str(outdir / "candidate_recovery_rows.csv"),
        str(outdir / "missing_expected.csv"),
        str(outdir / "summary.json"),
        "```",
        "",
        "## Next step",
        "",
        "Use `candidate_recovery_rows.csv` as the input table for B21.2 selector-v2 prior/symmetry scoring. This rerun is a small candidate-recovery panel, not a full FFHQ100 validation rerun.",
        "",
    ]
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"[write] {outdir / 'candidate_recovery_rows.csv'}")
    print(f"[write] {outdir / 'missing_expected.csv'}")
    print(f"[write] {outdir / 'summary.json'}")
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()
