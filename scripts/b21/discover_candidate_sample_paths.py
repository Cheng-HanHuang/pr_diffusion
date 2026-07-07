#!/usr/bin/env python3
"""B21.2 prerequisite: discover whether old replay CSVs expose candidate sample paths.

The B21.2 selector-v2 replay needs candidate images so that prior scores can be
computed for both x and rot180(x).  Some old B19/B20 summary CSVs contain only
policy-level rows, while B20 long CSVs may point to detail CSVs that contain the
actual sample path.  This helper audits what is available without launching GPU
jobs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PATH_KEYWORDS = ["sample_path", "recon_path", "png", "jpg", "image_path", "path", "file"]


def split_items(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path, max_rows: Optional[int] = None) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = []
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(dict(row))
        return fields, rows


def is_image_path(text: str) -> bool:
    if not text:
        return False
    p = Path(text)
    return p.suffix.lower() in IMAGE_EXTS


def existing_image_from_row(row: Dict[str, str]) -> Tuple[str, str]:
    for key, value in row.items():
        lk = key.lower()
        if not any(k in lk for k in PATH_KEYWORDS):
            continue
        if value and is_image_path(value) and Path(value).exists():
            return value, key
    for key, value in row.items():
        if value and is_image_path(value) and Path(value).exists():
            return value, key
    return "", ""


def resolve_from_detail_csv(path_text: str, max_rows: int = 100) -> Dict[str, object]:
    if not path_text:
        return {"detail_status": "no_csv_path"}
    path = Path(path_text)
    if not path.exists():
        return {"detail_status": "csv_missing", "detail_csv": path_text}
    try:
        fields, rows = read_csv_rows(path, max_rows=max_rows)
    except Exception as exc:
        return {"detail_status": "csv_read_error", "detail_csv": path_text, "detail_error": repr(exc)}
    candidates = []
    for row in rows:
        sample_path, sample_col = existing_image_from_row(row)
        if sample_path:
            candidates.append((sample_path, sample_col, row))
    out: Dict[str, object] = {
        "detail_status": "ok" if candidates else "no_existing_image_path",
        "detail_csv": path_text,
        "detail_fields": ",".join(fields),
        "detail_rows_scanned": len(rows),
        "detail_sample_candidates": len(candidates),
    }
    if candidates:
        sample_path, sample_col, row = candidates[0]
        out.update(
            {
                "sample_path": sample_path,
                "sample_path_source_col": f"detail:{sample_col}",
                "detail_psnr": row.get("psnr", row.get("selected_psnr", row.get("final_psnr", ""))),
                "detail_exact_operator_loss": row.get("exact_operator_loss", row.get("loss", "")),
                "detail_sqrt_loss_over_y_norm": row.get("sqrt_loss_over_y_norm", ""),
            }
        )
    return out


def expand_csvs(base: Path, globs_text: str) -> List[Path]:
    paths: List[Path] = []
    for pat in split_items(globs_text):
        full = pat if os.path.isabs(pat) else str(base / pat)
        paths.extend(Path(x) for x in glob.glob(full, recursive=True))
    return sorted(set(p for p in paths if p.exists() and p.is_file()))


def row_contains_targets(row: Dict[str, str], targets: set[str]) -> bool:
    text = " ".join(str(v) for v in row.values())
    return any(t in text for t in targets)


def audit_csv(path: Path, targets: set[str], max_detail_rows: int) -> List[Dict[str, object]]:
    try:
        fields, rows = read_csv_rows(path)
    except Exception as exc:
        return [{"source_csv": str(path), "error": repr(exc)}]
    out: List[Dict[str, object]] = []
    path_like_cols = [c for c in fields if any(k in c.lower() for k in PATH_KEYWORDS)]
    for row_idx, row in enumerate(rows):
        if targets and not row_contains_targets(row, targets):
            continue
        sample_path, sample_col = existing_image_from_row(row)
        resolved: Dict[str, object] = {}
        if not sample_path and row.get("csv_path"):
            resolved = resolve_from_detail_csv(row.get("csv_path", ""), max_rows=max_detail_rows)
            sample_path = str(resolved.get("sample_path", ""))
            sample_col = str(resolved.get("sample_path_source_col", ""))
        image_id = row.get("image_id", "")
        rec: Dict[str, object] = {
            "source_csv": str(path),
            "source_row": row_idx,
            "image_id": image_id,
            "dataset": row.get("dataset", ""),
            "meas_seed": row.get("meas_seed", ""),
            "run_seed_or_seed": row.get("run_seed", row.get("seed", "")),
            "policy": row.get("policy", ""),
            "variant": row.get("variant", row.get("arm", "")),
            "selected_run": row.get("selected_run", row.get("run_index", "")),
            "candidate_runs": row.get("candidate_runs", ""),
            "kept_runs": row.get("kept_runs", ""),
            "psnr": row.get("psnr", row.get("selected_psnr", row.get("oracleK_psnr", ""))),
            "good25": row.get("good25", ""),
            "selected_bad25": row.get("selected_bad25", ""),
            "csv_path": row.get("csv_path", ""),
            "path_like_cols_in_source": ",".join(path_like_cols),
            "sample_path_available": bool(sample_path),
            "sample_path": sample_path,
            "sample_path_source_col": sample_col,
        }
        rec.update(resolved)
        out.append(rec)
    return out


def render_report(outdir: Path, rows: Sequence[Dict[str, object]]) -> str:
    total = len(rows)
    available = sum(1 for r in rows if r.get("sample_path_available"))
    by_csv = Counter(Path(str(r.get("source_csv", ""))).name for r in rows)
    by_csv_available = Counter(Path(str(r.get("source_csv", ""))).name for r in rows if r.get("sample_path_available"))
    by_image = Counter(str(r.get("image_id", "")) for r in rows)
    by_image_available = Counter(str(r.get("image_id", "")) for r in rows if r.get("sample_path_available"))

    lines = [
        "# B21.2 candidate sample-path discovery",
        "",
        "Status: generated by `scripts/b21/discover_candidate_sample_paths.py`.",
        "",
        "## Summary",
        "",
        f"- Target rows discovered: `{total}`",
        f"- Rows with usable sample image path: `{available}`",
        f"- Rows without usable sample image path: `{total - available}`",
        "",
        "## Counts by image",
        "",
        "| image_id | rows | sample_path rows |",
        "|---|---:|---:|",
    ]
    for image_id in sorted(by_image):
        lines.append(f"| `{image_id}` | {by_image[image_id]} | {by_image_available[image_id]} |")
    lines.extend(["", "## Counts by source CSV", "", "| source CSV | rows | sample_path rows |", "|---|---:|---:|"])
    for name in sorted(by_csv):
        lines.append(f"| `{name}` | {by_csv[name]} | {by_csv_available[name]} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "B21.2 selector-v2 replay requires candidate images. If old B19.16/B19.20 rows have zero `sample_path` rows here, exact prior-score replay cannot be performed from those CSVs alone. In that case, use this report to decide whether to locate original DAPS result folders or rerun a small replay panel with explicit sample-path logging.",
            "",
            "Artifacts:",
            "",
            "```text",
            str(outdir / "candidate_sample_path_inventory.csv"),
            str(outdir / "candidate_sample_path_summary.json"),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover candidate sample paths for B21.2 selector-v2 replay")
    ap.add_argument("--b19_base", default="/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
    ap.add_argument("--targets", default="00046,00171,00480,00746,00971,00136,00154,00253")
    ap.add_argument(
        "--csv_globs",
        default=(
            "B19_16A*_per_image.csv,B19_16B*_per_image.csv,B19_16D*_per_image.csv,"
            "B19_20*.csv,B20_11_00046_meas5001_lf_guidance_long.csv,"
            "B20_12A_multiimage_heldout4_64seed_3arm_long.csv"
        ),
    )
    ap.add_argument("--max_detail_rows", type=int, default=200)
    ap.add_argument("--outdir", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_2_candidate_path_discovery")
    ap.add_argument("--report_path", default="docs/b21/b21_2_candidate_path_discovery.md")
    args = ap.parse_args()

    base = Path(args.b19_base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    targets = set(split_items(args.targets))
    csv_paths = expand_csvs(base, args.csv_globs)

    all_rows: List[Dict[str, object]] = []
    for path in csv_paths:
        all_rows.extend(audit_csv(path, targets=targets, max_detail_rows=args.max_detail_rows))

    write_csv(outdir / "candidate_sample_path_inventory.csv", all_rows)
    summary = {
        "csv_count": len(csv_paths),
        "row_count": len(all_rows),
        "sample_path_available_count": sum(1 for r in all_rows if r.get("sample_path_available")),
        "targets": sorted(targets),
        "csv_paths": [str(p) for p in csv_paths],
    }
    (outdir / "candidate_sample_path_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(outdir, all_rows)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"[write] {outdir / 'candidate_sample_path_inventory.csv'}")
    print(f"[write] {outdir / 'candidate_sample_path_summary.json'}")
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()
