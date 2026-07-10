#!/usr/bin/env python3
"""Repair missing PSNR fields in B21.4 matched-pair CSVs.

Some older B20/B21 selector CSVs store PSNR only inside a JSON-like
`psnr_metrics_json` column rather than a numeric `psnr_recomputed_from_png`
column.  The B21.4 collector originally treated that JSON string as nonnumeric,
which can produce NaN PSNRs even though the reconstruction and exact loss are
valid.

This utility rereads the source selector CSVs listed in `base_csv_path` and
`lf_csv_path`, extracts PSNR robustly, and writes a repaired pair table.  It does
not change the clean-free selector quantities; PSNR remains diagnostic only.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd


NUMERIC_PSNR_COLUMNS = [
    "psnr_recomputed_from_png",
    "selector_psnr_recomputed_from_png",
    "selected_psnr_recomputed_from_png",
    "psnr",
    "psnr_x",
    "final_psnr",
    "selected_psnr",
]
JSON_COLUMNS = [
    "psnr_metrics_json",
    "metrics_json",
    "selector_metrics_json",
    "selected_metrics_json",
]


def finite_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def recursive_find_psnr(obj: Any) -> Optional[float]:
    if isinstance(obj, dict):
        preferred = [
            "psnr_recomputed_from_png",
            "selector_psnr_recomputed_from_png",
            "selected_psnr_recomputed_from_png",
            "psnr",
            "psnr_x",
            "final_psnr",
        ]
        for k in preferred:
            if k in obj:
                v = finite_float(obj[k])
                if v is not None:
                    return v
        for k, v0 in obj.items():
            if "psnr" in str(k).lower():
                v = finite_float(v0)
                if v is not None:
                    return v
        for v0 in obj.values():
            v = recursive_find_psnr(v0)
            if v is not None:
                return v
    elif isinstance(obj, (list, tuple)):
        for v0 in obj:
            v = recursive_find_psnr(v0)
            if v is not None:
                return v
    return None


def parse_jsonish_psnr(s: Any) -> Optional[float]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    text = str(s).strip()
    if not text:
        return None
    # Occasionally the field may already just be a number.
    v = finite_float(text)
    if v is not None:
        return v
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(text)
            v = recursive_find_psnr(obj)
            if v is not None:
                return v
        except Exception:
            pass
    # Last-resort regex for strings like "psnr: 31.2" or "'psnr': 31.2".
    m = re.search(r"psnr[^0-9+\-.]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text, re.I)
    if m:
        return finite_float(m.group(1))
    return None


def extract_psnr_from_selector_csv(path: Any) -> Tuple[Optional[float], str]:
    if path is None or (isinstance(path, float) and math.isnan(path)):
        return None, "missing_path"
    p = Path(str(path))
    if not p.exists():
        return None, f"missing_file:{p}"
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return None, f"read_error:{e}"
    if df.empty:
        return None, "empty_csv"
    row = df.iloc[0]
    for col in NUMERIC_PSNR_COLUMNS:
        if col in row.index:
            v = finite_float(row[col])
            if v is not None:
                return v, col
    for col in JSON_COLUMNS:
        if col in row.index:
            v = parse_jsonish_psnr(row[col])
            if v is not None:
                return v, col
    # Some CSVs may have a single JSON-ish column under an unexpected name.
    for col in row.index:
        if "json" in str(col).lower() or "metric" in str(col).lower():
            v = parse_jsonish_psnr(row[col])
            if v is not None:
                return v, col
    return None, "not_found"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pairs", required=True)
    ap.add_argument("--output_pairs", required=True)
    ap.add_argument("--report_json", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.input_pairs)
    report: Dict[str, Any] = {
        "input_pairs": args.input_pairs,
        "output_pairs": args.output_pairs,
        "rows": int(len(df)),
        "base_nan_before": int(pd.to_numeric(df.get("base_psnr"), errors="coerce").isna().sum()),
        "lf_nan_before": int(pd.to_numeric(df.get("lf_psnr"), errors="coerce").isna().sum()),
        "base_repaired": 0,
        "lf_repaired": 0,
        "base_still_nan": 0,
        "lf_still_nan": 0,
        "sources": {},
        "errors": [],
    }

    for side in ["base", "lf"]:
        psnr_col = f"{side}_psnr"
        csv_col = f"{side}_csv_path"
        source_col = f"{side}_psnr_repair_source"
        if source_col not in df.columns:
            df[source_col] = ""
        if psnr_col not in df.columns:
            df[psnr_col] = math.nan
        for idx, row in df.iterrows():
            current = finite_float(row.get(psnr_col, math.nan))
            if current is not None:
                continue
            v, source = extract_psnr_from_selector_csv(row.get(csv_col, ""))
            df.at[idx, source_col] = source
            report["sources"][source] = int(report["sources"].get(source, 0)) + 1
            if v is not None:
                df.at[idx, psnr_col] = v
                report[f"{side}_repaired"] += 1
            else:
                report["errors"].append({
                    "row": int(idx),
                    "side": side,
                    "csv_path": str(row.get(csv_col, "")),
                    "source": source,
                })

    # Keep dependent diagnostic columns in sync when available.
    if "delta_lf_minus_base_psnr" in df.columns:
        df["delta_lf_minus_base_psnr"] = pd.to_numeric(df["lf_psnr"], errors="coerce") - pd.to_numeric(df["base_psnr"], errors="coerce")

    report["base_nan_after"] = int(pd.to_numeric(df.get("base_psnr"), errors="coerce").isna().sum())
    report["lf_nan_after"] = int(pd.to_numeric(df.get("lf_psnr"), errors="coerce").isna().sum())

    out = Path(args.output_pairs)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[write] {out}")
    if args.report_json:
        rp = Path(args.report_json)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[write] {rp}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
