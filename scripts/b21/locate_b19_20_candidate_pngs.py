#!/usr/bin/env python3
"""Locate B19.20 candidate PNGs outside replay CSVs.

B21.2 selector-v2 needs actual candidate images.  The B19.20 replay CSVs record
policy rows but no sample paths, so this helper searches likely DAPS/result roots
for PNG/JPG files whose path encodes target image IDs and optionally run seeds or
measurement seeds.  It is intentionally read-only and no-GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def split_items(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def walk_images(root: Path, max_files: int = 0) -> Iterable[Path]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".aider.tags.cache.v4"}]
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                yield Path(dirpath) / name
                count += 1
                if max_files > 0 and count >= max_files:
                    return


def infer_image_id(path: str, targets: set[str]) -> str:
    for t in sorted(targets):
        if t in path:
            return t
    return ""


def infer_int_after(patterns: Sequence[str], text: str) -> str:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""


def is_sample_png(path: str) -> bool:
    p = Path(path)
    parts = set(p.parts)
    if p.name.startswith("grid_results"):
        return False
    if "trajectory" in parts:
        return False
    return "samples" in parts


def token_match(path: str, tokens: Sequence[str], case_insensitive: bool = True) -> bool:
    if not tokens:
        return True
    hay = path.lower() if case_insensitive else path
    for tok in tokens:
        needle = tok.lower() if case_insensitive else tok
        if needle and needle in hay:
            return True
    return False


def looks_b19_20(path: str) -> bool:
    # runseed4400 alone is not enough: several B19.16/B20 outputs use that seed.
    return token_match(path, ["b19_20", "ffhq100"], case_insensitive=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Locate candidate PNG/JPG paths for B19.20/B21.2")
    ap.add_argument("--targets", default="00046,00136,00154,00171,00253,00480,00746,00971")
    ap.add_argument(
        "--roots",
        default=(
            "/egr/research-pac/huang248/pr_diffusion_b19_solver/external/daps/results,"
            "/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver"
        ),
    )
    ap.add_argument(
        "--require_any",
        default="B19_20,b19_20,ffhq100",
        help="Comma-separated path tokens. Default is strict B19.20/FFHQ100 search. Use an empty string to disable.",
    )
    ap.add_argument("--sample_only", action="store_true", help="Keep only files under samples/ and drop grid_results/trajectory PNGs.")
    ap.add_argument("--maybe_b19_20_only", action="store_true", help="Keep only paths that look like B19.20/FFHQ100 outputs.")
    ap.add_argument("--max_files_per_root", type=int, default=0)
    ap.add_argument("--outdir", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_2_b19_20_candidate_png_locator")
    ap.add_argument("--report_path", default="docs/b21/b21_2_b19_20_candidate_png_locator.md")
    args = ap.parse_args()

    targets = set(split_items(args.targets))
    roots = [Path(x) for x in split_items(args.roots)]
    require_any = split_items(args.require_any)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    root_stats: List[Dict[str, object]] = []
    rejected = Counter()
    for root in roots:
        if not root.exists():
            root_stats.append({"root": str(root), "exists": False, "scanned_images": 0, "matches": 0})
            continue
        scanned = 0
        matched = 0
        for p in walk_images(root, max_files=args.max_files_per_root):
            scanned += 1
            s = str(p)
            image_id = infer_image_id(s, targets)
            if not image_id:
                rejected["non_target_image"] += 1
                continue
            if require_any and not token_match(s, require_any, case_insensitive=True):
                rejected["missing_required_token"] += 1
                continue
            maybe_b19 = looks_b19_20(s)
            if args.maybe_b19_20_only and not maybe_b19:
                rejected["not_maybe_b19_20"] += 1
                continue
            if args.sample_only and not is_sample_png(s):
                rejected["not_sample_png"] += 1
                continue
            matched += 1
            rows.append(
                {
                    "path": s,
                    "root": str(root),
                    "image_id": image_id,
                    "meas_seed": infer_int_after([r"meas(\d{4,})", r"meas_seed[_-]?(\d{4,})"], s),
                    "run_seed": infer_int_after([r"runseed(\d{4,})", r"seed(\d{4,})"], s),
                    "run_index": infer_int_after([r"run(\d{4})", r"run[_-]?(\d+)"], s),
                    "is_sample_png": is_sample_png(s),
                    "maybe_b19_20": maybe_b19,
                }
            )
        root_stats.append({"root": str(root), "exists": True, "scanned_images": scanned, "matches": matched})

    write_csv(outdir / "candidate_png_matches.csv", rows)
    by_image = Counter(str(r["image_id"]) for r in rows)
    by_root = Counter(str(r["root"]) for r in rows)
    by_maybe = Counter(str(r["maybe_b19_20"]) for r in rows)
    sample_count = sum(1 for r in rows if r.get("is_sample_png"))
    summary = {
        "total_matches": len(rows),
        "sample_png_matches": sample_count,
        "maybe_b19_20_counts": dict(sorted(by_maybe.items())),
        "targets": sorted(targets),
        "require_any": require_any,
        "sample_only": bool(args.sample_only),
        "maybe_b19_20_only": bool(args.maybe_b19_20_only),
        "root_stats": root_stats,
        "matches_by_image": dict(sorted(by_image.items())),
        "matches_by_root": dict(sorted(by_root.items())),
        "rejected_counts": dict(sorted(rejected.items())),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# B21.2 B19.20 candidate PNG locator",
        "",
        "Status: generated by `scripts/b21/locate_b19_20_candidate_pngs.py`.",
        "",
        f"Total matches: `{len(rows)}`",
        f"Sample PNG matches: `{sample_count}`",
        f"maybe_b19_20 counts: `{dict(sorted(by_maybe.items()))}`",
        f"require_any: `{require_any}`",
        f"sample_only: `{bool(args.sample_only)}`",
        f"maybe_b19_20_only: `{bool(args.maybe_b19_20_only)}`",
        "",
        "## Root stats",
        "",
        "| root | exists | scanned images | matches |",
        "|---|---|---:|---:|",
    ]
    for r in root_stats:
        lines.append(f"| `{r['root']}` | {r['exists']} | {r['scanned_images']} | {r['matches']} |")
    lines.extend(["", "## Matches by image", "", "| image_id | matches |", "|---|---:|"])
    for image_id, count in sorted(by_image.items()):
        lines.append(f"| `{image_id}` | {count} |")
    lines.extend(["", "Artifacts:", "", "```text", str(outdir / "candidate_png_matches.csv"), str(outdir / "summary.json"), "```", ""])
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[write] {outdir / 'candidate_png_matches.csv'}")
    print(f"[write] {outdir / 'summary.json'}")
    print(f"[write] {args.report_path}")


if __name__ == "__main__":
    main()
