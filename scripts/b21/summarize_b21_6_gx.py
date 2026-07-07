#!/usr/bin/env python3
"""Build the runbook-matched B21.6 GX summary table.

This consumes the offline forensics artifacts produced by
`scripts/b21/hard_attractor_forensics.py` and computes the predeclared
agglomerative average-linkage cluster table at cuts 0.4x, 0.5x, and 0.6x of the
per-image median pairwise identity/rot180-reduced pixel distance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


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


def ffloat(x: object, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def avg_linkage(ids: List[str], dist: Dict[Tuple[str, str], float], threshold: float) -> List[List[str]]:
    clusters: List[List[str]] = [[x] for x in ids]

    def d_between(a: List[str], b: List[str]) -> float:
        vals = []
        for x in a:
            for y in b:
                key = (x, y) if x < y else (y, x)
                if key in dist:
                    vals.append(dist[key])
        if not vals:
            return math.inf
        return float(np.mean(vals))

    while True:
        best_i = best_j = -1
        best_d = math.inf
        n = len(clusters)
        for i in range(n):
            for j in range(i + 1, n):
                dij = d_between(clusters[i], clusters[j])
                if dij < best_d:
                    best_d = dij
                    best_i, best_j = i, j
        if best_i < 0 or best_d > threshold:
            break
        merged = clusters[best_i] + clusters[best_j]
        new_clusters = []
        for k, c in enumerate(clusters):
            if k not in {best_i, best_j}:
                new_clusters.append(c)
        new_clusters.append(merged)
        clusters = new_clusters
    clusters.sort(key=lambda c: (-len(c), sorted(c)[0]))
    return clusters


def build_tables(forensics_dir: Path, factors: Sequence[float]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    cand_rows = read_csv(forensics_dir / "candidate_index.csv")
    pair_rows = read_csv(forensics_dir / "distance_pairs.csv")
    by_id = {r["analysis_id"]: r for r in cand_rows}
    ids_by_image: Dict[str, List[str]] = defaultdict(list)
    for r in cand_rows:
        ids_by_image[r["image_id"]].append(r["analysis_id"])

    dist_by_image: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    lf_by_image: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    for r in pair_rows:
        a = r["a"]
        b = r["b"]
        key = (a, b) if a < b else (b, a)
        image = r["image_id"]
        dist_by_image[image][key] = ffloat(r.get("pixel_rms_rot_reduced"))
        lf_by_image[image][key] = ffloat(r.get("lfmag_rms"))

    gx_rows: List[Dict[str, object]] = []
    sensitivity_rows: List[Dict[str, object]] = []
    for image_id, ids in sorted(ids_by_image.items()):
        vals = [v for v in dist_by_image[image_id].values() if math.isfinite(v)]
        lf_vals = [v for v in lf_by_image[image_id].values() if math.isfinite(v)]
        med = float(np.median(vals)) if vals else math.nan
        lf_med = float(np.median(lf_vals)) if lf_vals else math.nan
        for factor in factors:
            threshold = factor * med if math.isfinite(med) else math.nan
            clusters = avg_linkage(ids, dist_by_image[image_id], threshold=threshold) if math.isfinite(threshold) else [[x] for x in ids]
            largest = clusters[0] if clusters else []
            largest_seed_set = sorted(set(by_id[x].get("seed", "") for x in largest))
            largest_arm_set = sorted(set(by_id[x].get("arm", by_id[x].get("variant", "")) for x in largest))
            largest_psnr = [ffloat(by_id[x].get("psnr")) for x in largest]
            largest_psnr = [x for x in largest_psnr if math.isfinite(x)]
            largest_share = len(largest) / len(ids) if ids else 0.0
            row = {
                "image_id": image_id,
                "factor": factor,
                "n_bad": len(ids),
                "median_pairwise_pixel_dist": med,
                "threshold": threshold,
                "median_pairwise_lfmag_dist": lf_med,
                "n_clusters": len(clusters),
                "largest_cluster_size": len(largest),
                "largest_share": largest_share,
                "cross_seed_shared": "Y" if len(largest_seed_set) > 1 else "N",
                "cross_arm_shared": "Y" if len(largest_arm_set) > 1 else "N",
                "largest_n_seeds": len(largest_seed_set),
                "largest_n_arms": len(largest_arm_set),
                "largest_psnr_min": min(largest_psnr) if largest_psnr else math.nan,
                "largest_psnr_mean": float(np.mean(largest_psnr)) if largest_psnr else math.nan,
                "largest_psnr_max": max(largest_psnr) if largest_psnr else math.nan,
                "repulsion_candidate": "Y" if largest_share >= 0.5 and len(largest_seed_set) > 1 else "N",
                "dominant_type": "contact_sheet_review_required",
                "largest_cluster_ids_first20": ",".join(largest[:20]),
                "largest_seeds_first20": ",".join(largest_seed_set[:20]),
                "largest_arms": ",".join(largest_arm_set),
            }
            sensitivity_rows.append(row)
            if abs(factor - 0.5) < 1e-9:
                gx_rows.append(row)
    return gx_rows, sensitivity_rows


def render_report(outdir: Path, gx_rows: Sequence[Dict[str, object]], sensitivity_rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "# B21.6 GX summary",
        "",
        "Status: generated by `scripts/b21/summarize_b21_6_gx.py` from the B21.6 forensics artifacts.",
        "",
        "Primary cut: agglomerative average-linkage at `0.5 x median pairwise identity/rot180-reduced pixel distance` per image. Sensitivity rows for `0.4x` and `0.6x` are written to CSV.",
        "",
        "## GX table",
        "",
        "| image | n_bad | n_clusters | largest_share | cross_seed_shared | cross_arm_shared | repulsion_candidate | dominant_type |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for r in gx_rows:
        lines.append(
            f"| `{r['image_id']}` | {r['n_bad']} | {r['n_clusters']} | {float(r['largest_share']):.3f} | {r['cross_seed_shared']} | {r['cross_arm_shared']} | {r['repulsion_candidate']} | {r['dominant_type']} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append("`dominant_type` is deliberately left as `contact_sheet_review_required`; it should be filled only after visually inspecting the contact sheets. This keeps the numeric GX pass separate from human visual labeling.")
    lines.extend(["", "Artifacts:", "", "```text", str(outdir / "gx_summary.csv"), str(outdir / "gx_sensitivity.csv"), "```", ""])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize B21.6 forensics into the GX gate table")
    ap.add_argument("--forensics_dir", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_6_hard_attractor_forensics")
    ap.add_argument("--outdir", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_6_hard_attractor_forensics")
    ap.add_argument("--factors", default="0.4,0.5,0.6")
    ap.add_argument("--report_path", default="docs/b21/b21_6_gx_summary.md")
    args = ap.parse_args()
    forensics_dir = Path(args.forensics_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    gx_rows, sensitivity_rows = build_tables(forensics_dir, factors=factors)
    write_csv(outdir / "gx_summary.csv", gx_rows)
    write_csv(outdir / "gx_sensitivity.csv", sensitivity_rows)
    report = render_report(outdir, gx_rows, sensitivity_rows)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"[write] {outdir / 'gx_summary.csv'}")
    print(f"[write] {outdir / 'gx_sensitivity.csv'}")
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()
