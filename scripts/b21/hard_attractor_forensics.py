#!/usr/bin/env python3
"""B21.6 hard-image bad-attractor forensics.

This script analyzes already-produced sample PNGs for hard FFHQ images.  It is a
no/low-GPU offline analysis: load candidate images, compute identity/rot180
reduced pixel distances, low-frequency magnitude distances, simple connected
components, nearest neighbors, and contact sheets.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PATH_KEYS = ["sample_path", "recon_path", "png", "jpg", "image_path", "path", "file"]


def split_items(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def safe_float(x: object, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


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
    return Path(text).suffix.lower() in IMAGE_EXTS


def existing_image_from_row(row: Dict[str, str]) -> Tuple[str, str]:
    for key, value in row.items():
        lk = key.lower()
        if any(k in lk for k in PATH_KEYS) and value and is_image_path(value) and Path(value).exists():
            return value, key
    for key, value in row.items():
        if value and is_image_path(value) and Path(value).exists():
            return value, key
    return "", ""


def resolve_sample_from_detail_csv(path_text: str, max_rows: int = 200) -> Dict[str, object]:
    if not path_text:
        return {"detail_status": "no_csv_path"}
    path = Path(path_text)
    if not path.exists():
        return {"detail_status": "csv_missing", "detail_csv": path_text}
    try:
        fields, rows = read_csv_rows(path, max_rows=max_rows)
    except Exception as exc:
        return {"detail_status": "csv_read_error", "detail_csv": path_text, "detail_error": repr(exc)}
    for row in rows:
        sample_path, sample_col = existing_image_from_row(row)
        if sample_path:
            return {
                "detail_status": "ok",
                "detail_csv": path_text,
                "detail_fields": ",".join(fields),
                "detail_rows_scanned": len(rows),
                "sample_path": sample_path,
                "sample_path_source_col": f"detail:{sample_col}",
                "detail_psnr": row.get("psnr", row.get("selected_psnr", row.get("final_psnr", ""))),
                "exact_operator_loss": row.get("exact_operator_loss", row.get("loss", "")),
                "sqrt_loss_over_y_norm": row.get("sqrt_loss_over_y_norm", ""),
            }
    return {"detail_status": "no_existing_image_path", "detail_csv": path_text, "detail_fields": ",".join(fields), "detail_rows_scanned": len(rows)}


def expand_csvs(base: Path, globs_text: str) -> List[Path]:
    paths: List[Path] = []
    for pat in split_items(globs_text):
        full = pat if os.path.isabs(pat) else str(base / pat)
        paths.extend(Path(x) for x in glob.glob(full, recursive=True))
    return sorted(set(p for p in paths if p.exists() and p.is_file()))


def collect_candidates(
    b19_base: Path,
    csv_globs: str,
    targets: set[str],
    bad_psnr_threshold: float,
    include_good: bool,
    max_detail_rows: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    candidates: List[Dict[str, object]] = []
    metrics_only: List[Dict[str, object]] = []
    csv_paths = expand_csvs(b19_base, csv_globs)
    seen_paths = set()
    for csv_path in csv_paths:
        try:
            _fields, rows = read_csv_rows(csv_path)
        except Exception as exc:
            metrics_only.append({"source_csv": str(csv_path), "error": repr(exc)})
            continue
        for row_i, row in enumerate(rows):
            image_id = row.get("image_id", "")
            if image_id not in targets:
                continue
            psnr = safe_float(row.get("psnr", row.get("selected_psnr", "nan")))
            if not include_good and (not math.isfinite(psnr) or psnr >= bad_psnr_threshold):
                continue
            sample_path, sample_col = existing_image_from_row(row)
            resolved: Dict[str, object] = {}
            if not sample_path and row.get("csv_path"):
                resolved = resolve_sample_from_detail_csv(row.get("csv_path", ""), max_rows=max_detail_rows)
                sample_path = str(resolved.get("sample_path", ""))
                sample_col = str(resolved.get("sample_path_source_col", ""))
            rec: Dict[str, object] = {
                "candidate_id": "",
                "source_csv": str(csv_path),
                "source_row": row_i,
                "image_id": image_id,
                "meas_seed": row.get("meas_seed", ""),
                "seed": row.get("seed", row.get("run_seed", "")),
                "variant": row.get("variant", row.get("arm", row.get("policy", ""))),
                "arm": row.get("arm", ""),
                "ann_steps": row.get("ann_steps", ""),
                "diff_steps": row.get("diff_steps", ""),
                "psnr": psnr,
                "good25": row.get("good25", ""),
                "csv_path": row.get("csv_path", ""),
                "sample_path": sample_path,
                "sample_path_source_col": sample_col,
                "exact_operator_loss": row.get("exact_operator_loss", resolved.get("exact_operator_loss", "")),
                "sqrt_loss_over_y_norm": row.get("sqrt_loss_over_y_norm", resolved.get("sqrt_loss_over_y_norm", "")),
                "detail_status": resolved.get("detail_status", "direct" if sample_path else "missing"),
            }
            if sample_path and Path(sample_path).exists():
                if sample_path in seen_paths:
                    continue
                seen_paths.add(sample_path)
                rec["candidate_id"] = f"c{len(candidates):06d}"
                candidates.append(rec)
            else:
                metrics_only.append(rec)
    return candidates, metrics_only


def cap_candidates(cands: Sequence[Dict[str, object]], max_per_image: int) -> List[Dict[str, object]]:
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for c in cands:
        by_image[str(c.get("image_id", ""))].append(dict(c))
    out: List[Dict[str, object]] = []
    for image_id, rows in sorted(by_image.items()):
        rows.sort(key=lambda r: (safe_float(r.get("psnr")), str(r.get("variant")), str(r.get("seed"))))
        if max_per_image <= 0 or len(rows) <= max_per_image:
            keep = rows
        else:
            # Keep the worst half and a uniform tail across the rest, so both
            # catastrophic attractors and moderate failures remain visible.
            worst_n = max_per_image // 2
            keep = rows[:worst_n]
            rest = rows[worst_n:]
            if rest:
                idxs = np.linspace(0, len(rest) - 1, max_per_image - len(keep)).round().astype(int).tolist()
                keep.extend(rest[i] for i in sorted(set(idxs)))
            keep = keep[:max_per_image]
        out.extend(keep)
    for i, row in enumerate(out):
        row["analysis_id"] = f"a{i:06d}"
        row["truncated_for_pairwise"] = len(by_image[str(row.get("image_id", ""))]) > max_per_image > 0
    return out


def load_image01(path: str, size: int = 128) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size and max(img.size) != size:
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        img = canvas
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def rot180(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr[::-1, ::-1, :])


def pixel_dist_rot_reduced(a: np.ndarray, b: np.ndarray) -> Tuple[float, str]:
    d0 = float(np.sqrt(np.mean((a - b) ** 2)))
    d1 = float(np.sqrt(np.mean((a - rot180(b)) ** 2)))
    if d1 < d0:
        return d1, "rot180"
    return d0, "identity"


def lf_feature(arr: np.ndarray, radius_frac: float) -> np.ndarray:
    # Use RGB low-frequency magnitude on the centered measurement grid of the
    # loaded sample itself.  This is for clustering/forensics, not exact scoring.
    h, w, c = arr.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    mask = np.sqrt(xx * xx + yy * yy) <= radius_frac
    feats = []
    for ch in range(c):
        F = np.fft.fft2(arr[:, :, ch], norm="ortho")
        feats.append(np.abs(F)[mask].astype(np.float32))
    feat = np.concatenate(feats)
    norm = float(np.linalg.norm(feat))
    if norm > 0:
        feat = feat / norm
    return feat


def connected_components(ids: List[str], pairs: Sequence[Tuple[str, str, float]], threshold: float) -> Dict[str, int]:
    parent = {x: x for x in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, d in pairs:
        if d <= threshold:
            union(a, b)
    roots = {}
    labels = {}
    next_label = 0
    for x in ids:
        r = find(x)
        if r not in roots:
            roots[r] = next_label
            next_label += 1
        labels[x] = roots[r]
    return labels


def auto_threshold(values: Sequence[float], q: float, fallback: float) -> float:
    vals = [v for v in values if math.isfinite(v) and v > 0]
    if not vals:
        return fallback
    return float(np.quantile(np.asarray(vals, dtype=np.float32), q))


def compute_distances(cands: List[Dict[str, object]], image_size: int, lf_radius_frac: float, pixel_threshold: float, lf_threshold: float, q: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, float]]:
    arrays: Dict[str, np.ndarray] = {}
    lf_feats: Dict[str, np.ndarray] = {}
    for c in cands:
        aid = str(c["analysis_id"])
        arr = load_image01(str(c["sample_path"]), size=image_size)
        arrays[aid] = arr
        lf_feats[aid] = lf_feature(arr, radius_frac=lf_radius_frac)

    pair_rows: List[Dict[str, object]] = []
    pixel_pairs_by_image: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    lf_pairs_by_image: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    pixel_values: List[float] = []
    lf_values: List[float] = []
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for c in cands:
        by_image[str(c["image_id"])].append(c)

    for image_id, rows in by_image.items():
        rows = sorted(rows, key=lambda r: str(r["analysis_id"]))
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a = str(rows[i]["analysis_id"])
                b = str(rows[j]["analysis_id"])
                pd, align = pixel_dist_rot_reduced(arrays[a], arrays[b])
                lfd = float(np.sqrt(np.mean((lf_feats[a] - lf_feats[b]) ** 2)))
                pixel_values.append(pd)
                lf_values.append(lfd)
                pixel_pairs_by_image[image_id].append((a, b, pd))
                lf_pairs_by_image[image_id].append((a, b, lfd))
                pair_rows.append({"image_id": image_id, "a": a, "b": b, "pixel_rms_rot_reduced": pd, "pixel_best_alignment": align, "lfmag_rms": lfd})

    pix_thr = pixel_threshold if pixel_threshold > 0 else auto_threshold(pixel_values, q=q, fallback=0.15)
    lf_thr = lf_threshold if lf_threshold > 0 else auto_threshold(lf_values, q=q, fallback=0.02)

    pixel_labels: Dict[str, int] = {}
    lf_labels: Dict[str, int] = {}
    combined_labels: Dict[str, int] = {}
    for image_id, rows in by_image.items():
        ids = [str(r["analysis_id"]) for r in rows]
        pixel_labels.update(connected_components(ids, pixel_pairs_by_image[image_id], pix_thr))
        lf_labels.update(connected_components(ids, lf_pairs_by_image[image_id], lf_thr))
        combined_pairs = []
        for pr, lr in zip(pixel_pairs_by_image[image_id], lf_pairs_by_image[image_id]):
            a, b, pd = pr
            _, _, lfd = lr
            combined_pairs.append((a, b, 0.0 if (pd <= pix_thr and lfd <= lf_thr) else 1.0))
        combined_labels.update(connected_components(ids, combined_pairs, 0.0))

    # Nearest neighbors by pixel and LF distance.
    nearest_rows: List[Dict[str, object]] = []
    for c in cands:
        aid = str(c["analysis_id"])
        image_id = str(c["image_id"])
        same = [p for p in pair_rows if p["image_id"] == image_id and (p["a"] == aid or p["b"] == aid)]
        if not same:
            continue
        pix = min(same, key=lambda r: float(r["pixel_rms_rot_reduced"]))
        lf = min(same, key=lambda r: float(r["lfmag_rms"]))
        nearest_rows.append(
            {
                "analysis_id": aid,
                "image_id": image_id,
                "nearest_pixel_id": pix["b"] if pix["a"] == aid else pix["a"],
                "nearest_pixel_dist": pix["pixel_rms_rot_reduced"],
                "nearest_pixel_alignment": pix["pixel_best_alignment"],
                "nearest_lf_id": lf["b"] if lf["a"] == aid else lf["a"],
                "nearest_lf_dist": lf["lfmag_rms"],
            }
        )

    for c in cands:
        aid = str(c["analysis_id"])
        c["pixel_cluster"] = pixel_labels.get(aid, -1)
        c["lfmag_cluster"] = lf_labels.get(aid, -1)
        c["combined_cluster"] = combined_labels.get(aid, -1)

    return pair_rows, nearest_rows, {"pixel_threshold": pix_thr, "lfmag_threshold": lf_thr, "lf_radius_frac": lf_radius_frac}


def make_contact_sheets(cands: Sequence[Dict[str, object]], outdir: Path, thumb: int, max_per_sheet: int) -> List[str]:
    sheet_dir = outdir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    by_image: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for c in cands:
        by_image[str(c["image_id"])].append(c)
    paths: List[str] = []
    for image_id, rows in sorted(by_image.items()):
        rows = sorted(rows, key=lambda r: (int(r.get("combined_cluster", -1)), safe_float(r.get("psnr")), str(r.get("variant")), str(r.get("seed"))))[:max_per_sheet]
        if not rows:
            continue
        label_h = 34
        cols = min(8, len(rows))
        rows_n = int(math.ceil(len(rows) / cols))
        sheet = Image.new("RGB", (cols * thumb, rows_n * (thumb + label_h)), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for idx, rec in enumerate(rows):
            img = Image.open(str(rec["sample_path"])).convert("RGB")
            img.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            x = (idx % cols) * thumb
            y = (idx // cols) * (thumb + label_h)
            sheet.paste(img, (x + (thumb - img.width) // 2, y))
            label = f"{rec['analysis_id']} C{rec.get('combined_cluster')}"
            label2 = f"{rec.get('variant')} s{rec.get('seed')} {safe_float(rec.get('psnr')):.1f}"
            draw.text((x + 2, y + thumb + 1), label[:24], fill=(0, 0, 0))
            draw.text((x + 2, y + thumb + 16), label2[:24], fill=(0, 0, 0))
        path = sheet_dir / f"b21_6_{image_id}_contact_sheet.jpg"
        sheet.save(path, quality=90)
        paths.append(str(path))
    return paths


def cluster_summary(cands: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for c in cands:
        groups[(str(c["image_id"]), int(c.get("combined_cluster", -1)))].append(c)
    rows = []
    for (image_id, cluster), members in sorted(groups.items()):
        psnrs = [safe_float(m.get("psnr")) for m in members if math.isfinite(safe_float(m.get("psnr")))]
        variants = Counter(str(m.get("variant", "")) for m in members)
        seeds = sorted(set(str(m.get("seed", "")) for m in members))
        rows.append(
            {
                "image_id": image_id,
                "combined_cluster": cluster,
                "size": len(members),
                "psnr_min": min(psnrs) if psnrs else math.nan,
                "psnr_mean": float(np.mean(psnrs)) if psnrs else math.nan,
                "psnr_max": max(psnrs) if psnrs else math.nan,
                "variants": json.dumps(dict(variants), sort_keys=True),
                "n_seeds": len(seeds),
                "seeds_first20": ",".join(seeds[:20]),
            }
        )
    return rows


def render_report(outdir: Path, cands: Sequence[Dict[str, object]], metrics_only: Sequence[Dict[str, object]], clusters: Sequence[Dict[str, object]], thresholds: Dict[str, float], sheets: Sequence[str]) -> str:
    by_image = Counter(str(c.get("image_id", "")) for c in cands)
    by_image_metric = Counter(str(c.get("image_id", "")) for c in metrics_only)
    lines = [
        "# B21.6 hard-image bad-attractor forensics",
        "",
        "Status: generated by `scripts/b21/hard_attractor_forensics.py`.",
        "",
        "## Summary",
        "",
        f"- Sample-image candidates analyzed: `{len(cands)}`",
        f"- Metrics-only rows without usable sample image: `{len(metrics_only)}`",
        f"- Pixel cluster threshold: `{thresholds.get('pixel_threshold')}`",
        f"- LF-magnitude cluster threshold: `{thresholds.get('lfmag_threshold')}`",
        f"- LF radius fraction: `{thresholds.get('lf_radius_frac')}`",
        "",
        "## Counts by image",
        "",
        "| image_id | sample rows | metrics-only rows |",
        "|---|---:|---:|",
    ]
    for image_id in sorted(set(by_image) | set(by_image_metric)):
        lines.append(f"| `{image_id}` | {by_image[image_id]} | {by_image_metric[image_id]} |")
    lines.extend(["", "## Combined cluster summary", "", "| image_id | cluster | size | psnr min | psnr mean | psnr max | n seeds |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in clusters[:200]:
        lines.append(
            f"| `{row['image_id']}` | {row['combined_cluster']} | {row['size']} | {float(row['psnr_min']):.3f} | {float(row['psnr_mean']):.3f} | {float(row['psnr_max']):.3f} | {row['n_seeds']} |"
        )
    lines.extend(["", "## Contact sheets", ""])
    for p in sheets:
        lines.append(f"- `{p}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "```text",
            str(outdir / "candidate_index.csv"),
            str(outdir / "metrics_only_rows.csv"),
            str(outdir / "distance_pairs.csv"),
            str(outdir / "nearest_neighbors.csv"),
            str(outdir / "cluster_summary.csv"),
            "```",
            "",
            "## Interpretation notes",
            "",
            "This is a forensic clustering pass, not a frozen selector. Clusters are based on sample PNGs already present in the filesystem and use identity/rot180-reduced pixel distance plus low-frequency magnitude distance. If a B19.20 or B19.16 row lacks a sample path, it is counted as metrics-only and cannot contribute to image-space clustering without locating or regenerating the candidate PNG.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="B21.6 hard-image bad-attractor forensics")
    ap.add_argument("--b19_base", default="/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
    ap.add_argument("--targets", default="00046,00171,00480,00746,00971")
    ap.add_argument("--bad_psnr_threshold", type=float, default=25.0)
    ap.add_argument("--include_good", action="store_true")
    ap.add_argument("--max_per_image", type=int, default=120)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--lf_radius_frac", type=float, default=0.12)
    ap.add_argument("--pixel_threshold", type=float, default=-1.0)
    ap.add_argument("--lf_threshold", type=float, default=-1.0)
    ap.add_argument("--auto_quantile", type=float, default=0.15)
    ap.add_argument("--max_detail_rows", type=int, default=200)
    ap.add_argument("--thumb", type=int, default=112)
    ap.add_argument("--max_sheet", type=int, default=80)
    ap.add_argument(
        "--csv_globs",
        default="B20_11_00046_meas5001_lf_guidance_long.csv,B20_12A_multiimage_heldout4_64seed_3arm_long.csv,B19_20*.csv",
    )
    ap.add_argument("--outdir", default="/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_6_hard_attractor_forensics")
    ap.add_argument("--report_path", default="docs/b21/b21_6_hard_attractor_forensics.md")
    args = ap.parse_args()

    b19_base = Path(args.b19_base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    targets = set(split_items(args.targets))

    cands_all, metrics_only = collect_candidates(
        b19_base=b19_base,
        csv_globs=args.csv_globs,
        targets=targets,
        bad_psnr_threshold=args.bad_psnr_threshold,
        include_good=bool(args.include_good),
        max_detail_rows=args.max_detail_rows,
    )
    cands = cap_candidates(cands_all, max_per_image=args.max_per_image)
    pair_rows, nearest_rows, thresholds = compute_distances(cands, image_size=args.image_size, lf_radius_frac=args.lf_radius_frac, pixel_threshold=args.pixel_threshold, lf_threshold=args.lf_threshold, q=args.auto_quantile)
    clusters = cluster_summary(cands)
    sheets = make_contact_sheets(cands, outdir=outdir, thumb=args.thumb, max_per_sheet=args.max_sheet)

    write_csv(outdir / "candidate_index.csv", cands)
    write_csv(outdir / "metrics_only_rows.csv", metrics_only)
    write_csv(outdir / "distance_pairs.csv", pair_rows)
    write_csv(outdir / "nearest_neighbors.csv", nearest_rows)
    write_csv(outdir / "cluster_summary.csv", clusters)
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "candidate_rows_total_before_cap": len(cands_all),
                "candidate_rows_analyzed": len(cands),
                "metrics_only_rows": len(metrics_only),
                "targets": sorted(targets),
                "thresholds": thresholds,
                "contact_sheets": sheets,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = render_report(outdir, cands, metrics_only, clusters, thresholds, sheets)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"[write] {outdir / 'candidate_index.csv'}")
    print(f"[write] {outdir / 'distance_pairs.csv'}")
    print(f"[write] {outdir / 'nearest_neighbors.csv'}")
    print(f"[write] {outdir / 'cluster_summary.csv'}")
    print(f"[write] {outdir / 'summary.json'}")
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()
