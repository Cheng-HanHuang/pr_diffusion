#!/usr/bin/env python3
"""Class-level failure taxonomy for reliability-oriented phase retrieval.

This script summarizes a diagnostic trace by image and separates failures into
classes useful for the reliability framework:

1. candidate-generation difficulty: no / few candidates exceed threshold;
2. seed availability difficulty: only a few seeds have any successful branch;
3. selector difficulty: oracle succeeds but current selector fails;
4. statistic ambiguity: successful and failed candidates overlap in selector
   statistic / residual features;
5. residual ambiguity: final measurement residual does not cleanly imply PSNR.

The script is intentionally class-level.  It should be used to understand what
kind of image / trace is difficult, not to patch a single image.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

Row = Dict[str, object]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[WARN] no rows for {path}")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fget(row: Dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        v = row.get(key, "")
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default


def finite(xs: Iterable[float]) -> List[float]:
    out = []
    for x in xs:
        try:
            fx = float(x)
        except Exception:
            continue
        if math.isfinite(fx):
            out.append(fx)
    return out


def fmean(xs: Iterable[float]) -> float:
    ys = finite(xs)
    return mean(ys) if ys else math.nan


def fmedian(xs: Iterable[float]) -> float:
    ys = finite(xs)
    return median(ys) if ys else math.nan


def fmin(xs: Iterable[float]) -> float:
    ys = finite(xs)
    return min(ys) if ys else math.nan


def fmax(xs: Iterable[float]) -> float:
    ys = finite(xs)
    return max(ys) if ys else math.nan


def corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    xv = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    mx, my = mean(xv), mean(yv)
    vx = sum((x - mx) ** 2 for x in xv)
    vy = sum((y - my) ** 2 for y in yv)
    if vx <= 0 or vy <= 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    npos = sum(1 for y, _ in pairs if y == 1)
    nneg = sum(1 for y, _ in pairs if y == 0)
    if npos == 0 or nneg == 0:
        return math.nan
    pairs = sorted(pairs, key=lambda ys: ys[1])
    # average ranks for ties
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    rank_sum_pos = sum(r for r, (y, _) in zip(ranks, pairs) if y == 1)
    return (rank_sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def find_trace_files(paths: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pstr in paths:
        p = Path(pstr).expanduser()
        if p.is_file():
            files.append(p)
        else:
            files.extend(Path(x) for x in glob.glob(str(p / "**/*_run_trace_summary.csv"), recursive=True))
            files.extend(Path(x) for x in glob.glob(str(p / "**/run_trace_summary.csv"), recursive=True))
    seen = set()
    out: List[Path] = []
    for p in files:
        k = str(p.resolve()) if p.exists() else str(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def image_id(name: str) -> str:
    m = re.search(r"(\d{5})\.png$", name)
    return m.group(1) if m else name


def select_current(rows: List[Row], stat_key: str) -> Row:
    by_cfg: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_cfg[str(r["config_tag"])].append(r)
    cfg_stats = []
    for cfg, gr in by_cfg.items():
        cfg_stats.append((fmean(fget(r, stat_key) for r in gr), cfg))
    cfg_stats.sort()
    cfg = cfg_stats[0][1]
    gr = by_cfg[cfg]
    return min(gr, key=lambda r: fget(r, stat_key))


def summarize_image(rows: List[Row], threshold: float, stat_key: str, psnr_key: str) -> Row:
    img = image_id(str(rows[0]["image_basename"]))
    psnrs = [fget(r, psnr_key) for r in rows]
    labels = [1 if p >= threshold else 0 for p in psnrs]
    seeds = sorted(set(str(r["seed"]) for r in rows))
    seed_success = 0
    for s in seeds:
        if max(fget(r, psnr_key) for r in rows if str(r["seed"]) == s) >= threshold:
            seed_success += 1
    oracle = max(rows, key=lambda r: fget(r, psnr_key))
    selected = select_current(rows, stat_key)
    good = [r for r in rows if fget(r, psnr_key) >= threshold]
    bad = [r for r in rows if fget(r, psnr_key) < threshold]

    stat_auc = auc(labels, [-fget(r, stat_key) for r in rows])
    lf_auc = auc(labels, [-fget(r, "raw_noisy_lowfreq_mag_l2") for r in rows])
    full_auc = auc(labels, [-fget(r, "raw_noisy_mag_l2") for r in rows])

    # Overlap diagnostics: if the best bad statistic/residual is better than many good ones,
    # selection is ambiguous.
    best_bad_stat = fmin(fget(r, stat_key) for r in bad)
    worst_good_stat = fmax(fget(r, stat_key) for r in good)
    best_bad_lf = fmin(fget(r, "raw_noisy_lowfreq_mag_l2") for r in bad)
    worst_good_lf = fmax(fget(r, "raw_noisy_lowfreq_mag_l2") for r in good)
    stat_overlap = int(math.isfinite(best_bad_stat) and math.isfinite(worst_good_stat) and best_bad_stat <= worst_good_stat)
    lf_overlap = int(math.isfinite(best_bad_lf) and math.isfinite(worst_good_lf) and best_bad_lf <= worst_good_lf)

    cand_rate = sum(labels) / len(labels) if labels else math.nan
    seed_rate = seed_success / len(seeds) if seeds else math.nan
    selector_fail = int(fget(oracle, psnr_key) >= threshold and fget(selected, psnr_key) < threshold)
    oracle_fail = int(fget(oracle, psnr_key) < threshold)

    classes: List[str] = []
    if oracle_fail:
        classes.append("oracle_failure_no_good_candidate")
    if cand_rate < 0.10:
        classes.append("thin_candidate_pool")
    if seed_rate < 0.75:
        classes.append("seed_sensitive")
    if selector_fail:
        classes.append("selector_failure_given_oracle_success")
    if stat_overlap or lf_overlap:
        classes.append("stat_residual_overlap")
    if not classes:
        classes.append("currently_reliable")

    return {
        "image_id": img,
        "image_basename": rows[0]["image_basename"],
        "threshold_db": threshold,
        "n_candidates": len(rows),
        "n_seeds": len(seeds),
        "n_success_candidates": sum(labels),
        "candidate_success_rate": cand_rate,
        "n_success_seeds": seed_success,
        "seed_success_rate": seed_rate,
        "oracle_psnr": fget(oracle, psnr_key),
        "oracle_config": oracle.get("config_tag", ""),
        "oracle_seed": oracle.get("seed", ""),
        "selected_psnr": fget(selected, psnr_key),
        "selected_config": selected.get("config_tag", ""),
        "selected_seed": selected.get("seed", ""),
        "selector_regret": fget(oracle, psnr_key) - fget(selected, psnr_key),
        "oracle_failure": oracle_fail,
        "selector_failure_given_oracle_success": selector_fail,
        "stat_auc_lower_is_better": stat_auc,
        "lf_resid_auc_lower_is_better": lf_auc,
        "full_resid_auc_lower_is_better": full_auc,
        "corr_stat_psnr": corr([fget(r, stat_key) for r in rows], psnrs),
        "corr_lf_resid_psnr": corr([fget(r, "raw_noisy_lowfreq_mag_l2") for r in rows], psnrs),
        "good_stat_mean": fmean(fget(r, stat_key) for r in good),
        "bad_stat_mean": fmean(fget(r, stat_key) for r in bad),
        "good_lf_resid_mean": fmean(fget(r, "raw_noisy_lowfreq_mag_l2") for r in good),
        "bad_lf_resid_mean": fmean(fget(r, "raw_noisy_lowfreq_mag_l2") for r in bad),
        "stat_overlap": stat_overlap,
        "lf_resid_overlap": lf_overlap,
        "failure_classes": ";".join(classes),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze class-level reliability failure modes.")
    ap.add_argument("--roots_or_traces", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thresholds", default="25,28,30")
    ap.add_argument("--selector_stat", default="post_winner_lf_mse_mean")
    ap.add_argument("--psnr_key", default="raw_psnr")
    args = ap.parse_args()

    traces = find_trace_files(args.roots_or_traces)
    if not traces:
        raise FileNotFoundError("No trace summary CSVs found")
    rows: List[Row] = []
    for p in traces:
        rows.extend(read_csv(p))
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    by_img: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        if r.get("image_basename") and r.get("seed") and r.get("config_tag"):
            by_img[image_id(str(r["image_basename"]))].append(r)

    out: List[Row] = []
    for tau in thresholds:
        for _, gr in sorted(by_img.items()):
            out.append(summarize_image(gr, tau, args.selector_stat, args.psnr_key))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "reliability_failure_taxonomy_by_image.csv", out)

    # Aggregate class counts.
    agg: Dict[Tuple[float, str], int] = defaultdict(int)
    for r in out:
        for c in str(r["failure_classes"]).split(";"):
            agg[(float(r["threshold_db"]), c)] += 1
    agg_rows = [
        {"threshold_db": tau, "failure_class": c, "n_images": n}
        for (tau, c), n in sorted(agg.items())
    ]
    write_csv(outdir / "reliability_failure_taxonomy_counts.csv", agg_rows)

    print(f"Loaded {len(rows)} rows from {len(traces)} traces")
    print(f"Wrote taxonomy to: {outdir}")
    print("\nTaxonomy counts:")
    for r in agg_rows:
        print(r)


if __name__ == "__main__":
    main()
