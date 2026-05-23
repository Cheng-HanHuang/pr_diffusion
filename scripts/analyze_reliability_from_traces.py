#!/usr/bin/env python3
"""Reliability analysis for multi-lambda diffusion phase retrieval traces.

This script implements the first algorithm/testing step suggested by
`docs/probabilistic_reliability_note.md`:

1. simulate fixed and adaptive compute policies from existing trace summaries;
2. calibrate whether `post_winner_lf_mse_mean` separates good and bad candidates;
3. summarize hard-image candidate availability.

Input files are the `*_run_trace_summary.csv` files produced by
`analyze_ffhq_diagnostic_trace_nopandas.py`.  You may pass either CSV files or
run roots containing these CSVs.

The script does not use ground truth for the executable selector decision.  PSNR
is used only for offline evaluation and for optional calibration of risk
thresholds.
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fget(row: Dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        v = row.get(key, "")
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default


def finite(xs: Iterable[float]) -> List[float]:
    return [float(x) for x in xs if math.isfinite(float(x))]


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


def fstd(xs: Iterable[float]) -> float:
    ys = finite(xs)
    return stdev(ys) if len(ys) > 1 else 0.0


def quantile(xs: Sequence[float], q: float) -> float:
    ys = sorted(finite(xs))
    if not ys:
        return math.nan
    if q <= 0:
        return ys[0]
    if q >= 1:
        return ys[-1]
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx, my = mean(xvals), mean(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def rankdata(xs: Sequence[float]) -> List[float]:
    indexed = sorted((x, i) for i, x in enumerate(xs))
    ranks = [0.0] * len(xs)
    j = 0
    while j < len(indexed):
        k = j + 1
        while k < len(indexed) and indexed[k][0] == indexed[j][0]:
            k += 1
        avg_rank = (j + 1 + k) / 2.0
        for _, idx in indexed[j:k]:
            ranks[idx] = avg_rank
        j = k
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    rx = rankdata([p[0] for p in pairs])
    ry = rankdata([p[1] for p in pairs])
    return pearson(rx, ry)


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mann-Whitney AUC. Higher score should mean more likely positive."""
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    n_pos = sum(1 for y, _ in pairs if y == 1)
    n_neg = sum(1 for y, _ in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    sorted_pairs = sorted((s, y) for y, s in pairs)
    ranks = rankdata([s for s, _ in sorted_pairs])
    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, sorted_pairs) if y == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def find_trace_files(paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for pstr in paths:
        p = Path(pstr).expanduser()
        if p.is_file():
            out.append(p)
            continue
        pats = [
            "**/*_run_trace_summary.csv",
            "**/run_trace_summary.csv",
        ]
        found: List[Path] = []
        for pat in pats:
            found.extend(Path(x) for x in glob.glob(str(p / pat), recursive=True))
        out.extend(sorted(set(found)))
    # Deduplicate while preserving order.
    seen = set()
    unique: List[Path] = []
    for p in out:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def normalize_config(config_tag: str) -> str:
    tag = str(config_tag).lower()
    if tag.endswith("_lf") or tag == "lf" or re.search(r"(^|_)lf($|_)", tag):
        return "LF"
    # Tags usually look like full25_s..._s2_lam0p005.
    m = re.search(r"lam([0-9]+(?:p|\.)[0-9]+)", tag)
    if m:
        val = m.group(1).replace("p", ".")
        try:
            f = float(val)
            if abs(f - 0.005) < 1e-9:
                return "S2_0.005"
            if abs(f - 0.01) < 1e-9:
                return "S2_0.01"
            if abs(f - 0.02) < 1e-9:
                return "S2_0.02"
            if abs(f - 0.05) < 1e-9:
                return "S2_0.05"
            if abs(f - 0.1) < 1e-9:
                return "S2_0.1"
            return f"S2_{f:g}"
        except Exception:
            pass
    if "s2" in tag and "001" in tag:
        return "S2_0.01"
    return str(config_tag)


def load_candidates(trace_files: Sequence[Path], stat_key: str, psnr_key: str, dedupe: bool) -> List[Row]:
    rows: List[Row] = []
    for path in trace_files:
        for r in read_csv(path):
            image = str(r.get("image_basename", ""))
            seed = str(r.get("seed", ""))
            config_tag = str(r.get("config_tag", ""))
            cfg = normalize_config(config_tag)
            stat = fget(r, stat_key)
            psnr = fget(r, psnr_key)
            if not image or not seed or not math.isfinite(stat) or not math.isfinite(psnr):
                continue
            row: Row = dict(r)
            row.update(
                {
                    "trace_file": str(path),
                    "image_basename": image,
                    "seed": seed,
                    "seed_int": int(float(seed)) if str(seed).replace(".", "", 1).isdigit() else seed,
                    "config_tag_original": config_tag,
                    "config_family": cfg,
                    "selector_stat": stat,
                    "raw_psnr_for_analysis": psnr,
                }
            )
            rows.append(row)

    if not dedupe:
        return rows

    # Keep one row per image/seed/config family.  Prefer the row with finite PSNR
    # and lowest selector statistic if duplicates appear from repeated analyses.
    by_key: Dict[Tuple[str, str, str], Row] = {}
    for r in rows:
        key = (str(r["image_basename"]), str(r["seed"]), str(r["config_family"]))
        old = by_key.get(key)
        if old is None or fget(r, "selector_stat") < fget(old, "selector_stat"):
            by_key[key] = r
    return list(by_key.values())


def select_by_stat(pool: List[Row]) -> Optional[Row]:
    valid = [r for r in pool if math.isfinite(fget(r, "selector_stat"))]
    if not valid:
        return None
    return min(valid, key=lambda r: fget(r, "selector_stat"))


def oracle_by_psnr(pool: List[Row]) -> Optional[Row]:
    valid = [r for r in pool if math.isfinite(fget(r, "raw_psnr_for_analysis"))]
    if not valid:
        return None
    return max(valid, key=lambda r: fget(r, "raw_psnr_for_analysis"))


def summarize_policy(image_rows: List[Row], threshold: float) -> Row:
    psnrs = [fget(r, "selected_psnr") for r in image_rows]
    oracle_psnrs = [fget(r, "oracle_psnr") for r in image_rows]
    budgets = [fget(r, "n_seeds_used") for r in image_rows]
    failures = [r for r in image_rows if fget(r, "selected_psnr") < threshold]
    oracle_failures = [r for r in image_rows if fget(r, "oracle_psnr") < threshold]
    selector_failures = [
        r for r in image_rows
        if fget(r, "oracle_psnr") >= threshold and fget(r, "selected_psnr") < threshold
    ]
    worst = min(image_rows, key=lambda r: fget(r, "selected_psnr")) if image_rows else {}
    return {
        "n_images": len(image_rows),
        "psnr_mean": fmean(psnrs),
        "psnr_median": fmedian(psnrs),
        "psnr_min": fmin(psnrs),
        "psnr_max": fmax(psnrs),
        "psnr_std": fstd(psnrs),
        "n_images_below_threshold": len(failures),
        "threshold_db": threshold,
        "oracle_psnr_mean": fmean(oracle_psnrs),
        "oracle_psnr_min": fmin(oracle_psnrs),
        "oracle_failures": len(oracle_failures),
        "selector_failures_given_oracle_success": len(selector_failures),
        "avg_seeds_used": fmean(budgets),
        "median_seeds_used": fmedian(budgets),
        "max_seeds_used": fmax(budgets),
        "worst_image": worst.get("image_basename", ""),
        "worst_selected_psnr": fget(worst, "selected_psnr"),
        "worst_oracle_psnr": fget(worst, "oracle_psnr"),
        "failed_images": ";".join(str(r.get("image_basename", "")) for r in failures),
        "oracle_failed_images": ";".join(str(r.get("image_basename", "")) for r in oracle_failures),
        "selector_failed_images": ";".join(str(r.get("image_basename", "")) for r in selector_failures),
    }


def calibration(rows: List[Row], thresholds: Sequence[float], stat_key_out: str) -> Tuple[List[Row], List[Row]]:
    global_rows: List[Row] = []
    by_image_rows: List[Row] = []
    by_image: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_image[str(r["image_basename"])].append(r)

    for tau in thresholds:
        labels = [1 if fget(r, "raw_psnr_for_analysis") >= tau else 0 for r in rows]
        stats = [fget(r, "selector_stat") for r in rows]
        psnrs = [fget(r, "raw_psnr_for_analysis") for r in rows]
        good_stats = [s for s, y in zip(stats, labels) if y == 1]
        bad_stats = [s for s, y in zip(stats, labels) if y == 0]
        global_rows.append(
            {
                "scope": "global",
                "threshold_db": tau,
                "n_candidates": len(rows),
                "n_success": sum(labels),
                "success_rate": sum(labels) / len(labels) if labels else math.nan,
                "stat_key": stat_key_out,
                "stat_mean_success": fmean(good_stats),
                "stat_median_success": fmedian(good_stats),
                "stat_mean_failure": fmean(bad_stats),
                "stat_median_failure": fmedian(bad_stats),
                "stat_q50_success": quantile(good_stats, 0.50),
                "stat_q75_success": quantile(good_stats, 0.75),
                "stat_q90_success": quantile(good_stats, 0.90),
                "stat_q95_success": quantile(good_stats, 0.95),
                "pearson_stat_vs_psnr": pearson(stats, psnrs),
                "spearman_stat_vs_psnr": spearman(stats, psnrs),
                "auc_neg_stat_predicts_success": auc_score(labels, [-s for s in stats]),
            }
        )
        for img, gr in sorted(by_image.items()):
            labels_i = [1 if fget(r, "raw_psnr_for_analysis") >= tau else 0 for r in gr]
            stats_i = [fget(r, "selector_stat") for r in gr]
            psnrs_i = [fget(r, "raw_psnr_for_analysis") for r in gr]
            good_i = [s for s, y in zip(stats_i, labels_i) if y == 1]
            bad_i = [s for s, y in zip(stats_i, labels_i) if y == 0]
            by_image_rows.append(
                {
                    "image_basename": img,
                    "threshold_db": tau,
                    "n_candidates": len(gr),
                    "n_success": sum(labels_i),
                    "success_rate": sum(labels_i) / len(labels_i) if labels_i else math.nan,
                    "stat_mean_success": fmean(good_i),
                    "stat_median_success": fmedian(good_i),
                    "stat_mean_failure": fmean(bad_i),
                    "stat_median_failure": fmedian(bad_i),
                    "auc_neg_stat_predicts_success": auc_score(labels_i, [-s for s in stats_i]),
                    "pearson_stat_vs_psnr": pearson(stats_i, psnrs_i),
                    "spearman_stat_vs_psnr": spearman(stats_i, psnrs_i),
                    "best_psnr": fmax(psnrs_i),
                    "best_stat": fmin(stats_i),
                }
            )
    return global_rows, by_image_rows


def candidate_availability(rows: List[Row], thresholds: Sequence[float], hard_images: Sequence[str]) -> List[Row]:
    by_image: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_image[str(r["image_basename"])].append(r)
    out: List[Row] = []
    for tau in thresholds:
        for img, gr in sorted(by_image.items()):
            seeds = sorted(set(str(r["seed"]) for r in gr))
            cfgs = sorted(set(str(r["config_family"]) for r in gr))
            success_rows = [r for r in gr if fget(r, "raw_psnr_for_analysis") >= tau]
            seeds_success = sorted(set(str(r["seed"]) for r in success_rows))
            by_cfg_counts = {}
            for cfg in cfgs:
                cfg_rows = [r for r in gr if str(r["config_family"]) == cfg]
                by_cfg_counts[f"success_count_{cfg}"] = sum(
                    1 for r in cfg_rows if fget(r, "raw_psnr_for_analysis") >= tau
                )
            best = oracle_by_psnr(gr)
            selected = select_by_stat(gr)
            row: Row = {
                "image_basename": img,
                "threshold_db": tau,
                "is_requested_hard_image": img in hard_images,
                "n_candidates": len(gr),
                "n_success": len(success_rows),
                "candidate_success_rate": len(success_rows) / len(gr) if gr else math.nan,
                "n_seeds": len(seeds),
                "n_seeds_with_any_success": len(seeds_success),
                "seed_success_rate": len(seeds_success) / len(seeds) if seeds else math.nan,
                "n_configs": len(cfgs),
                "best_raw_psnr": fget(best or {}, "raw_psnr_for_analysis"),
                "best_seed": (best or {}).get("seed", ""),
                "best_config_family": (best or {}).get("config_family", ""),
                "selected_by_stat_raw_psnr": fget(selected or {}, "raw_psnr_for_analysis"),
                "selected_by_stat_seed": (selected or {}).get("seed", ""),
                "selected_by_stat_config_family": (selected or {}).get("config_family", ""),
                "selector_regret_vs_oracle": fget(best or {}, "raw_psnr_for_analysis") - fget(selected or {}, "raw_psnr_for_analysis"),
                "seeds_with_success": ",".join(seeds_success),
            }
            row.update(by_cfg_counts)
            out.append(row)
    return out


def make_seed_orders(seeds: Sequence[str], n_orders: int, rng_seed: int) -> List[List[str]]:
    seeds = list(seeds)
    if not seeds:
        return []
    if n_orders <= 0:
        return [sorted(seeds, key=lambda s: int(float(s)) if str(s).isdigit() else str(s))]
    # For small seed sets and very high n_orders, enumerate all permutations.
    max_perm = math.factorial(len(seeds)) if len(seeds) <= 8 else None
    if max_perm is not None and max_perm <= n_orders:
        return [list(p) for p in itertools.permutations(seeds)]
    rng = random.Random(rng_seed)
    orders: List[List[str]] = []
    seen = set()
    base = list(seeds)
    for _ in range(n_orders * 10):
        cur = list(base)
        rng.shuffle(cur)
        key = tuple(cur)
        if key not in seen:
            seen.add(key)
            orders.append(cur)
        if len(orders) >= n_orders:
            break
    if not orders:
        orders.append(sorted(seeds))
    return orders


def simulate_once_for_image(
    gr: List[Row],
    seed_order: Sequence[str],
    threshold: float,
    policy: str,
    accept_stat_threshold: Optional[float],
    start_k: int,
    add_k: int,
    max_k: int,
) -> Row:
    image = str(gr[0]["image_basename"]) if gr else ""
    seeds_available = [s for s in seed_order if any(str(r["seed"]) == s for r in gr)]
    if not seeds_available:
        return {"image_basename": image, "selected_psnr": math.nan, "oracle_psnr": math.nan, "n_seeds_used": 0}

    def pool_for(k: int) -> List[Row]:
        use = set(seeds_available[: min(k, len(seeds_available))])
        return [r for r in gr if str(r["seed"]) in use]

    if policy.startswith("fixed"):
        try:
            k = int(policy.replace("fixed", ""))
        except Exception:
            k = start_k
        k = min(max(1, k), len(seeds_available))
        pool = pool_for(k)
        reason = "fixed_budget"
    elif policy == "oracle_until_success":
        # Diagnostic lower bound: this uses PSNR and is not executable.
        k = 0
        pool: List[Row] = []
        reason = "max_budget"
        for k in range(1, min(max_k, len(seeds_available)) + 1):
            pool = pool_for(k)
            oracle = oracle_by_psnr(pool)
            if oracle is not None and fget(oracle, "raw_psnr_for_analysis") >= threshold:
                reason = "oracle_success"
                break
    elif policy == "adaptive_stat_threshold":
        if accept_stat_threshold is None or not math.isfinite(accept_stat_threshold):
            raise ValueError("adaptive_stat_threshold requires accept_stat_threshold")
        k = min(max(1, start_k), len(seeds_available))
        reason = "max_budget"
        while True:
            pool = pool_for(k)
            selected = select_by_stat(pool)
            if selected is not None and fget(selected, "selector_stat") <= accept_stat_threshold:
                reason = "accepted_by_stat_threshold"
                break
            if k >= min(max_k, len(seeds_available)):
                reason = "max_budget"
                break
            k = min(k + add_k, max_k, len(seeds_available))
    else:
        raise ValueError(f"Unknown policy: {policy}")

    selected = select_by_stat(pool)
    oracle = oracle_by_psnr(pool)
    return {
        "image_basename": image,
        "policy": policy,
        "accept_stat_threshold": accept_stat_threshold if accept_stat_threshold is not None else math.nan,
        "n_seeds_used": len(set(str(r["seed"]) for r in pool)),
        "n_candidates_used": len(pool),
        "stop_reason": reason,
        "selected_seed": (selected or {}).get("seed", ""),
        "selected_config_family": (selected or {}).get("config_family", ""),
        "selected_stat": fget(selected or {}, "selector_stat"),
        "selected_psnr": fget(selected or {}, "raw_psnr_for_analysis"),
        "selected_success": fget(selected or {}, "raw_psnr_for_analysis") >= threshold,
        "oracle_seed": (oracle or {}).get("seed", ""),
        "oracle_config_family": (oracle or {}).get("config_family", ""),
        "oracle_psnr": fget(oracle or {}, "raw_psnr_for_analysis"),
        "oracle_success": fget(oracle or {}, "raw_psnr_for_analysis") >= threshold,
        "selector_regret": fget(oracle or {}, "raw_psnr_for_analysis") - fget(selected or {}, "raw_psnr_for_analysis"),
        "available_seed_order_prefix": ",".join(seeds_available[: len(set(str(r["seed"]) for r in pool))]),
    }


def simulate_policies(
    rows: List[Row],
    threshold: float,
    n_orders: int,
    rng_seed: int,
    start_k: int,
    add_k: int,
    max_k: int,
    threshold_quantiles: Sequence[float],
) -> Tuple[List[Row], List[Row]]:
    by_image: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_image[str(r["image_basename"])].append(r)
    all_seeds = sorted(set(str(r["seed"]) for r in rows), key=lambda s: int(float(s)) if str(s).isdigit() else str(s))
    seed_orders = make_seed_orders(all_seeds, n_orders=n_orders, rng_seed=rng_seed)

    success_stats = [fget(r, "selector_stat") for r in rows if fget(r, "raw_psnr_for_analysis") >= threshold]
    accept_thresholds: List[Tuple[str, Optional[float]]] = []
    for q in threshold_quantiles:
        accept_thresholds.append((f"adaptive_stat_q{int(round(q*100)):02d}", quantile(success_stats, q)))

    policy_specs: List[Tuple[str, Optional[float]]] = []
    for k in [1, 2, 4, 6, 8, 10]:
        if k <= len(all_seeds):
            policy_specs.append((f"fixed{k}", None))
    policy_specs.append(("oracle_until_success", None))
    policy_specs.extend((name, thr) for name, thr in accept_thresholds)

    image_rows: List[Row] = []
    summary_groups: Dict[Tuple[str, int, str], List[Row]] = defaultdict(list)

    for order_idx, order in enumerate(seed_orders):
        for policy_name, accept_thr in policy_specs:
            actual_policy = "adaptive_stat_threshold" if policy_name.startswith("adaptive_stat") else policy_name
            per_image: List[Row] = []
            for img, gr in sorted(by_image.items()):
                row = simulate_once_for_image(
                    gr,
                    order,
                    threshold=threshold,
                    policy=actual_policy,
                    accept_stat_threshold=accept_thr,
                    start_k=start_k,
                    add_k=add_k,
                    max_k=max_k,
                )
                row.update(
                    {
                        "policy_display": policy_name,
                        "order_idx": order_idx,
                        "threshold_db": threshold,
                        "full_seed_order": ",".join(order),
                    }
                )
                per_image.append(row)
                image_rows.append(row)
            summary_groups[(policy_name, order_idx, ",".join(order))].extend(per_image)

    summary_rows: List[Row] = []
    for (policy_name, order_idx, order_str), gr in sorted(summary_groups.items()):
        s = summarize_policy(gr, threshold)
        s.update({"policy_display": policy_name, "order_idx": order_idx, "full_seed_order": order_str})
        summary_rows.append(s)

    # Add average-over-orders summary for quick reading.
    by_policy: Dict[str, List[Row]] = defaultdict(list)
    for r in summary_rows:
        by_policy[str(r["policy_display"])].append(r)
    avg_rows: List[Row] = []
    for policy_name, gr in sorted(by_policy.items()):
        avg_rows.append(
            {
                "policy_display": policy_name,
                "order_idx": "mean_over_orders",
                "n_orders": len(gr),
                "threshold_db": threshold,
                "psnr_mean": fmean(fget(r, "psnr_mean") for r in gr),
                "psnr_min_mean_over_orders": fmean(fget(r, "psnr_min") for r in gr),
                "psnr_min_worst_over_orders": fmin(fget(r, "psnr_min") for r in gr),
                "n_images_below_threshold_mean": fmean(fget(r, "n_images_below_threshold") for r in gr),
                "n_images_below_threshold_max": fmax(fget(r, "n_images_below_threshold") for r in gr),
                "oracle_failures_mean": fmean(fget(r, "oracle_failures") for r in gr),
                "selector_failures_given_oracle_success_mean": fmean(fget(r, "selector_failures_given_oracle_success") for r in gr),
                "avg_seeds_used": fmean(fget(r, "avg_seeds_used") for r in gr),
                "max_seeds_used_max": fmax(fget(r, "max_seeds_used") for r in gr),
                "failed_images_union": ";".join(sorted(set(
                    img for r in gr for img in str(r.get("failed_images", "")).split(";") if img
                ))),
            }
        )
    summary_rows.extend(avg_rows)
    return image_rows, summary_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Reliability simulation/calibration from trace summaries.")
    ap.add_argument("--roots_or_traces", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--selector_stat", default="post_winner_lf_mse_mean")
    ap.add_argument("--psnr_key", default="raw_psnr")
    ap.add_argument("--thresholds", default="25,28,30")
    ap.add_argument("--primary_threshold", type=float, default=25.0)
    ap.add_argument("--dedupe", action="store_true", help="Deduplicate image/seed/config-family rows")
    ap.add_argument("--hard_images", default="00028,00005,00013,00034,00027,00007,00000")
    ap.add_argument("--n_seed_orders", type=int, default=200)
    ap.add_argument("--rng_seed", type=int, default=20260523)
    ap.add_argument("--adaptive_start_k", type=int, default=2)
    ap.add_argument("--adaptive_add_k", type=int, default=2)
    ap.add_argument("--adaptive_max_k", type=int, default=10)
    ap.add_argument("--threshold_quantiles", default="0.50,0.75,0.90,0.95")
    args = ap.parse_args()

    trace_files = find_trace_files(args.roots_or_traces)
    if not trace_files:
        raise FileNotFoundError("No *_run_trace_summary.csv files found")
    print("Trace files:")
    for p in trace_files:
        print(f"  {p}")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    hard_images = [x.strip() for x in args.hard_images.split(",") if x.strip()]
    threshold_quantiles = [float(x.strip()) for x in args.threshold_quantiles.split(",") if x.strip()]

    rows = load_candidates(trace_files, args.selector_stat, args.psnr_key, dedupe=args.dedupe)
    if not rows:
        raise RuntimeError("No usable candidate rows loaded")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save normalized candidate rows for inspection.
    write_csv(outdir / "normalized_candidates.csv", rows)

    cal_global, cal_by_image = calibration(rows, thresholds, args.selector_stat)
    write_csv(outdir / "selector_calibration_global.csv", cal_global)
    write_csv(outdir / "selector_calibration_by_image.csv", cal_by_image)

    availability = candidate_availability(rows, thresholds, hard_images)
    write_csv(outdir / "candidate_availability_by_image.csv", availability)
    write_csv(outdir / "hard_image_candidate_availability.csv", [r for r in availability if r.get("is_requested_hard_image")])

    adaptive_image, adaptive_summary = simulate_policies(
        rows,
        threshold=args.primary_threshold,
        n_orders=args.n_seed_orders,
        rng_seed=args.rng_seed,
        start_k=args.adaptive_start_k,
        add_k=args.adaptive_add_k,
        max_k=args.adaptive_max_k,
        threshold_quantiles=threshold_quantiles,
    )
    write_csv(outdir / "adaptive_policy_image_level.csv", adaptive_image)
    write_csv(outdir / "adaptive_policy_summary.csv", adaptive_summary)

    print(f"\nLoaded candidates: {len(rows)}")
    print(f"Images: {len(set(str(r['image_basename']) for r in rows))}")
    print(f"Seeds: {sorted(set(str(r['seed']) for r in rows))}")
    print(f"Config families: {sorted(set(str(r['config_family']) for r in rows))}")
    print(f"\nWrote reliability analysis to: {outdir}")
    print("Key files:")
    print("  selector_calibration_global.csv")
    print("  selector_calibration_by_image.csv")
    print("  candidate_availability_by_image.csv")
    print("  hard_image_candidate_availability.csv")
    print("  adaptive_policy_summary.csv")
    print("  adaptive_policy_image_level.csv")

    # Print compact primary-threshold calibration and best policy summaries.
    print("\nPrimary calibration rows:")
    for r in cal_global:
        if abs(float(r["threshold_db"]) - args.primary_threshold) < 1e-9:
            print(r)
    print("\nAdaptive policy mean-over-orders summary:")
    for r in adaptive_summary:
        if r.get("order_idx") == "mean_over_orders":
            print(r)


if __name__ == "__main__":
    main()
