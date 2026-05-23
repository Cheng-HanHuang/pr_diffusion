#!/usr/bin/env python3
"""Offline selector-policy sweep for diffusion phase-retrieval reliability traces.

This script consumes one or more `*_run_trace_summary.csv` files and simulates
non-ground-truth selector variants.  It is intended to answer the question:

  If the candidate pool already contains good candidates, can a better selector
  choose them more reliably than min(post_winner_lf_mse_mean)?

The script never uses PSNR to make executable selector choices.  PSNR is used
only for offline evaluation, oracle diagnostics, and regret reporting.

Typical input:
  hard_ablation_diag_run_trace_summary.csv

Typical outputs:
  selector_policy_image_level.csv
  selector_policy_summary.csv
  selector_policy_by_threshold.csv
  selector_policy_failures.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
                seen.add(key)
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fget(row: Dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        val = row.get(key, "")
        if val == "" or val is None:
            return default
        return float(val)
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


def fstd(xs: Iterable[float]) -> float:
    ys = finite(xs)
    return stdev(ys) if len(ys) > 1 else 0.0


def find_trace_files(paths: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for item in paths:
        p = Path(item).expanduser()
        if p.is_file():
            files.append(p)
            continue
        pats = ["**/*_run_trace_summary.csv", "**/run_trace_summary.csv"]
        for pat in pats:
            files.extend(Path(x) for x in glob.glob(str(p / pat), recursive=True))
    seen = set()
    unique: List[Path] = []
    for p in files:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def parse_branch(config_tag: str) -> Row:
    tag = str(config_tag)
    low = tag.lower()
    if re.search(r"(^|_)lf($|_)", low) or low.endswith("_lf") or low == "lf":
        lam = 0.0
        family = "LF"
    else:
        lam = math.nan
        m = re.search(r"lam([0-9]+(?:p|\.)[0-9]+)", low)
        if m:
            try:
                lam = float(m.group(1).replace("p", "."))
            except Exception:
                lam = math.nan
        family = f"S2_{lam:g}" if math.isfinite(lam) else tag
    proj_start = math.nan
    mps = re.search(r"(?:ps|start)([0-9]+)", low)
    if mps:
        try:
            proj_start = float(mps.group(1))
        except Exception:
            pass
    soft = math.nan
    hard = math.nan
    ms = re.search(r"soft([0-9]+)", low)
    mh = re.search(r"hard([0-9]+)", low)
    if ms:
        soft = float(ms.group(1))
    if mh:
        hard = float(mh.group(1))
    return {
        "branch_family": family,
        "branch_lambda": lam,
        "branch_proj_start": proj_start,
        "branch_soft": soft,
        "branch_hard": hard,
    }


def load_rows(
    trace_files: Sequence[Path],
    *,
    stat_key: str,
    psnr_key: str,
    lf_resid_key: str,
    full_resid_key: str,
    dedupe: bool,
) -> List[Row]:
    rows: List[Row] = []
    for path in trace_files:
        for src in read_csv(path):
            image = str(src.get("image_basename", ""))
            seed = str(src.get("seed", ""))
            config = str(src.get("config_tag", ""))
            stat = fget(src, stat_key)
            psnr = fget(src, psnr_key)
            if not image or not seed or not config or not math.isfinite(stat) or not math.isfinite(psnr):
                continue
            out: Row = dict(src)
            out.update(parse_branch(config))
            out.update(
                {
                    "trace_file": str(path),
                    "image_basename": image,
                    "seed": seed,
                    "config_tag": config,
                    "selector_stat": stat,
                    "eval_psnr": psnr,
                    "lf_resid": fget(src, lf_resid_key),
                    "full_resid": fget(src, full_resid_key),
                    "post_full_stat": fget(src, "post_winner_full_mse_mean"),
                    "post_lf_margin": fget(src, "post_lf_mse_margin_mean"),
                    "pre_lf_margin": fget(src, "pre_lf_mse_margin_mean"),
                    "winner_lf_frac_pre": fget(src, "winner_is_lf_best_frac_pre"),
                    "winner_lf_frac_all": fget(src, "winner_is_lf_best_frac_all"),
                    "last_lf_mse": fget(src, "last_winner_lf_mse"),
                    "last_full_mse": fget(src, "last_winner_full_mse"),
                }
            )
            rows.append(out)

    if not dedupe:
        return rows

    by_key: Dict[Tuple[str, str, str], Row] = {}
    for r in rows:
        key = (str(r["image_basename"]), str(r["seed"]), str(r["config_tag"]))
        old = by_key.get(key)
        if old is None or fget(r, "selector_stat") < fget(old, "selector_stat"):
            by_key[key] = r
    return list(by_key.values())


def norm_value(rows: Sequence[Row], row: Row, key: str) -> float:
    x = fget(row, key)
    vals = finite(fget(r, key) for r in rows)
    if not math.isfinite(x) or not vals:
        return 1e6
    med = median(vals)
    scale = abs(med) if abs(med) > 1e-12 else max(abs(v) for v in vals) if vals else 1.0
    if scale <= 1e-12:
        scale = 1.0
    return x / scale


def combined_score(row: Row, rows: Sequence[Row], weights: Dict[str, float]) -> float:
    total = 0.0
    for key, w in weights.items():
        total += float(w) * norm_value(rows, row, key)
    return total


def by_config(rows: Sequence[Row]) -> Dict[str, List[Row]]:
    out: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        out[str(r["config_tag"])].append(r)
    return out


def config_stats(rows: Sequence[Row]) -> Dict[str, float]:
    return {cfg: fmean(fget(r, "selector_stat") for r in gr) for cfg, gr in by_config(rows).items()}


def top_configs(rows: Sequence[Row], n: int) -> List[str]:
    stats = config_stats(rows)
    return [cfg for cfg, _ in sorted(stats.items(), key=lambda kv: kv[1])[: max(1, n)]]


def choose_min(rows: Sequence[Row], key: str) -> Optional[Row]:
    valid = [r for r in rows if math.isfinite(fget(r, key))]
    if not valid:
        return None
    return min(valid, key=lambda r: fget(r, key))


def choose_max(rows: Sequence[Row], key: str) -> Optional[Row]:
    valid = [r for r in rows if math.isfinite(fget(r, key))]
    if not valid:
        return None
    return max(valid, key=lambda r: fget(r, key))


def topk_by(rows: Sequence[Row], key: str, k: int) -> List[Row]:
    valid = [r for r in rows if math.isfinite(fget(r, key))]
    return sorted(valid, key=lambda r: fget(r, key))[: max(1, k)]


def policy_current(rows: Sequence[Row]) -> Optional[Row]:
    cfg = top_configs(rows, 1)[0]
    return choose_min([r for r in rows if str(r["config_tag"]) == cfg], "selector_stat")


def policy_config_bestofk_oracle(rows: Sequence[Row]) -> Optional[Row]:
    # Diagnostic only: config chosen by stat, seed chosen by PSNR.
    cfg = top_configs(rows, 1)[0]
    return choose_max([r for r in rows if str(r["config_tag"]) == cfg], "eval_psnr")


def policy_oracle(rows: Sequence[Row]) -> Optional[Row]:
    return choose_max(rows, "eval_psnr")


def make_policy_top1_topk_rerank(k: int, rerank_key: str) -> Callable[[Sequence[Row]], Optional[Row]]:
    def fn(rows: Sequence[Row]) -> Optional[Row]:
        cfg = top_configs(rows, 1)[0]
        cfg_rows = [r for r in rows if str(r["config_tag"]) == cfg]
        cand = topk_by(cfg_rows, "selector_stat", k)
        return choose_min(cand, rerank_key) or choose_min(cand, "selector_stat")
    return fn


def make_policy_topn_cfg_topk_rerank(n_cfg: int, k: int, rerank_key: str) -> Callable[[Sequence[Row]], Optional[Row]]:
    def fn(rows: Sequence[Row]) -> Optional[Row]:
        cfgs = set(top_configs(rows, n_cfg))
        cand: List[Row] = []
        for cfg in cfgs:
            cfg_rows = [r for r in rows if str(r["config_tag"]) == cfg]
            cand.extend(topk_by(cfg_rows, "selector_stat", k))
        return choose_min(cand, rerank_key) or choose_min(cand, "selector_stat")
    return fn


def make_policy_combined(n_cfg: int, weights: Dict[str, float], topk_per_cfg: Optional[int] = None) -> Callable[[Sequence[Row]], Optional[Row]]:
    def fn(rows: Sequence[Row]) -> Optional[Row]:
        cfgs = set(top_configs(rows, n_cfg))
        cand: List[Row] = []
        for cfg in cfgs:
            cfg_rows = [r for r in rows if str(r["config_tag"]) == cfg]
            if topk_per_cfg is None:
                cand.extend(cfg_rows)
            else:
                cand.extend(topk_by(cfg_rows, "selector_stat", topk_per_cfg))
        valid = [r for r in cand if math.isfinite(fget(r, "selector_stat"))]
        if not valid:
            return None
        return min(valid, key=lambda r: combined_score(r, valid, weights))
    return fn


def build_policies() -> Dict[str, Callable[[Sequence[Row]], Optional[Row]]]:
    policies: Dict[str, Callable[[Sequence[Row]], Optional[Row]]] = {
        "current_config_mean_stat_seed_min_stat": policy_current,
        "diagnostic_selected_config_oracle_seed": policy_config_bestofk_oracle,
        "diagnostic_oracle_all_candidates": policy_oracle,
        "top1cfg_top2stat_rerank_lf_resid": make_policy_top1_topk_rerank(2, "lf_resid"),
        "top1cfg_top3stat_rerank_lf_resid": make_policy_top1_topk_rerank(3, "lf_resid"),
        "top1cfg_top2stat_rerank_full_resid": make_policy_top1_topk_rerank(2, "full_resid"),
        "top1cfg_top3stat_rerank_full_resid": make_policy_top1_topk_rerank(3, "full_resid"),
        "top2cfg_top2stat_rerank_lf_resid": make_policy_topn_cfg_topk_rerank(2, 2, "lf_resid"),
        "top2cfg_top3stat_rerank_lf_resid": make_policy_topn_cfg_topk_rerank(2, 3, "lf_resid"),
        "top2cfg_top3stat_rerank_full_resid": make_policy_topn_cfg_topk_rerank(2, 3, "full_resid"),
        "top2cfg_combined_stat_lf": make_policy_combined(2, {"selector_stat": 1.0, "lf_resid": 1.0}),
        "top2cfg_combined_stat_full": make_policy_combined(2, {"selector_stat": 1.0, "full_resid": 1.0}),
        "top2cfg_combined_stat_lf_full": make_policy_combined(2, {"selector_stat": 1.0, "lf_resid": 0.75, "full_resid": 0.25}),
        "top3cfg_combined_stat_lf": make_policy_combined(3, {"selector_stat": 1.0, "lf_resid": 1.0}),
        "top3cfg_top3stat_combined_stat_lf_full": make_policy_combined(3, {"selector_stat": 1.0, "lf_resid": 0.75, "full_resid": 0.25}, topk_per_cfg=3),
        "allcfg_combined_stat_lf_full": make_policy_combined(999, {"selector_stat": 1.0, "lf_resid": 0.75, "full_resid": 0.25}),
    }
    return policies


def evaluate_policy(rows: List[Row], policy_name: str, policy_fn: Callable[[Sequence[Row]], Optional[Row]]) -> List[Row]:
    out: List[Row] = []
    grouped: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        grouped[str(r["image_basename"])].append(r)
    for image, gr in sorted(grouped.items()):
        selected = policy_fn(gr)
        oracle = policy_oracle(gr)
        cfg_rank = top_configs(gr, min(5, len(set(str(r["config_tag"]) for r in gr))))
        cfg_stat = config_stats(gr)
        row: Row = {
            "policy": policy_name,
            "image_basename": image,
            "n_candidates": len(gr),
            "n_configs": len(set(str(r["config_tag"]) for r in gr)),
            "n_seeds": len(set(str(r["seed"]) for r in gr)),
            "selected_psnr": fget(selected or {}, "eval_psnr"),
            "selected_seed": (selected or {}).get("seed", ""),
            "selected_config_tag": (selected or {}).get("config_tag", ""),
            "selected_branch_family": (selected or {}).get("branch_family", ""),
            "selected_lambda": fget(selected or {}, "branch_lambda"),
            "selected_proj_start": fget(selected or {}, "branch_proj_start"),
            "selected_stat": fget(selected or {}, "selector_stat"),
            "selected_lf_resid": fget(selected or {}, "lf_resid"),
            "selected_full_resid": fget(selected or {}, "full_resid"),
            "oracle_psnr": fget(oracle or {}, "eval_psnr"),
            "oracle_seed": (oracle or {}).get("seed", ""),
            "oracle_config_tag": (oracle or {}).get("config_tag", ""),
            "oracle_branch_family": (oracle or {}).get("branch_family", ""),
            "oracle_lambda": fget(oracle or {}, "branch_lambda"),
            "oracle_proj_start": fget(oracle or {}, "branch_proj_start"),
            "regret_vs_oracle": fget(oracle or {}, "eval_psnr") - fget(selected or {}, "eval_psnr"),
            "top_config_by_stat": cfg_rank[0] if cfg_rank else "",
            "top_config_stat": cfg_stat.get(cfg_rank[0], math.nan) if cfg_rank else math.nan,
            "second_config_by_stat": cfg_rank[1] if len(cfg_rank) > 1 else "",
            "second_config_stat": cfg_stat.get(cfg_rank[1], math.nan) if len(cfg_rank) > 1 else math.nan,
            "config_stat_margin_2_minus_1": (cfg_stat.get(cfg_rank[1], math.nan) - cfg_stat.get(cfg_rank[0], math.nan)) if len(cfg_rank) > 1 else math.nan,
            "top5_configs_by_stat": ";".join(cfg_rank),
        }
        out.append(row)
    return out


def summarize(image_rows: List[Row], thresholds: Sequence[float]) -> Tuple[List[Row], List[Row]]:
    by_policy: Dict[str, List[Row]] = defaultdict(list)
    for r in image_rows:
        by_policy[str(r["policy"])].append(r)

    summary: List[Row] = []
    by_threshold: List[Row] = []
    for policy, gr in sorted(by_policy.items()):
        psnrs = [fget(r, "selected_psnr") for r in gr]
        regrets = [fget(r, "regret_vs_oracle") for r in gr]
        worst = min(gr, key=lambda r: fget(r, "selected_psnr")) if gr else {}
        summary.append(
            {
                "policy": policy,
                "n_images": len(gr),
                "psnr_mean": fmean(psnrs),
                "psnr_median": fmedian(psnrs),
                "psnr_min": fmin(psnrs),
                "psnr_max": fmax(psnrs),
                "psnr_std": fstd(psnrs),
                "mean_regret_vs_oracle": fmean(regrets),
                "max_regret_vs_oracle": fmax(regrets),
                "worst_image": worst.get("image_basename", ""),
                "worst_image_psnr": fget(worst, "selected_psnr"),
                "worst_image_oracle_psnr": fget(worst, "oracle_psnr"),
            }
        )
        for tau in thresholds:
            failures = [r for r in gr if fget(r, "selected_psnr") < tau]
            oracle_failures = [r for r in gr if fget(r, "oracle_psnr") < tau]
            selector_failures = [r for r in gr if fget(r, "oracle_psnr") >= tau and fget(r, "selected_psnr") < tau]
            by_threshold.append(
                {
                    "policy": policy,
                    "threshold_db": tau,
                    "n_images": len(gr),
                    "n_below_threshold": len(failures),
                    "n_oracle_below_threshold": len(oracle_failures),
                    "n_selector_fail_given_oracle_success": len(selector_failures),
                    "failed_images": ";".join(str(r.get("image_basename", "")) for r in failures),
                    "selector_failed_images": ";".join(str(r.get("image_basename", "")) for r in selector_failures),
                    "oracle_failed_images": ";".join(str(r.get("image_basename", "")) for r in oracle_failures),
                }
            )
    return summary, by_threshold


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate selector policy variants from trace summaries.")
    ap.add_argument("--roots_or_traces", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--selector_stat", default="post_winner_lf_mse_mean")
    ap.add_argument("--psnr_key", default="raw_psnr")
    ap.add_argument("--lf_resid_key", default="raw_noisy_lowfreq_mag_l2")
    ap.add_argument("--full_resid_key", default="raw_noisy_mag_l2")
    ap.add_argument("--thresholds", default="25,28,30")
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    traces = find_trace_files(args.roots_or_traces)
    if not traces:
        raise FileNotFoundError("No *_run_trace_summary.csv files found")
    print("Trace files:")
    for t in traces:
        print(f"  {t}")

    rows = load_rows(
        traces,
        stat_key=args.selector_stat,
        psnr_key=args.psnr_key,
        lf_resid_key=args.lf_resid_key,
        full_resid_key=args.full_resid_key,
        dedupe=bool(args.dedupe),
    )
    if not rows:
        raise RuntimeError("No usable rows loaded")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(outdir / "selector_policy_normalized_candidates.csv", rows)

    policies = build_policies()
    image_rows: List[Row] = []
    for name, fn in policies.items():
        image_rows.extend(evaluate_policy(rows, name, fn))

    summary, by_threshold = summarize(image_rows, thresholds)
    failures = [r for r in image_rows if fget(r, "selected_psnr") < max(thresholds)]

    write_csv(outdir / "selector_policy_image_level.csv", image_rows)
    write_csv(outdir / "selector_policy_summary.csv", summary)
    write_csv(outdir / "selector_policy_by_threshold.csv", by_threshold)
    write_csv(outdir / "selector_policy_failures.csv", failures)

    print(f"\nLoaded rows: {len(rows)}")
    print(f"Images: {sorted(set(str(r['image_basename']) for r in rows))}")
    print(f"Configs: {len(set(str(r['config_tag']) for r in rows))}")
    print(f"Seeds: {sorted(set(str(r['seed']) for r in rows))}")
    print(f"\nWrote selector policy sweep to: {outdir}")
    print("\nSummary:")
    for row in sorted(summary, key=lambda r: (fget(r, "psnr_min"), fget(r, "psnr_mean")), reverse=True):
        print(row)
    print("\nThreshold diagnostics:")
    for row in by_threshold:
        if abs(float(row["threshold_db"]) - 28.0) < 1e-9:
            print(row)


if __name__ == "__main__":
    main()
