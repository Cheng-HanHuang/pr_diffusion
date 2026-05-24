#!/usr/bin/env python3
"""Extended selector-policy sweep for 00013-style seed-ranking failures.

This script extends `simulate_selector_policy_variants.py` with policies motivated
by the 00013 margin-recovery result:

- wider top-k seed pools, because the best 00013 seed was rank 4 by
  post_winner_lf_mse_mean inside the selected config;
- residual reranking inside top-k pools;
- rank-aggregation policies;
- explicit risk diagnostics that say when the current selector should defer and
  request more seeds/configs instead of finalizing.

The script is offline: it consumes `*_run_trace_summary.csv` and uses PSNR only
for evaluation, never for executable selection decisions.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Import helper functions from the base selector simulator in the same directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import simulate_selector_policy_variants as base  # type: ignore

Row = Dict[str, object]


def rank_rows(rows: Sequence[Row], key: str, ascending: bool = True) -> Dict[int, int]:
    valid = [(i, base.fget(r, key)) for i, r in enumerate(rows)]
    valid = [(i, v) for i, v in valid if math.isfinite(v)]
    valid.sort(key=lambda iv: iv[1], reverse=not ascending)
    return {i: rank + 1 for rank, (i, _) in enumerate(valid)}


def selected_config_rows(rows: Sequence[Row], n_cfg: int = 1) -> List[Row]:
    cfgs = set(base.top_configs(rows, n_cfg))
    return [r for r in rows if str(r["config_tag"]) in cfgs]


def choose_by_rank_sum(rows: Sequence[Row], keys: Sequence[Tuple[str, bool]], weights: Optional[Sequence[float]] = None) -> Optional[Row]:
    valid = list(rows)
    if not valid:
        return None
    weights = list(weights) if weights is not None else [1.0] * len(keys)
    rank_maps = [rank_rows(valid, key, ascending=asc) for key, asc in keys]
    best = None
    best_score = math.inf
    for i, r in enumerate(valid):
        score = 0.0
        ok = True
        for w, ranks in zip(weights, rank_maps):
            if i not in ranks:
                ok = False
                break
            score += float(w) * ranks[i]
        if ok and score < best_score:
            best_score = score
            best = r
    return best


def make_top1cfg_topk_rerank(k: int, rerank_key: str) -> Callable[[Sequence[Row]], Optional[Row]]:
    def policy(rows: Sequence[Row]) -> Optional[Row]:
        cfg_rows = selected_config_rows(rows, 1)
        cand = base.topk_by(cfg_rows, "selector_stat", k)
        return base.choose_min(cand, rerank_key) or base.choose_min(cand, "selector_stat")
    return policy


def make_top2cfg_topk_rerank(k: int, rerank_key: str) -> Callable[[Sequence[Row]], Optional[Row]]:
    def policy(rows: Sequence[Row]) -> Optional[Row]:
        cand: List[Row] = []
        for cfg in base.top_configs(rows, 2):
            cfg_rows = [r for r in rows if str(r["config_tag"]) == cfg]
            cand.extend(base.topk_by(cfg_rows, "selector_stat", k))
        return base.choose_min(cand, rerank_key) or base.choose_min(cand, "selector_stat")
    return policy


def make_top1cfg_topk_second_best_resid(k: int, resid_key: str) -> Callable[[Sequence[Row]], Optional[Row]]:
    """Choose the second-lowest residual among top-k stat seeds.

    This is not proposed as a final method.  It is a diagnostic policy that asks
    whether the absolute residual minimum is a misleading overfit candidate.
    """
    def policy(rows: Sequence[Row]) -> Optional[Row]:
        cfg_rows = selected_config_rows(rows, 1)
        cand = base.topk_by(cfg_rows, "selector_stat", k)
        valid = [r for r in cand if math.isfinite(base.fget(r, resid_key))]
        valid.sort(key=lambda r: base.fget(r, resid_key))
        if len(valid) >= 2:
            return valid[1]
        return valid[0] if valid else base.choose_min(cand, "selector_stat")
    return policy


def make_rank_agg_policy(n_cfg: int, topk_per_cfg: int, stat_w: float, lf_w: float, full_w: float) -> Callable[[Sequence[Row]], Optional[Row]]:
    def policy(rows: Sequence[Row]) -> Optional[Row]:
        cand: List[Row] = []
        for cfg in base.top_configs(rows, n_cfg):
            cfg_rows = [r for r in rows if str(r["config_tag"]) == cfg]
            cand.extend(base.topk_by(cfg_rows, "selector_stat", topk_per_cfg))
        return choose_by_rank_sum(
            cand,
            keys=[("selector_stat", True), ("lf_resid", True), ("full_resid", True)],
            weights=[stat_w, lf_w, full_w],
        )
    return policy


def make_stat_band_policy(n_cfg: int, band_rel: float, rerank_key: str) -> Callable[[Sequence[Row]], Optional[Row]]:
    """Keep candidates whose stat is within a relative band of config-min stat, then rerank.

    Motivation: the 00013 good seed can be slightly worse by trajectory statistic
    than the minimum-stat seed.  A band lets nearby seeds compete by final
    residual-like features.
    """
    def policy(rows: Sequence[Row]) -> Optional[Row]:
        cand: List[Row] = []
        for cfg in base.top_configs(rows, n_cfg):
            cfg_rows = [r for r in rows if str(r["config_tag"]) == cfg]
            stats = [base.fget(r, "selector_stat") for r in cfg_rows]
            stats = [x for x in stats if math.isfinite(x)]
            if not stats:
                continue
            best = min(stats)
            cutoff = best * (1.0 + float(band_rel))
            cand.extend([r for r in cfg_rows if base.fget(r, "selector_stat") <= cutoff])
        return base.choose_min(cand, rerank_key) or base.choose_min(cand, "selector_stat")
    return policy


def make_sharp_safe_policy() -> Callable[[Sequence[Row]], Optional[Row]]:
    """Prefer sharp >28-oriented branches, fallback to stable-safe branch class.

    This is heuristic and non-oracle.  It uses branch metadata only:
    - sharp: lambda in [0.08, 0.20] and proj_start >= 400
    - safe:  lambda around 0.15, proj_start around 350, soft=8, based on the
      observed stable >25 behavior.
    """
    def is_sharp(r: Row) -> bool:
        lam = base.fget(r, "branch_lambda")
        ps = base.fget(r, "branch_proj_start")
        return math.isfinite(lam) and 0.08 <= lam <= 0.20 and math.isfinite(ps) and ps >= 400

    def is_safe(r: Row) -> bool:
        lam = base.fget(r, "branch_lambda")
        ps = base.fget(r, "branch_proj_start")
        soft = base.fget(r, "branch_soft")
        return math.isfinite(lam) and abs(lam - 0.15) < 1e-9 and abs(ps - 350) < 1e-9 and abs(soft - 8) < 1e-9

    def policy(rows: Sequence[Row]) -> Optional[Row]:
        sharp = [r for r in rows if is_sharp(r)]
        if sharp:
            # Use top-4-by-stat within sharp candidates and rerank by LF residual.
            cand = base.topk_by(sharp, "selector_stat", 4)
            return base.choose_min(cand, "lf_resid") or base.choose_min(cand, "selector_stat")
        safe = [r for r in rows if is_safe(r)]
        if safe:
            return base.choose_min(safe, "selector_stat")
        return base.policy_current(rows)
    return policy


def extended_policies() -> Dict[str, Callable[[Sequence[Row]], Optional[Row]]]:
    policies: Dict[str, Callable[[Sequence[Row]], Optional[Row]]] = {
        "current_config_mean_stat_seed_min_stat": base.policy_current,
        "diagnostic_selected_config_oracle_seed": base.policy_config_bestofk_oracle,
        "diagnostic_oracle_all_candidates": base.policy_oracle,
    }

    # Wider top-k policies.  k=4 is specifically motivated by 00013: the best
    # seed was rank 4 by post_winner_lf_mse_mean in the selected config.
    for k in [4, 5, 6, 8]:
        policies[f"top1cfg_top{k}stat_rerank_lf_resid"] = make_top1cfg_topk_rerank(k, "lf_resid")
        policies[f"top1cfg_top{k}stat_rerank_full_resid"] = make_top1cfg_topk_rerank(k, "full_resid")
        policies[f"top1cfg_top{k}stat_second_lf_resid"] = make_top1cfg_topk_second_best_resid(k, "lf_resid")
        policies[f"top2cfg_top{k}stat_rerank_lf_resid"] = make_top2cfg_topk_rerank(k, "lf_resid")
        policies[f"top2cfg_top{k}stat_rerank_full_resid"] = make_top2cfg_topk_rerank(k, "full_resid")

    # Rank aggregation over stat + residuals.
    policies["top1cfg_top4_rankagg_stat1_lf1_full0p25"] = make_rank_agg_policy(1, 4, 1.0, 1.0, 0.25)
    policies["top1cfg_top5_rankagg_stat1_lf1_full0p25"] = make_rank_agg_policy(1, 5, 1.0, 1.0, 0.25)
    policies["top1cfg_top6_rankagg_stat1_lf1_full0p25"] = make_rank_agg_policy(1, 6, 1.0, 1.0, 0.25)
    policies["top2cfg_top4_rankagg_stat1_lf1_full0p25"] = make_rank_agg_policy(2, 4, 1.0, 1.0, 0.25)
    policies["top3cfg_top4_rankagg_stat1_lf1_full0p25"] = make_rank_agg_policy(3, 4, 1.0, 1.0, 0.25)

    # Stat-band policies: let near-tied stat candidates compete by residual.
    for band in [0.02, 0.04, 0.06, 0.08]:
        tag = str(band).replace(".", "p")
        policies[f"top1cfg_statband{tag}_rerank_lf_resid"] = make_stat_band_policy(1, band, "lf_resid")
        policies[f"top1cfg_statband{tag}_rerank_full_resid"] = make_stat_band_policy(1, band, "full_resid")
        policies[f"top2cfg_statband{tag}_rerank_lf_resid"] = make_stat_band_policy(2, band, "lf_resid")

    policies["sharp_safe_branch_heuristic"] = make_sharp_safe_policy()
    return policies


def risk_diagnostics(rows: List[Row], thresholds: Sequence[float]) -> List[Row]:
    grouped: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        grouped[str(r["image_basename"])].append(r)

    out: List[Row] = []
    for image, gr in sorted(grouped.items()):
        cfg_rank = base.top_configs(gr, min(5, len(set(str(r["config_tag"]) for r in gr))))
        cfg_stat = base.config_stats(gr)
        top_cfg = cfg_rank[0]
        top_cfg_rows = [r for r in gr if str(r["config_tag"]) == top_cfg]
        stat_sorted = sorted(top_cfg_rows, key=lambda r: base.fget(r, "selector_stat"))
        lf_sorted = sorted(top_cfg_rows, key=lambda r: base.fget(r, "lf_resid"))
        full_sorted = sorted(top_cfg_rows, key=lambda r: base.fget(r, "full_resid"))
        current = stat_sorted[0] if stat_sorted else None
        oracle = base.policy_oracle(gr)
        config_oracle = base.choose_max(top_cfg_rows, "eval_psnr")
        stat1 = base.fget(stat_sorted[0], "selector_stat") if len(stat_sorted) >= 1 else math.nan
        stat2 = base.fget(stat_sorted[1], "selector_stat") if len(stat_sorted) >= 2 else math.nan
        seed_margin_abs = stat2 - stat1 if math.isfinite(stat1) and math.isfinite(stat2) else math.nan
        seed_margin_rel = seed_margin_abs / abs(stat1) if math.isfinite(seed_margin_abs) and abs(stat1) > 1e-12 else math.nan
        cfg_margin_abs = cfg_stat.get(cfg_rank[1], math.nan) - cfg_stat.get(cfg_rank[0], math.nan) if len(cfg_rank) > 1 else math.nan
        cfg_margin_rel = cfg_margin_abs / abs(cfg_stat.get(cfg_rank[0], math.nan)) if len(cfg_rank) > 1 and abs(cfg_stat.get(cfg_rank[0], math.nan)) > 1e-12 else math.nan

        best_stat_seed = str((stat_sorted[0] if stat_sorted else {}).get("seed", ""))
        best_lf_seed = str((lf_sorted[0] if lf_sorted else {}).get("seed", ""))
        best_full_seed = str((full_sorted[0] if full_sorted else {}).get("seed", ""))
        disagreement = int(len({best_stat_seed, best_lf_seed, best_full_seed}) > 1)

        # A conservative non-oracle defer rule.  These thresholds are meant for
        # diagnosis, not yet final algorithm promotion.
        defer_reasons: List[str] = []
        if math.isfinite(cfg_margin_rel) and cfg_margin_rel < 0.02:
            defer_reasons.append("small_config_margin")
        if math.isfinite(seed_margin_rel) and seed_margin_rel < 0.02:
            defer_reasons.append("small_seed_margin")
        if disagreement:
            defer_reasons.append("stat_residual_disagreement")
        if base.fget(current or {}, "eval_psnr") < max(thresholds):
            # Evaluation-only label, written separately so it is not confused as
            # an executable rule.
            eval_current_fails_max_threshold = 1
        else:
            eval_current_fails_max_threshold = 0

        row: Row = {
            "image_basename": image,
            "top_config": top_cfg,
            "top_config_stat": cfg_stat.get(top_cfg, math.nan),
            "second_config": cfg_rank[1] if len(cfg_rank) > 1 else "",
            "second_config_stat": cfg_stat.get(cfg_rank[1], math.nan) if len(cfg_rank) > 1 else math.nan,
            "config_margin_abs": cfg_margin_abs,
            "config_margin_rel": cfg_margin_rel,
            "best_stat_seed": best_stat_seed,
            "best_lf_resid_seed": best_lf_seed,
            "best_full_resid_seed": best_full_seed,
            "seed_margin_abs": seed_margin_abs,
            "seed_margin_rel": seed_margin_rel,
            "stat_residual_disagreement": disagreement,
            "defer_by_nonoracle_risk_rule": int(bool(defer_reasons)),
            "defer_reasons": ";".join(defer_reasons),
            "current_selected_psnr_eval": base.fget(current or {}, "eval_psnr"),
            "selected_config_oracle_psnr_eval": base.fget(config_oracle or {}, "eval_psnr"),
            "oracle_all_psnr_eval": base.fget(oracle or {}, "eval_psnr"),
            "eval_current_fails_max_threshold": eval_current_fails_max_threshold,
            "top5_configs_by_stat": ";".join(cfg_rank),
        }
        for tau in thresholds:
            row[f"current_success_ge_{tau:g}"] = base.fget(current or {}, "eval_psnr") >= tau
            row[f"selected_config_oracle_success_ge_{tau:g}"] = base.fget(config_oracle or {}, "eval_psnr") >= tau
            row[f"oracle_success_ge_{tau:g}"] = base.fget(oracle or {}, "eval_psnr") >= tau
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extended selector-policy sweep for 00013-style failures.")
    ap.add_argument("--roots_or_traces", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--selector_stat", default="post_winner_lf_mse_mean")
    ap.add_argument("--psnr_key", default="raw_psnr")
    ap.add_argument("--lf_resid_key", default="raw_noisy_lowfreq_mag_l2")
    ap.add_argument("--full_resid_key", default="raw_noisy_mag_l2")
    ap.add_argument("--thresholds", default="25,28,30")
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    traces = base.find_trace_files(args.roots_or_traces)
    if not traces:
        raise FileNotFoundError("No *_run_trace_summary.csv files found")
    print("Trace files:")
    for t in traces:
        print(f"  {t}")

    rows = base.load_rows(
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

    policies = extended_policies()
    image_rows: List[Row] = []
    for name, fn in policies.items():
        image_rows.extend(base.evaluate_policy(rows, name, fn))

    summary, by_threshold = base.summarize(image_rows, thresholds)
    failures = [r for r in image_rows if base.fget(r, "selected_psnr") < max(thresholds)]
    risk_rows = risk_diagnostics(rows, thresholds)

    base.write_csv(outdir / "extended_selector_policy_image_level.csv", image_rows)
    base.write_csv(outdir / "extended_selector_policy_summary.csv", summary)
    base.write_csv(outdir / "extended_selector_policy_by_threshold.csv", by_threshold)
    base.write_csv(outdir / "extended_selector_policy_failures.csv", failures)
    base.write_csv(outdir / "extended_selector_risk_diagnostics.csv", risk_rows)

    print(f"\nLoaded rows: {len(rows)}")
    print(f"Images: {sorted(set(str(r['image_basename']) for r in rows))}")
    print(f"Configs: {len(set(str(r['config_tag']) for r in rows))}")
    print(f"Seeds: {sorted(set(str(r['seed']) for r in rows))}")
    print(f"\nWrote extended selector sweep to: {outdir}")
    print("\nBest summaries by min PSNR:")
    for row in sorted(summary, key=lambda r: (base.fget(r, "psnr_min"), base.fget(r, "psnr_mean")), reverse=True)[:20]:
        print(row)
    print("\nRisk diagnostics:")
    for row in risk_rows:
        print(row)


if __name__ == "__main__":
    main()
