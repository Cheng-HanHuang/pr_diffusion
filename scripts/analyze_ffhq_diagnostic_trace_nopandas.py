#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import glob
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


def open_text(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return open(path, "r", newline="", encoding="utf-8")


def read_csv(path: str):
    with open_text(path) as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[WARN] no rows for {path}")
        return
    fieldnames, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fget(r, k, default=math.nan):
    try:
        v = r.get(k, "")
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default


def finite(xs):
    return [x for x in xs if math.isfinite(x)]


def safe_mean(xs):
    xs = finite(xs)
    return mean(xs) if xs else math.nan


def safe_median(xs):
    xs = finite(xs)
    return median(xs) if xs else math.nan


def safe_min(xs):
    xs = finite(xs)
    return min(xs) if xs else math.nan


def safe_max(xs):
    xs = finite(xs)
    return max(xs) if xs else math.nan


def safe_std(xs):
    xs = finite(xs)
    return stdev(xs) if len(xs) > 1 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Root containing diagnostic_* folders")
    p.add_argument("--out_prefix", default=None)
    p.add_argument("--primary_alignment", default="raw")
    args = p.parse_args()

    root = Path(args.root)
    out_prefix = Path(args.out_prefix) if args.out_prefix else root / "diagnostic"

    step_files = sorted(glob.glob(str(root / "**" / "step_trace.csv.gz"), recursive=True))
    final_files = sorted(glob.glob(str(root / "**" / "final_metrics.csv"), recursive=True))
    cand_files = sorted(glob.glob(str(root / "**" / "candidate_trace.csv.gz"), recursive=True))
    print("step files:", len(step_files))
    for f in step_files:
        print(" ", f)
    print("final files:", len(final_files))
    for f in final_files:
        print(" ", f)
    print("candidate files:", len(cand_files))
    for f in cand_files:
        print(" ", f)

    if not step_files or not final_files:
        raise RuntimeError("Need at least one step_trace.csv.gz and one final_metrics.csv")

    step_rows = []
    for path in step_files:
        for r in read_csv(path):
            r["source_file"] = path
            step_rows.append(r)

    final_rows = []
    for path in final_files:
        for r in read_csv(path):
            r["source_file"] = path
            final_rows.append(r)

    # Count candidate rows without keeping all candidate rows in memory.
    candidate_counts = defaultdict(int)
    candidate_source_counts = defaultdict(lambda: defaultdict(int))
    if cand_files:
        for path in cand_files:
            with open_text(path) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    run_id = r.get("run_id", "")
                    candidate_counts[run_id] += 1
                    candidate_source_counts[run_id][r.get("candidate_source", "")] += 1

    # Aggregate step-level trace features per run.
    by_run = defaultdict(list)
    for r in step_rows:
        by_run[r.get("run_id", "")].append(r)

    trace_summary = []
    for run_id, gr in sorted(by_run.items()):
        gr.sort(key=lambda r: int(float(r.get("step_index", 0))))
        pre = [r for r in gr if int(float(r.get("pre_projection_stage", 0))) == 1]
        post = [r for r in gr if int(float(r.get("pre_projection_stage", 0))) == 0]
        rep = gr[0]

        def vals(rows, key):
            return [fget(r, key) for r in rows]

        adaptive_used = [fget(r, "adaptive_used", 0.0) for r in gr]
        winner_is_lf = [fget(r, "winner_is_lf_best", 0.0) for r in gr]
        winner_memory = [1.0 if r.get("winner_source") == "memory" else 0.0 for r in gr]
        winner_fresh = [1.0 if r.get("winner_source") == "fresh" else 0.0 for r in gr]
        winner_legacy = [1.0 if r.get("winner_source") == "legacy_eps_prev" else 0.0 for r in gr]

        row = {
            "run_id": run_id,
            "config_tag": rep.get("config_tag", ""),
            "image_basename": rep.get("image_basename", ""),
            "seed": rep.get("seed", ""),
            "score_mode": rep.get("score_mode", ""),
            "score_reg_lambda_schedule": rep.get("score_reg_lambda_schedule", ""),
            "adaptive_s2_margin": rep.get("adaptive_s2_margin", ""),
            "noise_memory_k": rep.get("noise_memory_k", ""),
            "n_steps": len(gr),
            "n_pre_steps": len(pre),
            "n_post_steps": len(post),
            "candidate_rows": candidate_counts.get(run_id, 0),
            "winner_is_lf_best_frac_all": safe_mean(winner_is_lf),
            "winner_is_lf_best_frac_pre": safe_mean([fget(r, "winner_is_lf_best", 0.0) for r in pre]),
            "adaptive_used_frac_all": safe_mean(adaptive_used),
            "adaptive_used_frac_pre": safe_mean([fget(r, "adaptive_used", 0.0) for r in pre]),
            "winner_memory_frac_all": safe_mean(winner_memory),
            "winner_memory_frac_pre": safe_mean([1.0 if r.get("winner_source") == "memory" else 0.0 for r in pre]),
            "winner_fresh_frac_all": safe_mean(winner_fresh),
            "winner_legacy_frac_all": safe_mean(winner_legacy),
            "pre_winner_lf_mse_mean": safe_mean(vals(pre, "winner_lf_mse_vs_observation")),
            "pre_winner_lf_mse_median": safe_median(vals(pre, "winner_lf_mse_vs_observation")),
            "pre_winner_lf_mse_min": safe_min(vals(pre, "winner_lf_mse_vs_observation")),
            "pre_winner_lf_mse_max": safe_max(vals(pre, "winner_lf_mse_vs_observation")),
            "pre_winner_full_mse_mean": safe_mean(vals(pre, "winner_full_mse_vs_observation")),
            "pre_lf_mse_margin_mean": safe_mean(vals(pre, "lf_mse_margin_best_to_second")),
            "pre_lf_mse_margin_median": safe_median(vals(pre, "lf_mse_margin_best_to_second")),
            "pre_lf_mse_margin_min": safe_min(vals(pre, "lf_mse_margin_best_to_second")),
            "pre_lf_mse_margin_max": safe_max(vals(pre, "lf_mse_margin_best_to_second")),
            "pre_lf_l2_margin_mean": safe_mean(vals(pre, "lf_l2_margin_best_to_second")),
            "pre_final_score_margin_mean": safe_mean(vals(pre, "final_score_margin_best_to_second")),
            "post_winner_lf_mse_mean": safe_mean(vals(post, "winner_lf_mse_vs_observation")),
            "post_lf_mse_margin_mean": safe_mean(vals(post, "lf_mse_margin_best_to_second")),
            "last_winner_lf_mse": fget(gr[-1], "winner_lf_mse_vs_observation"),
            "last_winner_full_mse": fget(gr[-1], "winner_full_mse_vs_observation"),
            "last_lf_mse_margin": fget(gr[-1], "lf_mse_margin_best_to_second"),
        }
        for source, cnt in sorted(candidate_source_counts.get(run_id, {}).items()):
            row[f"candidate_source_count_{source}"] = cnt
        trace_summary.append(row)

    # Merge final raw/rot/resolve metrics into trace summary.
    final_by_run_align = {(r.get("run_id", ""), r.get("alignment_mode", "")): r for r in final_rows}
    merged = []
    for r in trace_summary:
        out = dict(r)
        for align in sorted(set(fr.get("alignment_mode", "") for fr in final_rows)):
            fr = final_by_run_align.get((r["run_id"], align))
            if fr:
                out[f"{align}_psnr"] = fget(fr, "psnr")
                out[f"{align}_ssim"] = fget(fr, "ssim")
                out[f"{align}_lpips"] = fget(fr, "lpips")
                out[f"{align}_noisy_lowfreq_mag_l2"] = fget(fr, "noisy_lowfreq_mag_l2")
                out[f"{align}_noisy_mag_l2"] = fget(fr, "noisy_mag_l2")
        merged.append(out)

    # Config/image-level best-of-k using primary alignment.
    primary = [r for r in final_rows if r.get("alignment_mode") == args.primary_alignment]
    by_cfg_img = defaultdict(list)
    for r in primary:
        by_cfg_img[(r.get("config_tag", ""), r.get("image_basename", ""))].append(r)

    image_best = []
    for (tag, img), gr in sorted(by_cfg_img.items()):
        best = max(gr, key=lambda r: fget(r, "psnr"))
        psnrs = [fget(r, "psnr") for r in gr]
        image_best.append({
            "config_tag": tag,
            "image_basename": img,
            "n_seeds": len(set(r.get("seed", "") for r in gr)),
            "best_seed": best.get("seed", ""),
            "bestofk_psnr": fget(best, "psnr"),
            "bestofk_ssim_at_best_psnr": fget(best, "ssim"),
            "bestofk_lpips_at_best_psnr": fget(best, "lpips"),
            "psnr_mean_over_seeds": safe_mean(psnrs),
            "psnr_min_over_seeds": safe_min(psnrs),
        })

    by_cfg = defaultdict(list)
    for r in image_best:
        by_cfg[r["config_tag"]].append(r)
    config_summary = []
    for tag, gr in sorted(by_cfg.items()):
        psnr = [fget(r, "bestofk_psnr") for r in gr]
        worst = min(gr, key=lambda r: fget(r, "bestofk_psnr"))
        config_summary.append({
            "config_tag": tag,
            "alignment_mode": args.primary_alignment,
            "n_images": len(gr),
            "bestofk_psnr_mean": safe_mean(psnr),
            "bestofk_psnr_median": safe_median(psnr),
            "bestofk_psnr_min": safe_min(psnr),
            "bestofk_psnr_max": safe_max(psnr),
            "bestofk_psnr_std": safe_std(psnr),
            "n_images_below20": sum(1 for x in psnr if x < 20),
            "n_images_below25": sum(1 for x in psnr if x < 25),
            "worst_image": worst.get("image_basename", ""),
            "worst_image_bestofk_psnr": fget(worst, "bestofk_psnr"),
        })

    write_csv(Path(str(out_prefix) + "_run_trace_summary.csv"), merged)
    write_csv(Path(str(out_prefix) + "_image_bestofk.csv"), image_best)
    write_csv(Path(str(out_prefix) + "_config_summary.csv"), config_summary)

    print("\nConfig summary:")
    cols = [
        "config_tag", "alignment_mode", "n_images", "bestofk_psnr_mean",
        "bestofk_psnr_median", "bestofk_psnr_min", "n_images_below20",
        "n_images_below25", "worst_image", "worst_image_bestofk_psnr",
    ]
    print(",".join(cols))
    for r in config_summary:
        print(",".join(str(r.get(c, "")) for c in cols))

    print("\nSaved:")
    print(" ", Path(str(out_prefix) + "_run_trace_summary.csv"))
    print(" ", Path(str(out_prefix) + "_image_bestofk.csv"))
    print(" ", Path(str(out_prefix) + "_config_summary.csv"))


if __name__ == "__main__":
    main()
