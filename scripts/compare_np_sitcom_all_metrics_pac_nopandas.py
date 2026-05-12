#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


TAG_TO_SIGMA = {
    "sigma000": 0.00,
    "sigma0001": 0.001,
    "sigma001": 0.01,
    "sigma005": 0.05,
    "sigma010": 0.10,
    "sigma020": 0.20,
    "sigma050": 0.50,
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")

    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fnum(x):
    if x is None or x == "" or x == "NA":
        return None
    return float(x)


def basename_key(path_or_name: str) -> str:
    # Strip directory and also strip SITCOM numeric prefix if present:
    # 00000_ILSVRC2012_val_00001436.JPEG -> ILSVRC2012_val_00001436.JPEG
    b = os.path.basename(path_or_name)
    if "_" in b and b[:5].isdigit():
        return b[6:]
    return b


def discover_np_run_csvs(np_root: Path) -> dict[float, Path]:
    """Find one NP run-level CSV per sigma tag."""
    out = {}
    for tag, sigma in TAG_TO_SIGMA.items():
        d = np_root / tag
        if not d.exists():
            print(f"[warn] Missing NP dir for {tag}: {d}")
            continue

        candidates = sorted(d.rglob("*run_level*.csv"), key=lambda p: (p.stat().st_size, p.name), reverse=True)
        if not candidates:
            print(f"[warn] No run_level CSV found under {d}")
            continue

        out[sigma] = candidates[0]
        print(f"[info] NP sigma={sigma:g}: {candidates[0]}")
    return out


def load_np_run_level(np_root: Path) -> list[dict]:
    csvs = discover_np_run_csvs(np_root)
    rows = []

    for sigma, path in sorted(csvs.items()):
        for r in read_csv(path):
            align = r.get("alignment_mode", "")
            if align not in {"raw", "rot180", "resolve"}:
                continue
            rows.append({
                "method": "np",
                "sigma": sigma,
                "alignment_mode": align,
                "image_key": basename_key(r["image_basename"]),
                "image_basename": os.path.basename(r["image_basename"]),
                "run_id": r.get("seed", ""),
                "psnr": fnum(r.get("psnr")),
                "ssim": fnum(r.get("ssim")),
                "lpips": fnum(r.get("lpips")),
                "runtime_s": fnum(r.get("runtime_s")),
                "source_csv": str(path),
            })

    if not rows:
        raise RuntimeError(f"No NP run-level rows found under {np_root}")
    return rows


def load_sitcom_run_level(path: Path) -> list[dict]:
    rows = []
    for r in read_csv(path):
        align = r.get("alignment_mode", "")
        if align not in {"raw", "rot180", "flip_best", "flip_by_psnr"}:
            continue
        rows.append({
            "method": "sitcom_ode",
            "sigma": float(r["sigma"]),
            "alignment_mode": align,
            "image_index": int(r["image_index"]),
            "image_key": basename_key(r["image_basename"]),
            "image_basename": r["image_basename"],
            "run_id": r.get("run", ""),
            "psnr": fnum(r.get("psnr")),
            "ssim": fnum(r.get("ssim")),
            "lpips": fnum(r.get("lpips")),
            "sample_path": r.get("sample_path", ""),
        })

    if not rows:
        raise RuntimeError(f"No SITCOM run-level rows found in {path}")
    return rows


def add_np_flip_best(np_rows: list[dict]) -> list[dict]:
    """Add metric-wise flip_best rows from raw/rot180 NP run-level rows.

    This assumes raw and rot180 rows share sigma, image, and seed/run_id.
    """
    out = list(np_rows)

    by_key = defaultdict(dict)
    for r in np_rows:
        if r["alignment_mode"] in {"raw", "rot180"}:
            key = (r["sigma"], r["image_key"], r["run_id"])
            by_key[key][r["alignment_mode"]] = r

    for key, pair in by_key.items():
        if "raw" not in pair or "rot180" not in pair:
            continue
        raw = pair["raw"]
        rot = pair["rot180"]

        out.append({
            "method": "np",
            "sigma": raw["sigma"],
            "alignment_mode": "flip_best",
            "image_key": raw["image_key"],
            "image_basename": raw["image_basename"],
            "run_id": raw["run_id"],
            "psnr": max(raw["psnr"], rot["psnr"]),
            "ssim": max(raw["ssim"], rot["ssim"]),
            "lpips": min(raw["lpips"], rot["lpips"]),
            "runtime_s": raw["runtime_s"],
            "source_csv": raw.get("source_csv", ""),
        })

        # Physically consistent orientation chosen by PSNR.
        chosen = raw if raw["psnr"] >= rot["psnr"] else rot
        out.append({
            **chosen,
            "alignment_mode": "flip_by_psnr",
        })

    return out


def summarize_image_level(run_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in run_rows:
        key = (r["method"], r["sigma"], r["alignment_mode"], r["image_key"], r["image_basename"])
        groups[key].append(r)

    rows = []
    for key, g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        method, sigma, alignment, image_key, image_basename = key
        psnr_vals = [r["psnr"] for r in g if r["psnr"] is not None]
        ssim_vals = [r["ssim"] for r in g if r["ssim"] is not None]
        lpips_vals = [r["lpips"] for r in g if r["lpips"] is not None]
        runtime_vals = [r["runtime_s"] for r in g if r.get("runtime_s") is not None]

        rows.append({
            "method": method,
            "sigma": sigma,
            "alignment_mode": alignment,
            "image_key": image_key,
            "image_basename": image_basename,
            "n_runs": len(g),
            "psnr_mean": mean(psnr_vals),
            "psnr_median": median(psnr_vals),
            "psnr_best": max(psnr_vals),
            "ssim_mean": mean(ssim_vals),
            "ssim_median": median(ssim_vals),
            "ssim_best": max(ssim_vals),
            "lpips_mean": mean(lpips_vals),
            "lpips_median": median(lpips_vals),
            "lpips_best": min(lpips_vals),
            "runtime_s_mean": mean(runtime_vals) if runtime_vals else "",
        })
    return rows


def summarize_condition_level(image_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in image_rows:
        key = (r["method"], r["sigma"], r["alignment_mode"])
        groups[key].append(r)

    rows = []
    for key, g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        method, sigma, alignment = key
        rows.append({
            "method": method,
            "sigma": sigma,
            "alignment_mode": alignment,
            "n_images": len(g),
            "psnr_best_mean": mean(float(r["psnr_best"]) for r in g),
            "psnr_best_median": median(float(r["psnr_best"]) for r in g),
            "ssim_best_mean": mean(float(r["ssim_best"]) for r in g),
            "ssim_best_median": median(float(r["ssim_best"]) for r in g),
            "lpips_best_mean": mean(float(r["lpips_best"]) for r in g),
            "lpips_best_median": median(float(r["lpips_best"]) for r in g),
            "psnr_mean_mean": mean(float(r["psnr_mean"]) for r in g),
            "ssim_mean_mean": mean(float(r["ssim_mean"]) for r in g),
            "lpips_mean_mean": mean(float(r["lpips_mean"]) for r in g),
            "runtime_s_mean": mean(float(r["runtime_s_mean"]) for r in g if r["runtime_s_mean"] != "") if any(r["runtime_s_mean"] != "" for r in g) else "",
        })
    return rows


def compare_methods(image_rows: list[dict], alignments=("raw", "flip_best")) -> list[dict]:
    by_key = {}
    for r in image_rows:
        if r["alignment_mode"] not in alignments:
            continue
        key = (r["method"], float(r["sigma"]), r["alignment_mode"], r["image_key"])
        by_key[key] = r

    # Match images present in both methods.
    out = []
    sigmas = sorted({float(r["sigma"]) for r in image_rows})
    image_keys = sorted({r["image_key"] for r in image_rows})

    for sigma in sigmas:
        for alignment in alignments:
            for image_key in image_keys:
                s = by_key.get(("sitcom_ode", sigma, alignment, image_key))
                n = by_key.get(("np", sigma, alignment, image_key))
                if s is None or n is None:
                    continue
                out.append({
                    "sigma": sigma,
                    "alignment_mode": alignment,
                    "image_key": image_key,
                    "sitcom_psnr_best": s["psnr_best"],
                    "np_psnr_best": n["psnr_best"],
                    "delta_psnr_best": float(n["psnr_best"]) - float(s["psnr_best"]),
                    "sitcom_ssim_best": s["ssim_best"],
                    "np_ssim_best": n["ssim_best"],
                    "delta_ssim_best": float(n["ssim_best"]) - float(s["ssim_best"]),
                    "sitcom_lpips_best": s["lpips_best"],
                    "np_lpips_best": n["lpips_best"],
                    "delta_lpips_best": float(n["lpips_best"]) - float(s["lpips_best"]),
                    "sitcom_psnr_mean": s["psnr_mean"],
                    "np_psnr_mean": n["psnr_mean"],
                    "delta_psnr_mean": float(n["psnr_mean"]) - float(s["psnr_mean"]),
                    "sitcom_ssim_mean": s["ssim_mean"],
                    "np_ssim_mean": n["ssim_mean"],
                    "delta_ssim_mean": float(n["ssim_mean"]) - float(s["ssim_mean"]),
                    "sitcom_lpips_mean": s["lpips_mean"],
                    "np_lpips_mean": n["lpips_mean"],
                    "delta_lpips_mean": float(n["lpips_mean"]) - float(s["lpips_mean"]),
                })
    return out


def compare_condition(per_image_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in per_image_rows:
        groups[(float(r["sigma"]), r["alignment_mode"])].append(r)

    rows = []
    for key, g in sorted(groups.items()):
        sigma, alignment = key
        rows.append({
            "sigma": sigma,
            "alignment_mode": alignment,
            "n_images": len(g),
            "delta_psnr_best_mean": mean(float(r["delta_psnr_best"]) for r in g),
            "delta_psnr_best_median": median(float(r["delta_psnr_best"]) for r in g),
            "psnr_win_rate": sum(1 for r in g if float(r["delta_psnr_best"]) > 0) / len(g),
            "delta_ssim_best_mean": mean(float(r["delta_ssim_best"]) for r in g),
            "delta_ssim_best_median": median(float(r["delta_ssim_best"]) for r in g),
            "ssim_win_rate": sum(1 for r in g if float(r["delta_ssim_best"]) > 0) / len(g),
            "delta_lpips_best_mean": mean(float(r["delta_lpips_best"]) for r in g),
            "delta_lpips_best_median": median(float(r["delta_lpips_best"]) for r in g),
            # For LPIPS lower is better, so win means delta < 0.
            "lpips_win_rate": sum(1 for r in g if float(r["delta_lpips_best"]) < 0) / len(g),
        })
    return rows


def failure_bin_summary(per_image_rows: list[dict]) -> list[dict]:
    rows = []
    groups = defaultdict(list)
    for r in per_image_rows:
        groups[(float(r["sigma"]), r["alignment_mode"])].append(r)

    for key, g in sorted(groups.items()):
        sigma, alignment = key
        sorted_g = sorted(g, key=lambda r: float(r["sitcom_psnr_best"]))
        bins = [
            ("bottom5_by_sitcom_psnr", sorted_g[:5]),
            ("middle15_by_sitcom_psnr", sorted_g[5:20]),
            ("top5_by_sitcom_psnr", sorted_g[20:25]),
        ]
        for bin_name, b in bins:
            rows.append({
                "sigma": sigma,
                "alignment_mode": alignment,
                "bin": bin_name,
                "n_images": len(b),
                "sitcom_psnr_best_mean": mean(float(r["sitcom_psnr_best"]) for r in b),
                "np_psnr_best_mean": mean(float(r["np_psnr_best"]) for r in b),
                "delta_psnr_best_mean": mean(float(r["delta_psnr_best"]) for r in b),
                "psnr_win_rate": sum(1 for r in b if float(r["delta_psnr_best"]) > 0) / len(b),
                "sitcom_ssim_best_mean": mean(float(r["sitcom_ssim_best"]) for r in b),
                "np_ssim_best_mean": mean(float(r["np_ssim_best"]) for r in b),
                "delta_ssim_best_mean": mean(float(r["delta_ssim_best"]) for r in b),
                "ssim_win_rate": sum(1 for r in b if float(r["delta_ssim_best"]) > 0) / len(b),
                "sitcom_lpips_best_mean": mean(float(r["sitcom_lpips_best"]) for r in b),
                "np_lpips_best_mean": mean(float(r["np_lpips_best"]) for r in b),
                "delta_lpips_best_mean": mean(float(r["delta_lpips_best"]) for r in b),
                "lpips_win_rate": sum(1 for r in b if float(r["delta_lpips_best"]) < 0) / len(b),
            })
    return rows


def write_markdown(out_path: Path, cond_comp: list[dict], bin_rows: list[dict]) -> None:
    lines = []
    lines.append("# ImageNet-25 NP vs SITCOM-ODE: all-metric comparison\n")
    lines.append("Metrics use image-level best-of-4 aggregation: PSNR/SSIM use max over runs; LPIPS uses min over runs. `flip_best` is metric-wise best over raw and rot180.\n")

    for alignment in ["raw", "flip_best"]:
        lines.append(f"## {alignment} condition-level differences\n")
        lines.append("| sigma | ΔPSNR ↑ | PSNR win | ΔSSIM ↑ | SSIM win | ΔLPIPS ↓ | LPIPS win |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in cond_comp:
            if r["alignment_mode"] != alignment:
                continue
            lines.append(
                f"| {float(r['sigma']):.3g} | "
                f"{float(r['delta_psnr_best_mean']):+.3f} | {100*float(r['psnr_win_rate']):.0f}% | "
                f"{float(r['delta_ssim_best_mean']):+.3f} | {100*float(r['ssim_win_rate']):.0f}% | "
                f"{float(r['delta_lpips_best_mean']):+.3f} | {100*float(r['lpips_win_rate']):.0f}% |"
            )
        lines.append("")

    lines.append("## Failure-bin summary, flip_best\n")
    lines.append("| sigma | bin | ΔPSNR ↑ | PSNR win | ΔSSIM ↑ | SSIM win | ΔLPIPS ↓ | LPIPS win |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in bin_rows:
        if r["alignment_mode"] != "flip_best":
            continue
        lines.append(
            f"| {float(r['sigma']):.3g} | {r['bin']} | "
            f"{float(r['delta_psnr_best_mean']):+.3f} | {100*float(r['psnr_win_rate']):.0f}% | "
            f"{float(r['delta_ssim_best_mean']):+.3f} | {100*float(r['ssim_win_rate']):.0f}% | "
            f"{float(r['delta_lpips_best_mean']):+.3f} | {100*float(r['lpips_win_rate']):.0f}% |"
        )
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np_root", type=Path, default=Path("/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_imagenet25_noise_sweep_B_fast"))
    ap.add_argument("--sitcom_run_level", type=Path, default=Path("/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/postprocess_sitcom_ode_imagenet25_flip/sitcom_ode_imagenet25_run_level_flip.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_vs_sitcom_all_metrics"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    np_runs = add_np_flip_best(load_np_run_level(args.np_root))
    sitcom_runs = load_sitcom_run_level(args.sitcom_run_level)

    run_all = sitcom_runs + np_runs
    image_all = summarize_image_level(run_all)
    cond_all = summarize_condition_level(image_all)
    per_image = compare_methods(image_all, alignments=("raw", "flip_best"))
    cond_comp = compare_condition(per_image)
    bin_comp = failure_bin_summary(per_image)

    write_csv(args.outdir / "all_methods_run_level_selected.csv", run_all)
    write_csv(args.outdir / "all_methods_image_level_metrics.csv", image_all)
    write_csv(args.outdir / "all_methods_condition_level_metrics.csv", cond_all)
    write_csv(args.outdir / "np_vs_sitcom_per_image_all_metrics.csv", per_image)
    write_csv(args.outdir / "np_vs_sitcom_condition_delta_all_metrics.csv", cond_comp)
    write_csv(args.outdir / "np_vs_sitcom_failure_bins_all_metrics.csv", bin_comp)
    write_markdown(args.outdir / "np_vs_sitcom_all_metrics_summary.md", cond_comp, bin_comp)

    print("\n[done] wrote outputs under:")
    print(args.outdir)

    print("\n=== condition deltas: NP - SITCOM, best-of-4 metrics ===")
    print("sigma, align, dPSNR, psnr_win, dSSIM, ssim_win, dLPIPS, lpips_win")
    for r in cond_comp:
        print(
            f"{float(r['sigma']):.3g}, {r['alignment_mode']}, "
            f"{float(r['delta_psnr_best_mean']):+.3f}, {100*float(r['psnr_win_rate']):.0f}%, "
            f"{float(r['delta_ssim_best_mean']):+.3f}, {100*float(r['ssim_win_rate']):.0f}%, "
            f"{float(r['delta_lpips_best_mean']):+.3f}, {100*float(r['lpips_win_rate']):.0f}%"
        )


if __name__ == "__main__":
    main()
