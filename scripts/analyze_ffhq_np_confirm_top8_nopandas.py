#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path("/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411")
CONFIRM_ROOT = ROOT / "np_ffhq_confirm_top8"

OUT_CONFIG = ROOT / "np_ffhq_confirm_top8_ranked_bestof4_summary.csv"
OUT_IMAGE = ROOT / "np_ffhq_confirm_top8_image_level_bestof4.csv"
OUT_RUNS = ROOT / "np_ffhq_confirm_top8_merged_run_level.csv"

TAG_RE = re.compile(
    r"top(?P<rank>\d+)_s(?P<score_label>[^_]+)_p(?P<proj_label>[^_]+)_"
    r"start(?P<proj_start>\d+)_soft(?P<soft_k>\d+)_hard(?P<hard_k>\d+)"
)


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[WARN] no rows to write: {path}")
        return

    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fget(row: dict, key: str, default: float = math.nan) -> float:
    try:
        val = row.get(key, "")
        if val == "" or val is None:
            return default
        return float(val)
    except Exception:
        return default


def safe_stdev(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    return stdev(vals)


def label_to_float(label: str) -> float:
    if label in {"full", "fullfull"}:
        return 0.72
    return float(label)


def find_tag(path: str) -> str:
    parts = Path(path).parts
    for p in parts:
        if p.startswith("top") and "_start" in p and "_soft" in p:
            return p
    return ""


def parse_tag(path: str) -> dict:
    tag = find_tag(path)
    m = TAG_RE.search(tag)
    if not m:
        return {
            "config_tag": tag,
            "top_rank_from_triage": "",
            "score_label": "",
            "proj_label": "",
            "proj_start": "",
            "soft_k": "",
            "hard_k": "",
        }
    d = m.groupdict()
    return {
        "config_tag": tag,
        "top_rank_from_triage": int(d["rank"]),
        "score_label": d["score_label"],
        "proj_label": d["proj_label"],
        "score_radius_from_tag": label_to_float(d["score_label"]),
        "proj_radius_from_tag": label_to_float(d["proj_label"]),
        "proj_start": int(d["proj_start"]),
        "soft_k": int(d["soft_k"]),
        "hard_k": int(d["hard_k"]),
    }


def load_runs() -> list[dict]:
    files = sorted(glob.glob(str(CONFIRM_ROOT / "**" / "run_level.csv"), recursive=True))
    print(f"Found run_level.csv files: {len(files)}")

    rows = []
    for path in files:
        meta = parse_tag(path)
        for row in read_csv(path):
            row.update(meta)
            row["source_file"] = path

            # Fill from row if available, otherwise from tag.
            row["score_radius"] = row.get("score_radius") or row.get("score_radius_from_tag", "")
            row["proj_radius"] = row.get("proj_radius") or row.get("proj_radius_from_tag", "")
            row["num_candidates_soft"] = row.get("num_candidates_soft") or row.get("soft_k", "")
            row["num_candidates_hard"] = row.get("num_candidates_hard") or row.get("hard_k", "")
            rows.append(row)
    return rows


def main() -> None:
    rows = load_runs()
    if not rows:
        raise RuntimeError(f"No run_level.csv files found under {CONFIRM_ROOT}")

    # This confirmation was intended to use rot180 only.
    rot_rows = [r for r in rows if r.get("alignment_mode") == "rot180"]
    print("Total rows:", len(rows))
    print("Rot180 rows:", len(rot_rows))

    # Group by config.
    config_cols = [
        "config_tag",
        "top_rank_from_triage",
        "score_label",
        "proj_label",
        "score_radius",
        "proj_radius",
        "proj_start",
        "soft_k",
        "hard_k",
        "num_candidates_soft",
        "num_candidates_hard",
        "alignment_mode",
        "oversample",
        "measurement_noise_std",
    ]

    by_config_image: dict[tuple, list[dict]] = defaultdict(list)
    by_config: dict[tuple, list[dict]] = defaultdict(list)

    for r in rot_rows:
        ckey = tuple(str(r.get(c, "")) for c in config_cols)
        img = r.get("image_basename", "")
        by_config[ckey].append(r)
        by_config_image[(ckey, img)].append(r)

    image_best_rows = []
    for (ckey, img), gr in by_config_image.items():
        best = max(gr, key=lambda r: fget(r, "psnr"))
        out = {c: ckey[i] for i, c in enumerate(config_cols)}
        out.update(
            {
                "image_basename": img,
                "best_seed": best.get("seed", ""),
                "best_psnr": fget(best, "psnr"),
                "best_ssim": fget(best, "ssim"),
                "runtime_s_for_best_seed": fget(best, "runtime_s"),
                "n_seeds": len(set(r.get("seed", "") for r in gr)),
                "psnr_mean_over_seeds": mean([fget(r, "psnr") for r in gr]),
                "psnr_median_over_seeds": median([fget(r, "psnr") for r in gr]),
                "psnr_min_over_seeds": min([fget(r, "psnr") for r in gr]),
                "psnr_max_over_seeds": max([fget(r, "psnr") for r in gr]),
            }
        )
        image_best_rows.append(out)

    # Summarize config using image-level best-of-4.
    image_best_by_config: dict[tuple, list[dict]] = defaultdict(list)
    for r in image_best_rows:
        ckey = tuple(str(r.get(c, "")) for c in config_cols)
        image_best_by_config[ckey].append(r)

    summary_rows = []
    for ckey, imgs in image_best_by_config.items():
        all_runs = by_config[ckey]
        psnr_b4 = [fget(r, "best_psnr") for r in imgs]
        ssim_b4 = [fget(r, "best_ssim") for r in imgs]
        psnr_all = [fget(r, "psnr") for r in all_runs]
        ssim_all = [fget(r, "ssim") for r in all_runs]
        runtimes = [fget(r, "runtime_s") for r in all_runs]
        seed_counts = sorted(set(r.get("seed", "") for r in all_runs))

        worst_img_row = min(imgs, key=lambda r: fget(r, "best_psnr"))
        best_img_row = max(imgs, key=lambda r: fget(r, "best_psnr"))

        out = {c: ckey[i] for i, c in enumerate(config_cols)}
        out.update(
            {
                "n_images": len(imgs),
                "n_runs": len(all_runs),
                "n_seeds": len(seed_counts),
                "seeds": ",".join(seed_counts),

                # Main best-of-4 metrics.
                "bestof4_psnr_mean": mean(psnr_b4),
                "bestof4_psnr_median": median(psnr_b4),
                "bestof4_psnr_min": min(psnr_b4),
                "bestof4_psnr_max": max(psnr_b4),
                "bestof4_psnr_std": safe_stdev(psnr_b4),
                "bestof4_ssim_mean": mean(ssim_b4),
                "bestof4_ssim_median": median(ssim_b4),
                "bestof4_ssim_min": min(ssim_b4),

                # All-run metrics, useful for stability.
                "allrun_psnr_mean": mean(psnr_all),
                "allrun_psnr_median": median(psnr_all),
                "allrun_psnr_min": min(psnr_all),
                "allrun_psnr_std": safe_stdev(psnr_all),
                "allrun_ssim_mean": mean(ssim_all),

                "runtime_s_mean_per_reconstruction": mean(runtimes),
                "runtime_s_median_per_reconstruction": median(runtimes),

                "worst_image": worst_img_row.get("image_basename", ""),
                "worst_image_bestof4_psnr": fget(worst_img_row, "best_psnr"),
                "best_image": best_img_row.get("image_basename", ""),
                "best_image_bestof4_psnr": fget(best_img_row, "best_psnr"),
            }
        )
        summary_rows.append(out)

    summary_rows.sort(
        key=lambda r: (
            fget(r, "bestof4_psnr_mean"),
            fget(r, "bestof4_psnr_median"),
            fget(r, "bestof4_psnr_min"),
            fget(r, "bestof4_ssim_mean"),
        ),
        reverse=True,
    )

    write_csv(OUT_CONFIG, summary_rows)
    write_csv(OUT_IMAGE, image_best_rows)
    write_csv(OUT_RUNS, rows)

    print("\n=== Top configs by image-level best-of-4 PSNR ===")
    cols = [
        "config_tag",
        "n_images",
        "n_runs",
        "bestof4_psnr_mean",
        "bestof4_psnr_median",
        "bestof4_psnr_min",
        "bestof4_psnr_std",
        "bestof4_ssim_mean",
        "allrun_psnr_mean",
        "runtime_s_mean_per_reconstruction",
        "worst_image",
        "worst_image_bestof4_psnr",
    ]
    print(",".join(cols))
    for r in summary_rows:
        print(",".join(str(r.get(c, "")) for c in cols))

    print("\nSaved:")
    print(" ", OUT_CONFIG)
    print(" ", OUT_IMAGE)
    print(" ", OUT_RUNS)


if __name__ == "__main__":
    main()
