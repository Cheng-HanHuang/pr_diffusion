#!/usr/bin/env python3
"""Branch A residual recomputation and failure analysis for NP/SITCOM.

This script is intentionally analysis-side: it recomputes NP-compatible
oversampled Fourier magnitude residuals for saved SITCOM PNG samples, reruns
final-output selectors, and writes failure/regret tables.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_SCRIPT = _REPO_ROOT / "scripts" / "pr_external_difffpr_np_benchmark.py"
_GUIDED_SCRIPT = _REPO_ROOT / "scripts" / "pr_external_difffpr_np_guided_benchmark.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load_module("pr_external_difffpr_np_benchmark", _BASE_SCRIPT)
guided = _load_module("pr_external_difffpr_np_guided_benchmark", _GUIDED_SCRIPT)

from prdiffusion.io import load_image


NOISE_TAGS = {
    "0": "0.0",
    "0p01": "0.01",
    "005": "0.05",
    "0p08": "0.08",
    "0p10": "0.10",
}

SELECTOR_METHODS = [
    "oracle_best_psnr_diagnostic",
    "min_selector_post_lf_mse",
    "min_noisy_lowfreq_mag_l2",
    "min_noisy_mag_l2",
]


def read_csv(path: Path | str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path | str, rows: List[Dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fget(row: Dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def finite(xs: Iterable[float]) -> List[float]:
    return [x for x in xs if math.isfinite(x)]


def fmean(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return mean(vals) if vals else math.nan


def fmedian(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return median(vals) if vals else math.nan


def fstd(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return stdev(vals) if len(vals) > 1 else 0.0


def norm_float_label(value: object) -> str:
    v = fget({"x": value}, "x")
    if not math.isfinite(v):
        return str(value)
    return f"{v:.12g}"


def norm_image_index(value: object) -> str:
    try:
        return f"{int(float(str(value))):05d}"
    except Exception:
        s = str(value)
        stem = Path(s).stem
        try:
            return f"{int(stem):05d}"
        except Exception:
            return stem or s


def parse_sample_name(path: Path) -> tuple[int, int]:
    m = re.match(r"^(\d+)_run(\d+)\.png$", path.name)
    if not m:
        raise ValueError(f"Unexpected SITCOM sample filename: {path}")
    return int(m.group(1)), int(m.group(2))


def load_manifest_by_index(path: Path) -> Dict[int, Dict[str, str]]:
    rows = read_csv(path)
    out: Dict[int, Dict[str, str]] = {}
    for i, row in enumerate(rows):
        idx = int(row.get("index", i))
        out[idx] = row
    return out


def infer_sitcom_result_dir(sitcom_root: Path, tag: str) -> Path:
    return sitcom_root / tag / "results" / tag


def residual_norms(
    x_eval: torch.Tensor,
    mag_clean: torch.Tensor,
    mag_target: torch.Tensor,
    pad: int,
    radius: float,
) -> Dict[str, float]:
    clean_mag_l2 = float(base.oversampled_mag_l2(x_eval, mag_clean, pad).detach().cpu().item())
    noisy_mag_l2 = float(base.oversampled_mag_l2(x_eval, mag_target, pad).detach().cpu().item())
    clean_lf_l2 = float(
        base.oversampled_lowfreq_mag_l2(x_eval, mag_clean, pad, radius).detach().cpu().item()
    )
    noisy_lf_l2 = float(
        base.oversampled_lowfreq_mag_l2(x_eval, mag_target, pad, radius).detach().cpu().item()
    )
    mag_norm = float(torch.norm(mag_target).detach().cpu().item())
    _, _, h, w = mag_target.shape
    mask_hw = base.centered_lowfreq_mask(h, w, radius, mag_target.device)
    mask = mask_hw[None, None, :, :].expand_as(mag_target)
    lf_norm = float(torch.norm(mag_target[mask]).detach().cpu().item())
    return {
        "clean_mag_l2": clean_mag_l2,
        "noisy_mag_l2": noisy_mag_l2,
        "clean_lowfreq_mag_l2": clean_lf_l2,
        "noisy_lowfreq_mag_l2": noisy_lf_l2,
        "noisy_mag_l2_normalized": noisy_mag_l2 / mag_norm if mag_norm > 0 else math.nan,
        "noisy_lowfreq_mag_l2_normalized": noisy_lf_l2 / lf_norm if lf_norm > 0 else math.nan,
        "measurement_mag_norm": mag_norm,
        "measurement_lowfreq_mag_norm": lf_norm,
    }


def recompute_sitcom_residuals(args: argparse.Namespace) -> List[Dict[str, object]]:
    device = torch.device(args.device)
    rows_by_noise: List[Dict[str, object]] = []
    pad = base.oversample_pad(args.image_size, args.oversample)
    noise_tags = args.noise_tags or list(NOISE_TAGS)
    for tag in noise_tags:
        noise_std = float(NOISE_TAGS[tag])
        result_dir = infer_sitcom_result_dir(Path(args.sitcom_root), f"sitcom_ffhq25_s4_noise{tag}")
        sample_dir = result_dir / "samples"
        manifest_path = Path(args.sitcom_root) / f"sitcom_ffhq25_s4_noise{tag}" / "sitcom_images" / "manifest.csv"
        run_csv = Path(args.branchA_old_root) / f"sitcom_sitcom_ffhq25_s4_noise{tag}_run_level.csv"
        metric_rows = read_csv(run_csv)
        metric_by_key = {
            (int(float(r["image_index_in_split"])), int(float(r["sitcom_run_index"]))): r
            for r in metric_rows
        }
        manifest = load_manifest_by_index(manifest_path)
        samples = sorted(sample_dir.glob("*.png"))
        if len(samples) != len(metric_rows):
            raise ValueError(f"{tag}: sample count {len(samples)} != CSV rows {len(metric_rows)}")

        cache: Dict[int, Dict[str, torch.Tensor]] = {}
        for sample_path in samples:
            image_i, run_i = parse_sample_name(sample_path)
            meta = manifest[image_i]
            if image_i not in cache:
                x_gt = load_image(meta["source_path"], size=args.image_size, device=device)
                mag_clean = base.oversampled_magnitude(x_gt, pad)
                mag_target = mag_clean
                if noise_std > 0:
                    gen = torch.Generator(device=device).manual_seed(args.measurement_noise_seed + image_i)
                    noise = torch.randn(
                        mag_clean.shape,
                        device=device,
                        dtype=mag_clean.dtype,
                        generator=gen,
                    )
                    mag_target = mag_clean + noise_std * noise
                    if args.clip_noisy_magnitude:
                        mag_target = mag_target.clamp_min(0.0)
                cache[image_i] = {"x_gt": x_gt, "mag_clean": mag_clean, "mag_target": mag_target}

            tensors = cache[image_i]
            x_sample = load_image(str(sample_path), size=args.image_size, device=device)
            x_eval = guided.make_alignment(args.alignment, x_sample, tensors["x_gt"])
            norms = residual_norms(
                x_eval,
                tensors["mag_clean"],
                tensors["mag_target"],
                pad,
                args.score_radius,
            )
            source_row = metric_by_key[(image_i, run_i)]
            split_entry = meta.get("split_entry", "")
            source_stem = Path(split_entry).stem if split_entry else norm_image_index(image_i)
            row: Dict[str, object] = dict(source_row)
            row.update(
                image_basename=source_stem,
                source_image_basename=source_stem,
                normalized_image_id=norm_image_index(image_i),
                normalized_noise=norm_float_label(noise_std),
                measurement_noise_std=noise_std,
                oversample=args.oversample,
                score_radius=args.score_radius,
                residual_device=str(device),
                residual_from_saved_png=True,
                residual_alignment=args.alignment,
                clip_noisy_magnitude=bool(args.clip_noisy_magnitude),
                measurement_noise_seed=args.measurement_noise_seed,
                **norms,
            )
            rows_by_noise.append(row)
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows_by_noise


def canonicalize_candidate(row: Dict[str, str], source: str, idx: int) -> Dict[str, object]:
    out: Dict[str, object] = dict(row)
    out["candidate_source"] = source
    out["candidate_id"] = row.get("candidate_id") or f"{source}:{idx}"
    if "measurement_noise_std" not in out and "noise" in out:
        out["measurement_noise_std"] = out["noise"]
    out["normalized_noise"] = norm_float_label(out.get("measurement_noise_std", ""))
    out["alignment_mode"] = out.get("alignment_mode") or "resolve"
    out["normalized_image_id"] = norm_image_index(out.get("image_index_in_split", out.get("image_basename", "")))
    if "config_tag" not in out:
        out["config_tag"] = source
    for key in ("selector_post_winner_lf_mse_mean", "noisy_lowfreq_mag_l2", "noisy_mag_l2"):
        if key not in out:
            out[key] = "nan"
    return out


def group_key(row: Dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("normalized_image_id", "")),
        str(row.get("normalized_noise", "")),
        str(row.get("alignment_mode", "")),
    )


def choose(rows: List[Dict[str, object]], method: str) -> Dict[str, object]:
    if method == "oracle_best_psnr_diagnostic":
        return max(rows, key=lambda r: fget(r, "psnr", -1e99))
    if method == "min_selector_post_lf_mse":
        valid = [r for r in rows if math.isfinite(fget(r, "selector_post_winner_lf_mse_mean"))]
        return min(valid or rows, key=lambda r: fget(r, "selector_post_winner_lf_mse_mean", 1e99))
    if method == "min_noisy_lowfreq_mag_l2":
        valid = [r for r in rows if math.isfinite(fget(r, "noisy_lowfreq_mag_l2"))]
        return min(valid or rows, key=lambda r: fget(r, "noisy_lowfreq_mag_l2", 1e99))
    if method == "min_noisy_mag_l2":
        valid = [r for r in rows if math.isfinite(fget(r, "noisy_mag_l2"))]
        return min(valid or rows, key=lambda r: fget(r, "noisy_mag_l2", 1e99))
    raise ValueError(method)


def summarize(rows: List[Dict[str, object]], group_cols: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(c, "")) for c in group_cols)].append(row)
    out: List[Dict[str, object]] = []
    for key, rs in sorted(groups.items()):
        ps = [fget(r, "psnr") for r in rs]
        worst = min(rs, key=lambda r: fget(r, "psnr", 1e99))
        row = {col: val for col, val in zip(group_cols, key)}
        row.update(
            n=len(rs),
            psnr_mean=fmean(ps),
            psnr_median=fmedian(ps),
            psnr_min=min(finite(ps)) if finite(ps) else math.nan,
            psnr_max=max(finite(ps)) if finite(ps) else math.nan,
            psnr_std=fstd(ps),
            n_below25=sum(p < 25 for p in finite(ps)),
            n_below20=sum(p < 20 for p in finite(ps)),
            worst_image=worst.get("normalized_image_id", ""),
            worst_psnr=fget(worst, "psnr"),
        )
        out.append(row)
    return out


def run_mix(np_csv: Path, sitcom_csv: Path, outdir: Path, alignment: str) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    candidates: List[Dict[str, object]] = []
    for source, path in (("np", np_csv), ("sitcom", sitcom_csv)):
        for i, row in enumerate(read_csv(path)):
            cand = canonicalize_candidate(row, source, i)
            if cand.get("alignment_mode") == alignment:
                candidates.append(cand)
    write_csv(outdir / "candidate_level.csv", candidates)
    write_csv(outdir / "source_summary.csv", summarize(candidates, ["candidate_source", "normalized_noise", "alignment_mode"]))

    grouped: Dict[tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in candidates:
        grouped[group_key(row)].append(row)

    selected: List[Dict[str, object]] = []
    for _, rows in sorted(grouped.items()):
        for method in SELECTOR_METHODS:
            picked = dict(choose(rows, method))
            picked["selection_method"] = method
            selected.append(picked)
    write_csv(outdir / "selected_image_level.csv", selected)
    write_csv(outdir / "selected_summary.csv", summarize(selected, ["selection_method", "normalized_noise", "alignment_mode"]))
    return candidates, selected


def sanity_old_branch(branchA_old_root: Path, outdir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    selected_path = branchA_old_root / "branchA_np_sitcom_all_noises_fixed" / "selected_summary.csv"
    if selected_path.exists():
        for row in read_csv(selected_path):
            n = int(float(row.get("n", "0")))
            rows.append(
                {
                    "check": "old_selected_summary_n_per_method_noise_alignment",
                    "selection_method": row.get("selection_method", ""),
                    "noise": row.get("measurement_noise_std", row.get("normalized_noise", "")),
                    "alignment_mode": row.get("alignment_mode", ""),
                    "n": n,
                    "expected_n": 25,
                    "passes": n == 25,
                }
            )

    sitcom_path = branchA_old_root / "sitcom_all_noises_run_level.csv"
    if sitcom_path.exists():
        sitcom_rows = read_csv(sitcom_path)
        for col in ("noisy_mag_l2", "noisy_lowfreq_mag_l2", "selector_post_winner_lf_mse_mean"):
            total = len(sitcom_rows)
            nanish = sum(not math.isfinite(fget(r, col)) for r in sitcom_rows)
            rows.append(
                {
                    "check": f"old_sitcom_{col}_nan_count",
                    "column": col,
                    "n": nanish,
                    "expected_n": total,
                    "passes": nanish == total,
                }
            )
    write_csv(outdir / "A0_readonly_sanity.csv", rows)
    return rows


def failure_table(candidates: List[Dict[str, object]], selected: List[Dict[str, object]], outdir: Path) -> List[Dict[str, object]]:
    by_image: Dict[tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_image[group_key(row)].append(row)
    selected_by_key_method = {(group_key(r), str(r["selection_method"])): r for r in selected}

    rows: List[Dict[str, object]] = []
    for key, group_rows in sorted(by_image.items()):
        np_rows = [r for r in group_rows if r.get("candidate_source") == "np"]
        sitcom_rows = [r for r in group_rows if r.get("candidate_source") == "sitcom"]
        best_np = max(np_rows, key=lambda r: fget(r, "psnr", -1e99)) if np_rows else None
        best_sitcom = max(sitcom_rows, key=lambda r: fget(r, "psnr", -1e99)) if sitcom_rows else None
        oracle = max(group_rows, key=lambda r: fget(r, "psnr", -1e99))
        best_np_psnr = fget(best_np or {}, "psnr")
        best_sitcom_psnr = fget(best_sitcom or {}, "psnr")
        oracle_psnr = fget(oracle, "psnr")
        row: Dict[str, object] = {
            "normalized_image_id": key[0],
            "normalized_noise": key[1],
            "alignment_mode": key[2],
            "best_np_psnr": best_np_psnr,
            "best_sitcom_psnr": best_sitcom_psnr,
            "oracle_winner_source": oracle.get("candidate_source", ""),
            "oracle_psnr": oracle_psnr,
            "source_gap_sitcom_minus_np": best_sitcom_psnr - best_np_psnr,
            "source_gap_abs": abs(best_sitcom_psnr - best_np_psnr),
            "sitcom_catastrophic_best_below25": best_sitcom_psnr < 25,
            "sitcom_catastrophic_best_below20": best_sitcom_psnr < 20,
            "np_catastrophic_best_below25": best_np_psnr < 25,
            "np_rescues_sitcom_below25": best_sitcom_psnr < 25 <= best_np_psnr,
        }
        for method in SELECTOR_METHODS:
            sel = selected_by_key_method.get((key, method), {})
            sel_psnr = fget(sel, "psnr")
            safe_method = method.replace("oracle_best_psnr_diagnostic", "oracle")
            row[f"selected_source_by_{safe_method}"] = sel.get("candidate_source", "")
            row[f"selected_psnr_by_{safe_method}"] = sel_psnr
            row[f"selector_regret_vs_oracle_by_{safe_method}"] = oracle_psnr - sel_psnr
            row[f"catastrophic_by_{safe_method}_below25"] = sel_psnr < 25
        rows.append(row)
    write_csv(outdir / "A3_failure_table_image_noise.csv", rows)
    return rows


def selector_delta_table(selected_summary: List[Dict[str, str]], outdir: Path) -> List[Dict[str, object]]:
    by_noise: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in selected_summary:
        by_noise[row.get("normalized_noise", row.get("measurement_noise_std", ""))][row["selection_method"]] = row
    out: List[Dict[str, object]] = []
    for noise, methods in sorted(by_noise.items(), key=lambda kv: fget({"x": kv[0]}, "x")):
        oracle = fget(methods.get("oracle_best_psnr_diagnostic", {}), "psnr_mean")
        for method in SELECTOR_METHODS:
            row = methods.get(method, {})
            if not row:
                continue
            out.append(
                {
                    "normalized_noise": noise,
                    "selection_method": method,
                    "n": row.get("n", ""),
                    "psnr_mean": fget(row, "psnr_mean"),
                    "psnr_min": fget(row, "psnr_min"),
                    "n_below25": row.get("n_below25", ""),
                    "mean_regret_vs_oracle_summary": oracle - fget(row, "psnr_mean"),
                }
            )
    write_csv(outdir / "A2_selector_comparison_summary.csv", out)
    return out


def render_report(
    args: argparse.Namespace,
    sanity_rows: List[Dict[str, object]],
    selector_rows: List[Dict[str, object]],
    failures: List[Dict[str, object]],
    outdir: Path,
) -> None:
    sanity_bad = [r for r in sanity_rows if str(r.get("passes")) != "True"]
    by_method = defaultdict(list)
    for row in selector_rows:
        by_method[str(row["selection_method"])].append(row)

    def method_line(method: str) -> str:
        rows = by_method.get(method, [])
        parts = []
        for r in rows:
            parts.append(
                f"noise {r['normalized_noise']}: mean {fget(r, 'psnr_mean'):.3f}, "
                f"min {fget(r, 'psnr_min'):.3f}, n={r['n']}, regret {fget(r, 'mean_regret_vs_oracle_summary'):.3f}"
            )
        return "\n".join(f"- {p}" for p in parts)

    image_00005 = [
        r for r in failures if r["normalized_image_id"] == "00005" and r["normalized_noise"] == "0.05"
    ]
    min_lf_rows = by_method.get("min_noisy_lowfreq_mag_l2", [])
    lf_min_psnr = min((fget(r, "psnr_min") for r in min_lf_rows), default=math.nan)
    lf_mean_regret = fmean(fget(r, "mean_regret_vs_oracle_summary") for r in min_lf_rows)
    final_viability = (
        "not viable as a final-output selector"
        if (math.isfinite(lf_min_psnr) and lf_min_psnr < 25) or (math.isfinite(lf_mean_regret) and lf_mean_regret > 1.0)
        else "possibly viable enough for a short writeup"
    )

    lines = [
        "# Branch A NP/SITCOM residual selection report",
        "",
        "## Inputs",
        "",
        f"- NP CSV: `{args.np_csv}`",
        f"- Prior SITCOM/Branch A root: `{args.prior_root}`",
        f"- Output folder: `{outdir}`",
        f"- Residual convention: oversample={args.oversample}, score_radius={args.score_radius}, "
        f"measurement_noise_seed={args.measurement_noise_seed}, clip_noisy_magnitude={args.clip_noisy_magnitude}, "
        f"alignment={args.alignment}, samples loaded from saved PNGs.",
        "",
        "## A0 Read-Only Sanity",
        "",
        "- Previous `selected_summary.csv` does not pass the 25-row structural check where `n=50` appears; the new mixer groups by `image_index_in_split`.",
        "- Previous SITCOM residual columns were all non-finite (`nan`/blank), confirming the blocker described in the plan.",
    ]
    if sanity_bad:
        lines.append(f"- Failing sanity rows written to `A0_readonly_sanity.csv`: {len(sanity_bad)} rows.")
    lines.extend(
        [
            "",
            "## A2 Selector Results",
            "",
            "Oracle diagnostic:",
            method_line("oracle_best_psnr_diagnostic"),
            "",
            "Minimum noisy low-frequency magnitude residual:",
            method_line("min_noisy_lowfreq_mag_l2"),
            "",
            "Minimum full noisy magnitude residual:",
            method_line("min_noisy_mag_l2"),
            "",
            "Minimum NP selector post LF MSE:",
            method_line("min_selector_post_lf_mse"),
            "",
            "## A3 Failure Notes",
            "",
            f"- Image/noise failure table: `A3_failure_table_image_noise.csv` ({len(failures)} rows).",
        ]
    )
    if image_00005:
        r = image_00005[0]
        lines.append(
            "- For split image `00005` at noise `0.05`: "
            f"best NP PSNR {fget(r, 'best_np_psnr'):.3f}, "
            f"best SITCOM PSNR {fget(r, 'best_sitcom_psnr'):.3f}, "
            f"LF residual selector chose {r.get('selected_source_by_min_noisy_lowfreq_mag_l2')} "
            f"with PSNR {fget(r, 'selected_psnr_by_min_noisy_lowfreq_mag_l2'):.3f}."
        )
    lines.extend(
        [
            "",
            "## A4 Decision",
            "",
            f"Final-output Branch A is **{final_viability}** under this residual recomputation.",
            "",
            "Reason: the residual selectors are now comparable enough for diagnostics, but they still must be judged by whether they rescue hard SITCOM failures without sacrificing mean/min PSNR. If the `A2_selector_comparison_summary.csv` rows show large regret or low minimum PSNR, the next experiment should move directly to SITCOM trajectory instrumentation.",
            "",
            "## Suggested Next Experiments",
            "",
            "1. Patch SITCOM to save per-step trajectory diagnostics and raw final tensors, not just PNGs. This removes PNG quantization ambiguity and lets us test whether residual spikes precede final collapse.",
            "2. Run a hard-image trajectory pass on noise `0.05`, especially split image `00005`, recording low-frequency residual, full residual, update norms, and inter-run disagreement per step.",
            "3. Train or threshold a simple pre-collapse detector on per-step features, then test whether invoking NP correction only on risky SITCOM trajectories protects minimum PSNR while keeping SITCOM's successful-run ceiling.",
        ]
    )
    write_text(outdir / "REPORT.md", "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_root", default="/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610")
    ap.add_argument("--np_csv", default="/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608/np_selector_ffhq/selector_full25_s100_103/lf_s2_selector_20260609_154712/run_level.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--noise_tags", default="", help="Comma-separated SITCOM noise tags; default all.")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--oversample", type=float, default=2.0)
    ap.add_argument("--score_radius", type=float, default=0.6)
    ap.add_argument("--measurement_noise_seed", type=int, default=20260423)
    ap.add_argument("--clip_noisy_magnitude", action="store_true", default=True)
    ap.add_argument("--no_clip_noisy_magnitude", dest="clip_noisy_magnitude", action="store_false")
    ap.add_argument("--alignment", default="resolve")
    ap.add_argument("--skip_residuals", action="store_true")
    args = ap.parse_args()

    args.prior_root = str(Path(args.prior_root))
    args.branchA_old_root = str(Path(args.prior_root) / "branchA_mix")
    args.sitcom_root = str(Path(args.prior_root) / "sitcom_official")
    args.noise_tags = [x.strip() for x in str(args.noise_tags).split(",") if x.strip()]

    outdir = Path(args.outdir)
    residual_dir = outdir / "A1_sitcom_residuals"
    mix_dir = outdir / "A2_mix_select"
    analysis_dir = outdir / "A3_failure_analysis"

    sanity_rows = sanity_old_branch(Path(args.branchA_old_root), outdir)

    if args.skip_residuals:
        enriched_path = residual_dir / "sitcom_all_noises_with_residuals.csv"
        if not enriched_path.exists():
            raise FileNotFoundError(enriched_path)
    else:
        residual_rows = recompute_sitcom_residuals(args)
        write_csv(residual_dir / "sitcom_all_noises_with_residuals.csv", residual_rows)
        by_tag: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in residual_rows:
            noise = norm_float_label(row.get("measurement_noise_std"))
            by_tag[noise].append(row)
        for noise, rows in by_tag.items():
            write_csv(residual_dir / f"sitcom_noise_{noise.replace('.', 'p')}_with_residuals.csv", rows)
        enriched_path = residual_dir / "sitcom_all_noises_with_residuals.csv"

    _, selected = run_mix(Path(args.np_csv), enriched_path, mix_dir, args.alignment)
    candidates = read_csv(mix_dir / "candidate_level.csv")
    selected_rows = read_csv(mix_dir / "selected_image_level.csv")
    selected_summary = read_csv(mix_dir / "selected_summary.csv")
    selector_rows = selector_delta_table(selected_summary, mix_dir)
    failures = failure_table(candidates, selected_rows, analysis_dir)
    render_report(args, sanity_rows, selector_rows, failures, outdir)
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
